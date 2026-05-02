from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SOURCE_FILE = DATA_DIR / "ecommerce_customer_churn_dataset.csv"
EVENTS_FILE = DATA_DIR / "fintech_user_events.csv"
PROCESSED_FILE = DATA_DIR / "processed_fintech_data.csv"

RNG = np.random.default_rng(42)


PRODUCTS = [
    "payments",
    "savings",
    "credit",
    "insurance",
    "investments",
    "billpay",
]
CHANNELS = ["app", "web"]


def _hash_user_id(idx: int, salt: str = "insightforge_fintech") -> str:
    raw = f"{salt}_{idx}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _to_rate_0_1(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").fillna(0)
    if x.max() > 1:
        x = x / 100.0
    return x.clip(0, 1)


def _safe_qcut_int(series: pd.Series, q: int, labels: list[int], use_rank: bool = False) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if use_rank:
        s = s.rank(method="first")
    if s.notna().sum() == 0:
        return pd.Series([labels[len(labels) // 2]] * len(s), index=s.index, dtype="int64")
    try:
        out = pd.qcut(s, q=q, labels=labels, duplicates="drop")
        if out.isna().any():
            out = out.astype("object").fillna(labels[len(labels) // 2])
        return out.astype(int)
    except Exception:
        return pd.Series([labels[len(labels) // 2]] * len(s), index=s.index, dtype="int64")


def _assign_lifecycle_stage(days_since_last_active: float, tenure_days: float, events_30d: float) -> str:
    if tenure_days <= 30:
        return "New"
    if days_since_last_active > 60:
        return "Dormant"
    if days_since_last_active > 30:
        return "At Risk"
    if events_30d >= 12:
        return "Active"
    return "Engaged"


def _assign_product_mix_segment(n_products_used: float, product_diversity: float) -> str:
    if n_products_used >= 4 and product_diversity >= 1.2:
        return "Power User"
    if n_products_used >= 3:
        return "Multi Product"
    if n_products_used == 2:
        return "Dual Product"
    return "Single Product"


def _assign_rfm_segment(rfm_score: float) -> str:
    rfm = int(round(rfm_score))
    if rfm >= 10:
        return "High Value"
    if rfm >= 7:
        return "Potential Loyalists"
    if rfm >= 5:
        return "At Risk"
    return "Low Value"


def build_event_dataset(df: pd.DataFrame) -> pd.DataFrame:
    event_rows: list[dict] = []
    base_date = pd.Timestamp("2026-04-01")

    for i, row in df.reset_index(drop=True).iterrows():
        def _num(name: str, default: float) -> float:
            v = pd.to_numeric(row.get(name, default), errors="coerce")
            return float(default if pd.isna(v) else v)

        user_id = _hash_user_id(i)
        membership_years = max(_num("Membership_Years", 1.0), 0.1)
        logins = max(int(round(_num("Login_Frequency", 5))), 1)
        purchases = max(int(round(_num("Total_Purchases", 3))), 1)
        support_calls = max(int(round(_num("Customer_Service_Calls", 1))), 0)
        reviews = max(int(round(_num("Product_Reviews_Written", 1))), 0)
        days_since_last = max(int(round(_num("Days_Since_Last_Purchase", 30))), 0)
        churned = int(round(_num("Churned", 0)))
        avg_order = max(_num("Average_Order_Value", 50), 1.0)

        total_events = max(logins + purchases + support_calls + reviews, 5)
        start_days_ago = int(membership_years * 365)

        product_count = int(np.clip(1 + purchases // 8 + logins // 20, 1, len(PRODUCTS)))
        user_products = RNG.choice(PRODUCTS, size=product_count, replace=False)

        for e in range(total_events):
            if e < logins:
                event_type = "login"
                amount = 0.0
            elif e < logins + purchases:
                event_type = "txn"
                amount = float(np.clip(RNG.normal(avg_order, avg_order * 0.35), 1.0, None))
            elif e < logins + purchases + support_calls:
                event_type = "support_ticket"
                amount = 0.0
            else:
                event_type = "engagement_event"
                amount = 0.0

            low = max(min(days_since_last, start_days_ago), 1)
            high = max(start_days_ago + 1, low + 1)
            days_back = int(RNG.integers(low, high))
            if churned == 1:
                days_back = int(days_back + RNG.integers(5, 40))

            event_rows.append(
                {
                    "user_id": user_id,
                    "event_time": base_date - pd.Timedelta(days=days_back),
                    "event_type": event_type,
                    "product_category": str(RNG.choice(user_products)),
                    "amount": round(amount, 2),
                    "channel": str(RNG.choice(CHANNELS, p=[0.75, 0.25])),
                    "partner_id": f"partner_{(i % 12) + 1:02d}",
                    "device_type": str(RNG.choice(["android", "ios", "web"], p=[0.38, 0.37, 0.25])),
                    "geo_bucket": str(row.get("Country", "unknown")).lower().replace(" ", "_"),
                    "churned_label": churned,
                }
            )

    events = pd.DataFrame(event_rows).sort_values(["user_id", "event_time"]).reset_index(drop=True)
    return events


def build_processed_fintech(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    data = df.copy().reset_index(drop=True)
    data["user_id"] = [_hash_user_id(i) for i in range(len(data))]

    data["Discount_Usage_Rate"] = _to_rate_0_1(data["Discount_Usage_Rate"])
    data["Returns_Rate"] = _to_rate_0_1(data["Returns_Rate"])
    data["Email_Open_Rate"] = _to_rate_0_1(data["Email_Open_Rate"])
    data["Cart_Abandonment_Rate"] = _to_rate_0_1(data["Cart_Abandonment_Rate"])

    data["recency_score"] = _safe_qcut_int(data["Days_Since_Last_Purchase"], 4, [4, 3, 2, 1], use_rank=False)
    data["frequency_score"] = _safe_qcut_int(data["Total_Purchases"], 4, [1, 2, 3, 4], use_rank=True)
    data["monetary_score"] = _safe_qcut_int(data["Average_Order_Value"], 4, [1, 2, 3, 4], use_rank=False)
    data["RFM_score"] = data["recency_score"] + data["frequency_score"] + data["monetary_score"]
    data["rfm_segment"] = data["RFM_score"].apply(_assign_rfm_segment)

    data["tenure_days"] = (pd.to_numeric(data["Membership_Years"], errors="coerce").fillna(0) * 365).round().clip(lower=1)
    data["days_since_last_active"] = pd.to_numeric(data["Days_Since_Last_Purchase"], errors="coerce").fillna(0).clip(lower=0)
    data["events_last_30d"] = (pd.to_numeric(data["Login_Frequency"], errors="coerce").fillna(0) * 0.55).round().clip(lower=0)
    data["events_prev_30d"] = (pd.to_numeric(data["Login_Frequency"], errors="coerce").fillna(0) * 0.45).round().clip(lower=0)
    data["activity_trend"] = (data["events_last_30d"] - data["events_prev_30d"]) / (data["events_prev_30d"] + 1)

    data["lifecycle_stage"] = data.apply(
        lambda r: _assign_lifecycle_stage(r["days_since_last_active"], r["tenure_days"], r["events_last_30d"]),
        axis=1,
    )

    product_agg = (
        events.groupby("user_id")
        .agg(
            n_products_used=("product_category", "nunique"),
            dominant_product=("product_category", lambda s: s.mode().iloc[0] if not s.mode().empty else "payments"),
            total_events=("event_type", "count"),
            txn_events=("event_type", lambda s: int((s == "txn").sum())),
            total_amount=("amount", "sum"),
        )
        .reset_index()
    )

    merged = data.merge(product_agg, on="user_id", how="left")
    merged["n_products_used"] = merged["n_products_used"].fillna(1).astype(int)
    merged["total_events"] = merged["total_events"].fillna(0)
    merged["txn_events"] = merged["txn_events"].fillna(0)
    merged["total_amount"] = merged["total_amount"].fillna(0.0)
    merged["product_diversity"] = merged["n_products_used"] / np.log1p(merged["total_events"] + 1)
    merged["product_mix_segment"] = merged.apply(
        lambda r: _assign_product_mix_segment(r["n_products_used"], r["product_diversity"]),
        axis=1,
    )

    merged["engagement_score"] = (
        pd.to_numeric(merged["Login_Frequency"], errors="coerce").fillna(0)
        + pd.to_numeric(merged["Session_Duration_Avg"], errors="coerce").fillna(0)
        + pd.to_numeric(merged["Pages_Per_Session"], errors="coerce").fillna(0)
        + pd.to_numeric(merged["Email_Open_Rate"], errors="coerce").fillna(0)
        + pd.to_numeric(merged["Social_Media_Engagement_Score"], errors="coerce").fillna(0)
    ) / 5

    merged["risk_score"] = (
        merged["Cart_Abandonment_Rate"] + merged["Returns_Rate"] + pd.to_numeric(merged["Customer_Service_Calls"], errors="coerce").fillna(0)
    ) / 3
    safe_total_purchases = pd.to_numeric(merged["Total_Purchases"], errors="coerce").fillna(0).clip(lower=0)
    merged["support_intensity"] = pd.to_numeric(merged["Customer_Service_Calls"], errors="coerce").fillna(0) / (
        safe_total_purchases + 1
    )

    merged["churned_next_30d"] = pd.to_numeric(merged["Churned"], errors="coerce").fillna(0).astype(int)
    merged["segment_key"] = (
        merged["rfm_segment"].astype(str)
        + " | "
        + merged["lifecycle_stage"].astype(str)
        + " | "
        + merged["product_mix_segment"].astype(str)
    )

    keep_cols = [
        "user_id",
        "partner_id",
        "Country",
        "Age",
        "Gender",
        "rfm_segment",
        "RFM_score",
        "recency_score",
        "frequency_score",
        "monetary_score",
        "lifecycle_stage",
        "tenure_days",
        "days_since_last_active",
        "events_last_30d",
        "events_prev_30d",
        "activity_trend",
        "product_mix_segment",
        "n_products_used",
        "product_diversity",
        "dominant_product",
        "txn_events",
        "total_amount",
        "engagement_score",
        "risk_score",
        "support_intensity",
        "segment_key",
        "churned_next_30d",
    ]

    # partner_id comes from event data; populate through first event record if missing after merge
    partner_map = events.groupby("user_id")["partner_id"].first().reset_index()
    merged = merged.merge(partner_map, on="user_id", how="left")
    merged["partner_id"] = merged["partner_id"].fillna("partner_00")

    out = merged[keep_cols].copy()
    return out


def main() -> None:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Missing source file: {SOURCE_FILE}")

    base_df = pd.read_csv(SOURCE_FILE)
    events_df = build_event_dataset(base_df)
    processed_df = build_processed_fintech(base_df, events_df)

    events_df.to_csv(EVENTS_FILE, index=False)
    processed_df.to_csv(PROCESSED_FILE, index=False)

    print(f"Created: {EVENTS_FILE} ({len(events_df):,} rows)")
    print(f"Created: {PROCESSED_FILE} ({len(processed_df):,} rows)")


if __name__ == "__main__":
    main()
