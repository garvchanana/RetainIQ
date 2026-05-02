import pandas as pd
import numpy as np


def _safe_qcut(series: pd.Series, q: int, labels: list[int], use_rank: bool = False) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")

    if use_rank:
        s = s.rank(method="first")

    if s.notna().sum() == 0:
        return pd.Series([labels[len(labels)//2]] * len(s), index=s.index)

    try:
        return pd.qcut(s, q=q, labels=labels, duplicates="drop").astype(int)
    except Exception:
        return pd.Series([labels[len(labels)//2]] * len(s), index=s.index)


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # -----------------------------
    # Skew handling
    # -----------------------------
    skew_cols = ["Total_Purchases", "Lifetime_Value", "Credit_Balance"]
    for col in skew_cols:
        if col in data.columns:
            data[col] = np.log1p(pd.to_numeric(data[col], errors="coerce").clip(lower=0))

    # -----------------------------
    # Rate normalization
    # -----------------------------
    rate_cols = ["Cart_Abandonment_Rate", "Discount_Usage_Rate", "Returns_Rate", "Email_Open_Rate"]

    for col in rate_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

            if data[col].max() > 1:
                data[col] = data[col] / data[col].max()

    # -----------------------------
    # Numeric conversion
    # -----------------------------
    for col in data.columns:
        if col in ["Gender", "Country", "City", "Signup_Quarter"]:
            continue
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # -----------------------------
    # Safe cleaning (NO row drop)
    # -----------------------------
    num_cols = data.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        data[col] = data[col].fillna(data[col].median())
        data[col] = data[col].clip(lower=0)

    # -----------------------------
    # RFM
    # -----------------------------
    data["recency_score"] = _safe_qcut(data["Days_Since_Last_Purchase"], 4, [4,3,2,1])
    data["frequency_score"] = _safe_qcut(data["Total_Purchases"], 4, [1,2,3,4], use_rank=True)
    data["monetary_score"] = _safe_qcut(data["Average_Order_Value"], 4, [1,2,3,4])

    data["RFM_score"] = (
        data["recency_score"] +
        data["frequency_score"] +
        data["monetary_score"]
    )

    # -----------------------------
    # Behavioral Features
    # -----------------------------
    data["engagement_score"] = (
        data["Login_Frequency"] +
        data["Session_Duration_Avg"] +
        data["Pages_Per_Session"] +
        data["Email_Open_Rate"] +
        data["Social_Media_Engagement_Score"]
    ) / 5

    data["risk_score"] = (
        data["Cart_Abandonment_Rate"] +
        data["Returns_Rate"] +
        data["Customer_Service_Calls"]
    ) / 3

    data["loyalty_score"] = (
        data["Membership_Years"] +
        data["Total_Purchases"]
    ) / 2

    data["friction_score"] = (
        data["Cart_Abandonment_Rate"] +
        data["Returns_Rate"]
    ) / 2

    data["support_intensity"] = (
        data["Customer_Service_Calls"] / (data["Total_Purchases"] + 1)
    )

    data["value_per_purchase"] = (
        data["Lifetime_Value"] / (data["Total_Purchases"] + 1)
    )

    return data