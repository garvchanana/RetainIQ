from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "xgboost_model.pkl"
FEATURES_PATH = BASE_DIR / "model" / "features.pkl"
SAMPLE_PATH = BASE_DIR / "data" / "sample_input.csv"

DEFAULT_CHURN_THRESHOLD = 0.45


st.set_page_config(
    page_title="InsightForge AI - Customer Decision Engine",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    /* Fix metric card styling */
    div[data-testid="stMetric"] {
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 0.8rem;
        background: #FFFFFF;
        color: #111827 !important;  /* 🔥 FIX TEXT VISIBILITY */
    }

    div[data-testid="stMetric"] label {
        color: #6B7280 !important;  /* label text */
        font-size: 0.9rem;
    }

    div[data-testid="stMetric"] div {
        color: #111827 !important;  /* main value */
        font-weight: bold;
    }

    .app-title {
        font-size: 2.0rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #FFFFFF;
    }

    .app-subtitle {
        color: #9CA3AF;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, list(features)


def _safe_qcut(series: pd.Series, q: int, labels: list[int], use_rank: bool = False) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if use_rank:
        s = s.rank(method="first")

    if s.notna().sum() == 0:
        return pd.Series([labels[len(labels) // 2]] * len(s), index=s.index, dtype="int64")

    unique_vals = s.nunique(dropna=True)
    if unique_vals < 2:
        fallback = int(np.median(labels))
        return pd.Series([fallback] * len(s), index=s.index, dtype="int64")

    try:
        out = pd.qcut(s, q=q, labels=labels, duplicates="drop")
        if out.isna().any():
            fallback = int(np.median(labels))
            out = out.astype("object").fillna(fallback)
        return out.astype(int)
    except Exception:
        ranked = s.rank(pct=True)
        cuts = np.linspace(0, 1, len(labels) + 1)
        bucket_idx = np.digitize(ranked.fillna(0.5), cuts[1:-1], right=True)
        mapped = np.array(labels, dtype=int)[np.clip(bucket_idx, 0, len(labels) - 1)]
        return pd.Series(mapped, index=s.index, dtype="int64")


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    skew_cols = ["Total_Purchases", "Lifetime_Value", "Credit_Balance"]
    for col in skew_cols:
        if col in data.columns:
            data[col] = np.log1p(pd.to_numeric(data[col], errors="coerce").clip(lower=0))

    rate_cols = ["Cart_Abandonment_Rate", "Discount_Usage_Rate", "Returns_Rate", "Email_Open_Rate"]
    for col in rate_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").clip(0, 1)

    for col in data.columns:
        if col in ["Gender", "Country", "City", "Signup_Quarter", "segment_label", "churn_risk", "recommendation", "secondary_insight"]:
            continue
        try:
            data[col] = pd.to_numeric(data[col])
        except Exception:
            pass

    # ==============================
    # SAFE AGE HANDLING (NO ROW DROP)
    # ==============================
    if "Age" in data.columns:
        data["Age"] = pd.to_numeric(data["Age"], errors="coerce")
        data["Age"] = data["Age"].clip(lower=18, upper=80)

    # ==============================
    # SAFE NUMERIC HANDLING (NO ROW DROP)
    # ==============================
    num_cols = data.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

        # Replace NaN with median
        data[col] = data[col].fillna(data[col].median())

        # Clip negative values instead of dropping
        data[col] = data[col].clip(lower=0)


    required = {
        "Days_Since_Last_Purchase",
        "Total_Purchases",
        "Average_Order_Value",
        "Login_Frequency",
        "Session_Duration_Avg",
        "Pages_Per_Session",
        "Email_Open_Rate",
        "Social_Media_Engagement_Score",
        "Cart_Abandonment_Rate",
        "Returns_Rate",
        "Customer_Service_Calls",
        "Membership_Years",
        "Lifetime_Value",
    }

    missing_required = [c for c in required if c not in data.columns]
    if missing_required:
        raise ValueError(f"Missing required columns for feature engineering: {', '.join(sorted(missing_required))}")

    data["recency_score"] = _safe_qcut(data["Days_Since_Last_Purchase"], q=4, labels=[4, 3, 2, 1], use_rank=False)
    data["frequency_score"] = _safe_qcut(data["Total_Purchases"], q=4, labels=[1, 2, 3, 4], use_rank=True)
    data["monetary_score"] = _safe_qcut(data["Average_Order_Value"], q=4, labels=[1, 2, 3, 4], use_rank=False)

    data["RFM_score"] = (
        data["recency_score"].astype(int)
        + data["frequency_score"].astype(int)
        + data["monetary_score"].astype(int)
    )

    data["engagement_score"] = (
        pd.to_numeric(data["Login_Frequency"], errors="coerce")
        + pd.to_numeric(data["Session_Duration_Avg"], errors="coerce")
        + pd.to_numeric(data["Pages_Per_Session"], errors="coerce")
        + pd.to_numeric(data["Email_Open_Rate"], errors="coerce")
        + pd.to_numeric(data["Social_Media_Engagement_Score"], errors="coerce")
    ) / 5

    data["risk_score"] = (
        pd.to_numeric(data["Cart_Abandonment_Rate"], errors="coerce")
        + pd.to_numeric(data["Returns_Rate"], errors="coerce")
        + pd.to_numeric(data["Customer_Service_Calls"], errors="coerce")
    ) / 3

    data["loyalty_score"] = (
        pd.to_numeric(data["Membership_Years"], errors="coerce")
        + pd.to_numeric(data["Total_Purchases"], errors="coerce")
    ) / 2

    data["friction_score"] = (
        pd.to_numeric(data["Cart_Abandonment_Rate"], errors="coerce")
        + pd.to_numeric(data["Returns_Rate"], errors="coerce")
    ) / 2

    data["support_intensity"] = (
        pd.to_numeric(data["Customer_Service_Calls"], errors="coerce")
        / (pd.to_numeric(data["Total_Purchases"], errors="coerce") + 1)
    )

    data["value_per_purchase"] = (
        pd.to_numeric(data["Lifetime_Value"], errors="coerce")
        / (pd.to_numeric(data["Total_Purchases"], errors="coerce") + 1)
    )

    return data


def segment_from_rfm(rfm_score: float) -> str:
    rfm_int = int(round(rfm_score))
    if rfm_int == 10:
        return "High Value"
    if rfm_int == 7:
        return "Potential Loyalists"
    if rfm_int == 5:
        return "At Risk"
    return "Low Value"


def build_thresholds(data: pd.DataFrame) -> dict:
    return {
        "engagement_low": float(data["engagement_score"].quantile(0.25)),
        "friction_high": float(data["friction_score"].quantile(0.75)),
        "support_high": float(data["support_intensity"].quantile(0.75)),
        "loyalty_high": float(data["loyalty_score"].quantile(0.75)),
        "value_high": float(data["value_per_purchase"].quantile(0.75)),
        "risk_high": float(data["risk_score"].quantile(0.75)),
    }


def generate_decision(row: pd.Series, t: dict, churn_threshold: float) -> tuple[str, str]:
    high_churn = row["churn_probability"] >= churn_threshold
    low_churn = row["churn_probability"] < churn_threshold
    high_value = row["segment_label"] == "High Value"
    low_engagement = row["engagement_score"] <= t["engagement_low"]
    high_friction = row["friction_score"] >= t["friction_high"]
    high_support = row["support_intensity"] >= t["support_high"]
    high_loyalty = row["loyalty_score"] >= t["loyalty_high"]
    high_value_per_purchase = row["value_per_purchase"] >= t["value_high"]

    if high_churn and high_value:
        primary = "Offer retention incentive (discount/loyalty program)"
    elif high_churn and low_engagement:
        primary = "Trigger re-engagement campaign (email/push)"
    elif high_friction:
        primary = "Improve checkout/user experience"
    elif high_support:
        primary = "Proactive customer support intervention"
    elif high_loyalty and low_churn:
        primary = "Upsell or premium offering"
    elif high_value_per_purchase:
        primary = "Target with premium bundles"
    else:
        primary = "Maintain lifecycle nurture sequence with personalized touchpoints"

    if high_churn:
        secondary = "Churn risk is elevated; prioritize action within this cycle."
    elif row["risk_score"] >= t["risk_high"]:
        secondary = "Behavioral risk indicators are trending up across service and return activity."
    elif high_loyalty:
        secondary = "Strong loyalty indicators suggest readiness for higher-tier products."
    elif low_engagement:
        secondary = "Engagement is soft; optimize content cadence and message relevance."
    else:
        secondary = "Profile is relatively stable; monitor trend shifts weekly."

    return primary, secondary


def prepare_model_input(data: pd.DataFrame, model_features: list[str]) -> pd.DataFrame:
    frame = data.copy()
    for col in model_features:
        if col not in frame.columns:
            frame[col] = 0.0
    return frame[model_features].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _get_ground_truth(df: pd.DataFrame) -> pd.Series | None:
    for col in ["churn", "Churned", "churned"]:
        if col in df.columns:
            y = pd.to_numeric(df[col], errors="coerce")
            if y.notna().sum() == 0:
                return None
            y = (y > 0).astype(int)
            return y
    return None


def find_best_threshold(y_true: pd.Series, y_prob: pd.Series) -> dict | None:
    valid = pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).dropna()
    if valid.empty or valid["y_true"].nunique() < 2:
        return None

    thresholds = np.arange(0.05, 0.951, 0.005)
    best_row = None

    for thr in thresholds:
        pred = (valid["y_prob"] >= thr).astype(int)
        f1 = f1_score(valid["y_true"], pred, zero_division=0)
        acc = accuracy_score(valid["y_true"], pred)
        precision = precision_score(valid["y_true"], pred, zero_division=0)
        recall = recall_score(valid["y_true"], pred, zero_division=0)

        row = {
            "threshold": float(thr),
            "f1": float(f1),
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
        }

        if best_row is None:
            best_row = row
        else:
            if (row["f1"] > best_row["f1"]) or (
                np.isclose(row["f1"], best_row["f1"]) and row["accuracy"] > best_row["accuracy"]
            ):
                best_row = row

    return best_row


def main():
    st.markdown("""
    <div style="padding: 1.2rem 0;">
        <div class="app-title">InsightForge AI</div>
        <div class="app-subtitle">
            Transforming customer behavior into <b>actionable business decisions</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    with st.sidebar:
        st.markdown("## ⚙️ Control Panel")

        st.markdown("### 📂 Data Source")

        data_option = st.radio(
            "Select Dataset",
            [
                "Sample Input",
                "Preprocessed Data",
                "Raw Ecommerce Data"
            ]
        )

        st.markdown("---")

        st.markdown("### 🎯 Churn Threshold")
        threshold_input = st.slider(
            "Set churn classification threshold",
            min_value=0.05,
            max_value=0.95,
            value=DEFAULT_CHURN_THRESHOLD,
            step=0.01,
        )
        auto_use_best_threshold = st.checkbox(
            "Use best threshold from labeled data",
            value=False,
        )

        st.markdown("---")

        st.markdown("### ℹ️ Info")
        st.info("Decision-ready churn intelligence with calibrated risk thresholds, actionable interventions, and measurable business impact.")
    
    raw_df = None
    source_name = None

    if data_option == "Sample Input":
        raw_df = pd.read_csv("data/sample_input.csv")
        source_name = "Sample Input"

    elif data_option == "Preprocessed Data":
        raw_df = pd.read_csv("data/processed_data.csv")
        source_name = "Preprocessed Data"

    elif data_option == "Raw Ecommerce Data":
        raw_df = pd.read_csv(r"D:\Python\Data Science Projects\InsightForge AI\data\ecommerce_customer_churn_dataset.csv")
        source_name = "Raw Dataset"

    if raw_df is None:
        st.info("Upload a CSV from the sidebar, or enable sample data.")
        st.stop()

    st.caption(f"Data source: {source_name} | Rows: {len(raw_df):,}")

    model, model_features = load_assets()

    try:
        scored_df = apply_feature_engineering(raw_df)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    if scored_df.empty:
        st.error("No valid rows after data consistency checks. Please review the input file.")
        st.stop()

    X_infer = prepare_model_input(scored_df, model_features)
    scored_df["churn_probability"] = model.predict_proba(X_infer)[:, 1]

    y_true = _get_ground_truth(raw_df)
    best_threshold = None
    if y_true is not None and len(y_true) == len(scored_df):
        best_threshold = find_best_threshold(y_true.reset_index(drop=True), scored_df["churn_probability"].reset_index(drop=True))

    selected_threshold = threshold_input
    if auto_use_best_threshold and best_threshold is not None:
        selected_threshold = best_threshold["threshold"]

    scored_df["churn_risk"] = np.where(scored_df["churn_probability"] >= selected_threshold, "High", "Low")
    scored_df["segment_label"] = scored_df["RFM_score"].apply(segment_from_rfm)

    thresholds = build_thresholds(scored_df)
    decisions = scored_df.apply(lambda row: generate_decision(row, thresholds, selected_threshold), axis=1)
    scored_df["recommendation"] = decisions.apply(lambda x: x[0])
    scored_df["secondary_insight"] = decisions.apply(lambda x: x[1])

    scored_df = scored_df.reset_index(drop=True)
    scored_df["customer_id"] = scored_df.index + 1

    st.markdown("### 📊 Key Metrics")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Customers", f"{len(scored_df):,}")
    m2.metric("Avg Churn Risk", f"{scored_df['churn_probability'].mean():.2%}")
    m3.metric("High-Risk", f"{(scored_df['churn_risk'] == 'High').sum():,}")
    m4.metric("Threshold", f"{selected_threshold:.2f}")

    if best_threshold is not None:
        st.markdown("### ✅ Best Threshold Analysis")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Threshold", f"{best_threshold['threshold']:.3f}")
        c2.metric("F1 Score", f"{best_threshold['f1']:.3f}")
        c3.metric("Accuracy", f"{best_threshold['accuracy']:.3f}")
        c4.metric("Precision / Recall", f"{best_threshold['precision']:.3f} / {best_threshold['recall']:.3f}")
    else:
        st.info("Best-threshold analysis is available when churn labels are present in the input data.")

    st.markdown(" ")

    st.markdown("### 🔥 High-Risk Customer Insights")
    st.caption("Top customers requiring immediate intervention")
    top10 = scored_df.sort_values("churn_probability", ascending=False).head(10)
    st.dataframe(
        top10[["customer_id", "segment_label", "churn_probability", "churn_risk", "recommendation", "secondary_insight"]],
        use_container_width=True,
        hide_index=True,
    )

    
    st.markdown("### 📈 Churn Risk Distribution")
    st.caption("Distribution of predicted churn probabilities across the customer base, highlighting risk concentration and decision thresholds.")
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(
        scored_df["churn_probability"],
        bins=20,
        color="#2563EB",
        alpha=0.85
    )

    ax.axvline(
        selected_threshold,
        color="#DC2626",
        linestyle="--",
        linewidth=2,
        label="Risk Threshold"
    )

    ax.set_title("Churn Probability Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Churn Probability")
    ax.set_ylabel("Customer Count")

    ax.legend()
    ax.grid(alpha=0.2)

    st.pyplot(fig)

    st.markdown("""
    This visualization shows how customers are distributed based on their predicted churn probability.  
    - The **right side** represents high-risk customers  
    - The **dotted line** indicates the decision threshold used for action  
    """)

    high_risk_pct = (scored_df["churn_probability"] >= selected_threshold).mean()

    if high_risk_pct >= 0.5:
        st.error("🚨 Critical churn risk detected. Immediate retention strategy required across segments.")

    elif high_risk_pct >= 0.3:
        st.warning("⚠️ Elevated churn risk. Focus on high-value and high-risk customer segments.")

    elif high_risk_pct >= 0.15:
        st.info("🔍 Moderate churn signals detected. Targeted engagement strategies recommended.")

    else:
        st.success("✅ Customer base is stable. Maintain current engagement and monitor trends.")

    st.markdown("### 🧠 Segment Risk Analysis")
    st.caption("Identifying high-risk customer segments for targeted intervention")

    segment_risk = scored_df.groupby("segment_label")["churn_probability"].mean()
    for seg, val in segment_risk.items():
        if val >= 0.35:
            st.warning(f"⚠️ {seg}: {val:.2f} → High churn risk (priority segment)")
        elif val >= 0.2:
            st.info(f"🔍 {seg}: {val:.2f} → Moderate risk")
        else:
            st.success(f"✅ {seg}: {val:.2f} → Stable segment")

    st.markdown("### 📌 Executive Decision Summary")
    st.caption("Prioritized intervention buckets and segment risk concentration.")

    scored_df["priority_bucket"] = np.select(
        [
            (scored_df["churn_risk"] == "High") & (scored_df["segment_label"] == "High Value"),
            (scored_df["churn_risk"] == "High") & (scored_df["engagement_score"] <= thresholds["engagement_low"]),
            (scored_df["friction_score"] >= thresholds["friction_high"]),
            (scored_df["support_intensity"] >= thresholds["support_high"]),
            (scored_df["churn_risk"] == "Low") & (scored_df["loyalty_score"] >= thresholds["loyalty_high"]),
        ],
        [
            "Immediate Retention (High-Value Churn Risk)",
            "Re-Engagement Campaign Priority",
            "Checkout/Experience Optimization",
            "Proactive Support Intervention",
            "Upsell/Premium Opportunity",
        ],
        default="Monitor & Nurture",
    )

    bucket_summary = (
        scored_df["priority_bucket"]
        .value_counts()
        .rename_axis("priority_bucket")
        .reset_index(name="customers")
    )
    bucket_summary["share_pct"] = (bucket_summary["customers"] / len(scored_df) * 100).round(2)

    total_customers = len(scored_df)
    high_risk_count = int((scored_df["churn_risk"] == "High").sum())
    priority_mask = (
        (scored_df["churn_risk"] == "High")
        & (
            (scored_df["engagement_score"] <= thresholds["engagement_low"])
            | (scored_df["friction_score"] >= thresholds["friction_high"])
            | (scored_df["support_intensity"] >= thresholds["support_high"])
        )
    )
    priority_count = int(priority_mask.sum())

    critical_mask = priority_mask & (
        scored_df["churn_probability"] >= min(0.95, selected_threshold + 0.15)
    )
    critical_count = int(critical_mask.sum())

    vip_mask = critical_mask & (scored_df["segment_label"] == "High Value")
    vip_count = int(vip_mask.sum())

    funnel_df = pd.DataFrame(
        {
            "stage": [
                "Total Customer Base",
                "High-Risk Customers",
                "Priority Intervention",
                "Critical Immediate Action",
                "High-Value Critical Retention",
            ],
            "count": [
                total_customers,
                high_risk_count,
                priority_count,
                critical_count,
                vip_count,
            ],
        }
    )
    funnel_df["share_pct"] = (funnel_df["count"] / max(total_customers, 1) * 100).round(2)

    left, right = st.columns([1.5, 1])
    with left:
        st.markdown("#### Intervention Funnel")
        max_count = max(funnel_df["count"].max(), 1)
        fig_funnel, ax_funnel = plt.subplots(figsize=(9, 4.8))
        y_pos = np.arange(len(funnel_df))
        counts = funnel_df["count"].values
        left_offsets = (max_count - counts) / 2

        ax_funnel.barh(
            y_pos,
            counts,
            left=left_offsets,
            color=["#2563EB", "#1D4ED8", "#0284C7", "#0EA5E9", "#22C55E"],
            alpha=0.95,
        )
        ax_funnel.set_yticks(y_pos)
        ax_funnel.set_yticklabels(funnel_df["stage"])
        ax_funnel.invert_yaxis()
        ax_funnel.set_xlim(0, max_count)
        ax_funnel.set_xlabel("Customers")
        ax_funnel.set_title("Customer Risk-to-Action Funnel", fontsize=12, fontweight="bold")
        ax_funnel.grid(axis="x", alpha=0.2)

        for i, row in funnel_df.iterrows():
            ax_funnel.text(
                max_count * 0.98,
                i,
                f"{int(row['count']):,}  ({row['share_pct']:.1f}%)",
                va="center",
                ha="right",
                fontsize=9,
                color="white",
                fontweight="bold",
            )

        st.pyplot(fig_funnel)

    with right:
        st.markdown("#### Priority Strategy Mix")
        st.dataframe(bucket_summary, use_container_width=True, hide_index=True)
        st.markdown("#### Segment x Risk")
        segment_risk_matrix = pd.crosstab(scored_df["segment_label"], scored_df["churn_risk"])
        st.dataframe(segment_risk_matrix, use_container_width=True)

    if y_true is not None and len(y_true) == len(scored_df):
        y_eval = y_true.reset_index(drop=True)
        y_pred = (scored_df["churn_probability"].reset_index(drop=True) >= selected_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_eval, y_pred).ravel()
        total_pos = tp + fn
        total_neg = tn + fp
        tpr = tp / total_pos if total_pos else 0.0
        tnr = tn / total_neg if total_neg else 0.0
        fpr = fp / total_neg if total_neg else 0.0
        fnr = fn / total_pos if total_pos else 0.0

        st.markdown("#### 🧩 Confusion Metrics")
        c_left, c_right = st.columns([1.2, 1])

        with c_left:
            cm = np.array([[tn, fp], [fn, tp]])
            fig_cm, ax_cm = plt.subplots(figsize=(5.2, 4.2))
            im = ax_cm.imshow(cm, cmap="Blues")
            ax_cm.set_xticks([0, 1], labels=["Pred: No Churn", "Pred: Churn"])
            ax_cm.set_yticks([0, 1], labels=["Actual: No Churn", "Actual: Churn"])
            ax_cm.set_title("Confusion Matrix", fontsize=11, fontweight="bold")
            for i in range(2):
                for j in range(2):
                    ax_cm.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="#0B1220", fontweight="bold")
            plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
            st.pyplot(fig_cm)

        with c_right:
            r1, r2 = st.columns(2)
            r1.metric("True Positive Rate", f"{tpr:.3f}")
            r2.metric("True Negative Rate", f"{tnr:.3f}")
            r3, r4 = st.columns(2)
            r3.metric("False Positive Rate", f"{fpr:.3f}")
            r4.metric("False Negative Rate", f"{fnr:.3f}")
            st.caption(f"TP: {tp:,} | FP: {fp:,} | TN: {tn:,} | FN: {fn:,}")

    summary_export = scored_df[
        ["customer_id", "segment_label", "churn_probability", "churn_risk", "priority_bucket", "recommendation"]
    ].copy()
    summary_export = summary_export.sort_values("churn_probability", ascending=False).head(200)
    csv_data = summary_export.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Prioritized Action List (Top 200)",
        data=csv_data,
        file_name="InsightForge_prioritized_actions.csv",
        mime="text/csv",
    )

if __name__ == "__main__":
    main()