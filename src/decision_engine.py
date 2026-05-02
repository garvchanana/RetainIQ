import pandas as pd


# ==============================
# GENERATE BUSINESS DECISION
# ==============================

def generate_decision(row: pd.Series) -> tuple:
    churn = row.get("churn_probability", 0)
    segment = row.get("segment_label", "")
    engagement = row.get("engagement_score", 0)
    friction = row.get("friction_score", 0)
    support = row.get("support_intensity", 0)
    value = row.get("value_per_purchase", 0)

    # -----------------------------
    # PRIMARY DECISION LOGIC
    # -----------------------------

    if churn >= 0.7 and segment == "High Value":
        primary = "Offer retention incentives (loyalty rewards or exclusive benefits)"
        insight = "High-value users are at strong risk of churn"

    elif churn >= 0.6 and engagement < 0.3:
        primary = "Trigger re-engagement campaign (email/push notifications)"
        insight = "Low engagement is contributing to churn risk"

    elif friction > 0.5:
        primary = "Improve user experience (reduce friction in transactions/journey)"
        insight = "High friction signals poor user experience"

    elif support > 0.5:
        primary = "Initiate proactive customer support outreach"
        insight = "High support dependency indicates dissatisfaction"

    elif churn >= 0.5 and segment in ["At Risk", "Potential Loyalists"]:
        primary = "Provide targeted offers or personalized recommendations"
        insight = "Users are in transition stage and need engagement push"

    elif churn < 0.3 and segment == "High Value":
        primary = "Upsell premium products or services"
        insight = "Stable and valuable customers present growth opportunity"

    elif value > 0.7:
        primary = "Promote premium bundles or financial upgrades"
        insight = "High spending behavior detected"

    else:
        primary = "Maintain engagement with regular updates"
        insight = "No immediate risk detected"

    return primary, insight


# ==============================
# APPLY DECISION ENGINE
# ==============================

def apply_decision_engine(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    results = data.apply(generate_decision, axis=1)

    data["recommendation"] = [r[0] for r in results]
    data["secondary_insight"] = [r[1] for r in results]

    return data