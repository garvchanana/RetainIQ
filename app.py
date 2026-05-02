from __future__ import annotations

import json
import os
from dotenv import load_dotenv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from google import genai
from google.genai import errors as genai_errors

load_dotenv()


# -----------------------------
# Config
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

PROCESSED_FINTECH_PATH = DATA_DIR / "processed_fintech_data.csv"
SEGMENT_SCORES_PATH = DATA_DIR / "fintech_segment_scores.csv"
HYPOTHESES_PATH = DATA_DIR / "fintech_top3_churn_hypotheses.csv"
SCORED_USERS_PATH = DATA_DIR / "fintech_scored_users.csv"

DEFAULT_THRESHOLD = 0.45
BEST_THRESHOLD_NOTEBOOK = 0.29


st.set_page_config(
    page_title="RetainIQ",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 2.6rem; padding-bottom: 2rem;}
        div[data-testid="stMetric"] {
            border: 1px solid #243044;
            border-radius: 12px;
            padding: 0.35rem 0.55rem;
            background: linear-gradient(145deg, #111827 0%, #0B1220 100%);
            box-shadow: inset 0 0 0 1px rgba(59, 130, 246, 0.08);
        }
        div[data-testid="stMetricLabel"] {
            color: #9CA3AF !important;
        }
        div[data-testid="stMetricValue"] {
            color: #F9FAFB !important;
            font-size: 1.65rem !important;
        }
        div[data-testid="stMetricDelta"] {
            color: #93C5FD !important;
        }
        .title-wrap {margin-bottom: 0.35rem; margin-top: 0.25rem;}
        .app-title {font-size: 2rem; font-weight: 700; color: #F9FAFB; line-height: 1.2;}
        .app-subtitle {color: #9CA3AF; margin-top: 0.2rem;}
        .kpi-note {
            color: #9CA3AF;
            font-size: 0.82rem;
            line-height: 1.25;
            margin-top: 0.2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Utilities
# -----------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_all_data() -> dict[str, pd.DataFrame]:
    return {
        "processed": load_csv(PROCESSED_FINTECH_PATH),
        "segment_scores": load_csv(SEGMENT_SCORES_PATH),
        "hypotheses": load_csv(HYPOTHESES_PATH),
        "scored_users": load_csv(SCORED_USERS_PATH),
    }


def ensure_scored_users(processed: pd.DataFrame, scored_users: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if not scored_users.empty and "churn_probability" in scored_users.columns:
        df = scored_users.copy()
        # Always recompute risk from active threshold so dashboard stays synchronized.
        df["churn_risk"] = np.where(df["churn_probability"] >= threshold, "High", "Low")
        return df

    # Fallback: build heuristic probability if scored file is not available.
    req = ["risk_score", "support_intensity", "activity_trend", "RFM_score", "n_products_used"]
    missing = [c for c in req if c not in processed.columns]
    if missing:
        out = processed.copy()
        out["churn_probability"] = np.nan
        out["churn_risk"] = "Unknown"
        return out

    p = processed.copy()
    r = p["risk_score"].fillna(p["risk_score"].median())
    s = p["support_intensity"].fillna(p["support_intensity"].median())
    t = p["activity_trend"].fillna(0)
    f = p["RFM_score"].fillna(p["RFM_score"].median())
    m = p["n_products_used"].fillna(1)

    raw = 1.2 * r + 1.1 * s - 0.9 * t - 0.35 * f - 0.2 * m
    prob = 1 / (1 + np.exp(-raw))
    p["churn_probability"] = np.clip(prob, 0.001, 0.999)
    p["churn_risk"] = np.where(p["churn_probability"] >= threshold, "High", "Low")
    return p


def make_priority_bucket(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()

    for col in ["engagement_score", "risk_score", "support_intensity", "churn_probability", "rfm_segment", "churn_risk"]:
        if col not in out.columns:
            out[col] = np.nan

    eng_low = out["engagement_score"].quantile(0.25) if out["engagement_score"].notna().any() else 0
    risk_high = out["risk_score"].quantile(0.75) if out["risk_score"].notna().any() else 1
    support_high = out["support_intensity"].quantile(0.75) if out["support_intensity"].notna().any() else 1

    out["priority_bucket"] = np.select(
        [
            (out["churn_probability"] >= threshold) & (out["rfm_segment"] == "High Value"),
            (out["churn_probability"] >= threshold) & (out["engagement_score"] <= eng_low),
            (out["risk_score"] >= risk_high),
            (out["support_intensity"] >= support_high),
        ],
        [
            "Immediate Retention (High Value)",
            "Re-engagement Priority",
            "Risk Reduction Program",
            "Proactive Support Intervention",
        ],
        default="Monitor & Nurture",
    )
    return out


def render_empty_state() -> None:
    st.error("Missing required fintech data files. Please run notebook pipeline first.")
    st.info(
        "Required files: `processed_fintech_data.csv`, `fintech_segment_scores.csv`, "
        "`fintech_top3_churn_hypotheses.csv`, `fintech_scored_users.csv`"
    )


@st.cache_data(show_spinner=False, ttl=600)
def generate_ai_explanation(section: str, payload: dict, api_key: str) -> str:
    clean_key = (api_key or "").strip().replace("\r", "").replace("\n", "")
    client = genai.Client(api_key=clean_key)
    prompt = f"""
    You are a fintech retention intelligence analyst.

    Section: {section}
    Metrics: {json.dumps(payload)}

    Generate one sharp business insight in under 25 words.
    Focus on churn risk, user behavior, segment trends, or retention actions.
    Professional, executive-level, concise, and data-driven.
    No bullets, markdown, labels, or filler language.
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        text = (response.text or "").strip()
        return text if text else "AI explanation unavailable for this section."
    except genai_errors.ClientError as exc:
        raise RuntimeError(f"Gemini API request failed: {exc}") from exc


def safe_ai_explanation(section: str, payload: dict, api_key: str, fallback_text: str) -> str:
    try:
        return generate_ai_explanation(section, payload, api_key)
    except Exception:
        return fallback_text


# -----------------------------
# App
# -----------------------------
def main() -> None:
    st.markdown(
        """
        <div class='title-wrap'>
            <div class='app-title'>RetainIQ — Fintech Segmentation & Churn Intelligence Platform</div>
            <div class='app-subtitle'>Transforming behavioral user activity into churn intelligence, retention prioritization, and segment-driven decision analytics for fintech platforms.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    data = load_all_data()
    processed = data["processed"]
    segment_scores = data["segment_scores"]
    hypotheses = data["hypotheses"]
    scored_users = data["scored_users"]

    if processed.empty:
        render_empty_state()
        st.stop()

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    gemini_key = gemini_key.strip().replace("\r", "").replace("\n", "")
    if not gemini_key:
        st.error("GEMINI_API_KEY is required. Add it to .env in the project root and rerun.")
        st.stop()

    with st.sidebar:
        st.markdown("## ⚙️ Control Panel")

        threshold_slider = st.slider(
            "Churn risk threshold",
            min_value=0.05,
            max_value=0.95,
            value=DEFAULT_THRESHOLD,
            step=0.01,
        )
        use_best_threshold = st.checkbox(
            f"Use Best Threshold ({BEST_THRESHOLD_NOTEBOOK:.2f})",
            value=False,
            help="Switch to the calibrated best threshold for classification.",
        )
        threshold = BEST_THRESHOLD_NOTEBOOK if use_best_threshold else threshold_slider
        st.caption(
            f"Active threshold: {threshold:.2f} "
            f"({'Best' if use_best_threshold else 'Manual'})"
        )

        rfm_options = sorted(processed["rfm_segment"].dropna().astype(str).unique().tolist()) if "rfm_segment" in processed.columns else []
        life_options = sorted(processed["lifecycle_stage"].dropna().astype(str).unique().tolist()) if "lifecycle_stage" in processed.columns else []
        mix_options = sorted(processed["product_mix_segment"].dropna().astype(str).unique().tolist()) if "product_mix_segment" in processed.columns else []

        rfm_choice = st.selectbox("RFM Segment", ["All"] + rfm_options, index=0)
        life_choice = st.selectbox("Lifecycle Stage", ["All"] + life_options, index=0)
        mix_choice = st.selectbox("Product Mix Segment", ["All"] + mix_options, index=0)

        top_n_segments = st.slider("Top risky segments", 5, 30, 12, 1)
        top_n_users = st.slider("Prioritized users export", 50, 1000, 200, 50)

        st.markdown("---")
        st.caption(
            "Analytics controls for churn intelligence, segment monitoring, and retention prioritization."
        )

    scored = ensure_scored_users(processed, scored_users, threshold)
    scored = make_priority_bucket(scored, threshold)

    # Apply filters
    f = scored.copy()
    if rfm_choice != "All" and "rfm_segment" in f.columns:
        f = f[f["rfm_segment"].astype(str) == rfm_choice]
    if life_choice != "All" and "lifecycle_stage" in f.columns:
        f = f[f["lifecycle_stage"].astype(str) == life_choice]
    if mix_choice != "All" and "product_mix_segment" in f.columns:
        f = f[f["product_mix_segment"].astype(str) == mix_choice]

    if f.empty:
        st.warning("No rows match current filter selection.")
        st.stop()

    with st.sidebar:

        st.markdown("---")
        st.markdown("### 📦 Export Intelligence Reports")

        with st.expander("Download Reports"):

            prioritized = f.sort_values(
                "churn_probability",
                ascending=False
            ).head(top_n_users).copy()

            keep_cols = [
                "user_id",
                "segment_key",
                "rfm_segment",
                "lifecycle_stage",
                "product_mix_segment",
                "churn_probability",
                "churn_risk",
                "priority_bucket",
            ]

            keep_cols = [c for c in keep_cols if c in prioritized.columns]

            p_csv = prioritized[keep_cols].to_csv(index=False).encode("utf-8")

            st.download_button(
                "📥 Prioritized Users",
                data=p_csv,
                file_name="fintech_prioritized_users.csv",
                mime="text/csv",
                use_container_width=True
            )

            if not hypotheses.empty:
                h_csv = hypotheses.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "📥 Driver Hypotheses",
                    data=h_csv,
                    file_name="fintech_top3_churn_hypotheses.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            if not segment_scores.empty:
                s_csv = segment_scores.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "📥 Segment Risk Report",
                    data=s_csv,
                    file_name="fintech_segment_scores.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    top_bucket_current = (
        f["priority_bucket"].value_counts().index[0]
        if "priority_bucket" in f.columns and not f["priority_bucket"].empty
        else "Monitor & Nurture"
    )
    high_ratio_current = ((f["churn_risk"] == "High").mean() * 100) if "churn_risk" in f.columns else 0.0
    avg_prob_current = (f["churn_probability"].mean() * 100) if "churn_probability" in f.columns else 0.0

    # -----------------------------
    # 📌 Executive Snapshot
    # -----------------------------
    st.markdown("### 📌 Executive Snapshot")
    exec_payload = {
        "users_analyzed": int(len(f)),
        "avg_churn_probability": float(f["churn_probability"].mean()),
        "high_risk_users": int((f["churn_risk"] == "High").sum()),
        "threshold": float(threshold),
    }
    st.markdown(
        safe_ai_explanation(
            "Executive Snapshot",
            exec_payload,
            gemini_key,
            "A quick view of current portfolio exposure: coverage, average risk intensity, and active decision threshold.",
        )
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users Analyzed", f"{len(f):,}")
    c2.metric("Avg Churn Probability", f"{f['churn_probability'].mean():.2%}")
    c3.metric("High-Risk Users", f"{(f['churn_risk'] == 'High').sum():,}")
    c4.metric("Threshold", f"{threshold:.2f}")

    n1, n2, n3, n4 = st.columns(4)
    n1.markdown("<div class='kpi-note'>Total users included after current segment filters.</div>", unsafe_allow_html=True)
    n2.markdown("<div class='kpi-note'>Mean calibrated churn likelihood across selected users.</div>", unsafe_allow_html=True)
    n3.markdown("<div class='kpi-note'>Users at or above threshold; immediate retention pool.</div>", unsafe_allow_html=True)
    n4.markdown("<div class='kpi-note'>Decision cut-off used to classify High vs Low churn risk.</div>", unsafe_allow_html=True)

    # -----------------------------
    # 🧩 Segmentation Intelligence
    # -----------------------------
    st.divider()
    st.markdown("### 🧩 Segmentation Intelligence")
    seg_payload = {
        "rfm_segments": int(f["rfm_segment"].nunique()) if "rfm_segment" in f.columns else 0,
        "lifecycle_stages": int(f["lifecycle_stage"].nunique()) if "lifecycle_stage" in f.columns else 0,
        "product_mix_segments": int(f["product_mix_segment"].nunique()) if "product_mix_segment" in f.columns else 0,
        "avg_churn_probability": float(f["churn_probability"].mean()),
    }
    st.markdown(
        safe_ai_explanation(
            "Segmentation Intelligence",
            seg_payload,
            gemini_key,
            "This section explains where churn concentration is forming across behavior layers: RFM, lifecycle, and product mix.",
        )
    )
    left, right = st.columns([1.25, 1])

    with left:
        st.markdown("#### Segment Risk Heatmap")
        if {"rfm_segment", "lifecycle_stage", "churn_probability"}.issubset(f.columns):
            heat = f.pivot_table(
                index="rfm_segment",
                columns="lifecycle_stage",
                values="churn_probability",
                aggfunc="mean",
            )
            heat_pct = (heat * 100).round(2)
            fig_h, ax_h = plt.subplots(figsize=(8.2, 4.4))
            im = ax_h.imshow(heat_pct.fillna(0).values, cmap="YlGnBu", aspect="auto")
            ax_h.set_xticks(np.arange(len(heat_pct.columns)))
            ax_h.set_yticks(np.arange(len(heat_pct.index)))
            ax_h.set_xticklabels(heat_pct.columns)
            ax_h.set_yticklabels(heat_pct.index)
            ax_h.set_title("Segment Risk Heatmap (%)", fontsize=11, fontweight="bold")
            plt.setp(ax_h.get_xticklabels(), rotation=20, ha="right")

            for i in range(len(heat_pct.index)):
                for j in range(len(heat_pct.columns)):
                    val = heat_pct.iloc[i, j]
                    label = "—" if pd.isna(val) else f"{val:.2f}"
                    ax_h.text(j, i, label, ha="center", va="center", color="#0B1220", fontsize=9, fontweight="bold")

            cbar = fig_h.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)
            cbar.set_label("Churn Probability (%)")
            st.pyplot(fig_h)
        else:
            st.info("Required columns for heatmap are missing.")

    with right:
        st.markdown("#### Product Mix Risk")
        if {"product_mix_segment", "churn_probability"}.issubset(f.columns):
            mix = f.groupby("product_mix_segment", as_index=False).agg(
                avg_churn_probability=("churn_probability", "mean"),
                users=("user_id", "count"),
            )
            mix["user_share_pct"] = (mix["users"] / max(len(f), 1) * 100).round(2)
            mix["avg_churn_pct"] = (mix["avg_churn_probability"] * 100).round(2)
            mix["risk_load"] = (mix["users"] * mix["avg_churn_probability"]).round(2)
            mix = mix.sort_values(["avg_churn_probability", "users"], ascending=[False, False]).reset_index(drop=True)
            fig_mix, ax_mix_left = plt.subplots(figsize=(8.0, 4.4))
            x = np.arange(len(mix))

            bars = ax_mix_left.bar(
                x,
                mix["users"].values,
                color="#1D4ED8",
                alpha=0.85,
                width=0.62,
                label="Users",
            )
            ax_mix_left.set_ylabel("Users")
            ax_mix_left.set_xticks(x)
            ax_mix_left.set_xticklabels(mix["product_mix_segment"], rotation=20, ha="right")
            ax_mix_left.grid(axis="y", alpha=0.2)

            ax_mix_right = ax_mix_left.twinx()
            ax_mix_right.plot(
                x,
                mix["avg_churn_pct"].values,
                color="#DC2626",
                marker="o",
                linewidth=2.2,
                label="Avg Churn (%)",
            )
            ax_mix_right.set_ylabel("Avg Churn (%)")

            for i, row in mix.iterrows():
                ax_mix_right.text(
                    i,
                    row["avg_churn_pct"] + 0.6,
                    f"{row['avg_churn_pct']:.1f}%",
                    color="#DC2626",
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                )

            ax_mix_left.set_title("Product Mix Risk Composition", fontsize=11, fontweight="bold")
            left_handles, left_labels = ax_mix_left.get_legend_handles_labels()
            right_handles, right_labels = ax_mix_right.get_legend_handles_labels()
            ax_mix_left.legend(
                left_handles + right_handles,
                left_labels + right_labels,
                loc="upper right",
                bbox_to_anchor=(0.98, 0.98),
                frameon=True
            )
            st.pyplot(fig_mix)
        else:
            st.info("Product mix columns missing.")

    # -----------------------------
    # ⚙️ Churn Risk Engine
    # -----------------------------
    st.divider()
    st.markdown("### ⚙️ Churn Risk Engine")
    risk_payload = {
        "avg_churn_probability": float(f["churn_probability"].mean()),
        "high_risk_users": int((f["churn_risk"] == "High").sum()),
        "high_risk_ratio_pct": float(high_ratio_current),
        "threshold": float(threshold),
    }
    st.markdown(
        safe_ai_explanation(
            "Churn Risk Engine",
            risk_payload,
            gemini_key,
            "Probability distribution and risky-segment ranking help prioritize interventions with calibrated confidence.",
        )
    )
    c_left, c_right = st.columns([1.35, 1])

    with c_left:
        st.markdown("#### Probability Distribution")
        fig, ax = plt.subplots(figsize=(8.5, 4.1))
        ax.hist(f["churn_probability"].dropna(), bins=24, color="#2563EB", alpha=0.88)
        ax.axvline(threshold, color="#DC2626", linestyle="--", linewidth=2, label="Threshold")
        ax.set_xlabel("Churn Probability")
        ax.set_ylabel("Users")
        ax.set_title("Calibrated Churn Probability Distribution", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.2)
        ax.legend()
        st.pyplot(fig)

    with c_right:
        st.markdown("#### Top Risk Segments")
        seg_runtime = (
            f.groupby("segment_key", as_index=False)
            .agg(
                users=("user_id", "count"),
                avg_churn_probability=("churn_probability", "mean"),
                high_risk_users=("churn_risk", lambda x: int((x == "High").sum())),
            )
        )
        seg_runtime["high_risk_ratio"] = seg_runtime["high_risk_users"] / seg_runtime["users"]
        seg_runtime = seg_runtime.sort_values("avg_churn_probability", ascending=False).head(top_n_segments)
        st.dataframe(
            seg_runtime[["segment_key", "users", "avg_churn_probability", "high_risk_ratio"]],
            use_container_width=True,
            hide_index=True,
        )

    # -----------------------------
    # 🔬 Top-3 Driver Hypotheses
    # -----------------------------
    st.divider()
    st.markdown("### 🔬 Top-3 Churn Driver Hypotheses")
    hyp_payload = {
        "available_hypothesis_rows": int(len(hypotheses)),
        "unique_segments_with_hypotheses": int(hypotheses["segment_key"].nunique()) if not hypotheses.empty and "segment_key" in hypotheses.columns else 0,
        "avg_effect_gap": float(hypotheses["effect_gap"].mean()) if not hypotheses.empty and "effect_gap" in hypotheses.columns else 0.0,
    }
    st.markdown(
        safe_ai_explanation(
            "Top-3 Churn Driver Hypotheses",
            hyp_payload,
            gemini_key,
            "Each hypothesis is test-ready: it connects a churn driver to a measurable intervention experiment.",
        )
    )
    if hypotheses.empty:
        st.info("Hypothesis file not found. Run notebook export to generate `fintech_top3_churn_hypotheses.csv`.")
    else:
        seg_options = sorted(hypotheses["segment_key"].dropna().astype(str).unique().tolist())
        default_seg = seg_options[0] if seg_options else None
        selected_seg = st.selectbox("Select Segment", seg_options, index=0 if default_seg else None)
        h = hypotheses[hypotheses["segment_key"].astype(str) == str(selected_seg)].sort_values("rank")
        h = h.dropna(how="all")

        show_cols = [c for c in ["rank", "driver_feature", "effect_gap", "hypothesis"] if c in h.columns]
        visible_rows = max(len(h), 1)
        table_height = min(420, max(120, 42 + visible_rows * 38))
        st.dataframe(
            h[show_cols],
            use_container_width=True,
            hide_index=True,
            height=table_height,
        )

    # -----------------------------
    # 🚦 Action Funnel & Prioritization
    # -----------------------------
    st.divider()
    st.markdown("### 🚦 Action Funnel & Prioritization")
    total_users = len(f)
    high_risk = int((f["churn_risk"] == "High").sum())
    priority = int((f["priority_bucket"] != "Monitor & Nurture").sum())
    critical = int((f["churn_probability"] >= min(0.95, threshold + 0.15)).sum())
    hv_critical = int(((f["churn_probability"] >= min(0.95, threshold + 0.15)) & (f["rfm_segment"] == "High Value")).sum()) if "rfm_segment" in f.columns else 0
    funnel_payload = {
        "total_users": int(total_users),
        "high_risk_users": int(high_risk),
        "priority_users": int(priority),
        "critical_users": int(critical),
        "high_value_critical_users": int(hv_critical),
    }
    st.markdown(
        safe_ai_explanation(
            "Action Funnel & Prioritization",
            funnel_payload,
            gemini_key,
            "It converts segment risk into execution stages, from total users to high-value critical intervention targets.",
        )
    )

    funnel = pd.DataFrame(
        {
            "stage": [
                "Total Users",
                "High-Risk Users",
                "Priority Intervention",
                "Critical Immediate Action",
                "High-Value Critical",
            ],
            "count": [total_users, high_risk, priority, critical, hv_critical],
        }
    )
    funnel["share_pct"] = (funnel["count"] / max(total_users, 1) * 100).round(2)

    fl, fr = st.columns([1.45, 1])
    with fl:
        max_count = max(funnel["count"].max(), 1)
        fig2, ax2 = plt.subplots(figsize=(9.2, 4.8))
        y_pos = np.arange(len(funnel))
        left_offsets = (max_count - funnel["count"].values) / 2
        ax2.barh(
            y_pos,
            funnel["count"].values,
            left=left_offsets,
            color=["#1D4ED8", "#2563EB", "#0284C7", "#0EA5E9", "#22C55E"],
            alpha=0.95,
        )
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(funnel["stage"])
        ax2.invert_yaxis()
        ax2.set_xlim(0, max_count)
        ax2.set_xlabel("Users")
        ax2.set_title("Risk-to-Action Funnel", fontsize=11, fontweight="bold")
        ax2.grid(axis="x", alpha=0.2)

        for i, row in funnel.iterrows():
            ax2.text(
                max_count * 0.98,
                i,
                f"{int(row['count']):,} ({row['share_pct']:.1f}%)",
                va="center",
                ha="right",
                fontsize=9,
                color="white",
                fontweight="bold",
            )
        st.pyplot(fig2)

    with fr:
        st.markdown("#### Priority Buckets")
        bucket = (
            f["priority_bucket"]
            .value_counts()
            .rename_axis("priority_bucket")
            .reset_index(name="users")
        )
        bucket["share_pct"] = (bucket["users"] / len(f) * 100).round(2)
        st.dataframe(bucket, use_container_width=True, hide_index=True)

    # -----------------------------
    # ✅ Strategic Conclusion
    # -----------------------------
    st.divider()
    st.markdown("### ✅ Retention Command Center")
    conclusion_payload = {
        "primary_focus_bucket": str(top_bucket_current),
        "high_risk_share_pct": float(high_ratio_current),
        "avg_churn_probability_pct": float(avg_prob_current),
    }
    st.markdown(
        safe_ai_explanation(
            "Retention Command Center",
            conclusion_payload,
            gemini_key,
            "Final action view: where risk is concentrated and which intervention buckets should be funded first.",
        )
    )

    end_left, end_right = st.columns([1.15, 1])
    with end_left:
        if {"priority_bucket", "churn_probability"}.issubset(f.columns):
            action_impact = (
                f.groupby("priority_bucket", as_index=False)
                .agg(
                    users=("user_id", "count"),
                    avg_churn_probability=("churn_probability", "mean"),
                )
                .sort_values(["avg_churn_probability", "users"], ascending=[False, False])
            )
            action_impact["weighted_risk_load"] = action_impact["users"] * action_impact["avg_churn_probability"]
            st.markdown("#### Weighted Risk Load by Intervention")
            st.bar_chart(action_impact.set_index("priority_bucket")["weighted_risk_load"])
        else:
            st.info("Priority bucket columns are missing for final impact view.")

    with end_right:

        st.markdown("#### Retention Priority Quadrant")

        st.markdown(
            """
            <style>

            .quadrant-wrapper {
                display: grid;
                grid-template-columns: 1fr 1fr;
                border: 1px solid #243044;
                border-radius: 18px;
                overflow: hidden;
                margin-top: 0.6rem;
                background: #0B1220;
            }

            .quadrant-box {
                min-height: 190px;
                padding: 1.15rem;
                border-right: 1px solid #243044;
                border-bottom: 1px solid #243044;
                position: relative;
            }

            .quadrant-box:nth-child(2),
            .quadrant-box:nth-child(4) {
                border-right: none;
            }

            .quadrant-box:nth-child(3),
            .quadrant-box:nth-child(4) {
                border-bottom: none;
            }

            .quadrant-label {
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.5px;
                color: #60A5FA;
                margin-bottom: 0.6rem;
            }

            .quadrant-title {
                font-size: 1.08rem;
                font-weight: 700;
                color: #F9FAFB;
                margin-bottom: 0.75rem;
                line-height: 1.3;
            }

            .quadrant-metric {
                font-size: 2rem;
                font-weight: 800;
                color: #FFFFFF;
                line-height: 1;
                margin-bottom: 0.55rem;
            }

            .quadrant-sub {
                color: #9CA3AF;
                font-size: 0.82rem;
                line-height: 1.45;
            }

            .quadrant-axis-x {
                text-align: center;
                color: #9CA3AF;
                font-size: 0.78rem;
                margin-top: 0.6rem;
                letter-spacing: 0.4px;
            }

            .quadrant-axis-y {
                position: absolute;
                top: 50%;
                left: -2.1rem;
                transform: rotate(-90deg);
                color: #9CA3AF;
                font-size: 0.78rem;
                letter-spacing: 0.4px;
            }

            </style>
            """,
            unsafe_allow_html=True,
        )

        high_risk_high_value = len(
            f[
                (f["rfm_segment"] == "High Value") &
                (f["churn_risk"] == "High")
            ]
        )

        dormant_users = len(
            f[
                f["lifecycle_stage"] == "Dormant"
            ]
        )

        stable_users = len(
            f[
                (f["churn_risk"] == "Low") &
                (f["rfm_segment"] == "High Value")
            ]
        )

        reengage_users = len(
            f[
                (f["engagement_score"] < f["engagement_score"].median()) &
                (f["churn_risk"] == "High")
            ]
        )

        q1, q2 = st.columns(2)
        q3, q4 = st.columns(2)

        with q1:
            st.markdown(
                f"""
                <div class="quadrant-box">
                    <div class="quadrant-label">HIGH RISK • HIGH VALUE</div>
                    <div class="quadrant-title">
                        Immediate Retention
                    </div>
                    <div class="quadrant-metric">
                        {high_risk_high_value:,}
                    </div>
                    <div class="quadrant-sub">
                        Critical users requiring proactive retention intervention and personalized recovery campaigns.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with q2:
            st.markdown(
                f"""
                <div class="quadrant-box">
                    <div class="quadrant-label">LOW RISK • HIGH VALUE</div>
                    <div class="quadrant-title">
                        Loyalty Expansion
                    </div>
                    <div class="quadrant-metric">
                        {stable_users:,}
                    </div>
                    <div class="quadrant-sub">
                        Stable high-value users suitable for upsell, ecosystem expansion, and loyalty optimization.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with q3:
            st.markdown(
                f"""
                <div class="quadrant-box">
                    <div class="quadrant-label">HIGH RISK • LOW VALUE</div>
                    <div class="quadrant-title">
                        Automated Re-engagement
                    </div>
                    <div class="quadrant-metric">
                        {reengage_users:,}
                    </div>
                    <div class="quadrant-sub">
                        Low-engagement users requiring scalable behavioral nudges and lifecycle reactivation flows.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with q4:
            st.markdown(
                f"""
                <div class="quadrant-box">
                    <div class="quadrant-label">LOW RISK • LOW VALUE</div>
                    <div class="quadrant-title">
                        Monitor & Nurture
                    </div>
                    <div class="quadrant-metric">
                        {dormant_users:,}
                    </div>
                    <div class="quadrant-sub">
                        Stable portfolio users monitored through passive engagement and retention tracking systems.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="quadrant-axis-x">
                LOW VALUE ⟶ ⟶ ⟶ BUSINESS VALUE ⟶ ⟶ ⟶ HIGH VALUE
            </div>
            """,
            unsafe_allow_html=True,
        )
if __name__ == "__main__":
    main()