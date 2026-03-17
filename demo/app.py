import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="RAG Failure Modes Explorer",
    page_icon="🔬",
    layout="wide",
)

@st.cache_data
def load_data():
    exp1 = pd.read_csv("outputs/exp1_boundary_ecs.csv", encoding="latin-1")
    exp2 = pd.read_csv("outputs/exp2_distraction_ecs.csv", encoding="latin-1")
    exp3 = pd.read_csv("outputs/exp3_multihop_ecs.csv", encoding="utf-8")
    exp4 = pd.read_csv("outputs/exp4_chunksize_ecs.csv", encoding="utf-8")
    summary = pd.read_csv("analysis/summary_table.csv")
    return exp1, exp2, exp3, exp4, summary

exp1, exp2, exp3, exp4, summary = load_data()

EXP_MAP = {
    "Exp 1 - Chunk Boundary": exp1,
    "Exp 2 - Retrieval Distraction": exp2,
    "Exp 3 - Multi-hop vs Single-hop": exp3,
    "Exp 4 - Chunk Size": exp4,
}

st.sidebar.title("Navigation")
view = st.sidebar.radio(
    "View",
    ["📊 Project Overview", "🔍 Question Explorer", "🔴 Silent Failure Cases"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Exploring pre-computed results from 4 experiments on RAG failure modes.  \n"
    "No API keys or model loading required."
)

def metric_badge(label: str, value: float, good_thresh: float = 0.5) -> str:
    colour = "#2ecc71" if value >= good_thresh else "#e74c3c"
    return (
        f'<span style="background:{colour};color:white;padding:2px 8px;'
        f'border-radius:4px;font-size:0.85em;margin-right:4px;">'
        f'{label}: {value:.3f}</span>'
    )

# =============================================
# VIEW 1 - Project Overview
# =============================================

if view == "📊 Project Overview":
    st.title("Diagnosing RAG Failure Modes on Long-Document QA")
    st.markdown(
        """
        This project empirically diagnoses failure modes in a standard RAG pipeline on
        **QASPER** - a dataset of NLP research-paper QA pairs that require reading
        across multiple document sections. Four experiments examine chunk boundary
        fragmentation, retrieval distraction, multi-hop reasoning, and chunk-size
        sensitivity. A custom retrieval-grounded metric -
        **Evidence Coverage Score (ECS)** - detects silent failures that standard
        token-overlap faithfulness scores miss.
        """
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric(
        label="Silent Failure Rate - 512-token condition",
        value="21.5%",
        help="faith > 0.5 AND ECS < 0.5, exp4 512-token condition",
    )
    col2.metric(
        label="Silent Failure Rate - Multi-hop",
        value="27.8%",
        help="faith > 0.5 AND ECS < 0.5, multi-hop condition (Exp 3)",
    )
    col3.metric(
        label="IDK rate drop under boundary cuts",
        value="31.9% to 12.5%",
        help="Clean condition vs boundary-cut condition (Exp 1)",
    )

    st.markdown("---")
    st.subheader("Summary Results")

    display_cols = [c for c in summary.columns]
    st.dataframe(summary[display_cols], use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Mean ECS vs Mean Faithfulness by Condition")

    plot_df = summary.dropna(subset=["mean_proxy_faith"]).copy()

    if plot_df.empty:
        st.info(
            "proxy_faithfulness is only available for Exp 3 and Exp 4.  "
            "The bar chart below uses those conditions only."
        )
    else:
        conditions = plot_df["condition"].astype(str)
        x = np.arange(len(conditions))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 4))
        bars_ecs = ax.bar(x - width / 2, plot_df["mean_ecs"], width, label="Mean ECS", color="#3498db")
        bars_faith = ax.bar(x + width / 2, plot_df["mean_proxy_faith"], width, label="Mean Faithfulness", color="#e67e22")

        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=25, ha="right", fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title("Mean ECS vs Mean Proxy Faithfulness (Exp 3 & Exp 4)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown(
        """
        **Reading the chart:** ECS and faithfulness move in *opposite directions* as
        chunk size grows (Exp 4). Larger chunks improve retrieval coverage (ECS up)
        but hurt generation faithfulness (faith down). 512 tokens is the sweet spot on
        both dimensions for QASPER.
        """
    )

# =============================================
# VIEW 2 - Question Explorer
# =============================================
elif view == "🔍 Question Explorer":
    st.title("Question Explorer")
    st.caption("Browse individual questions across all four experiments.")

    col_filter1, col_filter2 = st.columns([1, 1])

    with col_filter1:
        exp_choice = st.selectbox("Select experiment", list(EXP_MAP.keys()))

    df_selected = EXP_MAP[exp_choice]
    conditions  = sorted(df_selected["condition"].unique().tolist())

    with col_filter2:
        cond_choice = st.selectbox("Select condition", conditions)

    df_view = df_selected[df_selected["condition"] == cond_choice].reset_index(drop=True)
    has_faith = "proxy_faithfulness" in df_view.columns

    st.markdown(f"**{len(df_view)} questions** in *{exp_choice} / {cond_choice}*")

    display_cols = ["question", "f1", "ecs"]
    if has_faith:
        display_cols.append("proxy_faithfulness")
    display_cols.append("idk")

    df_table = df_view[display_cols].copy()
    df_table["question"] = df_table["question"].str[:90] + "…"

    selected_rows = st.dataframe(
        df_table,
        use_container_width=True,
        hide_index=False,
        on_select="rerun",
        selection_mode="single-row",
    )

    chosen_indices = selected_rows.selection.rows if selected_rows.selection else []

    if chosen_indices:
        idx = chosen_indices[0]
        row = df_view.iloc[idx]

        st.markdown("---")
        st.subheader("Question Detail")

        # Silent failure flag
        is_silent_failure = (
            has_faith
            and row.get("proxy_faithfulness", 0) > 0.5
            and row.get("ecs", 1) < 0.5
        )
        if is_silent_failure:
            st.error("🔴 **Silent Failure** - faithfulness > 0.5 but ECS < 0.5")

        st.markdown(f"**Question:** {row['question']}")
        st.markdown(f"**Gold Answer:** {row.get('gold_answer', '-')}")
        st.markdown(f"**Predicted Answer:** {row.get('predicted_answer', '-')}")

        sections = row.get("retrieved_section_names", "-")
        st.markdown(f"**Retrieved Sections:** `{sections}`")

        st.markdown("**Scores:**")
        badge_html = metric_badge("F1", float(row.get("f1", 0)))
        badge_html += metric_badge("ECS", float(row.get("ecs", 0)))
        if has_faith:
            badge_html += metric_badge("Faithfulness", float(row.get("proxy_faithfulness", 0)))
        st.markdown(badge_html, unsafe_allow_html=True)

        idk_val = str(row.get("idk", "")).strip().lower()
        st.markdown(f"**IDK response:** {'✅ Yes' if idk_val == 'true' else '❌ No'}")
    else:
        st.info("Click a row to expand its full detail card.")

# =============================================
# VIEW 3 - Silent Failure Cases
# =============================================
elif view == "🔴 Silent Failure Cases":
    st.title("Cases Where the System Appears Faithful But Retrieved Wrong Context")
    st.markdown(
        """
        **Silent failures** are responses where proxy faithfulness > 0.5 (the answer
        looks grounded in retrieved context) but ECS < 0.5 (gold evidence paragraphs
        were never retrieved). Standard metrics would classify these as acceptable;
        ECS exposes them.

        *This view uses Exp 3 (multi-hop vs single-hop) only - the only experiment
        with both proxy_faithfulness and ECS, and the one with the highest silent
        failure rate.*
        """
    )

    st.markdown("---")

    sf_mask = (exp3["proxy_faithfulness"] > 0.5) & (exp3["ecs"] < 0.5)
    df_sf = exp3[sf_mask].reset_index(drop=True)

    total_exp3 = len(exp3)
    sf_count = len(df_sf)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Exp 3 questions", total_exp3)
    c2.metric("Silent failure cases", sf_count)
    c3.metric("Silent failure rate", f"{100 * sf_count / total_exp3:.1f}%")

    # Per-condition breakdown
    breakdown = (
        exp3.groupby("condition")
        .apply(lambda g: pd.Series({
            "n": len(g),
            "silent_count": ((g["proxy_faithfulness"] > 0.5) & (g["ecs"] < 0.5)).sum(),
            "silent_pct": 100 * ((g["proxy_faithfulness"] > 0.5) & (g["ecs"] < 0.5)).mean(),
        }))
        .reset_index()
    )
    st.subheader("By Condition")
    st.dataframe(breakdown, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.subheader("Faithfulness vs ECS - Silent Failure Quadrant")

    cond_vals = exp3["condition"].unique().tolist()
    colours = {"single_hop": "#3498db", "multi_hop": "#e74c3c"}
    fallback = ["#9b59b6", "#1abc9c", "#f39c12", "#2c3e50"]

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, cond in enumerate(cond_vals):
        subset = exp3[exp3["condition"] == cond]
        col = colours.get(cond, fallback[i % len(fallback)])
        ax.scatter(
            subset["proxy_faithfulness"],
            subset["ecs"],
            label=cond,
            color=col,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.4,
            s=60,
        )

    # Highlight the silent failure quadrant
    ax.axvline(x=0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(y=0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.fill_between(
        [0.5, 1.0], [0, 0], [0.5, 0.5],
        color="#e74c3c", alpha=0.07, label="Silent failure zone",
        zorder=0,
    )

    ax.set_xlabel("Proxy Faithfulness (answer-to-context overlap)", fontsize=11)
    ax.set_ylabel("ECS (gold evidence recall)", fontsize=11)
    ax.set_title("Faithfulness vs ECS - Exp 3 (multi-hop vs single-hop)", fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    ax.text(
        0.76, 0.05,
        "Looks faithful\nWrong context",
        fontsize=8.5, color="#c0392b",
        ha="center", style="italic",
    )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        """
        Points in the **lower-right quadrant** (faithfulness > 0.5, ECS < 0.5) are
        silent failures. The model generated an answer that appears grounded in its
        retrieved context - but the gold evidence paragraphs were never retrieved.
        Multi-hop questions are disproportionately represented in this quadrant.
        """
    )

    st.markdown("---")
    st.subheader("Silent Failure Row Browser")

    if df_sf.empty:
        st.info("No silent failure cases found with the current threshold (faith > 0.5, ECS < 0.5).")
    else:
        browse_cols = ["condition", "question", "proxy_faithfulness", "ecs", "f1"]
        df_browse   = df_sf[browse_cols].copy()
        df_browse["question"] = df_browse["question"].str[:90] + "…"

        sel = st.dataframe(
            df_browse,
            use_container_width=True,
            hide_index=False,
            on_select="rerun",
            selection_mode="single-row",
        )

        chosen = sel.selection.rows if sel.selection else []
        if chosen:
            row = df_sf.iloc[chosen[0]]
            st.markdown("---")
            st.markdown(f"**Question:** {row['question']}")
            st.markdown(f"**Gold Answer:** {row.get('gold_answer', '-')}")
            st.markdown(f"**Predicted Answer:** {row.get('predicted_answer', '-')}")
            st.markdown(f"**Retrieved Sections:** `{row.get('retrieved_section_names', '-')}`")
            badge_html = (
                metric_badge("F1", float(row["f1"]))
                + metric_badge("ECS", float(row["ecs"]))
                + metric_badge("Faithfulness", float(row["proxy_faithfulness"]))
            )
            st.markdown(badge_html, unsafe_allow_html=True)
        else:
            st.info("Click a row to see the full question detail.")