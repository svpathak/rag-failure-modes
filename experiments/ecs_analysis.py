import sys
import ast
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    TRAIN_JSON, DEV_JSON, OUTPUT_DIR, ANALYSIS_DIR, PLOTS_DIR,
    ECS_THRESHOLD, FAITH_THRESHOLD, CONDITION_TO_COLLECTION
)
from src.data_loader import load_papers, build_qa_records
from src.indexer import load_index

# Script-level config
EXP_CONFIGS = [
    {
        "name":          "exp1_boundary",
        "input_csv":     OUTPUT_DIR / "exp1_boundary.csv",
        "output_csv":    OUTPUT_DIR / "exp1_boundary_ecs.csv",
        "encoding":      "latin-1",
        "collection":    "qasper_chunks",
        "condition_col": "condition",
    },
    {
        "name":          "exp2_distraction",
        "input_csv":     OUTPUT_DIR / "exp2_distraction.csv",
        "output_csv":    OUTPUT_DIR / "exp2_distraction_ecs.csv",
        "encoding":      "latin-1",
        "collection":    "qasper_chunks",
        "condition_col": "condition",
    },
    {
        "name":          "exp3_multihop",
        "input_csv":     OUTPUT_DIR / "exp3_multihop.csv",
        "output_csv":    OUTPUT_DIR / "exp3_multihop_ecs.csv",
        "encoding":      "utf-8",
        "collection":    "qasper_chunks",
        "condition_col": "condition",
    },
    {
        "name":          "exp4_chunksize",
        "input_csv":     OUTPUT_DIR / "exp4_chunksize.csv",
        "output_csv":    OUTPUT_DIR / "exp4_chunksize_ecs.csv",
        "encoding":      "utf-8",
        "collection":    None,
        "condition_col": "condition",
    },
]

def load_csv_with_fallback(path, primary_enc="utf-8", fallback_enc="latin-1"):
    try:
        return pd.read_csv(path, encoding=primary_enc)
    except UnicodeDecodeError:
        print(f"  [warn] utf-8 failed for {path.name}, retrying with {fallback_enc}")
        return pd.read_csv(path, encoding=fallback_enc)


def flatten_gold_evidence(gold_evidence_lol):
    """Flatten list-of-lists (one per annotator), deduplicate, strip blanks."""
    return list({
        para
        for annotator in gold_evidence_lol
        for para in annotator
        if para and para.strip()
    })


def token_recall(gold_para, chunk_text):
    gold_tokens  = set(gold_para.lower().split())
    chunk_tokens = set(chunk_text.lower().split())
    if not gold_tokens:
        return 0.0
    return len(gold_tokens & chunk_tokens) / len(gold_tokens)


def evidence_coverage_score(retrieved_chunks, gold_evidence_paragraphs):
    if not gold_evidence_paragraphs:
        return 0.0
    covered = 0
    for para in gold_evidence_paragraphs:
        for chunk in retrieved_chunks:
            if token_recall(para, chunk["text"]) >= ECS_THRESHOLD:
                covered += 1
                break
    return covered / len(gold_evidence_paragraphs)


def fetch_chunks_from_chroma(collection, chunk_ids):
    try:
        results = collection.get(ids=chunk_ids, include=["documents", "metadatas"])
        return [
            {"text": doc, **meta}
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]
    except Exception as e:
        print(f"  [warn] ChromaDB fetch failed for ids {chunk_ids[:2]}...: {e}")
        return []


# ============================================================================
# TASK 1 - Build qa_lookup + join gold_evidence_flat onto all CSVs
# ============================================================================
print("\n" + "="*60)
print("TASK 1 - Setup + Join")
print("="*60)

print("Loading papers (train + dev)...")
papers = load_papers(TRAIN_JSON, DEV_JSON)
qa_records = build_qa_records(papers)

clean_records = [
    r for r in qa_records
    if not any("BIBREF" in ans or "TABREF" in ans for ans in r["gold_answers"])
]
print(f"  Total QA records: {len(qa_records)}")
print(f"  After BIBREF filter: {len(clean_records)}")

qa_lookup = {r["question_id"]: r for r in clean_records}
print(f"  qa_lookup size: {len(qa_lookup)} unique question_ids\n")

dataframes = {}

for cfg in EXP_CONFIGS:
    name = cfg["name"]
    print(f"[{name}] Loading {cfg['input_csv'].name}...")

    df = load_csv_with_fallback(cfg["input_csv"], fallback_enc=cfg["encoding"])
    print(f"  Rows: {len(df)} | Columns: {list(df.columns)}")

    # Normalize columns across experiments
    if "proxy_faithfulness" not in df.columns:
        df["proxy_faithfulness"] = float("nan")   # exp1, exp2
    if "exact_match" not in df.columns:
        df["exact_match"] = float("nan")          # exp3, exp4

    df["retrieved_chunk_ids_parsed"] = df["retrieved_chunk_ids"].apply(ast.literal_eval)

    def get_flat_evidence(qid):
        record = qa_lookup.get(qid)
        if record is None:
            return None
        return flatten_gold_evidence(record["gold_evidence"])

    df["gold_evidence_flat"] = df["question_id"].apply(get_flat_evidence)

    null_count = df["gold_evidence_flat"].isna().sum()
    if null_count > 0:
        missing_ids = df[df["gold_evidence_flat"].isna()]["question_id"].tolist()
        raise AssertionError(
            f"[{name}] {null_count} rows failed to join on question_id.\n"
            f"  Missing IDs (first 5): {missing_ids[:5]}\n"
            f"  Likely BIBREF-filtered or wrong split."
        )

    empty_count = df["gold_evidence_flat"].apply(lambda x: len(x) == 0).sum()
    if empty_count > 0:
        print(f"  [warn] {empty_count} rows have empty gold_evidence - ECS will be 0.0 for these.")

    print(f"  Join OK - no nulls. Avg evidence paras/row: "
          f"{df['gold_evidence_flat'].apply(len).mean():.2f}\n")

    dataframes[name] = (df, cfg)

print("Task 1 complete.\n")


# ============================================================================
# TASK 2 - Compute ECS on all 4 experiment CSVs
# ============================================================================
print("="*60)
print("TASK 2 - Compute ECS")
print("="*60)

print("Loading ChromaDB collections...")
loaded_collections = {}
for cname in ["qasper_chunks", "qasper_chunks_256", "qasper_chunks_1024"]:
    try:
        loaded_collections[cname] = load_index(collection_name=cname)
        print(f"  {cname}: {loaded_collections[cname].count()} chunks")
    except Exception as e:
        print(f"  [warn] Could not load {cname}: {e}")
print()

for cfg in EXP_CONFIGS:
    name = cfg["name"]
    df, _ = dataframes[name]
    print(f"[{name}] Computing ECS for {len(df)} rows...")

    ecs_scores = []

    for _, row in df.iterrows():
        chunk_ids = row["retrieved_chunk_ids_parsed"]

        # exp4: pick collection based on condition value
        if name == "exp4_chunksize":
            coll_name = CONDITION_TO_COLLECTION.get(str(row["condition"]), "qasper_chunks")
        else:
            coll_name = cfg["collection"]

        collection = loaded_collections.get(coll_name)
        if collection is None:
            ecs_scores.append(0.0)
            continue

        retrieved_chunks = fetch_chunks_from_chroma(collection, chunk_ids)
        ecs = evidence_coverage_score(retrieved_chunks, row["gold_evidence_flat"])
        ecs_scores.append(round(ecs, 4))

    df["ecs"] = ecs_scores
    df.to_csv(cfg["output_csv"], index=False, encoding="utf-8")
    print(f"  Mean ECS: {df['ecs'].mean():.4f} | Saved: {cfg['output_csv'].name}\n")

    dataframes[name] = (df, cfg)

print("Task 2 complete.\n")


# ============================================================================
# TASK 3 - Core comparison: ECS vs proxy_faithfulness (exp3 focus)
# ============================================================================
print("="*60)
print("TASK 3 - ECS vs proxy_faithfulness Analysis")
print("="*60)

df3, _ = dataframes["exp3_multihop"]

# Headline numbers ──────────────────────────────────────────────────────────
high_faith_low_ecs = (
    (df3["proxy_faithfulness"] > FAITH_THRESHOLD) &
    (df3["ecs"] < ECS_THRESHOLD)
)
print(f"\nExp3 overall (n={len(df3)}):")
print(f"  proxy_faith > {FAITH_THRESHOLD} but ECS < {ECS_THRESHOLD} : "
      f"{high_faith_low_ecs.sum()} rows ({100*high_faith_low_ecs.mean():.1f}%)")

for cond in df3["condition"].unique():
    mask = df3["condition"] == cond
    subset = df3[mask]
    hf_le = (
        (subset["proxy_faithfulness"] > FAITH_THRESHOLD) &
        (subset["ecs"] < ECS_THRESHOLD)
    )
    print(f"  [{cond}] n={len(subset)} | high_faith_low_ECS: "
          f"{hf_le.sum()} ({100*hf_le.mean():.1f}%)")

# All experiments: mean ECS vs mean proxy_faithfulness per condition
print("\nAll experiments - mean ECS vs mean proxy_faithfulness per condition:")
for name, (df, cfg) in dataframes.items():
    print(f"\n  {name}:")
    for cond in df[cfg["condition_col"]].unique():
        subset = df[df[cfg["condition_col"]] == cond]
        print(f"    [{cond}] n={len(subset)} | "
              f"mean_faith={subset['proxy_faithfulness'].mean():.4f} | "
              f"mean_ECS={subset['ecs'].mean():.4f}")

# Scatter plot: ECS vs proxy_faithfulness on exp3
fig, ax = plt.subplots(figsize=(8, 6))
colors = {"single_hop": "#2196F3", "multi_hop": "#F44336"}

for cond, grp in df3.groupby("condition"):
    ax.scatter(
        grp["proxy_faithfulness"], grp["ecs"],
        c=colors.get(cond, "grey"), label=cond, alpha=0.7, edgecolors="none", s=60
    )

ax.axvline(FAITH_THRESHOLD, color="grey", linestyle="--", linewidth=0.8)
ax.axhline(ECS_THRESHOLD,   color="grey", linestyle="--", linewidth=0.8)
ax.axvspan(FAITH_THRESHOLD, 1.0, ymin=0, ymax=0.5, alpha=0.08, color="#F44336")
ax.text(0.76, 0.15, "Silent\nfailure", transform=ax.transAxes,
        fontsize=9, color="#F44336", ha="center")

ax.set_xlabel("proxy_faithfulness", fontsize=11)
ax.set_ylabel("ECS (Evidence Coverage Score)", fontsize=11)
ax.set_title("ECS vs proxy_faithfulness - Exp3 (Multi-hop vs Single-hop)", fontsize=12)
ax.legend()
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "ecs_vs_faithfulness_exp3.png", dpi=150)
plt.close()
print(f"\n  Plot saved: plots/ecs_vs_faithfulness_exp3.png")

print("\nTask 3 complete.\n")


# ============================================================================
# TASK 4 - Summary table across all experiments and conditions
# ============================================================================
print("="*60)
print("TASK 4 - Summary Table")
print("="*60)

summary_rows = []

for name, (df, cfg) in dataframes.items():
    for cond in sorted(df[cfg["condition_col"]].unique()):
        subset = df[df[cfg["condition_col"]] == cond]
        hf_le = (
            (subset["proxy_faithfulness"] > FAITH_THRESHOLD) &
            (subset["ecs"] < ECS_THRESHOLD)
        )
        summary_rows.append({
            "experiment": name,
            "condition": cond,
            "n": len(subset),
            "mean_f1": round(subset["f1"].mean(), 4),
            "mean_proxy_faith": round(subset["proxy_faithfulness"].mean(), 4),
            "mean_ecs": round(subset["ecs"].mean(), 4),
            "high_faith_low_ecs_pct": round(100 * hf_le.mean(), 1),
        })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

print("\nTask 4 complete.\n")

# ============================================================================
# TASK 5 - Save all outputs
# ============================================================================
print("="*60)
print("TASK 5 - Save Outputs")
print("="*60)

# Summary table
summary_df.to_csv(ANALYSIS_DIR / "summary_table.csv", index=False)
print(f"  summary_table.csv saved")

# Plot: Mean ECS vs mean proxy_faithfulness across all conditions
fig, ax = plt.subplots(figsize=(10, 5))
x = range(len(summary_df))
x_labels = [f"{r['experiment']}\n[{r['condition']}]" for _, r in summary_df.iterrows()]

ax.bar(x, summary_df["mean_ecs"],        color="#2196F3", alpha=0.8, label="mean ECS")
ax.bar(x, summary_df["mean_proxy_faith"], color="#FF9800", alpha=0.5, label="mean proxy_faith",
       width=0.4)
ax.set_xticks(list(x))
ax.set_xticklabels(x_labels, fontsize=8)
ax.set_ylabel("Score")
ax.set_title("Mean ECS vs Mean proxy_faithfulness - All Conditions")
ax.legend()
ax.set_ylim(0, 1.0)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "ecs_by_condition_all_exps.png", dpi=150)
plt.close()
print(f"  ecs_by_condition_all_exps.png saved")

# Plot: F1 vs ECS scatter across all experiments
fig, ax = plt.subplots(figsize=(8, 6))
exp_colors = {
    "exp1_boundary":    "#9C27B0",
    "exp2_distraction": "#2196F3",
    "exp3_multihop":    "#F44336",
    "exp4_chunksize":   "#4CAF50",
}
for name, (df, _) in dataframes.items():
    ax.scatter(
        df["f1"], df["ecs"],
        c=exp_colors.get(name, "grey"), label=name, alpha=0.4, s=20, edgecolors="none"
    )
ax.set_xlabel("F1", fontsize=11)
ax.set_ylabel("ECS", fontsize=11)
ax.set_title("F1 vs ECS - All Experiments", fontsize=12)
ax.legend(fontsize=8)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "f1_vs_ecs_scatter.png", dpi=150)
plt.close()
print(f"  f1_vs_ecs_scatter.png saved")

print("\n" + "="*60)
print("Phase 3 complete.")
print("="*60)
print(f"\nOutputs:")
print(f"  outputs/exp1_boundary_ecs.csv")
print(f"  outputs/exp2_distraction_ecs.csv")
print(f"  outputs/exp3_multihop_ecs.csv")
print(f"  outputs/exp4_chunksize_ecs.csv")
print(f"  analysis/summary_table.csv")
print(f"  analysis/plots/ecs_vs_faithfulness_exp3.png")
print(f"  analysis/plots/ecs_by_condition_all_exps.png")
print(f"  analysis/plots/f1_vs_ecs_scatter.png")
