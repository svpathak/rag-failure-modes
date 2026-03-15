# Stress-Testing Long-Document RAG on Multi-Section Reasoning Tasks

Empirical study of RAG failure modes on long-document multi-section QA, with a retrieval-grounded evaluation metric (ECS) that detects silent failures standard metrics miss.

---

## Abstract

Standard RAG evaluation measures whether a generated answer is grounded in retrieved context - but not whether the retrieved context was correct in the first place. This project stress-tests a fixed-size chunking RAG pipeline on QASPER (NLP research paper QA) across four controlled experiments: chunk boundary fragmentation, retrieval distraction, multi-hop reasoning, and chunk-size sensitivity. A custom metric - Evidence Coverage Score (ECS) - is introduced to measure retrieval quality against gold evidence annotations. Results show that token-overlap faithfulness scores remain above 0.5 in 21.5% of cases where gold evidence was never retrieved, a failure class that standard metrics cannot detect.

---

## Key Findings

- **21.5%** of baseline responses appear faithful (token-overlap > 0.5) but retrieved wrong context (ECS < 0.5)
- **Chunk boundary cuts** cause confident wrong answers rather than abstentions - a harder-to-detect failure mode than IDK responses
- **Multi-hop queries** show 27.8% silent failure rate vs 20.9% for single-section queries
- **Larger chunks improve retrieval coverage (ECS) but degrade generation faithfulness** - 512 tokens is the sweet spot on QASPER
- **ECS perfectly separates retrieval hit from miss** (0.858 vs 0.000 in the distraction experiment) - validates it as a retrieval quality signal

---

## Results

| experiment | condition | n | mean_f1 | mean_proxy_faith | mean_ecs | high_faith_low_ecs_pct |
|---|---|---|---|---|---|---|
| baseline | clean_512 | 100 | 0.2077 | - | - | - |
| exp1_boundary | boundary_cut | 40 | 0.0378 | - | 0.6737 | - |
| exp1_boundary | clean | 160 | 0.0754 | - | 0.5501 | - |
| exp2_distraction | hit | 134 | 0.0657 | - | 0.8579 | - |
| exp2_distraction | miss | 66 | 0.0921 | - | 0.0000 | - |
| exp3_multihop | single_hop | 182 | 0.0676 | 0.5757 | 0.5904 | 20.9% |
| exp3_multihop | multi_hop | 18 | 0.1431 | 0.4967 | 0.4169 | 27.8% |
| exp4_chunksize | 256 | 200 | 0.1068 | 0.5083 | 0.4451 | 20.5% |
| exp4_chunksize | 512 | 200 | 0.0744 | 0.5676 | 0.5748 | 21.5% |
| exp4_chunksize | 1024 | 200 | 0.0556 | 0.4420 | 0.6497 | 16.0% |

*Exp 1 and Exp 2 have no proxy_faithfulness column; high_faith_low_ecs_pct shows 0.0 for these as an artifact of missing data, not a real finding.*

---

## The ECS Metric

**Evidence Coverage Score** measures what fraction of gold evidence paragraphs were actually retrieved, using QASPER's ground truth annotations rather than relying on the generated answer.

```
ECS = token_recall(gold_paragraphs, retrieved_chunks)
```

This is orthogonal to faithfulness. Faithfulness checks answer-to-context direction (is the answer grounded in what was retrieved?). ECS checks context-to-ground-truth direction (was the right context retrieved at all?). A response can score high on both, high on one and low on the other, or low on both. The high-faithfulness / low-ECS quadrant is the dangerous one - the system appears to be working while answering from wrong evidence.

---

## Stack

```
Dataset    : QASPER (1169 NLP research papers, multi-section QA)
Chunking   : TokenTextSplitter, 512 tokens, 50 overlap (deliberately naive)
Embeddings : all-MiniLM-L6-v2 via transformers
Vector DB  : ChromaDB (3 collections: 256 / 512 / 1024 tokens)
Generation : Groq API, llama-3.1-8b-instant, top_k=3
Metrics    : Token F1, proxy_faithfulness (token overlap), ECS (gold para recall)
```

---

## Run the Demo (no API keys required)

The demo is a pre-computed results explorer. It loads CSVs from `outputs/` and `analysis/` - no model loading, no ChromaDB queries, no environment setup beyond the packages below.

```bash
git clone https://github.com/svpathak/rag-stress-test
cd rag-stress-test
pip install streamlit pandas matplotlib
streamlit run demo/app.py
```

---

## Project Structure

```
rag-stress-test/
- analysis/
  - summary_table.csv
  - plots/
- data/
  - qasper-dev-v0.3.json
  - qasper-train-v0.3.json
- demo/
  - app.py
- experiments/
  - ecs_analysis.py
  - exp1_boundary.py
  - exp2_distraction.py
  - exp3_multihop.py
  - exp4_chunksize.py
- outputs/
  - exp1_boundary.csv
  - exp1_boundary_ecs.csv
  - exp2_distraction.csv
  - exp2_distraction_ecs.csv
  - exp3_multihop.csv
  - exp3_multihop_ecs.csv
  - exp4_chunksize.csv
  - exp4_chunksize_ecs.csv
  - results_baseline.csv
  - results_baseline_clean.csv
- scripts/
  - run_eval.py
- src/
  - __init__.py
  - chunker.py
  - data_loader.py
  - embedder.py
  - evaluator.py
  - generator.py
  - indexer.py
  - retriever.py
- config.py
- README.md
- requirements.txt
```

---

## Caveats

- Multi-hop experiment (Exp 3) has n=18 multi-hop questions. The 27.8% silent failure rate is directionally consistent with the hypothesis but should not be treated as a definitive estimate.
- Phase 2 absolute F1 values are lower than baseline due to BIBREF filtering and sampling variance. Within-experiment comparisons are the valid unit of analysis, not cross-experiment F1.
- ECS threshold of 0.5 for classifying silent failures is reasonable but arbitrary. The finding is robust to small threshold changes but the exact percentage shifts.

---

## Dataset

```
@inproceedings{dasigi-etal-2021-dataset,
  title     = {A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers},
  author    = {Dasigi, Pradeep and Lo, Kyle and Beltagy, Iz and Cohan, Arman and Smith, Noah A. and Gardner, Matt},
  booktitle = {Proceedings of NAACL 2021},
  year      = {2021}
}
```