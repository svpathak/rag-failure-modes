import sys
import csv
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TRAIN_JSON, DEV_JSON, OUTPUT_DIR, EVAL_SAMPLE_SIZE, EVAL_RANDOM_SEED
from src.data_loader import load_papers, build_qa_records, sample_records
from src.embedder import load_model
from src.indexer import load_index
from src.retriever import retrieve
from src.generator import get_llm_client, generate
from src.evaluator import compute_f1, compute_em


def token_overlap(text_a, text_b):
    def tokenize(t):
        return set(t.lower().strip().split())
    a, b = tokenize(text_a), tokenize(text_b)
    if not b:
        return 0.0
    return len(a & b) / len(b)


def evidence_retrieved(retrieved_chunks, gold_evidence_paragraphs, threshold=0.5):
    for chunk in retrieved_chunks:
        for para in gold_evidence_paragraphs:
            if token_overlap(chunk['text'], para) >= threshold:
                return True
    return False


if __name__ == "__main__":
    # --- 1. Data loading + filtering ---
    papers = load_papers(TRAIN_JSON, DEV_JSON)
    qa_records = build_qa_records(papers)
    clean_records = [
        r for r in qa_records
        if not any("BIBREF" in ans or "TABREF" in ans
                   for ans in r["gold_answers"])
    ]
    clean_records = sample_records(clean_records, EVAL_SAMPLE_SIZE, EVAL_RANDOM_SEED)
    print("Step 1: Done")

    # --- 2. Load embedder + ChromaDB collection ---
    tokenizer, model = load_model()
    collection = load_index()
    client = get_llm_client()
    print("Step 2: Done")

    # --- 3. Main loop ---
    results = []
    total = len(clean_records)
    for i, record in enumerate(clean_records):
        print(f"[{i+1}/{total}] {record['paper_id']} — {record['question'][:60]}")

        retrieved_chunks = retrieve(
            record['question'], record['paper_id'],
            collection, tokenizer, model
        )
        print(f"  retrieved: {len(retrieved_chunks)} chunks")

        # flatten gold evidence across annotators
        all_evidence = [para for annotator in record['gold_evidence'] for para in annotator]
        condition = "hit" if evidence_retrieved(retrieved_chunks, all_evidence) else "miss"
        print(f"  condition: {condition}")

        predicted_answer = generate(record['question'], retrieved_chunks, client)
        print(f"  answer: {predicted_answer[:80]}")

        best_gold = max(record['gold_answers'],
                        key=lambda a: compute_f1(predicted_answer, a))
        f1 = compute_f1(predicted_answer, best_gold)
        em = compute_em(predicted_answer, best_gold)
        idk = predicted_answer.strip().lower().startswith("i don't know") or \
              predicted_answer.strip().lower().startswith("i cannot")

        print(f"  f1: {f1:.3f} | em: {em} | idk: {idk}")

        results.append({
            "question_id": record['question_id'],
            "paper_id": record['paper_id'],
            "question": record['question'],
            "gold_answer": best_gold,
            "predicted_answer": predicted_answer,
            "retrieved_chunk_ids": str([c['chunk_id'] for c in retrieved_chunks]),
            "retrieved_section_names": str([c['section_name'] for c in retrieved_chunks]),
            "f1": f1,
            "exact_match": em,
            "idk": idk,
            "experiment": "exp2_distraction",
            "condition": condition
        })

    # --- 4. Write CSV ---
    with open(OUTPUT_DIR / "exp2_distraction.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Done. {len(results)} records written.")
    print(f"hit: {sum(1 for r in results if r['condition'] == 'hit')}")
    print(f"miss: {sum(1 for r in results if r['condition'] == 'miss')}")