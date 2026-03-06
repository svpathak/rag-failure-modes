import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TRAIN_JSON, DEV_JSON, CHROMA_DIR, CHROMA_COLLECTION
from src.data_loader import load_papers, build_qa_records
from src.chunker import chunk_papers
from src.embedder import load_model
from src.indexer import load_index
from src.retriever import retrieve
from src.generator import get_llm_client, generate
from src.evaluator import compute_f1, compute_em

def classify_boundary_cut(gold_evidence_paragraphs, paper_chunks):
    sorted_chunks = sorted(paper_chunks, key=lambda x: (x['section_idx'], x['chunk_index']))

    for para in gold_evidence_paragraphs:
        if len(para) < 100:
            continue # too short
        para_start = para[:50].strip()
        para_end = para[-50:].strip()

        for i, chunk in enumerate(sorted_chunks):
            if para_start in chunk['text']:
                if i < len(sorted_chunks) - 1:
                    next_chunk = sorted_chunks[i + 1]
                    same_section = next_chunk['section_idx'] == chunk['section_idx']
                    consecutive = next_chunk['chunk_index'] == chunk['chunk_index'] + 1
                    if same_section and consecutive and para_end in next_chunk['text']:
                        return "boundary_cut"

    return "clean"

if __name__ == "__main__":
    # --- 1. Data loading + filtering ---
    papers = load_papers(TRAIN_JSON, DEV_JSON)
    qa_records = build_qa_records(papers)
    clean_records = [
        r for r in qa_records
        if not any("BIBREF" in ans or "TABREF" in ans
                   for ans in r["gold_answers"])
    ]

    # --- 2. Build paper_id → chunks lookup ---
    all_chunks = chunk_papers(papers)  # <-- papers, not clean_records
    chunks_by_paper = {}
    for chunk in all_chunks:
        chunks_by_paper.setdefault(chunk['paper_id'], []).append(chunk)

    # --- 3. Load embedder + ChromaDB collection ---
    tokenizer, model = load_model()
    collection = load_index(CHROMA_DIR, CHROMA_COLLECTION)
    client = get_llm_client()

    # --- 4. Main loop ---
    results = []
    for record in clean_records:
        paper_chunks = chunks_by_paper.get(record['paper_id'], [])
        condition = classify_boundary_cut(record['gold_evidence'], paper_chunks)

        retrieved_chunks = retrieve(
            record['question'], record['paper_id'],
            collection, tokenizer, model
        )
        predicted_answer = generate(record['question'], retrieved_chunks, client)

        # pick best gold answer (highest F1 against prediction)
        best_gold = max(record['gold_answers'],
                        key=lambda a: compute_f1(predicted_answer, a))
        f1 = compute_f1(predicted_answer, best_gold)
        em = compute_em(predicted_answer, best_gold)
        idk = predicted_answer.strip().lower().startswith("i don't know") or \
              predicted_answer.strip().lower().startswith("i cannot")

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
            "experiment": "exp1_boundary",
            "condition": condition
        })

    # --- 5. Write CSV ---
    import csv, os
    os.makedirs("results", exist_ok=True)
    with open("results/exp1_boundary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Done. {len(results)} records written.")
    print(f"boundary_cut: {sum(1 for r in results if r['condition'] == 'boundary_cut')}")
    print(f"clean: {sum(1 for r in results if r['condition'] == 'clean')}")
