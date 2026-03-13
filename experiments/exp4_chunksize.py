import sys
import csv
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (TRAIN_JSON, DEV_JSON, OUTPUT_DIR, EVAL_SAMPLE_SIZE,
                    EVAL_RANDOM_SEED, CHUNK_CONDITIONS)
from src.data_loader import load_papers, build_qa_records, sample_records
from src.chunker import chunk_papers
from src.embedder import load_model
from src.indexer import build_index, load_index
from src.retriever import retrieve
from src.generator import get_llm_client, generate
from src.evaluator import compute_f1


def proxy_faithfulness(predicted_answer, retrieved_chunks):
    answer_tokens = set(predicted_answer.lower().split())
    context_tokens = set(
        " ".join(c['text'] for c in retrieved_chunks).lower().split()
    )
    if not answer_tokens:
        return 0.0
    return round(len(answer_tokens & context_tokens) / len(answer_tokens), 4)


if __name__ == "__main__":
    # --- 1. Data loading + filtering ---
    papers_list = load_papers(TRAIN_JSON, DEV_JSON)
    qa_records = build_qa_records(papers_list)
    clean_records = [
        r for r in qa_records
        if not any("BIBREF" in ans or "TABREF" in ans
                   for ans in r["gold_answers"])
    ]
    clean_records = sample_records(clean_records, EVAL_SAMPLE_SIZE, EVAL_RANDOM_SEED)
    print(f"Step 1: Done - {len(clean_records)} records")

    # --- 2. Load embedder + Groq client ---
    tokenizer, model = load_model()
    client = get_llm_client()
    print("Step 2: Done")

    # --- 3. Build missing indexes ---
    print("Building indexes...")
    for cond in CHUNK_CONDITIONS:
        cname = cond["collection_name"]
        try:
            col = load_index(collection_name=cname)
            print(f"  {cname} already exists ({col.count()} chunks), skipping.")
        except Exception:
            print(f"  Building {cname}...")
            chunks = chunk_papers(
                papers_list,
                chunk_size=cond["chunk_size"],
                chunk_overlap=cond["chunk_overlap"]
            )
            build_index(chunks, collection_name=cname)
            print(f"  {cname} built.")
    print("Step 3: Done")

    # --- 4. Main loop — one pass per chunk size condition ---
    results = []
    for cond in CHUNK_CONDITIONS:
        cname = cond["collection_name"]
        chunk_size = cond["chunk_size"]
        collection = load_index(collection_name=cname)
        print(f"\n--- Running condition: chunk_size={chunk_size} ---")

        total = len(clean_records)
        for i, record in enumerate(clean_records):
            print(f"  [{i+1}/{total}] {record['paper_id']} — {record['question'][:60]}")

            retrieved_chunks = retrieve(
                record['question'], record['paper_id'],
                collection, tokenizer, model
            )
            print(f"    retrieved: {len(retrieved_chunks)} chunks")

            predicted_answer = generate(record['question'], retrieved_chunks, client)
            print(f"    answer: {predicted_answer[:80]}")

            best_gold = max(record['gold_answers'],
                            key=lambda a: compute_f1(predicted_answer, a))
            f1 = compute_f1(predicted_answer, best_gold)
            idk = predicted_answer.strip().lower().startswith("i don't know") or \
                  predicted_answer.strip().lower().startswith("i cannot")
            faith = proxy_faithfulness(predicted_answer, retrieved_chunks)

            print(f"    f1: {f1:.3f} | idk: {idk} | faith: {faith:.3f}")

            results.append({
                "question_id": record['question_id'],
                "paper_id": record['paper_id'],
                "question": record['question'],
                "gold_answer": best_gold,
                "predicted_answer": predicted_answer,
                "retrieved_chunk_ids": str([c['chunk_id'] for c in retrieved_chunks]),
                "retrieved_section_names": str([c['section_name'] for c in retrieved_chunks]),
                "f1": f1,
                "idk": idk,
                "proxy_faithfulness": faith,
                "experiment": "exp4_chunksize",
                "condition": str(chunk_size)
            })

    # --- 5. Write CSV ---
    with open(OUTPUT_DIR / "exp4_chunksize.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone. {len(results)} total rows written.")

    # --- 6. Quick summary ---
    for cond in CHUNK_CONDITIONS:
        size = str(cond["chunk_size"])
        group = [r for r in results if r['condition'] == size]
        f1s = [r['f1'] for r in group]
        faiths = [r['proxy_faithfulness'] for r in group]
        idks = sum(1 for r in group if r['idk'])
        print(f"\nchunk_size={size} (n={len(group)})")
        print(f"  avg F1: {sum(f1s)/len(f1s):.4f}")
        print(f"  avg faith: {sum(faiths)/len(faiths):.4f}")
        print(f"  IDK: {idks} ({100*idks/len(group):.1f}%)")