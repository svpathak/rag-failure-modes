import sys
import csv
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TRAIN_JSON, DEV_JSON, OUTPUT_DIR, EVAL_SAMPLE_SIZE, EVAL_RANDOM_SEED
from src.data_loader import load_papers, build_qa_records, sample_records
from src.embedder import load_model
from src.indexer import load_index
from src.retriever import retrieve
from src.generator import get_llm_client, generate
from src.evaluator import compute_f1


def token_overlap(text_a, text_b):
    a = set(text_a.lower().strip().split())
    b = set(text_b.lower().strip().split())
    if not b:
        return 0.0
    return len(a & b) / len(b)


def match_para_to_section(para, sections, threshold=0.3):
    best_section = None
    best_score = 0.0
    for section in sections:
        score = token_overlap(para, section["text"])
        if score > best_score:
            best_score = score
            best_section = section["section_name"]
    return best_section if best_score >= threshold else None


def classify_hop_type(gold_evidence, sections):
    labels = []
    for annotator_paras in gold_evidence:
        sections_hit = set()
        for para in annotator_paras:
            if len(para.strip()) < 20:
                continue
            section = match_para_to_section(para, sections)
            if section:
                sections_hit.add(section)
        if len(sections_hit) >= 2:
            labels.append("multi_hop")
        else:
            labels.append("single_hop")
    return "multi_hop" if labels.count("multi_hop") > labels.count("single_hop") else "single_hop"


def proxy_faithfulness(predicted_answer, retrieved_chunks):
    """Fraction of answer tokens that appear in retrieved context — no LLM needed."""
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
    papers = {p['paper_id']: p for p in papers_list}
    qa_records = build_qa_records(papers_list)
    clean_records = [
        r for r in qa_records
        if not any("BIBREF" in ans or "TABREF" in ans
                   for ans in r["gold_answers"])
    ]
    clean_records = sample_records(clean_records, EVAL_SAMPLE_SIZE, EVAL_RANDOM_SEED)
    print(f"Step 1: Done - {len(clean_records)} records")

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

        paper = papers[record['paper_id']]
        sections = paper.get("sections", [])
        condition = classify_hop_type(record['gold_evidence'], sections)
        print(f"  condition: {condition}")

        retrieved_chunks = retrieve(
            record['question'], record['paper_id'],
            collection, tokenizer, model
        )
        print(f"  retrieved: {len(retrieved_chunks)} chunks")

        predicted_answer = generate(record['question'], retrieved_chunks, client)
        print(f"  answer: {predicted_answer[:80]}")

        best_gold = max(record['gold_answers'],
                        key=lambda a: compute_f1(predicted_answer, a))
        f1 = compute_f1(predicted_answer, best_gold)
        idk = predicted_answer.strip().lower().startswith("i don't know") or \
              predicted_answer.strip().lower().startswith("i cannot")
        faith = proxy_faithfulness(predicted_answer, retrieved_chunks)

        print(f"  f1: {f1:.3f} | idk: {idk} | faithfulness: {faith:.3f}")

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
            "experiment": "exp3_multihop",
            "condition": condition
        })

    print("Step 3: Done")

    # --- 4. Write CSV ---
    with open(OUTPUT_DIR / "exp3_multihop.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Done. {len(results)} records written.")
    print(f"single_hop: {sum(1 for r in results if r['condition'] == 'single_hop')}")
    print(f"multi_hop:  {sum(1 for r in results if r['condition'] == 'multi_hop')}")

    # --- 5. Quick summary ---
    single = [r for r in results if r['condition'] == 'single_hop']
    multi = [r for r in results if r['condition'] == 'multi_hop']
    for group, name in [(single, 'single_hop'), (multi, 'multi_hop')]:
        f1s = [r['f1'] for r in group]
        faiths = [r['proxy_faithfulness'] for r in group]
        print(f"\n{name} (n={len(group)})")
        print(f"  avg F1: {sum(f1s)/len(f1s):.4f}")
        print(f"  avg proxy_faith: {sum(faiths)/len(faiths):.4f}")
        print(f"  IDK: {sum(1 for r in group if r['idk'])}")