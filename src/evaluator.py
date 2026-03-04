import sys
import string
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TOP_K, RESULTS_FILE


def normalize(text):
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


def compute_f1(prediction, gold_answers):
    pred_tokens = normalize(prediction)
    best_f1 = 0.0

    for gold in gold_answers:
        gold_tokens = normalize(gold)

        common = set(pred_tokens) & set(gold_tokens)
        if not common:
            continue

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    return best_f1


def compute_em(prediction, gold_answers):
    pred_norm = " ".join(normalize(prediction))
    return float(any(pred_norm == " ".join(normalize(g)) for g in gold_answers))


def evaluate(qa_records, collection, tokenizer, model, llm_client):
    from src.retriever import retrieve
    from src.generator import generate

    results = []

    for i, record in enumerate(qa_records):
        question = record["question"]
        paper_id = record["paper_id"]
        gold = record["gold_answers"]

        retrieved = retrieve(question, paper_id, collection, tokenizer, model, TOP_K)

        # Generate
        try:
            predicted = generate(question, retrieved, llm_client)
        except Exception as e:
            print(f"[WARN] Generation failed for q {i}: {e}")
            predicted = ""

        # Score
        f1 = compute_f1(predicted, gold)
        em = compute_em(predicted, gold)

        results.append({
            "question_id": record["question_id"],
            "paper_id": paper_id,
            "question": question,
            "gold_answers": gold,
            "predicted_answer": predicted,
            "f1": round(f1, 4),
            "em": round(em, 4),
            "retrieved_chunk_ids": [c["chunk_id"] for c in retrieved]
        })

        print(f"[{i+1}/{len(qa_records)}] F1: {f1:.2f} | EM: {em:.0f} | Q: {question[:60]}")

    df = pd.DataFrame(results)

    avg_f1 = df["f1"].mean()
    avg_em = df["em"].mean()

    print("\n" + "=" * 50)
    print(f"Average F1 : {avg_f1:.4f}")
    print(f"Exact Match : {avg_em:.4f}")
    print("=" * 50)

    df.to_csv(RESULTS_FILE, index=False)
    print(f"\nResults saved to {RESULTS_FILE}")

    return df