import sys
import csv
import os
import ast
import asyncio
import functools
import time
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
from src.evaluator import compute_f1, compute_em


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


def build_ragas_llm():
    from ragas.llms import LangchainLLMWrapper
    from langchain_groq import ChatGroq

    chat = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.environ["GROQ_API_KEY"],
        max_tokens=2048
    )

    original_create = chat.client.create

    @functools.wraps(original_create)
    def patched_create(*args, **kwargs):
        kwargs.pop("n", None)
        time.sleep(20)
        return original_create(*args, **kwargs)

    original_async_create = chat.async_client.create

    async def patched_async_create(*args, **kwargs):
        kwargs.pop("n", None)
        await asyncio.sleep(20)
        return await original_async_create(*args, **kwargs)

    chat.client.create = patched_create
    chat.async_client.create = patched_async_create

    return LangchainLLMWrapper(chat)


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
            "retrieved_chunks_text": [c['text'] for c in retrieved_chunks],
            "retrieved_chunk_ids": str([c['chunk_id'] for c in retrieved_chunks]),
            "retrieved_section_names": str([c['section_name'] for c in retrieved_chunks]),
            "f1": f1,
            "exact_match": em,
            "idk": idk,
            "experiment": "exp3_multihop",
            "condition": condition
        })

    print("Step 3: Done")

    # --- 4. RAGAS evaluation ---
    print("Running RAGAS...")
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_recall
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.run_config import RunConfig
        from langchain_huggingface import HuggingFaceEmbeddings
        from datasets import Dataset

        groq_llm = build_ragas_llm()

        ragas_embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        )

        faithfulness.llm = groq_llm
        answer_relevancy.llm = groq_llm
        answer_relevancy.embeddings = ragas_embeddings
        context_recall.llm = groq_llm

        ragas_data = Dataset.from_list([{
            "question": r["question"],
            "answer": r["predicted_answer"],
            "contexts": r["retrieved_chunks_text"],
            "ground_truth": r["gold_answer"]
        } for r in results])

        ragas_scores = ragas_evaluate(
            ragas_data,
            metrics=[faithfulness, answer_relevancy, context_recall],
            run_config=RunConfig(
                max_workers=1,
                timeout=120,
                max_retries=0,
                max_wait=60
            )
        )
        scores_df = ragas_scores.to_pandas()

        for i, row in scores_df.iterrows():
            results[i]["ragas_faithfulness"] = round(row.get("faithfulness", 0.0), 4)
            results[i]["ragas_answer_relevancy"] = round(row.get("answer_relevancy", 0.0), 4)
            results[i]["ragas_context_recall"] = round(row.get("context_recall", 0.0), 4)

        print("RAGAS: Done")

    except Exception as e:
        print(f"[WARN] RAGAS failed: {e}")
        for r in results:
            r["ragas_faithfulness"] = None
            r["ragas_answer_relevancy"] = None
            r["ragas_context_recall"] = None

    # --- 5. Write CSV ---
    for r in results:
        r.pop("retrieved_chunks_text", None)

    with open(OUTPUT_DIR / "exp3_multihop.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Done. {len(results)} records written.")
    print(f"single_hop: {sum(1 for r in results if r['condition'] == 'single_hop')}")
    print(f"multi_hop: {sum(1 for r in results if r['condition'] == 'multi_hop')}")