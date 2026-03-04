import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TRAIN_JSON, DEV_JSON, EVAL_SAMPLE_SIZE, EVAL_RANDOM_SEED
from src.data_loader import load_papers, build_qa_records, sample_records
from src.indexer import load_index
from src.embedder import load_model
from src.generator import get_llm_client
from src.evaluator import evaluate


if __name__ == "__main__":
    print("Loading papers...")
    papers = load_papers(TRAIN_JSON, DEV_JSON)
    records = build_qa_records(papers)
    sample = sample_records(records, n=EVAL_SAMPLE_SIZE, seed=EVAL_RANDOM_SEED)
    print(f"Eval sample: {len(sample)} questions\n")

    print("Loading index...")
    collection = load_index()
    print(f"Index loaded: {collection.count()} chunks\n")

    print("Loading embedding model...")
    tokenizer, model = load_model()
    print("Model loaded\n")

    print("Loading Groq client...")
    llm_client = get_llm_client()
    print("Client loaded\n")

    print("Running evaluation...")
    df = evaluate(sample, collection, tokenizer, model, llm_client)