import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TOP_K
from src.embedder import get_embeddings


def retrieve(question, paper_id, collection, tokenizer, model, top_k=TOP_K):
    query_embedding = get_embeddings([question], tokenizer, model).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where={"paper_id": paper_id}
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            **results["metadatas"][0][i]
        })

    return chunks