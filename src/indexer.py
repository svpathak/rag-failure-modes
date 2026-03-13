import sys
from pathlib import Path
import chromadb

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHROMA_DIR, CHROMA_COLLECTION, BATCH_SIZE
from src.embedder import load_model, get_embeddings


def build_index(chunks, collection_name=None):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    name = collection_name if collection_name is not None else CHROMA_COLLECTION
    collection = client.get_or_create_collection(name)

    if collection.count() > 0:
        print(f"Index already exists with {collection.count()} chunks. Skipping.")
        return collection

    tokenizer, model = load_model()

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        embeddings = get_embeddings([c["text"] for c in batch], tokenizer, model).tolist()

        collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings,
            documents=[c["text"] for c in batch],
            metadatas=[{"paper_id": c["paper_id"], "section_name": c["section_name"],
                         "section_idx": c["section_idx"], "chunk_index": c["chunk_index"]}
                        for c in batch]
        )
        print(f"Indexed {min(i + BATCH_SIZE, len(chunks))} / {len(chunks)}")

    print(f"Done. Total indexed: {collection.count()}")
    return collection


def load_index(collection_name=None):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    name = collection_name if collection_name is not None else CHROMA_COLLECTION
    return client.get_collection(name)


if __name__ == "__main__":
    from config import TRAIN_JSON, DEV_JSON
    from data_loader import load_papers
    from chunker import chunk_papers

    papers = load_papers(TRAIN_JSON, DEV_JSON)
    chunks = chunk_papers(papers)
    build_index(chunks)