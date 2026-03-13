import sys
from pathlib import Path
from langchain_text_splitters import TokenTextSplitter

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_papers(papers, chunk_size=None, chunk_overlap=None):
    size = chunk_size if chunk_size is not None else CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else CHUNK_OVERLAP
    token_splitter = TokenTextSplitter(chunk_overlap=overlap, chunk_size=size)
    
    chunks = []
    for paper in papers:
        paper_id = paper["paper_id"]
        for section in paper["sections"]:
            section_idx = section["section_idx"]
            text = section["text"]
            if not text:
                continue
            splits = token_splitter.split_text(text)
            for split_idx, split in enumerate(splits):
                chunks.append({
                    "chunk_id": f"{paper_id}_{section_idx}_{split_idx}",
                    "paper_id": paper_id,
                    "section_name": section["section_name"],
                    "section_idx": section_idx,
                    "chunk_index": split_idx,
                    "text": split
                })
    return chunks

# For debugging/testing
if __name__ == "__main__":
    from config import TRAIN_JSON, DEV_JSON
    from src.data_loader import load_papers

    papers = load_papers(TRAIN_JSON, DEV_JSON)
    chunks = chunk_papers(papers)

    print(f"Total papers : {len(papers)}")
    print(f"Total chunks : {len(chunks)}")
    print(f"Avg chunks per paper : {len(chunks) / len(papers):.1f}")
    print()
    print("Sample chunk:")
    print(chunks[0])