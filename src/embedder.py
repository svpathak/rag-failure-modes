import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EMBEDDING_MODEL


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL)
    model.eval()
    return tokenizer, model


def get_embeddings(texts, tokenizer, model):
    encoded = tokenizer(texts, padding=True, truncation=True,
                        max_length=512, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded)
    # mean pooling
    embeddings = output.last_hidden_state.mean(dim=1).numpy().astype("float32")
    return embeddings
