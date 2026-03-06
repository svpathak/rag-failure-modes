from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
CHROMA_DIR = ROOT_DIR / "chroma_db"

TRAIN_JSON = DATA_DIR / "qasper-train-v0.3.json"
DEV_JSON = DATA_DIR / "qasper-dev-v0.3.json"

OUTPUT_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

# Chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Embedding
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 64

# Retrieval
TOP_K = 3
CHROMA_COLLECTION = "qasper_chunks"

# Generation
GROQ_MODEL = "llama-3.1-8b-instant"
MAX_OUTPUT_TOKENS = 256
TEMPERATURE = 0.0

# Evaluation
EVAL_SAMPLE_SIZE = 100 # questions for Phase 1 baseline
EVAL_RANDOM_SEED = 42
BASELINE_RESULTS_FILE  = OUTPUT_DIR / "results_baseline.csv"