
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

PDF_PATH = BASE_DIR /  "The Hitchhiker’s Guide to Agentic AI.pdf"
DATA_DIR = BASE_DIR / "rag_data"

CHUNKS_PATH = DATA_DIR / "chunks.json"
META_PATH = DATA_DIR / "metadata.json"
SEMANTIC_INDEX_PATH = DATA_DIR / "semantic-qwen3-embedding-0.6b.faiss"
CHUNK_TOKENIZER_PATH = DATA_DIR / "qwen_tokenizer" / "tokenizer.json"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 120
CHUNK_UNIT = "tokens"

OLLAMA_MODEL = "qwen3.5:0.8b"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
OLLAMA_EMBED_URL = "http://127.0.0.1:11434/api/embed"

BM25_TOP_K = 8
FINAL_TOP_K = 5
SEMANTIC_TOP_K = 8

DATA_DIR.mkdir(parents=True, exist_ok=True)
