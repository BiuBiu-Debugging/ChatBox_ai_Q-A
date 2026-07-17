import os
from dotenv import load_dotenv

load_dotenv()


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
VECTOR_DB_DIR = os.path.join(DATA_DIR, "vector_db")

TOP_K = int(os.getenv("TOP_K", "5"))

API_KEY = os.getenv("API_KEY", "")

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))


os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)


RATE_LIMIT = os.getenv("RATE_LIMIT", "30/minute")

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "qwen2.5:latest"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-embed-text:latest"
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))

CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


FILE_STATUSES = (
    "pending",
    "embedded",
    "failed"
)