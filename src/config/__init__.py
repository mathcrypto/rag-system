# Central settings (loaded once from .env)
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

# Models
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))

# Retrieval / index
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))
MULTI_QUERY_COUNT = int(os.getenv("MULTI_QUERY_COUNT", "5"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_docs")
PERSIST_DIR = Path(os.getenv("PERSIST_DIR", str(ROOT / "data" / "vectordb")))

# Reranking (Cohere)
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
RERANK_MODEL = os.getenv("RERANK_MODEL", "rerank-english-v3.0")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "3"))

# Hybrid fusion weights (BM25 + dense)
HYBRID_BM25_WEIGHT = float(os.getenv("HYBRID_BM25_WEIGHT", "0.5"))
HYBRID_DENSE_WEIGHT = float(os.getenv("HYBRID_DENSE_WEIGHT", "0.5"))
