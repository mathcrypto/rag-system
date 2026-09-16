# OpenAI embedding client (LangChain OpenAIEmbeddings)
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# Default matches langchain_openai / current OpenAI RAG examples.
DEFAULT_EMBED_MODEL = "text-embedding-3-small"


def get_embeddings(model: str = DEFAULT_EMBED_MODEL) -> OpenAIEmbeddings:
    """Return a LangChain OpenAIEmbeddings instance (reads OPENAI_API_KEY)."""
    return OpenAIEmbeddings(model=model)


if __name__ == "__main__":
    embeddings = get_embeddings()
    vector = embeddings.embed_query("hello from the RAG system")
    print(f"model={DEFAULT_EMBED_MODEL} dim={len(vector)}")
