# RETRIEVAL
#
# After indexing, every split lives as a point in a high-dimensional embedding
# space. For a question we embed it too (project it into that same space),
# search around it for nearby document vectors, and grab the closest ones.
# k = how many nearby neighbors to fetch (1, 2, 3, … n).
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from embedding.openai_embedder import get_embeddings

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

PERSIST_DIR = ROOT / "data" / "vectordb"
COLLECTION_NAME = "rag_docs"
DEFAULT_K = 4  # number of nearby neighbors to return


def get_vectorstore(persist_directory: Path = PERSIST_DIR) -> Chroma:
    """Open the persisted Chroma collection (indexed splits in embedding space)."""
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=get_embeddings(),
        collection_name=COLLECTION_NAME,
    )


def get_retriever(
    persist_directory: Path = PERSIST_DIR,
    k: int = DEFAULT_K,
) -> BaseRetriever:
    # Embed query into the same space as the indexed splits, then return the
    # k nearest neighbors (e.g. k=1 → only the single closest split).
    return get_vectorstore(persist_directory).as_retriever(
        search_kwargs={"k": k},
    )


def retrieve(
    query: str,
    k: int = DEFAULT_K,
    persist_directory: Path = PERSIST_DIR,
) -> list[Document]:
    """Project query into embedding space → fetch k nearby splits."""
    return get_retriever(persist_directory=persist_directory, k=k).invoke(query)


if __name__ == "__main__":
    hits = retrieve("What is RAG chunking?")
    print(f"Sample retrieval hits: {len(hits)} (k={DEFAULT_K})")
    if hits:
        print(hits[0].page_content[:200])
