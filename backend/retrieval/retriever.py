# Dense (vector) retrieval — reopen the Chroma index from build_index (step 4).
from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

import config
from embedding.openai_embedder import get_embeddings


def get_vectorstore(persist_directory: Path | None = None) -> Chroma:
    # Reload data/vectordb written during indexing (no re-load of data/raw/)
    persist_directory = persist_directory or config.PERSIST_DIR
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=get_embeddings(),
        collection_name=config.COLLECTION_NAME,
    )


def get_indexed_documents(persist_directory: Path | None = None) -> list[Document]:
    """Load all chunks currently stored in Chroma (used to build BM25)."""
    store = get_vectorstore(persist_directory)
    raw = store.get(include=["documents", "metadatas"])
    docs: list[Document] = []
    for text, meta in zip(raw.get("documents") or [], raw.get("metadatas") or []):
        if text:
            docs.append(Document(page_content=text, metadata=meta or {}))
    return docs


def get_retriever(
    persist_directory: Path | None = None,
    k: int | None = None,
) -> BaseRetriever:
    # Used in step 7: return top-k nearest chunks for a query
    k = config.RETRIEVAL_K if k is None else k
    return get_vectorstore(persist_directory).as_retriever(search_kwargs={"k": k})


def retrieve(
    query: str,
    k: int | None = None,
    persist_directory: Path | None = None,
) -> list[Document]:
    # Dense strategy path for step 7
    return get_retriever(persist_directory=persist_directory, k=k).invoke(query)
