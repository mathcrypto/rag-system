# RETRIEVAL — embed the query, return k nearest Chroma splits.
#
# Vector search: the query is embedded into the same space as the indexed
# chunks, then compared against the vector DB to find the most relevant
# information for the question. That similarity search uses Approximate
# Nearest Neighbors (ANN). Common tools: FAISS (large-scale) or ChromaDB
# (small–medium retrieval) — we use Chroma here.
from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

import config
from embedding.openai_embedder import get_embeddings


def get_retriever(
    persist_directory: Path | None = None,
    k: int | None = None,
) -> BaseRetriever:
    persist_directory = persist_directory or config.PERSIST_DIR
    k = config.RETRIEVAL_K if k is None else k
    store = Chroma(
        persist_directory=str(persist_directory),
        embedding_function=get_embeddings(),
        collection_name=config.COLLECTION_NAME,
    )
    return store.as_retriever(search_kwargs={"k": k})


def retrieve(
    query: str,
    k: int | None = None,
    persist_directory: Path | None = None,
) -> list[Document]:
    # return the top k nearest neighbors (chunks) on invoke
    return get_retriever(persist_directory=persist_directory, k=k).invoke(query)
