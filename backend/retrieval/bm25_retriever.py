# BM25 RETRIEVAL — sparse keyword search (exact terms, rare words, IDs).
from __future__ import annotations

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

import config
from retrieval.retriever import get_indexed_documents


def get_bm25_retriever(k: int | None = None) -> BaseRetriever:
    k = config.RETRIEVAL_K if k is None else k
    docs = get_indexed_documents()
    if not docs:
        raise ValueError("No indexed documents found. Run scripts/build_index.py first.")
    return BM25Retriever.from_documents(docs, k=k)


def retrieve_bm25(query: str, k: int | None = None) -> list[Document]:
    return get_bm25_retriever(k=k).invoke(query)
