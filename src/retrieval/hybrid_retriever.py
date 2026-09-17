# HYBRID RETRIEVAL — fuse BM25 (keywords) + dense (embeddings).
from __future__ import annotations

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

import config
from retrieval.bm25_retriever import get_bm25_retriever
from retrieval.retriever import get_retriever


def get_hybrid_retriever(k: int | None = None) -> BaseRetriever:
    # EnsembleRetriever merges ranked lists from both searchers (weighted fusion).
    k = config.RETRIEVAL_K if k is None else k
    dense = get_retriever(k=k)
    bm25 = get_bm25_retriever(k=k)
    return EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[config.HYBRID_BM25_WEIGHT, config.HYBRID_DENSE_WEIGHT],
    )


def retrieve_hybrid(query: str, k: int | None = None) -> list[Document]:
    return get_hybrid_retriever(k=k).invoke(query)
