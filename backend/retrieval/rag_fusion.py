# RAG-FUSION — one option for ask-time step 7 (strategy='rag_fusion').
#
# Differs from multi_query.py: that module unique-unions hits; this one
# re-ranks with RRF so docs that appear across many query lists rise.
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import config
from retrieval.retriever import get_retriever

_RAG_FUSION_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant that generates multiple search queries based on a single input query.
Generate multiple search queries related to: {question}
Output ({n} queries):"""
)


def reciprocal_rank_fusion(
    results: list[list[Document]],
    rrf_k: int = 60,
) -> list[Document]:
    """Fuse ranked lists with RRF: score += 1 / (rank + rrf_k)."""
    fused_scores: dict[str, float] = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            fused_scores[doc_str] = fused_scores.get(doc_str, 0.0) + 1.0 / (rank + rrf_k)

    return [
        loads(doc_str)
        for doc_str, _score in sorted(
            fused_scores.items(), key=lambda item: item[1], reverse=True
        )
    ]


def retrieve_rag_fusion(question: str, k: int | None = None) -> list[Document]:
    # 7a) LLM writes several search queries from the user question
    n = config.MULTI_QUERY_COUNT
    text = (
        _RAG_FUSION_PROMPT
        | ChatOpenAI(
            model=config.OPENAI_CHAT_MODEL,
            temperature=0,
        )
        | StrOutputParser()
    ).invoke({"question": question, "n": n})
    queries = [q.strip() for q in text.split("\n") if q.strip()]
    cleaned: list[str] = []
    for q in queries:
        line = q.lstrip("0123456789.-) ").strip()
        if line:
            cleaned.append(line)
    queries = cleaned[:n] or [question]

    # 7b) Open Chroma retriever and search once per query
    k = config.RETRIEVAL_K if k is None else k
    retriever = get_retriever(k=k)
    ranked_lists = retriever.map().invoke(queries)

    # 7c) Fuse ranked lists with Reciprocal Rank Fusion, keep top-k
    fused = reciprocal_rank_fusion(ranked_lists)
    return fused[:k]
