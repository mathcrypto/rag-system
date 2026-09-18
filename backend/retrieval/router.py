# ROUTING — pick the best retriever strategy for a question.
#
# Logical routing (prompted classification): an LLM reads the question and
# picks one of dense | bm25 | multi_query | rag_fusion | hybrid.
# This is not semantic routing (no embedding similarity to route examples).
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import config
from retrieval.bm25_retriever import retrieve_bm25
from retrieval.hybrid_retriever import retrieve_hybrid
from retrieval.multi_query import retrieve_multi_query
from retrieval.rag_fusion import retrieve_rag_fusion
from retrieval.retriever import retrieve

STRATEGIES = ("dense", "bm25", "multi_query", "rag_fusion", "hybrid")

_ROUTER_PROMPT = ChatPromptTemplate.from_template(
    """You route a user question to the best retrieval strategy.

Strategies:
- dense: semantic vector search (default for natural-language questions)
- bm25: keyword / exact-term search (IDs, error codes, rare exact phrases)
- multi_query: rewrite the question several ways, unique-union the hits
- rag_fusion: rewrite several ways, fuse ranked lists with Reciprocal Rank Fusion
- hybrid: combine BM25 keywords + dense semantics when both matter

Return ONLY one token from this list: dense, bm25, multi_query, rag_fusion, hybrid

Question: {question}"""
)


def route(question: str) -> str:
    # LLM chooses which retriever path to use.
    raw = (
        _ROUTER_PROMPT
        | ChatOpenAI(
            model=config.OPENAI_CHAT_MODEL,
            temperature=0,
        )
        | StrOutputParser()
    ).invoke({"question": question})
    choice = raw.strip().lower().split()[0].strip(".,;:\"'")
    return choice if choice in STRATEGIES else "dense"


def retrieve_with_strategy(
    question: str,
    strategy: str = "dense",
    k: int | None = None,
) -> list[Document]:
    # Ask-time step 7: dispatch to the chosen retriever
    k = config.RETRIEVAL_K if k is None else k
    if strategy == "bm25":
        return retrieve_bm25(question, k=k)
    if strategy == "multi_query":
        return retrieve_multi_query(question, k=k)
    if strategy == "rag_fusion":
        return retrieve_rag_fusion(question, k=k)
    if strategy == "hybrid":
        return retrieve_hybrid(question, k=k)
    return retrieve(question, k=k)


def retrieve_routed(question: str, k: int | None = None) -> tuple[str, list[Document]]:
    # Ask-time step 7 with logical routing: LLM picks strategy, then retrieve
    strategy = route(question)
    return strategy, retrieve_with_strategy(question, strategy=strategy, k=k)
