# ROUTING — pick the best retriever strategy for a question.
#
# Options: dense | bm25 | multi_query | hybrid
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import config
from retrieval.bm25_retriever import retrieve_bm25
from retrieval.hybrid_retriever import retrieve_hybrid
from retrieval.multi_query import retrieve_multi_query
from retrieval.retriever import retrieve

STRATEGIES = ("dense", "bm25", "multi_query", "hybrid")

_ROUTER_PROMPT = ChatPromptTemplate.from_template(
    """You route a user question to the best retrieval strategy.

Strategies:
- dense: semantic vector search (default for natural-language questions)
- bm25: keyword / exact-term search (IDs, error codes, rare exact phrases)
- multi_query: rewrite the question several ways for broader recall when phrasing may miss docs
- hybrid: combine BM25 keywords + dense semantics when both matter

Return ONLY one token from this list: dense, bm25, multi_query, hybrid

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
    k = config.RETRIEVAL_K if k is None else k
    if strategy == "bm25":
        return retrieve_bm25(question, k=k)
    if strategy == "multi_query":
        return retrieve_multi_query(question, k=k)
    if strategy == "hybrid":
        return retrieve_hybrid(question, k=k)
    return retrieve(question, k=k)


def retrieve_routed(question: str, k: int | None = None) -> tuple[str, list[Document]]:
    # Route then retrieve; returns (chosen_strategy, docs).
    strategy = route(question)
    return strategy, retrieve_with_strategy(question, strategy=strategy, k=k)
