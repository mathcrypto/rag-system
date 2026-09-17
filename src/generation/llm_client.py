# GENERATION — question → (route) retrieve → (optional rerank) → prompt → LLM.
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

import config
from generation.prompt_builder import PROMPT, format_docs
from reranking.cohere_reranker import rerank
from retrieval.router import retrieve_routed, retrieve_with_strategy


def _get_docs(
    question: str,
    k: int,
    strategy: str | None,
    use_routing: bool,
    use_rerank: bool,
) -> list[Document]:
    if use_routing:
        _chosen, docs = retrieve_routed(question, k=k)
    else:
        docs = retrieve_with_strategy(question, strategy=strategy or "dense", k=k)
    if use_rerank:
        docs = rerank(question, docs)
    return docs


def answer(
    question: str,
    k: int | None = None,
    strategy: str | None = None,
    multi_query: bool = False,
    use_routing: bool = False,
    use_rerank: bool = False,
) -> str:
    """
    strategy: dense | bm25 | multi_query | hybrid
    multi_query=True is a shortcut for strategy='multi_query'
    use_routing=True lets an LLM pick the strategy
    """
    k = config.RETRIEVAL_K if k is None else k
    if multi_query and strategy is None:
        strategy = "multi_query"

    llm = ChatOpenAI(
        model=config.OPENAI_CHAT_MODEL,
        temperature=config.OPENAI_TEMPERATURE,
    )
    get_context = RunnableLambda(
        lambda q: format_docs(
            _get_docs(
                q,
                k=k,
                strategy=strategy,
                use_routing=use_routing,
                use_rerank=use_rerank,
            )
        )
    )

    chain = (
        {"context": get_context, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain.invoke(question)
