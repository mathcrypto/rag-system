# GENERATION — question → (route) retrieve → (optional rerank) → prompt → LLM.
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

import config
from generation.ask_result import AskResult
from generation.prompt_builder import PROMPT, format_docs
from reranking.cohere_reranker import rerank
from retrieval.router import retrieve_routed, retrieve_with_strategy


def _get_docs(
    question: str,
    k: int,
    strategy: str | None,
    use_routing: bool,
    use_rerank: bool,
) -> tuple[str, list[Document]]:
    # 7) Retrieve top chunks from the saved Chroma index (strategy or LLM route)
    if use_routing:
        chosen, docs = retrieve_routed(question, k=k)
    else:
        chosen = strategy or "dense"
        docs = retrieve_with_strategy(question, strategy=chosen, k=k)

    # 8) Optional Cohere rerank — keep the best top_n chunks
    if use_rerank:
        docs = rerank(question, docs)
    return chosen, docs


def ask(
    question: str,
    k: int | None = None,
    strategy: str | None = None,
    multi_query: bool = False,
    use_routing: bool = False,
    use_rerank: bool | None = None,
) -> AskResult:
    """
    Production entry: retrieve → (rerank) → generate.
    Returns answer text plus strategy used and source chunks.
    """
    # 5) Resolve defaults (how many chunks, strategy shortcuts, rerank on/off)
    k = config.RETRIEVAL_K if k is None else k
    if multi_query and strategy is None:
        strategy = "multi_query"
    if use_rerank is None:
        use_rerank = bool(config.COHERE_API_KEY)

    chosen, docs = _get_docs(
        question,
        k=k,
        strategy=strategy,
        use_routing=use_routing,
        use_rerank=use_rerank,
    )

    # 6) Chat model that will write the final answer
    llm = ChatOpenAI(
        model=config.OPENAI_CHAT_MODEL,
        temperature=config.OPENAI_TEMPERATURE,
    )

    # 9–11) context + question → prompt → LLM → plain text
    context = format_docs(docs)
    text = (
        PROMPT | llm | StrOutputParser()
    ).invoke({"context": context, "question": question})

    return AskResult(answer=text, strategy=chosen, sources=docs)


def answer(
    question: str,
    k: int | None = None,
    strategy: str | None = None,
    multi_query: bool = False,
    use_routing: bool = False,
    use_rerank: bool | None = None,
) -> str:
    """Convenience wrapper — same as ask(...).answer (scripts / demos)."""
    return ask(
        question,
        k=k,
        strategy=strategy,
        multi_query=multi_query,
        use_routing=use_routing,
        use_rerank=use_rerank,
    ).answer
