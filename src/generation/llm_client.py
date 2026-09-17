# GENERATION — question → retrieve → prompt → LLM → answer string.
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

import config
from generation.prompt_builder import PROMPT, format_docs
from retrieval.multi_query import retrieve_multi_query
from retrieval.retriever import get_retriever


def answer(
    question: str,
    k: int | None = None,
    multi_query: bool = False,
) -> str:
    k = config.RETRIEVAL_K if k is None else k
    llm = ChatOpenAI(
        model=config.OPENAI_CHAT_MODEL,
        temperature=config.OPENAI_TEMPERATURE,
    )

    if multi_query:
        get_context = RunnableLambda(
            lambda q: format_docs(retrieve_multi_query(q, k=k))
        )
    else:
        get_context = get_retriever(k=k) | format_docs

    chain = (
        {"context": get_context, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    # invoke the chain with the question
    return chain.invoke(question)
