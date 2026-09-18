# MULTI-QUERY RETRIEVAL
#
# One entry point: expand question → retrieve for each phrasing → unique-union docs.
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import config
from retrieval.retriever import get_retriever

_MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template(
    """You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from a vector
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search.
Provide these alternative questions separated by newlines. Original question: {question}"""
)


def retrieve_multi_query(question: str, k: int | None = None) -> list[Document]:
    # 1) LLM rephrases the question into several search queries
    text = (
        _MULTI_QUERY_PROMPT
        | ChatOpenAI(
            model=config.OPENAI_CHAT_MODEL,
            temperature=config.OPENAI_TEMPERATURE,
        )
        | StrOutputParser()
    ).invoke({"question": question})
    queries = [q.strip() for q in text.split("\n") if q.strip()]
    if question.strip() and question.strip() not in queries:
        queries = [question.strip(), *queries]
    queries = queries[: config.MULTI_QUERY_COUNT + 1]

    # 2) Retrieve top-k for each query 
    k = config.RETRIEVAL_K if k is None else k
    retriever = get_retriever(k=k)
    ranked_lists = retriever.map().invoke(queries)

    # 3) Unique union via dumps/loads 
    flattened = [dumps(doc) for docs in ranked_lists for doc in docs]
    return [loads(doc) for doc in set(flattened)]
