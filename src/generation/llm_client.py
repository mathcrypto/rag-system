# GENERATION
#
# After retrieval we have relevant splits. Generation stuffs those into a prompt
# with the user question and asks the LLM to answer using only that context.
# LCEL wires it as: question → retrieve → format context → prompt → LLM → string.
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from generation.prompt_builder import format_docs, get_prompt
from retrieval.retriever import DEFAULT_K, get_retriever

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0


def get_llm(
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=temperature)


def get_rag_chain(k: int = DEFAULT_K):
    # Full RAG chain 
    #   {"context": retriever | format_docs, "question": RunnablePassthrough()}
    #   | prompt | llm | StrOutputParser()
    #
    # - retriever: embed question, fetch k nearby splits
    # - format_docs: turn Document list into a single {context} string
    # - RunnablePassthrough: forward the raw question into {question}
    # - prompt: fill the ChatPromptTemplate
    # - llm: generate the answer
    # - StrOutputParser: return plain text instead of an AIMessage
    retriever = get_retriever(k=k)
    prompt = get_prompt()
    llm = get_llm()

    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def answer(question: str, k: int = DEFAULT_K) -> str:
    # One-shot: invoke the chain with just a question string
    # (unlike prompt|llm which needs {"context": docs, "question": ...}).
    return get_rag_chain(k=k).invoke(question)


if __name__ == "__main__":
    question = "What is RAG chunking?"
    print(f"Q: {question}")
    print(f"A: {answer(question, k=1)}")
