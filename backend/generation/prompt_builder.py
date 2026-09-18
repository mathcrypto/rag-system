# PROMPT — answer only from {context} given {question}.
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

RAG_TEMPLATE = """Answer the question based only on the following context.
Reply in 2-3 concise sentences.

{context}

Question: {question}
"""

PROMPT = ChatPromptTemplate.from_template(RAG_TEMPLATE)


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)
