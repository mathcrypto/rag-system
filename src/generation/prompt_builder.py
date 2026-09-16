# PROMPT
#
# Builds the RAG prompt that tells the LLM: answer ONLY from retrieved context.
# Same idea as ChatPromptTemplate.from_template(...) in rag-from-scratch
# (or hub.pull("rlm/rag-prompt") — we keep the template local in git).
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

RAG_TEMPLATE = """Answer the question based only on the following context:
{context}

Question: {question}
"""


def get_prompt() -> ChatPromptTemplate:
    # Notebook: prompt = ChatPromptTemplate.from_template(template)
    return ChatPromptTemplate.from_template(RAG_TEMPLATE)


def format_docs(docs: list[Document]) -> str:
    # Retriever returns Document objects; the prompt needs a text {context}.
    return "\n\n".join(doc.page_content for doc in docs)


def build_messages(question: str, docs: list[Document]):
    # Manual path (notebook partial chain):
    #   chain = prompt | llm
    #   chain.invoke({"context": docs, "question": "..."})
    # Here we only build the filled messages (no LLM yet).
    prompt = get_prompt()
    return prompt.invoke(
        {
            "context": format_docs(docs),
            "question": question,
        }
    )


if __name__ == "__main__":
    from retrieval.retriever import retrieve

    question = "What is RAG chunking?"
    docs = retrieve(question, k=1)
    messages = build_messages(question, docs)
    print(messages)
