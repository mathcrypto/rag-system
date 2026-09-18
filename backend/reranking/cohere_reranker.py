# RERANKING — ask-time step 8 (after retrieve, before the answer prompt).
#
# Score (question, chunk) pairs more precisely and keep top_n.
from __future__ import annotations

from langchain_cohere import CohereRerank
from langchain_core.documents import Document

import config


def rerank(
    question: str,
    docs: list[Document],
    top_n: int | None = None,
) -> list[Document]:
    # Cohere cross-encoder style rerank; returns the best top_n chunks
    if not docs:
        return []
    top_n = config.RERANK_TOP_N if top_n is None else top_n
    compressor = CohereRerank(
        cohere_api_key=config.COHERE_API_KEY,
        model=config.RERANK_MODEL,
        top_n=min(top_n, len(docs)),
    )
    return list(compressor.compress_documents(documents=docs, query=question))
