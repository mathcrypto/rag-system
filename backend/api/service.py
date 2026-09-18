# RagService — production caller of generation.llm_client.ask.
from __future__ import annotations

from pathlib import Path

import config
from api.schemas import AskRequest, AskResponse, HealthResponse, SourceChunk
from generation.llm_client import ask
from retrieval.retriever import get_vectorstore


class RagService:
    """HTTP/API-facing wrapper around the RAG pipeline."""

    def health(self) -> HealthResponse:
        persist = Path(config.PERSIST_DIR)
        index_ready = False
        try:
            store = get_vectorstore()
            index_ready = store._collection.count() > 0
        except Exception:
            index_ready = persist.exists() and any(persist.iterdir())
        return HealthResponse(
            status="ok" if index_ready else "degraded",
            index_ready=index_ready,
            collection=config.COLLECTION_NAME,
        )

    def ask(self, body: AskRequest) -> AskResponse:
        result = ask(
            body.question,
            k=body.k,
            strategy=None if body.use_routing else body.strategy,
            use_routing=body.use_routing,
            use_rerank=body.use_rerank,
        )
        return AskResponse(
            answer=result.answer,
            strategy=result.strategy,
            sources=[
                SourceChunk(content=doc.page_content, metadata=dict(doc.metadata or {}))
                for doc in result.sources
            ],
        )
