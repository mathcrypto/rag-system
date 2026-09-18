# API schemas — request/response for the production HTTP layer.
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Strategy = Literal["dense", "bm25", "multi_query", "rag_fusion", "hybrid"]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    strategy: Strategy | None = Field(
        default="dense",
        description="Retriever strategy; ignored when use_routing is true",
    )
    use_routing: bool = Field(
        default=False,
        description="If true, LLM picks dense|bm25|multi_query|rag_fusion|hybrid",
    )
    use_rerank: bool | None = Field(
        default=None,
        description="None = on when COHERE_API_KEY is set",
    )
    k: int | None = Field(default=None, ge=1, le=20, description="Top-k chunks")


class SourceChunk(BaseModel):
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AskResponse(BaseModel):
    answer: str
    strategy: str
    sources: list[SourceChunk]


class HealthResponse(BaseModel):
    status: str
    index_ready: bool
    collection: str
