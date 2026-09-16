# Base embedding interface
from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    """Minimal interface for embedding text into vectors."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...
