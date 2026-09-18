# ASK RESULT — structured output for API / production callers.
from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document


@dataclass
class AskResult:
    answer: str
    strategy: str
    sources: list[Document] = field(default_factory=list)
