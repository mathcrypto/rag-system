# Retrieval smoke eval — dense search must hit a known chunk.
from __future__ import annotations

from retrieval.retriever import retrieve

# SQuAD Notre Dame passage in data/raw/squad_000.txt (after ingest).
SMOKE_QUESTION = "What is the Grotto at Notre Dame?"
SMOKE_NEEDLE = "Grotto"


def run_retrieval_smoke(k: int = 5) -> None:
    docs = retrieve(SMOKE_QUESTION, k=k)
    if not docs:
        raise AssertionError("dense retrieval returned no documents (is the index built?)")
    if not any(SMOKE_NEEDLE in (d.page_content or "") for d in docs):
        preview = [d.page_content[:80].replace("\n", " ") for d in docs]
        raise AssertionError(
            f"expected a chunk containing {SMOKE_NEEDLE!r} in top-{k}; got: {preview}"
        )
