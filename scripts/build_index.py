# INDEXING: load → split → embed into Chroma
from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma

import config
from embedding.openai_embedder import get_embeddings
from ingestion.loader import load_directory, load_sample
from ingestion.splitter import split_documents


def build_index(
    source_dir: Path | None = None,
    persist_directory: Path | None = None,
) -> Chroma:
    persist_directory = persist_directory or config.PERSIST_DIR
    documents = load_directory(source_dir) if source_dir else load_directory()
    if not documents:
        documents = [load_sample()]

    splits = split_documents(documents)
    persist_directory.mkdir(parents=True, exist_ok=True)

    return Chroma.from_documents(
        documents=splits,
        embedding=get_embeddings(),
        persist_directory=str(persist_directory),
        collection_name=config.COLLECTION_NAME,
    )


if __name__ == "__main__":
    vectorstore = build_index()
    print(f"Indexed {vectorstore._collection.count()} chunks → {config.PERSIST_DIR}")
