# INDEXING: load → split → embed into Chroma
from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma

import config
from embedding.openai_embedder import get_embeddings
from ingestion.loader import load_directory
from ingestion.splitter import split_documents


def build_index(
    source_dir: Path | None = None,
    persist_directory: Path | None = None,
) -> Chroma:
    # 1) Where to save the vector DB
    persist_directory = persist_directory or config.PERSIST_DIR

    # 2) Load docs from data/raw/ (or source_dir if given)
    documents = load_directory(source_dir) if source_dir else load_directory()
    if not documents:
        raise ValueError(f"No documents found in {source_dir or 'data/raw'}")

    # 3) Split into chunks for embedding
    splits = split_documents(documents)

    # 4) Embed chunks with OpenAI and write Chroma
    return Chroma.from_documents(
        documents=splits,
        embedding=get_embeddings(),
        persist_directory=str(persist_directory),
        collection_name=config.COLLECTION_NAME,
    )


if __name__ == "__main__":
    import os

    source = Path(os.environ["RAW_DIR"]) if os.environ.get("RAW_DIR") else None
    persist = Path(os.environ["PERSIST_DIR"]) if os.environ.get("PERSIST_DIR") else None
    vectorstore = build_index(source_dir=source, persist_directory=persist)
    print(f"Indexed {vectorstore._collection.count()} chunks → {persist or config.PERSIST_DIR}")
