# INDEXING (rag-from-scratch): load → split → embed into Chroma.
# Goal: turn raw documents into searchable vectors so retrieval is easy later.
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma

from embedding.openai_embedder import get_embeddings
from ingestion.loader import load_directory, load_sample
from ingestion.splitter import split_documents
from retrieval.retriever import COLLECTION_NAME, PERSIST_DIR, get_retriever

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def build_index(
    source_dir: Path | None = None,
    persist_directory: Path = PERSIST_DIR,
) -> Chroma:
    # Load raw documents (text, PDFs, etc.)
    documents = load_directory(source_dir) if source_dir else load_directory()
    if not documents:
        documents = [load_sample()]

    # Split into smaller chunks (tiktoken-sized pieces of the raw docs)
    splits = split_documents(documents)
    persist_directory.mkdir(parents=True, exist_ok=True)

    # Build the index: take every split, embed it, store it in embedding space
    # (with a link back to the raw chunk). That high-dim space is what we
    # search at query time.
    return Chroma.from_documents(
        documents=splits,
        embedding=get_embeddings(),
        persist_directory=str(persist_directory),
        collection_name=COLLECTION_NAME,
    )


if __name__ == "__main__":
    vectorstore = build_index()
    print(f"Indexed {vectorstore._collection.count()} chunks → {PERSIST_DIR}")

    # Query: embed the question into that same space, fetch k nearby splits.
    retriever = get_retriever(k=1)  # notebook often demos k=1
    hits = retriever.invoke("What is RAG chunking?")
    print(f"Sample retrieval hits: {len(hits)}")
    if hits:
        print(hits[0].page_content[:200])
