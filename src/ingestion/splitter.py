# Splits docs into chunks (token-aware for Voyage embeddings)
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import voyageai

from loader import load_directory, load_sample

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

EMBED_MODEL = "voyage-4"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64

_vo = voyageai.Client()


def _token_length(text: str) -> int:
    """Count tokens with the same tokenizer Voyage uses for embedding."""
    return _vo.count_tokens([text], model=EMBED_MODEL)


def get_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=_token_length,
    )


def split_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    splitter = get_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata = {
            **chunk.metadata,
            "chunk_index": i,
            "chunk_size_tokens": chunk_size,
            "chunk_overlap_tokens": chunk_overlap,
        }
    return chunks


if __name__ == "__main__":
    documents = load_directory()
    if not documents:
        documents = [load_sample()]

    chunks = split_documents(documents)
    print(f"docs: {len(documents)} | chunks: {len(chunks)}")
    print(chunks[0].page_content[:300])
    print(chunks[0].metadata)
