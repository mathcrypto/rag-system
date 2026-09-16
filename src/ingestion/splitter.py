# Splits docs into chunks (tiktoken-aware, matches rag-from-scratch indexing)
from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .loader import load_directory, load_sample

# Detailed indexing section in rag-from-scratch (not the overview's 1000/200 chars).
DEFAULT_CHUNK_SIZE = 300
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_ENCODING = "cl100k_base"  # OpenAI embeddings / GPT-3.5/4 tokenizer family


def get_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=DEFAULT_ENCODING,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
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
            "encoding": DEFAULT_ENCODING,
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
