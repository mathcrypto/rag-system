# OpenAI embedding client (LangChain OpenAIEmbeddings)
from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

import config


def get_embeddings(model: str | None = None) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=model or config.OPENAI_EMBED_MODEL)
