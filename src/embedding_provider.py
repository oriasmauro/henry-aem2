"""
Embedding providers for RAG, supporting OpenAI and local Sentence Transformers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from openai import OpenAI

logger = logging.getLogger("rag-embed")

OPENAI_PROVIDER = "openai"
SENTENCE_TRANSFORMERS_PROVIDER = "sentence-transformers"


class EmbeddingProvider(Protocol):
    name: str
    dim: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass
class OpenAIEmbeddingProvider:
    client: OpenAI
    model: str
    name: str = OPENAI_PROVIDER
    dim: int = 0

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        vectors = [item.embedding for item in response.data]
        if vectors and self.dim == 0:
            self.dim = len(vectors[0])
        return vectors


@dataclass
class SentenceTransformerProvider:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    name: str = SENTENCE_TRANSFORMERS_PROVIDER
    dim: int = 0
    _model: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Import lazily so the rest of the project can run without loading torch.
        from sentence_transformers import SentenceTransformer

        logger.info("Loading local embedding model: %s", self.model_name)
        self._model = SentenceTransformer(self.model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        vectors = embeddings.astype("float32").tolist()
        if vectors and self.dim == 0:
            self.dim = len(vectors[0])
        return vectors


def build_embedding_provider(
    provider_name: str,
    *,
    openai_client: OpenAI | None = None,
    openai_model: str | None = None,
    local_model: str | None = None,
) -> EmbeddingProvider:
    if provider_name == OPENAI_PROVIDER:
        if openai_client is None or not openai_model:
            raise ValueError("OpenAI provider requires client and model")
        return OpenAIEmbeddingProvider(client=openai_client, model=openai_model)

    if provider_name == SENTENCE_TRANSFORMERS_PROVIDER:
        if not local_model:
            raise ValueError("Sentence Transformers provider requires local_model")
        return SentenceTransformerProvider(model_name=local_model)

    raise ValueError(f"Unsupported embedding provider: {provider_name}")
