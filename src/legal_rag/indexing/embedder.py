"""Embedding model abstraction for legal document vectors."""

from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from legal_rag.config import settings

logger = logging.getLogger("legal_rag.indexing.embedder")


class LegalEmbedder:
    """Computes dense vector representations using SentenceTransformer."""

    _instance: LegalEmbedder | None = None

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode a list of texts into normalized embedding vectors."""
        if not texts:
            return np.empty((0, 384))
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings

    def embed_query(self, query: str) -> list[float]:
        """Encode a single query string into a vector list."""
        vec = self.model.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.tolist()
