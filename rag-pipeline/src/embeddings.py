"""
embeddings.py
Thin wrapper around a local sentence-transformers model so the rest of
the pipeline doesn't care which embedding model is plugged in.
Runs fully offline — no API key required.
"""

from __future__ import annotations

import os

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalized embeddings."""
        if not texts:
            return np.zeros((0, self._model.get_sentence_embedding_dimension()), dtype="float32")

        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # cosine similarity via dot product
        )
        return np.asarray(vectors, dtype="float32")

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()
