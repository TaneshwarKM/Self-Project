"""
vector_store.py
A minimal FAISS-backed vector store: indexes chunk embeddings and
supports top-k similarity search, with save/load to disk so the index
doesn't need to be rebuilt on every run.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss
import numpy as np

from src.chunking import Chunk


class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        # Inner product on normalized vectors == cosine similarity
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        assert len(chunks) == embeddings.shape[0], "chunks/embeddings length mismatch"
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[Chunk, float]]:
        if self.index.ntotal == 0:
            return []
        query = query_embedding.reshape(1, -1)
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, directory: str) -> None:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out / "index.faiss"))
        with open(out / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        with open(out / "meta.json", "w") as f:
            json.dump({"dimension": self.dimension, "count": len(self.chunks)}, f)

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        src = Path(directory)
        with open(src / "meta.json") as f:
            meta = json.load(f)
        store = cls(dimension=meta["dimension"])
        store.index = faiss.read_index(str(src / "index.faiss"))
        with open(src / "chunks.pkl", "rb") as f:
            store.chunks = pickle.load(f)
        return store
