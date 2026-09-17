"""
retriever.py
Combines dense (embedding) retrieval with a sparse BM25 pass and
reciprocal-rank fusion, then exposes a single retrieve() call the
pipeline uses. This hybrid approach catches exact keyword matches
(e.g. product codes, names) that pure embedding search can miss.
"""

from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.chunking import Chunk
from src.embeddings import Embedder
from src.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # fused relevance score, higher is better


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder, chunks: list[Chunk]):
        self.store = store
        self.embedder = embedder
        self.chunks = chunks
        tokenized = [c.text.lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None
        self._chunk_by_id = {c.chunk_id: c for c in chunks}

    def _dense_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        query_vec = self.embedder.embed_one(query)
        results = self.store.search(query_vec, top_k=top_k)
        return [(chunk.chunk_id, score) for chunk, score in results]

    def _sparse_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        ranked = sorted(
            zip((c.chunk_id for c in self.chunks), scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    @staticmethod
    def _reciprocal_rank_fusion(
        *ranked_lists: list[tuple[str, float]], k: int = 60
    ) -> dict[str, float]:
        fused: dict[str, float] = {}
        for ranked in ranked_lists:
            for rank, (chunk_id, _score) in enumerate(ranked):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        return fused

    def retrieve(self, query: str, top_k: int = 5, candidate_pool: int = 20) -> list[RetrievedChunk]:
        dense = self._dense_search(query, candidate_pool)
        sparse = self._sparse_search(query, candidate_pool)
        fused = self._reciprocal_rank_fusion(dense, sparse)

        ranked_ids = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            RetrievedChunk(chunk=self._chunk_by_id[cid], score=score)
            for cid, score in ranked_ids
            if cid in self._chunk_by_id
        ]
