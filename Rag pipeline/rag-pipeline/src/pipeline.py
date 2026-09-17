"""
pipeline.py
Top-level orchestrator: build an index from a document directory, then
answer queries against it. This is the single entry point the CLI,
Streamlit app, and evaluation script all use.
"""

from __future__ import annotations

import time
from pathlib import Path

from src.chunking import chunk_documents
from src.embeddings import Embedder
from src.generator import Generator
from src.ingest import load_documents
from src.retriever import HybridRetriever
from src.vector_store import VectorStore

DEFAULT_INDEX_DIR = "data/vector_store"


class RAGPipeline:
    def __init__(self, embedder: Embedder | None = None, generator: Generator | None = None):
        self.embedder = embedder or Embedder()
        self.generator = generator or Generator()
        self.store: VectorStore | None = None
        self.retriever: HybridRetriever | None = None

    def build_index(self, source_dir: str, chunk_size: int = 800, overlap: int = 100) -> int:
        """Ingest, chunk, embed, and index documents. Returns chunk count."""
        documents = load_documents(source_dir)
        chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)

        self.store = VectorStore(dimension=self.embedder.dimension)
        vectors = self.embedder.embed([c.text for c in chunks])
        self.store.add(chunks, vectors)
        self.retriever = HybridRetriever(self.store, self.embedder, chunks)
        return len(chunks)

    def save_index(self, directory: str = DEFAULT_INDEX_DIR) -> None:
        if self.store is None:
            raise RuntimeError("No index built yet — call build_index() first.")
        self.store.save(directory)

    def load_index(self, directory: str = DEFAULT_INDEX_DIR) -> None:
        self.store = VectorStore.load(directory)
        self.retriever = HybridRetriever(self.store, self.embedder, self.store.chunks)

    def query(self, question: str, top_k: int = 5) -> dict:
        if self.retriever is None:
            raise RuntimeError("No index loaded — call build_index() or load_index() first.")

        start = time.perf_counter()
        retrieved = self.retriever.retrieve(question, top_k=top_k)
        result = self.generator.generate(question, retrieved)
        result["latency_seconds"] = round(time.perf_counter() - start, 3)
        result["retrieved_chunks"] = [
            {"chunk_id": r.chunk.chunk_id, "score": round(r.score, 4), "text": r.chunk.text}
            for r in retrieved
        ]
        return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query the RAG pipeline from the CLI")
    parser.add_argument("--source", default="data/sample_docs", help="Directory of documents to index")
    parser.add_argument("--question", required=True, help="Question to ask")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    pipeline = RAGPipeline()
    n_chunks = pipeline.build_index(args.source)
    print(f"Indexed {n_chunks} chunks from {args.source}\n")

    result = pipeline.query(args.question, top_k=args.top_k)
    print(f"Q: {args.question}\n")
    print(f"A: {result['answer']}\n")
    print(f"Grounded: {result['grounded']} | Top score: {result['top_score']:.3f} | Latency: {result['latency_seconds']}s")
    print(f"Sources: {result['sources']}")
