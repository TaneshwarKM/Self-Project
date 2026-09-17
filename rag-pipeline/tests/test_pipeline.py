"""
test_pipeline.py
Lightweight unit tests for the parts of the pipeline that don't need
network access or API keys: chunking and the vector store.
Run with: pytest
"""

import numpy as np

from src.chunking import Chunk, chunk_documents, recursive_chunk
from src.ingest import RawDocument
from src.vector_store import VectorStore


def test_recursive_chunk_respects_size_limit():
    text = "word " * 500  # long single-paragraph text
    chunks = recursive_chunk(text, chunk_size=100)
    assert all(len(c) <= 100 + 20 for c in chunks)  # small slack for split boundaries
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")[: len("".join(chunks).replace(" ", ""))]


def test_chunk_documents_produces_ids_and_metadata():
    docs = [RawDocument(doc_id="a.txt", source_path="/tmp/a.txt", text="Sentence one. Sentence two. " * 50)]
    chunks = chunk_documents(docs, chunk_size=100, overlap=10)

    assert len(chunks) > 1
    assert all(c.doc_id == "a.txt" for c in chunks)
    assert all(c.chunk_id.startswith("a.txt::chunk-") for c in chunks)
    assert all("chunk_index" in c.metadata for c in chunks)


def test_vector_store_returns_nearest_neighbor():
    store = VectorStore(dimension=4)
    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", text="alpha"),
        Chunk(chunk_id="c2", doc_id="d1", text="beta"),
    ]
    vectors = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype="float32")
    store.add(chunks, vectors)

    query = np.array([0.9, 0.1, 0, 0], dtype="float32")
    results = store.search(query, top_k=1)

    assert len(results) == 1
    assert results[0][0].chunk_id == "c1"


def test_vector_store_empty_search_returns_empty_list():
    store = VectorStore(dimension=4)
    query = np.zeros(4, dtype="float32")
    assert store.search(query, top_k=5) == []
