"""
chunking.py
Splits normalized document text into overlapping chunks small enough
to embed and fit into an LLM context window, while trying to break on
natural boundaries (paragraphs, then sentences, then words).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.ingest import RawDocument

SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _split_on_separator(text: str, separator: str, chunk_size: int) -> list[str]:
    pieces = text.split(separator)
    chunks, current = [], ""
    for piece in pieces:
        candidate = f"{current}{separator}{piece}" if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


def recursive_chunk(text: str, chunk_size: int = 800, separators: list[str] | None = None) -> list[str]:
    """Recursively split text on decreasing granularity separators."""
    separators = separators or SEPARATORS
    if len(text) <= chunk_size or not separators:
        return [text] if text.strip() else []

    sep, rest = separators[0], separators[1:]
    pieces = _split_on_separator(text, sep, chunk_size)

    final: list[str] = []
    for piece in pieces:
        if len(piece) <= chunk_size:
            final.append(piece)
        else:
            final.extend(recursive_chunk(piece, chunk_size, rest))
    return final


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or len(chunks) < 2:
        return chunks
    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        tail = chunks[i - 1][-overlap:]
        overlapped.append(f"{tail} {chunks[i]}".strip())
    return overlapped


def chunk_documents(
    documents: list[RawDocument],
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[Chunk]:
    """Chunk every document, returning flat list of Chunk objects."""
    all_chunks: list[Chunk] = []

    for doc in documents:
        raw_pieces = recursive_chunk(doc.text, chunk_size=chunk_size)
        raw_pieces = _add_overlap(raw_pieces, overlap=overlap)
        raw_pieces = [re.sub(r"\s+", " ", p).strip() for p in raw_pieces if p.strip()]

        for i, piece in enumerate(raw_pieces):
            all_chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::chunk-{i}",
                    doc_id=doc.doc_id,
                    text=piece,
                    metadata={**doc.metadata, "chunk_index": i},
                )
            )

    return all_chunks
