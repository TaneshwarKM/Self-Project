"""
ingest.py
Loads raw documents (PDF, Markdown, HTML, TXT) from a directory and
normalizes them into plain text with lightweight metadata.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader


@dataclass
class RawDocument:
    doc_id: str
    source_path: str
    text: str
    metadata: dict = field(default_factory=dict)


def _load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _load_html(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _normalize(text: str) -> str:
    # Collapse excessive whitespace while preserving paragraph breaks.
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


LOADERS = {
    ".pdf": _load_pdf,
    ".html": _load_html,
    ".htm": _load_html,
    ".md": _load_text,
    ".txt": _load_text,
}


def load_documents(source_dir: str) -> list[RawDocument]:
    """Walk source_dir and load every supported file into a RawDocument."""
    documents: list[RawDocument] = []
    root = Path(source_dir)

    if not root.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        loader = LOADERS.get(path.suffix.lower())
        if loader is None:
            continue

        try:
            raw_text = loader(path)
        except Exception as exc:  # noqa: BLE001 — surface but don't crash ingestion
            print(f"[ingest] skipped {path}: {exc}")
            continue

        text = _normalize(raw_text)
        if not text:
            continue

        documents.append(
            RawDocument(
                doc_id=str(path.relative_to(root)),
                source_path=str(path),
                text=text,
                metadata={"filename": path.name, "extension": path.suffix.lower()},
            )
        )

    return documents


if __name__ == "__main__":
    import sys

    docs = load_documents(sys.argv[1] if len(sys.argv) > 1 else "data/sample_docs")
    print(f"Loaded {len(docs)} documents")
    for d in docs:
        print(f"  - {d.doc_id} ({len(d.text)} chars)")
