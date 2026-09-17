# RAG Pipeline — Retrieval-Augmented Generation from Scratch

A from-scratch RAG system: document ingestion → chunking → hybrid
(dense + sparse) retrieval → grounded generation with a confidence
guardrail against hallucination. Built to understand and demonstrate
every layer of a production RAG stack, not just wrap an API call.

## Why this exists

LLMs hallucinate or go stale on private/niche information because it
was never in their training data. This pipeline grounds answers in a
specific document set, refuses to answer when retrieval confidence is
low, and cites the exact source chunk behind every claim.

## Architecture

```
 documents (PDF/MD/HTML/TXT)
        │
        ▼
   ingest.py  ──▶  normalize text, strip boilerplate
        │
        ▼
  chunking.py ──▶  recursive chunking (800 chars, 100 overlap)
        │
        ▼
 embeddings.py ──▶  BAAI/bge-small-en-v1.5 (local, no API key)
        │
        ▼
vector_store.py ──▶  FAISS IndexFlatIP (cosine similarity)
        │
        ▼
 retriever.py  ──▶  hybrid: dense search + BM25, fused via
        │            reciprocal rank fusion
        ▼
 generator.py  ──▶  confidence gate → grounded prompt →
        │            Anthropic/OpenAI → cited answer
        ▼
     answer + sources + retrieval score + latency
```

## Features

- **Hybrid retrieval** — dense embeddings (semantic match) fused with
  BM25 (exact keyword match) via reciprocal rank fusion, so both
  paraphrased questions and exact-term lookups (IDs, codes, names)
  work well.
- **Hallucination guardrail** — if the top retrieval score falls below
  `MIN_RETRIEVAL_SCORE`, the pipeline refuses to answer instead of
  guessing.
- **Inline citations** — every generated answer cites the specific
  chunk ID it drew from.
- **Swappable LLM backend** — Anthropic or OpenAI via one env var, no
  code changes.
- **Local embeddings** — no API key required just to build an index.
- **Evaluation harness** (`evaluate.py`) — measures retrieval
  precision@k and refusal rate on a labeled question set, so quality
  claims are backed by numbers, not vibes.
- **Streamlit demo** — upload documents and query them from a browser.

## Setup

```bash
git clone <this-repo>
cd rag-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY or OPENAI_API_KEY
```

## Usage

**CLI**
```bash
python -m src.pipeline --source data/sample_docs --question "How many days of annual leave do employees get?"
```

**Streamlit demo**
```bash
streamlit run app.py
```

**Evaluation**
```bash
python evaluate.py --source data/sample_docs --eval-set data/eval_set.json
```

## Results

Measured with `evaluate.py` on the included 5-question eval set
against the two sample policy documents (`data/sample_docs/`):

| Metric | Value |
|---|---|
| Mean precision@5 | *run `evaluate.py` and fill in* |
| Mean query latency | *run `evaluate.py` and fill in* |
| Refusal rate (low-confidence questions) | *run `evaluate.py` and fill in* |

> Run `python evaluate.py` after adding your own document set and eval
> questions to get numbers for your own corpus — the ones above are a
> placeholder until you run it against your data, since retrieval
> quality depends entirely on the source documents.

## Project structure

```
rag-pipeline/
├── src/
│   ├── ingest.py        # load PDF/MD/HTML/TXT into normalized text
│   ├── chunking.py       # recursive chunking with overlap
│   ├── embeddings.py     # local sentence-transformers wrapper
│   ├── vector_store.py   # FAISS index + save/load
│   ├── retriever.py      # hybrid dense + BM25 retrieval
│   ├── generator.py      # grounded prompt + confidence gate + LLM call
│   └── pipeline.py       # orchestrates the full flow, CLI entry point
├── app.py                # Streamlit demo
├── evaluate.py           # retrieval precision / refusal rate eval
├── data/
│   ├── sample_docs/      # example documents to try the pipeline on
│   └── eval_set.json     # labeled questions for evaluation
├── tests/
│   └── test_pipeline.py  # unit tests for chunking + vector store
└── requirements.txt
```

## Design decisions worth noting

- **Why hybrid retrieval, not just embeddings?** Pure dense retrieval
  misses exact-match queries (policy numbers, product codes) that
  BM25 catches easily. Fusing both gets the benefits of each.
- **Why a confidence gate?** A RAG system that answers confidently
  from irrelevant context is worse than one that says "I don't know"
  — the gate trades a small amount of recall for a large drop in
  hallucinated answers.
- **Why local embeddings but a cloud LLM?** Embedding every chunk
  during indexing is high-volume and cheap to do locally; generation
  is low-volume and benefits from a stronger hosted model.

## Possible extensions

- Add a cross-encoder reranking stage on top of the current fusion step
- Add semantic chunking (embedding-based boundary detection) as an
  alternative to recursive character chunking
- Add conversation memory for multi-turn follow-up questions
- Swap FAISS for a hosted vector DB (Pinecone/Weaviate) for
  multi-user, incrementally-updated corpora

## License

MIT
