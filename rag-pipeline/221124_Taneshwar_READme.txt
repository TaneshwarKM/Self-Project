Working RAG pipeline, tested and ready to upload. It's below.

What's inside:

src/ingest.py — loads PDF/MD/HTML/TXT, normalizes text
src/chunking.py — recursive chunking with overlap
src/embeddings.py — local embeddings (no API key needed for indexing)
src/vector_store.py — FAISS index with save/load
src/retriever.py — hybrid dense + BM25 retrieval (fused via reciprocal rank fusion — catches both paraphrased and exact-keyword queries)
src/generator.py — grounded prompt + confidence gate (refuses to answer instead of hallucinating) + Anthropic/OpenAI backend
app.py — Streamlit demo (upload docs, ask questions, see retrieved evidence + scores)
evaluate.py — measures retrieval precision@k and refusal rate on a labeled eval set, so your resume numbers come from real measurements
tests/test_pipeline.py — unit tests, already run and passing (4/4)
2 sample policy documents + a 5-question eval set to try it immediately