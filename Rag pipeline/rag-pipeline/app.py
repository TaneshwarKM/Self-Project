"""
app.py
Streamlit demo UI: upload documents, build an index, ask questions,
and see retrieved sources alongside the grounded answer.
Run with: streamlit run app.py
"""

import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.pipeline import RAGPipeline

load_dotenv()

st.set_page_config(page_title="RAG Pipeline Demo", page_icon="🔎", layout="wide")
st.title("🔎 RAG Pipeline Demo")
st.caption("Upload documents, then ask questions grounded in their content.")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
    st.session_state.n_chunks = 0

with st.sidebar:
    st.header("1. Build index")
    uploaded_files = st.file_uploader(
        "Upload PDF / TXT / MD / HTML files",
        type=["pdf", "txt", "md", "html"],
        accept_multiple_files=True,
    )
    top_k = st.slider("Chunks to retrieve (top_k)", 1, 10, 5)

    if st.button("Build index", type="primary", disabled=not uploaded_files):
        with tempfile.TemporaryDirectory() as tmp_dir:
            for f in uploaded_files:
                (Path(tmp_dir) / f.name).write_bytes(f.getbuffer())

            with st.spinner("Ingesting, chunking, and embedding..."):
                pipeline = RAGPipeline()
                n_chunks = pipeline.build_index(tmp_dir)

            st.session_state.pipeline = pipeline
            st.session_state.n_chunks = n_chunks
        st.success(f"Indexed {st.session_state.n_chunks} chunks from {len(uploaded_files)} file(s).")

st.header("2. Ask a question")
question = st.text_input("Your question", placeholder="What does the document say about...?")
ask = st.button("Ask", disabled=st.session_state.pipeline is None)

if ask and question:
    with st.spinner("Retrieving and generating..."):
        result = st.session_state.pipeline.query(question, top_k=top_k)

    st.subheader("Answer")
    if result["grounded"]:
        st.write(result["answer"])
    else:
        st.warning(result["answer"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Grounded", "Yes" if result["grounded"] else "No")
    col2.metric("Top retrieval score", f"{result['top_score']:.3f}")
    col3.metric("Latency", f"{result['latency_seconds']}s")

    with st.expander("Retrieved chunks (evidence used)"):
        for chunk in result["retrieved_chunks"]:
            st.markdown(f"**{chunk['chunk_id']}** — score: {chunk['score']:.4f}")
            st.text(chunk["text"][:500])
            st.divider()
elif st.session_state.pipeline is None:
    st.info("Upload documents and build an index in the sidebar to get started.")
