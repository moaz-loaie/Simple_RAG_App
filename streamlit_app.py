"""Streamlit UI: build index from a PDF, ask questions, show answer and cited sources."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from langchain_community.vectorstores import Chroma

from rag.embeddings import make_embeddings
from rag.llm import get_resolved_chat_model, reset_llm_cache
from rag.rag import RAGPipeline


def _init_state() -> None:
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "persist_dir" not in st.session_state:
        st.session_state.persist_dir = None
    if "pdf_label" not in st.session_state:
        st.session_state.pdf_label = None


def main() -> None:
    st.set_page_config(page_title="Modular RAG", layout="wide")
    _init_state()
    st.title("PDF RAG")
    st.caption("OpenRouter when reachable; Ollama on connectivity failure. Embeddings: FastEmbed (local).")

    pipeline = RAGPipeline()
    emb = make_embeddings()

    with st.sidebar:
        st.subheader("Index")
        path_input = st.text_input("PDF path on disk", placeholder=r"D:\docs\file.pdf")
        uploaded = st.file_uploader("Or upload PDF", type=["pdf"])
        persist_override = st.text_input(
            "Chroma persist directory (optional)",
            help="Leave empty to use CHROMA_PERSIST_DIR from .env or ./chroma_db",
        )
        if st.button("Re-check LLM (OpenRouter / Ollama)", help="Clears provider cache after changing .env"):
            reset_llm_cache()
            try:
                r = get_resolved_chat_model()
                st.success(f"Using: {r.provider_name}")
            except Exception as e:
                st.error(str(e))

    persist = Path(persist_override.strip()).resolve() if persist_override.strip() else None

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Build / rebuild index", type="primary"):
            pdf_path: Path | None = None
            tmp_path: Path | None = None
            try:
                if uploaded is not None:
                    suffix = Path(uploaded.name).suffix or ".pdf"
                    fd, tmp_name = tempfile.mkstemp(suffix=suffix)
                    Path(tmp_name).unlink(missing_ok=True)
                    tmp_path = Path(tmp_name)
                    tmp_path.write_bytes(uploaded.getvalue())
                    pdf_path = tmp_path
                    label = uploaded.name
                elif path_input.strip():
                    pdf_path = Path(path_input.strip()).expanduser().resolve()
                    label = pdf_path.name
                else:
                    st.error("Provide a PDF path or upload a file.")
                    pdf_path = None
                    label = None

                if pdf_path is not None:
                    with st.spinner("Loading, chunking, embedding…"):
                        vs = pipeline.ingest_pdf(
                            pdf_path,
                            persist_directory=persist,
                            embedding=emb,
                        )
                    st.session_state.vector_store = vs
                    st.session_state.persist_dir = persist
                    st.session_state.pdf_label = label
                    st.success(f"Indexed: {label}")
            finally:
                if tmp_path is not None:
                    tmp_path.unlink(missing_ok=True)

    with col_b:
        if st.button("Load existing index from disk"):
            try:
                vs = pipeline.load_store(
                    persist_directory=persist,
                    embedding=emb,
                )
                st.session_state.vector_store = vs
                st.session_state.persist_dir = persist
                st.session_state.pdf_label = "(loaded from disk)"
                st.success("Loaded Chroma from persist directory.")
            except Exception as e:
                st.error(str(e))

    vs: Chroma | None = st.session_state.vector_store

    if vs is None:
        st.info("Build or load an index to enable questions.")
        return

    st.divider()
    q = st.text_input("Question", placeholder="What does the document say about …?")
    if st.button("Ask", type="primary") and q.strip():
        with st.spinner("Retrieving and generating…"):
            try:
                result = pipeline.query(vs, q.strip())
            except Exception as e:
                st.error(str(e))
                return
        st.subheader("Answer")
        st.write(result.answer)
        st.subheader("Sources used")
        st.caption("Page is 0-based (LangChain / PyPDF). Lower Chroma **distance** usually means a closer match.")
        for i, s in enumerate(result.sources, start=1):
            with st.expander(f"{i}. {s.display_name} — page {s.page} (label {s.page_label}) — distance {s.distance:.4f}"):
                st.text(s.excerpt)
                if s.source_path:
                    st.caption(s.source_path)

    st.divider()
    try:
        prov = get_resolved_chat_model().provider_name
    except Exception:
        prov = "(not resolved)"
    st.caption(f"LLM backend: **{prov}** · PDF: **{st.session_state.pdf_label or '—'}**")


if __name__ == "__main__":
    main()
