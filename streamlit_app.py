"""Streamlit UI: build index from a PDF, ask questions, show answer and cited sources."""

from __future__ import annotations

import gc
import tempfile
import time
import warnings
from pathlib import Path

import streamlit as st
from langchain_community.vectorstores import Chroma

from rag.embeddings import make_embeddings
from rag.llm import get_resolved_chat_model, reset_llm_cache
from rag.rag import RAGPipeline


@st.cache_resource(show_spinner=False)
def _cached_pipeline() -> RAGPipeline:
    return RAGPipeline()


@st.cache_resource(show_spinner=False)
def _cached_embeddings():
    """Loads FastEmbed + ONNX — only call when building/loading index or querying."""
    return make_embeddings()


@st.cache_resource(show_spinner=False)
def _cached_llm_provider_label() -> str:
    """OpenRouter probe / Ollama selection — runs once per process unless cache cleared."""
    try:
        return get_resolved_chat_model().provider_name
    except Exception:
        return "(not resolved)"


def _inject_ui_styles() -> None:
    st.markdown(
        """
<style>
  .block-container { padding-top: 1.25rem; padding-bottom: 3rem; max-width: 1100px; }
  h1 { font-weight: 700; letter-spacing: -0.03em; margin-bottom: 0.25rem; }
  [data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,0.08); }
  [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
  div[data-testid="stVerticalBlockBorderWrapper"] > div {
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
    border-radius: 12px !important;
    border-color: rgba(255,255,255,0.12) !important;
    padding: 1rem 1.1rem;
  }
</style>
""",
        unsafe_allow_html=True,
    )


def _unlink_temp_pdf(path: Path) -> None:
    """Windows often keeps PDF handles briefly after load; retry instead of failing the run."""
    for _ in range(8):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            gc.collect()
            time.sleep(0.12)
    try:
        path.unlink(missing_ok=True)
    except PermissionError:
        pass


def _init_state() -> None:
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "persist_dir" not in st.session_state:
        st.session_state.persist_dir = None
    if "pdf_label" not in st.session_state:
        st.session_state.pdf_label = None


def _clear_last_exchange() -> None:
    for k in ("last_q", "last_answer", "last_sources"):
        st.session_state.pop(k, None)


def _persist_from_session() -> Path | None:
    raw = (st.session_state.get("persist_override") or "").strip()
    if not raw:
        return None
    return Path(raw).resolve()


def _uploaded_from_session():
    return st.session_state.get("pdf_upload")


def _path_from_session() -> str:
    return (st.session_state.get("path_input") or "").strip()


@st.fragment
def _sidebar_fragment() -> None:
    """Must be invoked as: with st.sidebar: _sidebar_fragment()"""
    st.markdown("### Index")
    st.text_input(
        "PDF path on disk",
        placeholder="C:\\Users\\you\\Documents\\file.pdf",
        key="path_input",
    )
    st.file_uploader("Or upload PDF", type=["pdf"], key="pdf_upload")
    st.text_input(
        "Chroma persist directory (optional)",
        help="Leave empty to use CHROMA_PERSIST_DIR from .env or ./chroma_db",
        key="persist_override",
    )
    if st.button(
        "Re-check LLM (OpenRouter / Ollama)",
        help="Clears provider cache after changing .env",
        key="btn_recheck_llm",
    ):
        reset_llm_cache()
        try:
            _cached_llm_provider_label.clear()
        except Exception:
            pass
        try:
            r = get_resolved_chat_model()
            st.success(f"Using: {r.provider_name}")
        except Exception as e:
            st.error(str(e))


@st.fragment
def _main_fragment() -> None:
    """Main area only — sidebar lives in _sidebar_fragment (Streamlit API rule)."""
    pipeline = _cached_pipeline()
    persist = _persist_from_session()
    path_input = _path_from_session()
    uploaded = _uploaded_from_session()

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        if st.button("Build / rebuild index", type="primary", use_container_width=True, key="btn_build"):
            emb = _cached_embeddings()
            pdf_path: Path | None = None
            tmp_path: Path | None = None
            citation_label: str | None = None
            try:
                if uploaded is not None:
                    suffix = Path(uploaded.name).suffix or ".pdf"
                    with tempfile.NamedTemporaryFile(
                        suffix=suffix, delete=False
                    ) as tmp:
                        tmp.write(uploaded.getvalue())
                        tmp.flush()
                        tmp_path = Path(tmp.name)
                    pdf_path = tmp_path
                    label = uploaded.name
                    citation_label = uploaded.name
                elif path_input:
                    pdf_path = Path(path_input).expanduser().resolve()
                    label = pdf_path.name
                    citation_label = None
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
                            citation_source_label=citation_label,
                        )
                    st.session_state.vector_store = vs
                    st.session_state.persist_dir = persist
                    st.session_state.pdf_label = label
                    _clear_last_exchange()
                    st.success(f"Indexed: **{label}**")
            finally:
                if tmp_path is not None:
                    _unlink_temp_pdf(tmp_path)

    with c2:
        if st.button(
            "Load existing index from disk",
            use_container_width=True,
            key="btn_load",
        ):
            emb = _cached_embeddings()
            try:
                vs = pipeline.load_store(
                    persist_directory=persist,
                    embedding=emb,
                )
                st.session_state.vector_store = vs
                st.session_state.persist_dir = persist
                st.session_state.pdf_label = "(loaded from disk)"
                _clear_last_exchange()
                st.success("Loaded Chroma from persist directory.")
            except Exception as e:
                st.error(str(e))

    vs: Chroma | None = st.session_state.vector_store

    if vs is None:
        st.info("Build or load an index to enable questions.")
        st.caption(
            "Use **Re-check LLM** in the sidebar to verify OpenRouter or Ollama before querying."
        )
        return

    st.divider()

    prompt = st.chat_input(
        "Ask about this PDF…  (Enter to send · Shift+Enter for a new line)",
        key="chat_prompt",
        height=100,
    )
    if prompt and prompt.strip():
        qtext = prompt.strip()
        with st.spinner("Retrieving and generating…"):
            try:
                out = pipeline.query(vs, qtext)
                st.session_state.last_q = qtext
                st.session_state.last_answer = out.answer
                st.session_state.last_sources = [
                    {
                        "display_name": s.display_name,
                        "page": s.page,
                        "page_label": s.page_label,
                        "distance": float(s.distance),
                        "full_text": s.full_text,
                    }
                    for s in out.sources
                ]
            except Exception as e:
                _clear_last_exchange()
                st.error(str(e))

    hist = st.container(height=520, border=True)
    with hist:
        if (
            st.session_state.get("last_q")
            and "last_answer" in st.session_state
            and "last_sources" in st.session_state
        ):
            with st.chat_message("user"):
                st.markdown(st.session_state.last_q)
            with st.chat_message("assistant"):
                st.markdown(st.session_state.last_answer or "_No answer._")
            with st.expander("Supporting text from your PDF", expanded=False):
                for s in st.session_state.last_sources or []:
                    title = (
                        f"{s['display_name']} · page {s['page']} "
                        f"(printed {s.get('page_label') or '—'}) · distance {s['distance']:.4f}"
                    )
                    with st.expander(title, expanded=False):
                        st.caption("Exact chunk the model used as context (unchanged).")
                        st.text(s.get("full_text") or "")

    st.divider()
    pdf = st.session_state.pdf_label or "—"
    st.caption(f"LLM: **{_cached_llm_provider_label()}** · Active PDF: **{pdf}**")


def main() -> None:
    warnings.filterwarnings("ignore", message=".*thenlper/gte-large now uses mean pooling.*")
    warnings.filterwarnings("ignore", message=".*class `Chroma` was deprecated.*")
    st.set_page_config(
        page_title="PDF RAG",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_ui_styles()
    _init_state()
    st.markdown("# PDF RAG")
    st.caption(
        "OpenRouter when reachable · Ollama on connectivity failure · Local FastEmbed embeddings"
    )
    with st.sidebar:
        _sidebar_fragment()
    _main_fragment()


if __name__ == "__main__":
    main()
