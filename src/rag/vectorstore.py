"""Chroma vector store build / load."""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.config import Settings, get_settings


def persist_path(s: Settings | None = None) -> Path:
    cfg = s or get_settings()
    return cfg.chroma_persist_dir


def build_chroma(
    documents: list[Document],
    embedding: Embeddings,
    *,
    persist_directory: Path | str | None = None,
    s: Settings | None = None,
) -> Chroma:
    cfg = s or get_settings()
    path = Path(persist_directory) if persist_directory is not None else cfg.chroma_persist_dir
    path.mkdir(parents=True, exist_ok=True)
    return Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=str(path),
    )


def load_chroma(
    embedding: Embeddings,
    *,
    persist_directory: Path | str | None = None,
    s: Settings | None = None,
) -> Chroma:
    cfg = s or get_settings()
    path = Path(persist_directory) if persist_directory is not None else cfg.chroma_persist_dir
    if not path.exists():
        raise FileNotFoundError(f"Chroma persist directory missing: {path}")
    return Chroma(
        persist_directory=str(path),
        embedding_function=embedding,
    )
