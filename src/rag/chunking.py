"""Text splitting (notebook defaults)."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import Settings, get_settings


def make_text_splitter(s: Settings | None = None) -> RecursiveCharacterTextSplitter:
    cfg = s or get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
