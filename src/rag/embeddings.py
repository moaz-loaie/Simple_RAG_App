"""Local FastEmbed embeddings (no API key)."""

from __future__ import annotations

from langchain_community.embeddings import FastEmbedEmbeddings

from rag.config import Settings, get_settings


def make_embeddings(s: Settings | None = None) -> FastEmbedEmbeddings:
    cfg = s or get_settings()
    return FastEmbedEmbeddings(model_name=cfg.fastembed_model)
