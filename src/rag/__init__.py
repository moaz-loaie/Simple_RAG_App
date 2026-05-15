"""Modular RAG: ingest PDFs, index with Chroma + FastEmbed, query with OpenRouter or Ollama."""

from rag.rag import RAGPipeline, RAGResult, SourceChunk

__all__ = ["RAGPipeline", "RAGResult", "SourceChunk"]
