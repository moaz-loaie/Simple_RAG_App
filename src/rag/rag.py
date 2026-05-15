"""End-to-end RAG: retrieve with scores, then answer with a fixed context+question prompt."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.embeddings import Embeddings

from rag.chunking import make_text_splitter
from rag.config import Settings, get_settings
from rag.embeddings import make_embeddings
from rag.loaders import load_pdf
from rag.llm import get_resolved_chat_model
from rag.vectorstore import build_chroma, load_chroma

_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context.
Use Markdown when it helps readability: short headings (##), bullet lists for enumerations, and **bold** for important terms or names. Do not invent facts beyond the context.

Context:
{context}

Question: {question}
"""
)


@dataclass
class SourceChunk:
    """One retrieved chunk used as LLM context."""

    source_path: str
    display_name: str
    page: int | None
    page_label: str | None
    excerpt: str
    full_text: str
    distance: float


@dataclass
class RAGResult:
    answer: str
    sources: list[SourceChunk]
    llm_provider: str


class RAGPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def ingest_pdf(
        self,
        pdf_path: str | Path,
        *,
        persist_directory: Path | str | None = None,
        embedding: Embeddings | None = None,
        citation_source_label: str | None = None,
    ) -> Chroma:
        raw_docs = load_pdf(pdf_path)
        splitter = make_text_splitter(self.settings)
        chunks = splitter.split_documents(raw_docs)
        if citation_source_label:
            label = citation_source_label.strip()
            for ch in chunks:
                meta = dict(ch.metadata or {})
                meta["source"] = label
                ch.metadata = meta
        emb = embedding or make_embeddings(self.settings)
        return build_chroma(
            chunks,
            emb,
            persist_directory=persist_directory,
            s=self.settings,
        )

    def load_store(
        self,
        *,
        persist_directory: Path | str | None = None,
        embedding: Embeddings | None = None,
    ) -> Chroma:
        emb = embedding or make_embeddings(self.settings)
        return load_chroma(
            emb,
            persist_directory=persist_directory,
            s=self.settings,
        )

    def query(self, vector_store: Chroma, question: str) -> RAGResult:
        pairs = vector_store.similarity_search_with_score(
            question, k=self.settings.top_k
        )
        sources: list[SourceChunk] = []
        context_parts: list[str] = []
        max_e = self.settings.excerpt_max_chars

        for doc, dist in pairs:
            meta = doc.metadata or {}
            src = meta.get("source") or ""
            display = os.path.basename(str(src)) if src else "(unknown)"
            page = meta.get("page")
            if page is not None and not isinstance(page, int):
                try:
                    page = int(page)
                except (TypeError, ValueError):
                    page = None
            label = meta.get("page_label")
            label_s = str(label) if label is not None else None
            full_text = doc.page_content or ""
            excerpt = full_text[:max_e]
            if len(full_text) > max_e:
                excerpt = excerpt[:-1] + "…"
            sources.append(
                SourceChunk(
                    source_path=str(src),
                    display_name=display,
                    page=page,
                    page_label=label_s,
                    excerpt=excerpt if excerpt else "(empty chunk)",
                    full_text=full_text,
                    distance=float(dist),
                )
            )
            context_parts.append(full_text)

        resolved = get_resolved_chat_model(self.settings)
        chain = _PROMPT | resolved.llm | StrOutputParser()
        answer = chain.invoke(
            {
                "context": "\n\n".join(context_parts),
                "question": question,
            }
        )
        return RAGResult(
            answer=answer,
            sources=sources,
            llm_provider=resolved.provider_name,
        )
