"""Environment-backed settings (see `.env.example`)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str | None
    openrouter_base_url: str
    openrouter_model: str
    openrouter_http_referer: str
    openrouter_app_title: str
    ollama_base_url: str
    ollama_model: str
    request_timeout_seconds: float
    chroma_persist_dir: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int
    fastembed_model: str
    excerpt_max_chars: int
    openrouter_reasoning_enabled: bool


def get_settings() -> Settings:
    root = Path(__file__).resolve().parents[2]
    default_chroma = root / "chroma_db"
    persist = os.getenv("CHROMA_PERSIST_DIR", "").strip()
    if persist:
        p = Path(persist)
        chroma_dir = p.resolve() if p.is_absolute() else (root / p)
    else:
        chroma_dir = default_chroma

    key = os.getenv("OPENROUTER_API_KEY", "").strip() or None

    return Settings(
        openrouter_api_key=key,
        openrouter_base_url=os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ).rstrip("/"),
        openrouter_model=os.getenv("OPENROUTER_MODEL", "openrouter/free"),
        openrouter_http_referer=os.getenv(
            "OPENROUTER_HTTP_REFERER", "http://localhost:8501"
        ),
        openrouter_app_title=os.getenv("OPENROUTER_APP_TITLE", "RAG App"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip(
            "/"
        ),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        request_timeout_seconds=_env_float("LLM_REQUEST_TIMEOUT", 30.0),
        chroma_persist_dir=chroma_dir,
        chunk_size=_env_int("CHUNK_SIZE", 2000),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 500),
        top_k=_env_int("TOP_K", 3),
        fastembed_model=os.getenv("FASTEMBED_MODEL", "thenlper/gte-large"),
        excerpt_max_chars=_env_int("EXCERPT_MAX_CHARS", 320),
        openrouter_reasoning_enabled=_env_bool("OPENROUTER_REASONING_ENABLED", False),
    )
