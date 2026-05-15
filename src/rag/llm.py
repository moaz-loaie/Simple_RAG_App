"""OpenRouter-first chat LLM with Ollama fallback on connectivity failures only."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx
from langchain_community.chat_models import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, AuthenticationError

from rag.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class ResolvedChatModel:
    llm: BaseChatModel
    provider_name: str


def _connectivity_exc(exc: BaseException) -> bool:
    if isinstance(
        exc,
        (
            APIConnectionError,
            APITimeoutError,
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            TimeoutError,
            ConnectionError,
            OSError,
        ),
    ):
        return True
    err = f"{type(exc).__name__}: {exc}".lower()
    return any(
        s in err
        for s in (
            "connection",
            "timeout",
            "timed out",
            "name or service not known",
            "getaddrinfo failed",
            "nodename nor servname",
            "network is unreachable",
            "temporary failure",
        )
    )


def _auth_exc(exc: BaseException) -> bool:
    if isinstance(exc, AuthenticationError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (401, 403)
    err = f"{type(exc).__name__}: {exc}"
    return "401" in err or "403" in err or "invalid api key" in err.lower()


def resolve_chat_model(s: Settings | None = None) -> ResolvedChatModel:
    """Pick OpenRouter (ChatOpenAI) when the key is set and the API is reachable; else Ollama."""
    cfg = s or get_settings()

    if not cfg.openrouter_api_key:
        logger.info("OPENROUTER_API_KEY not set; using Ollama.")
        return ResolvedChatModel(
            llm=ChatOllama(
                base_url=cfg.ollama_base_url,
                model=cfg.ollama_model,
                temperature=0,
            ),
            provider_name="ollama (no OpenRouter key)",
        )

    llm = ChatOpenAI(
        base_url=cfg.openrouter_base_url,
        api_key=cfg.openrouter_api_key,
        model=cfg.openrouter_model,
        temperature=0,
        timeout=cfg.request_timeout_seconds,
        max_retries=0,
        default_headers={
            "HTTP-Referer": cfg.openrouter_http_referer,
            "X-Title": cfg.openrouter_app_title,
        },
    )

    try:
        llm.invoke(
            [HumanMessage(content="ping")],
            max_tokens=1,
        )
    except Exception as e:
        if _auth_exc(e):
            logger.warning("OpenRouter auth failed; not falling back to Ollama.")
            raise RuntimeError(
                "OpenRouter rejected the request (check OPENROUTER_API_KEY). "
                "Connectivity fallback does not apply to authentication errors."
            ) from e
        if _connectivity_exc(e):
            logger.warning("OpenRouter unreachable; falling back to Ollama: %s", e)
            return ResolvedChatModel(
                llm=ChatOllama(
                    base_url=cfg.ollama_base_url,
                    model=cfg.ollama_model,
                    temperature=0,
                ),
                provider_name="ollama (OpenRouter unreachable)",
            )
        raise

    return ResolvedChatModel(llm=llm, provider_name="openrouter")


_resolved: ResolvedChatModel | None = None


def get_resolved_chat_model(s: Settings | None = None) -> ResolvedChatModel:
    global _resolved
    if _resolved is None:
        _resolved = resolve_chat_model(s)
    return _resolved


def reset_llm_cache() -> None:
    """Test helper: force re-probe on next resolve."""
    global _resolved
    _resolved = None
