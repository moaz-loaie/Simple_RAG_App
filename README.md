# Modular RAG (OpenRouter + Ollama fallback)

PDF RAG matching [RAG.ipynb](RAG.ipynb): PyPDF → recursive split → **FastEmbed** (local) → **Chroma**. Questions use **OpenRouter** first; on connectivity failures only, **Ollama** is used. Answers include **retrieved sources** (file, page, excerpt, distance).

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- Python 3.13 (see `.python-version`)
- Optional: [Ollama](https://ollama.com/) with your `OLLAMA_MODEL` pulled, for fallback or when no OpenRouter key is set

## Setup

```powershell
cd D:\RAG
uv sync
```

Activate the virtual environment before running `python` or `streamlit` directly (optional if you use `uv run`, which uses `.venv` automatically):

```powershell
.\.venv\Scripts\Activate.ps1
```

Copy environment template and add your key:

```powershell
copy .env.example .env
# Edit .env — set OPENROUTER_API_KEY for OpenRouter; omit it to use Ollama only.
```

## Run the UI

From the repo root (with venv activated, or via `uv run`):

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```

Or:

```powershell
uv run streamlit run streamlit_app.py
```

In the sidebar, use **Re-check LLM** after changing `.env` so the OpenRouter/Ollama probe runs again.

## OpenRouter free models

Use a model id that includes `:free` if you want no billing, e.g. `meta-llama/llama-3.2-3b-instruct:free`. Rate limits may apply.

## Layout

- `src/rag/` — config, loaders, chunking, embeddings, vectorstore, LLM router, pipeline
- `streamlit_app.py` — UI at repo root
