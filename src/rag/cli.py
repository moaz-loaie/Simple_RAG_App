"""Console entry: `rag-ui` → Streamlit app at repo root."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parents[1]
    app = root / "streamlit_app.py"
    if not app.is_file():
        print(f"streamlit_app.py not found at {app}", file=sys.stderr)
        raise SystemExit(1)
    extra = list(sys.argv[1:])
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), *extra]
    raise SystemExit(subprocess.call(cmd))
