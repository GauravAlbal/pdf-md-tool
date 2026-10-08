#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/mamba/env/bin/pdftoppm" ]]; then
  export PATH="$ROOT/mamba/env/bin:$PATH"
fi

if [[ ! -d "$ROOT/.venv" ]]; then
  uv venv "$ROOT/.venv"
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
  uv pip install fastapi 'uvicorn[standard]' python-multipart pdf-inspector jinja2 aiofiles pydantic-settings olmocr firecrawl-anydoc
else
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8787}"
echo "PDF → Markdown UI  http://${HOST}:${PORT}"
exec python -m uvicorn app.main:app --host "$HOST" --port "$PORT" --reload
