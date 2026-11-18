#!/usr/bin/env bash
# Streamlit UI on :8501 (API: make api → :8000)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
exec streamlit run src/lexbridge/ui/app.py --server.port "${PORT:-8501}" --server.headless true
