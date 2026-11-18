#!/usr/bin/env bash
# Parse statutes, merge mappings, load judgments → data/processed/*.jsonl (+ mapping_conflicts.csv)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
python -c "from lexbridge.resources import ingest; from lexbridge.config import get_config; ingest(get_config().path('processed'), use_network=True)"
