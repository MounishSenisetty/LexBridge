#!/usr/bin/env bash
# Chunk, index (BM25 + dense), build the graph and the transferability cache.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
rm -f data/processed/transferability.jsonl data/processed/index.pkl data/processed/graph.gpickle
python -c "
from lexbridge.resources import load_resources
r = load_resources(build=False)
print(open('data/processed/index_meta.json').read())"
python benchmark/generate_candidates.py
