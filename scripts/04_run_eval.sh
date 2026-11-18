#!/usr/bin/env bash
# All systems + ablations → results/*.json, paper/tables/main.tex, paper/tables/ablation.tex
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
[ -f data/benchmark/lexbridge_qa_sample.jsonl ] || python benchmark/generate_candidates.py
python -m lexbridge.eval.ablations
