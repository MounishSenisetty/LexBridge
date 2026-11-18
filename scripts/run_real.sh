#!/usr/bin/env bash
# Full real-data run (CPU). Each step logs to results/real/logs/. Resumable: finished steps are skipped.
#   1 ingest  2 role tagger  3 index (rules-only C3)  4 C3 with the LLM  5 graph refresh
#   6 benchmark  7 resolver eval  8 retrieval eval + ablations (extractive generator)  9 generation eval (LLM)
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src LEXBRIDGE_CONFIG=configs/config.real.yaml
export LEXBRIDGE_TRANSFER=rules   # only step 4 (scripts/05_transferability.py) runs the C3 LLM
P=data/processed_real; R=results/real; L=$R/logs; mkdir -p "$L"
GEN_N=${GEN_N:-100}
step() { echo "[$(date +%H:%M:%S)] $1"; }
[ -f $P/judgments.jsonl ] || { step ingest; python -c "from lexbridge.resources import ingest; from lexbridge.config import get_config; ingest(get_config().path('processed'), use_network=True)" > $L/1_ingest.log 2>&1; }
[ -f $P/role_model.pkl ] || { step "role tagger"; python -m lexbridge.nlp.role_classifier > $L/2_roles.log 2>&1; }
[ -f $P/index.pkl ] || { step index; LEXBRIDGE_TRANSFER=rules python -c "
from lexbridge.resources import load_resources
r = load_resources(build=False)
for v in ('naive', 'noresolver'): r.index(v)
print(open('$P/index_meta.json').read())" > $L/3_index.log 2>&1; }
step "C3 transferability (LLM)"; LLM_BACKEND=hf python scripts/05_transferability.py > $L/4_transfer.log 2>&1
step "graph refresh"; python -c "
import pickle
from lexbridge.resources import load_resources
from lexbridge.index.graph import build_graph, save_graph
r = load_resources(build=False)
save_graph(build_graph(r.statutes, r.mappings, r.index('role').chunks, {k: v.label for k, v in r.transfer.items()}), __import__('pathlib').Path('$P/graph.gpickle'))
print('ok')" > $L/5_graph.log 2>&1
[ -f data/benchmark_real/lexbridge_qa_sample.jsonl ] || { step benchmark; python benchmark/generate_candidates.py --n 500 > $L/6_bench.log 2>&1; }
step "resolver eval"; for v in v1 v2 v2+prior; do python -m lexbridge.eval.resolver_eval --n 3000 --version $v --split test >> $L/7_resolver.log 2>&1; done
step "retrieval eval + ablations"; LLM_BACKEND=mock python -m lexbridge.eval.ablations > $L/8_ablations.log 2>&1
step "generation eval (n=$GEN_N)"; python -m lexbridge.eval.run_eval --system no_rag,hybrid,lexbridge --limit $GEN_N --llm hf --tag _gen > $L/9_gen.log 2>&1
python benchmark/export_annotation_sheets.py > $L/11_sheets.log 2>&1
python -m lexbridge.eval.ablations --gen-table $GEN_N "Qwen2.5-1.5B-Instruct" > $L/10_gentable.log 2>&1
step done
