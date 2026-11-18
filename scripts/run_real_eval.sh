#!/usr/bin/env bash
# Evaluation half of the real-data run (after scripts/run_real.sh steps 1–7). Reports the TEST split;
# design choices were made on the dev split (results/real/rerank_study_dev.json, resolver_eval_*_dev.json).
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src LEXBRIDGE_CONFIG=configs/config.real.yaml LEXBRIDGE_TRANSFER=rules
R=results/real; L=$R/logs; mkdir -p "$L"; GEN_N=${GEN_N:-60}
step() { echo "[$(date +%H:%M:%S)] $1"; }
if [ "${SKIP_RETRIEVAL:-0}" != 1 ]; then
step "retrieval eval + ablations (test)"; LLM_BACKEND=mock python -m lexbridge.eval.ablations --split test > $L/8_ablations.log 2>&1
step "significance"; for b in bm25 hybrid; do python -m lexbridge.eval.significance --system lexbridge --baseline $b > $L/8b_sig_$b.log 2>&1; done
fi
step "generation eval (test, n=$GEN_N, Qwen2.5-1.5B)"
python -m lexbridge.eval.run_eval --system no_rag,hybrid,lexbridge --split test --categories collision,cross_code_precedent,temporal \
  --limit $GEN_N --llm hf --tag _gen > $L/9_gen.log 2>&1
python -m lexbridge.eval.ablations --gen-table $GEN_N "Qwen2.5-1.5B-Instruct" > $L/10_gentable.log 2>&1
python -m lexbridge.eval.significance --system lexbridge --baseline hybrid --tag _gen; python -m lexbridge.eval.significance --system lexbridge --baseline no_rag --tag _gen > $L/10b_sig_gen.log 2>&1
python -m lexbridge.eval.answer_eval --tag _gen > $L/10c_answer_eval.log 2>&1
python benchmark/export_annotation_sheets.py > $L/11_sheets.log 2>&1
step done
