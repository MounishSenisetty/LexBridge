PY ?= python
export PYTHONPATH := src

.PHONY: install test ingest index demo api eval candidates clean

install:
	pip install -e ".[api,ui,dev]"

test:
	$(PY) -m pytest -q

ingest:
	bash scripts/01_ingest.sh

index:
	bash scripts/02_index.sh

api:
	$(PY) -m uvicorn lexbridge.api.server:app --port 8000

demo:
	bash scripts/03_demo.sh

candidates:
	$(PY) benchmark/generate_candidates.py

eval:
	bash scripts/04_run_eval.sh

clean:
	rm -rf data/processed/* results/*.json

# ---- real-data profile (CPU; see README "Results on real data") ----
.PHONY: fetch-real real real-eval
fetch-real:
	python scripts/fetch_indiacode.py && python scripts/fetch_judgments.py 1500 1500 0,1,2,3,4,5,6,7

real:
	bash scripts/run_real.sh

real-eval:
	bash scripts/run_real_eval.sh
