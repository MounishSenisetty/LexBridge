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
