# Data

Everything runs out of the box on **seed + fixture data** (`data/seed/`). Real data goes in `data/raw/`
(git-ignored) and is picked up automatically by `make ingest` when present.

| Folder | What to put there | Source |
|---|---|---|
| `raw/statutes/` | Bare-act PDFs or text. The filename must contain the code: `ipc.pdf`, `crpc.pdf`, `iea.pdf`, `bns.pdf`, `bnss.pdf`, `bsa.pdf` | [India Code](https://www.indiacode.nic.in): search each act, download the PDF |
| `raw/mappings/` | `ieee_*.xlsx` (or `.csv`) | IEEE DataPort, "Indian Criminal Law Transition Mapping Dataset" (login required) |
| — (auto) | HF dataset `nandhakumarg/IPC_and_BNS_transformation` | Downloaded by `datasets.load_dataset` when `sample_mode: false` in `configs/config.yaml` and the network allows it; otherwise skipped with a warning |
| `raw/ildc/` | ILDC CSV / JSON (`name`, `text`, …) | Exploration-Lab CJPE / ILDC GitHub (request form) |
| `raw/judgments/` | Any `.txt` / `.pdf` judgments | e.g. manually downloaded from indiankanoon.org; follow the site's terms, do not scrape |

Install the optional loaders with `pip install -e ".[data]"` (datasets, pandas, openpyxl, pypdf).

## Seed and fixtures (in repo)
- `seed/seed_mappings.csv`: 11 IPC→BNS rows, `verified=false`. **For unit tests only.**
- `seed/fixtures/statutes/{IPC,BNS}.txt`: abridged section texts for the seed rows. Labelled
  "test fixture — verify against the bare act". Do not cite them.
- `seed/fixtures/judgments.jsonl`: 14 **synthetic** judgment snippets (regenerate with
  `python data/seed/fixtures/make_fixtures.py`). Each carries hand-labelled `gold_sections`, which give
  benchmark qrels that do not depend on the system's own resolver.

## Outputs
`make ingest` → `processed/{statutes,mappings,judgments}.jsonl`, `processed/mapping_conflicts.csv`
`make index` → `processed/{chunks.jsonl,index.pkl,graph.gpickle,transferability.jsonl,index_meta.json}` and
`benchmark/{candidates,lexbridge_qa_sample}.jsonl`
