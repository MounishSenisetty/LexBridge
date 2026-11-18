# Data

There are two profiles:

- **Sample** (`configs/config.yaml`, the default) runs on the seed and fixture data in `data/seed/`. It needs no
  downloads and is what the tests use.
- **Real** (`LEXBRIDGE_CONFIG=configs/config.real.yaml`) reads `data/raw/` and writes `data/processed_real/`,
  `data/benchmark_real/` and `results/real/`. `bash scripts/run_real.sh` runs the whole pipeline.

`data/raw/` is git-ignored. The fetch scripts below recreate it.

## Real sources

| What | Source | How to get it | License / terms | Status |
|---|---|---|---|---|
| BNS, BNSS, BSA: section-level official text | India Code, DSpace 7 REST API at `indiacode.gov.in/server/api` | `python scripts/fetch_indiacode.py` → `raw/statutes/{BNS,BNSS,BSA}.indiacode.jsonl` | Government of India legislation | ✅ 358 / 531 / 170 sections |
| IPC, IEA: official PDFs | India Code repealed-acts items ("Rep., Act 45/47 of 2023") | same script → `raw/statutes/{ipc,iea}.indiacode.pdf`, parsed with `strict_order` | same | ✅ 553 / 181 sections |
| CrPC | India Code | same script | same | ❌ Only an OCR scan of the original 1974 gazette, unamended and heavily garbled. Not used. |
| IPC→BNS mapping (562 rows, with IPC and BNS texts) | HF `nandhakumarg/IPC_and_BNS_transformation` | `curl` the CSV to `raw/mappings/hf/` (see below) | Apache-2.0 | ✅ |
| IPC→BNS mapping, second source | IEEE DataPort "Indian Criminal Law Transition Mapping Dataset" | Manual download (login) → `raw/mappings/ieee_*.xlsx` | IEEE DataPort terms | ⏳ Needs your login |
| CrPC→BNSS and IEA→BSA mappings | — | No open source found | — | ❌ Gap |
| Judgments: 1,500 IPC-era + 1,462 BNS-era criminal decisions | HF `overthelex/indian-court-decisions` (eCourts and Supreme Court via the AWS Open Data registry, 1950–2026) | `python scripts/fetch_judgments.py 1500 1500 0,1,2,3,4,5,6,7` (reads parquet row groups over HTTP) | CC-BY-4.0 | ✅ |
| ILDC / IL-TUR (CJPE, rhetorical roles, LSI) | HF `Exploration-Lab/IL-TUR` | Gated: accept the terms on the dataset page while logged in, with a HF token in the environment. The loader reads `raw/ildc/iltur/*.parquet` | CC-BY-NC-SA-4.0 | ⏳ Terms not yet accepted |
| InJudgements (11,970 judgments, 1950–2017) | HF `opennyaiorg/InJudgements_dataset` | Gated, same as above → `raw/judgments/injudgements/*.parquet` | Apache-2.0 | ⏳ Terms not yet accepted |
| Rhetorical-role training data (Indian SC, sentence-level) | HF `subinay612/MARRO_Rhetorical-Role-Labeling` (`IN-*` splits) | `curl` the zips to `raw/roles/`, unzip to `raw/roles/in/` | **No license stated.** Used locally only, never committed. | ✅ 30,729 train / 2,602 test sentences |
| Indian Kanoon | indiankanoon.org | Manual only; follow the site's terms; no scraping | — | Not used |

HF mapping CSV:

```bash
mkdir -p data/raw/mappings/hf
curl -L -o data/raw/mappings/hf/hf_ipc_bns.csv "https://huggingface.co/datasets/nandhakumarg/IPC_and_BNS_transformation/resolve/main/IPC%20and%20BNS%20transformation%20.csv"
```

## Mapping checks

The ingest step compares every IPC→BNS row against the official India Code text of both sides
(`src/lexbridge/ingest/verify_mappings.py`) and writes `processed_real/mapping_verification.csv`:

| Status | Rows |
|---|---|
| consistent | 508 |
| suspect | 32 |
| unchecked (no official text for the IPC side) | 22 |

The suspect rows are mostly clear numbering errors in the HF dataset, where another BNS section reproduces the
IPC text almost verbatim:

- IPC 171A → HF says BNS 206; the official text matches **BNS 169** (100%).
- IPC 442–460 (house-trespass): HF is off by one, e.g. 443 → 331 where the text matches **BNS 330** (92%).
- IPC 368 → HF says 140(2); the text matches **BNS 142** (91%).
- Several BNS 2(N) definition clause numbers.

Splits are recognised as splits, not errors. For example, IPC 498A → BNS 85 has its definition of cruelty
moved to BNS 86.

**This is an automatic consistency check, not legal verification.** `SectionMapping.verified` stays `false` until
a person checks the row against the bare acts. The benchmark uses only `consistent` rows.

The check also caught an error in this repo's own fixture. BNS 302 reads "imprisonment of either description",
not "simple imprisonment", so IPC 298 → BNS 302 is identical. The fixture and the draft gold are now corrected.

## Seed and fixtures (in repo, sample profile)

- `seed/seed_mappings.csv`: 11 IPC→BNS rows. All 11 agree with the HF dataset and pass the automatic check.
- `seed/fixtures/statutes/{IPC,BNS}.txt`: abridged texts. Do not cite them.
- `seed/fixtures/judgments.jsonl`: 14 **synthetic** snippets with hand-labelled `gold_sections`.
