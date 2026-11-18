# LexBridge-QA annotation guidelines (v0.1)

Two annotators (ideally law students or practitioners) label each candidate independently. Target
Cohen's κ ≥ 0.7 per field before adjudication. **Not legal advice; annotators use the bare acts.**

## Sources you must use
- Bare acts from India Code (indiacode.nic.in): IPC 1860, CrPC 1973, Evidence Act 1872, BNS 2023,
  BNSS 2023, BSA 2023. Do **not** rely on the fixture texts in `data/seed/fixtures/` (abridged).
- The official MHA / Bureau of Police Research comparison tables for the mapping.

## Fields
| Field | Values | How to decide |
|---|---|---|
| `question_valid` | yes / no / fix | Is the question answerable and unambiguous? Use `fix` and rewrite in `notes`. |
| `gold_section` | e.g. `BNS_302` | The section the question is about, with its code. If the question names no code, decide from the offence date (rule below), else from the offence words. |
| `gold_governing` | old / new / both | Offence date before 1 July 2024 → `old`; on or after → `new`; no date → `both`. |
| `gold_tag` | DIRECTLY_APPLICABLE, APPLICABLE_WITH_CAVEAT, REVIEW_REQUIRED, NOT_APPLICABLE | Transfer questions only. See below. |
| `gold_docs` | `;`-separated doc_ids | Judgments a lawyer would cite for this question (any code). |

## Applicability tags
- **DIRECTLY_APPLICABLE** — ingredients and punishment are the same (identical or reworded only).
- **APPLICABLE_WITH_CAVEAT** — same ingredients but punishment changed, or the provision was merged/split.
  Precedent on ingredients still applies; sentencing precedent may not.
- **REVIEW_REQUIRED** — ingredients changed (e.g. IPC 124A → BNS 152). Read both texts.
- **NOT_APPLICABLE** — the offence has no counterpart (e.g. IPC 377 for adults).

## Collision category
The trap: same number, different offence (IPC 302 murder vs BNS 302 religious feelings). Check that
`gold_section` names the offence the *question* is about, not the more famous one.

## Process
1. Each annotator fills `annotation_sheet_template.csv` (one row per qid) without discussion.
2. Run `python -m lexbridge.eval.agreement a1.csv a2.csv --field gold_tag` (and the other fields).
3. Adjudicate disagreements; record the final label and a one-line reason.
4. Flip `status` from `draft` to `gold` only after adjudication.
