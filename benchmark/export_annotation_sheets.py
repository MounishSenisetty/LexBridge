"""Export annotation sheets for two annotators (κ via lexbridge.eval.agreement).

python benchmark/export_annotation_sheets.py   (uses the active config: sample or real)
  → <benchmark>/annotation_sheet_A.csv, annotation_sheet_B.csv  (identical; one per annotator)
  → <benchmark>/mapping_review.csv                          (real profile: suspect rows first)
Suggested values come from templates and silver labels. Annotators must check them against the bare acts and
the judgments, not copy them."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lexbridge.config import get_config  # noqa: E402

COLS = ["qid", "category", "question", "offence_date", "suggested_gold_section", "suggested_governing",
        "suggested_docs", "gold_section", "gold_governing", "gold_tag", "gold_docs", "question_valid", "notes"]


def main():
    cfg = get_config()
    bdir = cfg.path("benchmark")
    qs = [json.loads(l) for l in (bdir / "lexbridge_qa_sample.jsonl").read_text().splitlines() if l.strip()]
    for who in ("A", "B"):
        with open(bdir / f"annotation_sheet_{who}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            for q in qs:
                w.writerow({"qid": q["qid"], "category": q["category"], "question": q["question"],
                            "offence_date": q.get("offence_date") or "",
                            "suggested_gold_section": q.get("gold_section") or "",
                            "suggested_governing": q.get("gold_governing") or "",
                            "suggested_docs": ";".join((q.get("gold_docs") or [])[:10])})
    print(f"{len(qs)} questions → {bdir}/annotation_sheet_{{A,B}}.csv")
    ver = cfg.path("processed") / "mapping_verification.csv"
    if ver.exists():
        rows = list(csv.DictReader(open(ver)))
        order = {"suspect": 0, "unchecked": 1, "consistent": 2}
        rows.sort(key=lambda r: (order.get(r["status"], 3), r["old"]))
        cols = list(rows[0]) + ["reviewer_new_section", "reviewer_change_type", "reviewer_verified", "reviewer_notes"]
        with open(bdir / "mapping_review.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"{len(rows)} mappings → {bdir}/mapping_review.csv (suspect first)")


if __name__ == "__main__":
    main()
