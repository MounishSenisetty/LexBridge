"""Run a system over LexBridge-QA and write results/{system}.json.

python -m lexbridge.eval.run_eval --system lexbridge   (or no_rag | bm25 | dense | hybrid | all)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

from ..config import get_config
from ..resources import load_resources
from . import metrics as M

SYSTEMS = ["no_rag", "bm25", "dense", "hybrid", "lexbridge"]
K = 5


def load_questions(path: Path | None = None) -> list[dict]:
    path = path or get_config().path("benchmark") / "lexbridge_qa_sample.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `make candidates` first")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def evaluate(system, questions: list[dict]) -> dict:
    rows = []
    for q in questions:
        out = system.answer(q["question"], q.get("offence_date") if q["category"] != "temporal" else None)
        ranked = M.ranked_docs(out)
        gold = set(q.get("gold_docs") or [])
        row = {"qid": q["qid"], "category": q["category"], "resolved": out.get("resolved_sections"),
               "governing": out.get("governing_code", {}).get("governing"), "ranked_docs": ranked[:K],
               "faithfulness": out.get("faithfulness_score")}
        if q.get("gold_section"):
            row["code_correct"] = float(q["gold_section"] in (out.get("resolved_sections") or []))
        if gold and q["category"] in ("collision", "cross_code_precedent", "temporal"):
            row.update(recall=M.recall_at_k(ranked, gold, K), mrr=M.mrr(ranked, gold), ndcg=M.ndcg_at_k(ranked, gold, K))
            row["cit_p"], row["cit_r"] = M.citation_pr(out, gold)
        if q["category"] == "collision" and ranked:  # n/a for systems that retrieve nothing
            row["cer"] = M.collision_error(ranked, set(q.get("distractor_docs") or []), K)
        if q["category"] == "temporal" or q.get("gold_governing"):
            row["temporal_correct"] = float(row["governing"] == q["gold_governing"])
        if q["category"] == "transfer":
            tag = next((a.get("tag") for a in out.get("applicability", []) if a.get("old") == q["gold_section"]), None)
            row["pred_tag"], row["gold_tag"] = tag or "DIRECTLY_APPLICABLE", q.get("gold_tag")
        rows.append(row)

    def m(key, cat=None):
        return M.mean([r.get(key) for r in rows if cat is None or r["category"] == cat])

    tr = [r for r in rows if r["category"] == "transfer" and r.get("gold_tag")]
    summary = {
        "n": len(rows),
        f"CER@{K}": m("cer", "collision"),
        "code_resolution_acc": m("code_correct"),
        f"Recall@{K}": m("recall"), "MRR": m("mrr"), f"nDCG@{K}": m("ndcg"),
        "transfer_macroF1": M.macro_f1([r["gold_tag"] for r in tr], [r["pred_tag"] for r in tr]) if tr else None,
        "temporal_acc": m("temporal_correct"),
        "citation_P": m("cit_p"), "citation_R": m("cit_r"),
        "faithfulness": m("faithfulness"),
    }
    return {"summary": summary, "rows": rows}


def run(names: list[str], questions=None, res=None, write: bool = True) -> dict[str, dict]:
    from ..baselines import get_system
    res = res or load_resources()
    questions = questions or load_questions()
    out = {}
    out_dir = get_config().path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    for n in names:
        logger.info(f"evaluating {n} on {len(questions)} questions")
        r = evaluate(get_system(n, res), questions)
        r["system"] = n
        out[n] = r
        if write:
            (out_dir / f"{n.strip('-').replace(' ', '_')}.json").write_text(json.dumps(r, indent=2, default=str))
    return out


def fmt(v) -> str:
    return "--" if v is None else f"{v:.3f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="lexbridge", help=f"one of {SYSTEMS} or 'all'")
    a = ap.parse_args(argv)
    names = SYSTEMS if a.system == "all" else [a.system]
    results = run(names)
    for n, r in results.items():
        print(n, {k: fmt(v) if k != "n" else v for k, v in r["summary"].items()})


if __name__ == "__main__":
    main()
