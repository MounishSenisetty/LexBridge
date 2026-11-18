"""Generate DRAFT LexBridge-QA candidates from templates + mappings + fixture corpus.

Outputs data/benchmark/candidates.jsonl (all) and lexbridge_qa_sample.jsonl (stratified sample, default 50).
Gold relevant documents come from the hand-labelled ``gold_sections`` of fixture judgments, never from the
system's own resolver. All items carry status=draft until annotated (see annotation_guidelines.md)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lexbridge.config import get_config  # noqa: E402
from lexbridge.nlp.date_extractor import extract_dates  # noqa: E402
from lexbridge.resources import load_resources  # noqa: E402


def lower_title(t: str) -> str:
    return t.strip().lower() or "this offence"


def build(res, tpl: dict) -> list[dict]:
    transition = get_config().transition
    docs_by_sec: dict[str, set[str]] = {}
    for j in res.judgments:
        for s in j.gold_sections:
            docs_by_sec.setdefault(s, set()).add(j.doc_id)

    def docs(*ids):
        return sorted(set().union(*(docs_by_sec.get(i, set()) for i in ids if i)))

    items: list[dict] = []

    def add(cat, tid, q, **gold):
        items.append({"qid": f"{cat[:3].upper()}-{len(items) + 1:03d}", "category": cat, "template_id": tid,
                      "question": q, "status": "draft", **gold})

    # --- collision pairs: same number in IPC and BNS, different offence (heading overlap < 0.5)
    pairs = []
    lex = res.lexicon
    for sid in lex.tok:
        code, num = sid.split("_")
        if code != "IPC" or f"BNS_{num}" not in lex.tok:
            continue
        a, b = lex.tok[sid], lex.tok[f"BNS_{num}"]
        if a and b and len(a & b) / min(len(a), len(b)) < 0.5:
            pairs.append(num)
    for num in sorted(pairs):
        for code, other in (("BNS", "IPC"), ("IPC", "BNS")):
            sid, distract = f"{code}_{num}", f"{other}_{num}"
            cps = res.table.counterparts(sid)
            for t_i, t in enumerate(tpl["collision"]):
                add("collision", f"collision.{t_i}", t.format(num=num, code=code, code_long=tpl["code_long"][code]),
                    gold_section=sid, distractor_section=distract, gold_docs=docs(sid, *cps),
                    distractor_docs=docs(distract), gold_governing=None, offence_date=None)
        for d in (tpl["dates_before"][0], tpl["dates_after"][1]):
            dd = extract_dates(d)[0].value
            sid = f"{'IPC' if dd < transition else 'BNS'}_{num}"
            distract = f"{'BNS' if dd < transition else 'IPC'}_{num}"
            add("collision", "collision_implicit.0", tpl["collision_implicit"][0].format(num=num, date=d),
                gold_section=sid, distractor_section=distract, gold_docs=docs(sid, *res.table.counterparts(sid)),
                distractor_docs=docs(distract), gold_governing="old" if dd < transition else "new",
                offence_date=dd.isoformat())

    # --- cross-code precedent
    for m in res.mappings:
        if not m.new_id:
            continue
        for t_i, t in enumerate(tpl["cross_code_precedent"]):
            add("cross_code_precedent", f"cross_code_precedent.{t_i}",
                t.format(new=m.new_section, title=lower_title(m.title)),
                gold_section=m.new_id, gold_docs=docs(m.new_id, m.old_id), gold_governing=None, offence_date=None)

    # --- temporal
    for k, m in enumerate(res.mappings):
        if not m.new_id:
            continue
        before = tpl["dates_before"][k % len(tpl["dates_before"])]
        after = tpl["dates_after"][k % len(tpl["dates_after"])]
        for t_i, (t, d) in enumerate(zip(tpl["temporal"], (before, after) if k % 2 == 0 else (after, before))):
            dd = extract_dates(d)[0].value
            fam = "old" if dd < transition else "new"
            add("temporal", f"temporal.{t_i}", t.format(old=m.old_section, title=lower_title(m.title), date=d),
                gold_section=m.old_id if t_i == 1 else None, gold_governing=fam,
                gold_governing_sections=[m.old_id if fam == "old" else m.new_id],
                gold_docs=docs(m.old_id, m.new_id), offence_date=dd.isoformat())

    # --- transfer
    for m in res.mappings:
        add("transfer", "transfer.0", tpl["transfer"][0].format(old=m.old_section),
            gold_section=m.old_id, gold_tag=tpl["draft_transfer_gold"].get(m.old_id), gold_docs=docs(m.old_id, m.new_id),
            gold_governing=None, offence_date=None)
    return items


def stratified(items: list[dict], n: int) -> list[dict]:
    by: dict[str, list[dict]] = {}
    for it in items:
        by.setdefault(it["category"], []).append(it)
    out, i = [], 0
    while len(out) < min(n, len(items)):
        for cat in sorted(by):
            if i < len(by[cat]) and len(out) < n:
                out.append(by[cat][i])
        i += 1
    return sorted(out, key=lambda x: x["qid"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=get_config()["sample_sizes"]["questions"])
    ap.add_argument("--out", default=str(get_config().path("benchmark")))
    a = ap.parse_args(argv)
    tpl = yaml.safe_load(open(ROOT / "benchmark" / "templates.yaml"))
    items = build(load_resources(), tpl)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidates.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
    sample = stratified(items, a.n)
    (out / "lexbridge_qa_sample.jsonl").write_text("".join(json.dumps(i) + "\n" for i in sample))
    cats = {}
    for i in sample:
        cats[i["category"]] = cats.get(i["category"], 0) + 1
    print(f"{len(items)} draft candidates → {out/'candidates.jsonl'}; sample of {len(sample)} {cats}")


if __name__ == "__main__":
    main()
