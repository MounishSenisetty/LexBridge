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
                # A BNS question: precedents under EVERY IPC predecessor of the new section are relevant
            # (BNS 318 ← IPC 415/417/418/420), not only the row the question was generated from.
            gold_section=m.new_id, gold_docs=docs(m.new_id, *res.table.counterparts(m.new_id)),
            gold_governing=None, offence_date=None)

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


def split_of(key: str) -> str:
    """Deterministic 50/50 dev/test split (design decisions on dev only; report test)."""
    import hashlib
    return "dev" if int(hashlib.md5(key.encode()).hexdigest(), 16) % 2 == 0 else "test"


def silver_docs(res) -> dict[str, set[str]]:
    """section_id -> judgments that cite it WITH an explicit code ("Section 302 IPC"). Weak labels: they do
    not depend on the resolver's inference rules, but they are not human relevance judgements."""
    from lexbridge.nlp.section_ner import extract_sections
    out: dict[str, set[str]] = {}
    for j in res.judgments:
        for r in extract_sections(j.text):
            if r.code:
                out.setdefault(f"{r.code}_{r.section}", set()).add(j.doc_id)
    return out


def build_real(res, tpl: dict, per_cat: int = 125, min_docs: int = 3, seed: int = 13) -> list[dict]:
    import random
    from lexbridge.ingest.verify_mappings import heading_sim
    rnd = random.Random(seed)
    transition = get_config().transition
    silver = silver_docs(res)

    def docs(*ids):
        return sorted(set().union(*(silver.get(i, set()) for i in ids if i)))

    def low(t):
        return t.strip().rstrip(".").lower() or "this offence"

    items: list[dict] = []

    def add(cat, tid, q, **gold):
        qid = f"{cat[:3].upper()}-{len(items) + 1:04d}"
        items.append({"qid": qid, "category": cat, "template_id": tid, "question": q, "status": "draft",
                      "labels": "silver", "split": split_of(qid), **gold})

    store = res.store.by_id
    good = [m for m in res.mappings if m.auto_check == "consistent"]
    # --- collision: same number in IPC and BNS, unrelated official headings, both cited in the corpus
    pairs = []
    for sid, s in store.items():
        if s.code != "IPC" or f"BNS_{s.section}" not in store:
            continue
        b = store[f"BNS_{s.section}"]
        if heading_sim(s.title, b.title) < 0.3 and len(silver.get(sid, ())) >= min_docs and len(silver.get(b.id, ())) >= min_docs:
            pairs.append(s.section)
    rnd.shuffle(pairs)
    col = []
    for num in pairs:
        for code, other in (("BNS", "IPC"), ("IPC", "BNS")):
            sid, distract = f"{code}_{num}", f"{other}_{num}"
            cps = res.table.counterparts(sid)
            t_i = rnd.randrange(len(tpl["collision"]))
            col.append(("collision", f"collision.{t_i}",
                        tpl["collision"][t_i].format(num=num, code=code, code_long=tpl["code_long"][code]),
                        dict(gold_section=sid, distractor_section=distract, gold_docs=docs(sid, *cps),
                             distractor_docs=sorted(set(silver.get(distract, set())) - set(docs(sid, *cps))),
                             gold_governing=None, offence_date=None)))
        for d in (rnd.choice(tpl["dates_before"]), rnd.choice(tpl["dates_after"])):
            dd = extract_dates(d)[0].value
            fam_code, other = ("IPC", "BNS") if dd < transition else ("BNS", "IPC")
            sid = f"{fam_code}_{num}"
            col.append(("collision", "collision_implicit.0", tpl["collision_implicit"][0].format(num=num, date=d),
                        dict(gold_section=sid, distractor_section=f"{other}_{num}",
                             gold_docs=docs(sid, *res.table.counterparts(sid)),
                             distractor_docs=sorted(silver.get(f"{other}_{num}", set()) - set(docs(sid, *res.table.counterparts(sid)))),
                             gold_governing="old" if dd < transition else "new", offence_date=dd.isoformat())))
    for c in col[:per_cat]:
        add(c[0], c[1], c[2], **c[3])

    # --- cross-code precedent: BNS question; the IPC-era precedent must be found through the mapping
    cc = [m for m in good if m.new_id and len(silver.get(m.old_id, ())) >= min_docs]
    rnd.shuffle(cc)
    for m in cc[:per_cat]:
        t_i = rnd.randrange(len(tpl["cross_code_precedent"]))
        title = store[m.new_id].title if m.new_id in store else m.title
        add("cross_code_precedent", f"cross_code_precedent.{t_i}",
            tpl["cross_code_precedent"][t_i].format(new=m.new_section, title=low(title)),
            # A BNS question: precedents under EVERY IPC predecessor of the new section are relevant
            # (BNS 318 ← IPC 415/417/418/420), not only the row the question was generated from.
            gold_section=m.new_id, gold_docs=docs(m.new_id, *res.table.counterparts(m.new_id)),
            gold_governing=None, offence_date=None)

    # --- temporal: the date decides the governing provision
    tm = [m for m in good if m.new_id and len(silver.get(m.old_id, ())) + len(silver.get(m.new_id, ())) >= min_docs]
    rnd.shuffle(tm)
    for k, m in enumerate(tm[:per_cat]):
        d = rnd.choice(tpl["dates_before"] if k % 2 == 0 else tpl["dates_after"])
        dd = extract_dates(d)[0].value
        fam = "old" if dd < transition else "new"
        t_i = k % len(tpl["temporal"])
        title = store[m.old_id].title if m.old_id in store else m.title
        add("temporal", f"temporal.{t_i}", tpl["temporal"][t_i].format(old=m.old_section, title=low(title), date=d),
            gold_section=m.old_id if t_i == 1 else None, gold_governing=fam,
            gold_governing_sections=[m.old_id if fam == "old" else m.new_id],
            gold_docs=docs(m.old_id, m.new_id), offence_date=dd.isoformat())

    # --- transfer: gold tags need annotators; left empty (not scored) until annotated
    tr = [m for m in good if len(silver.get(m.old_id, ())) >= 1]
    rnd.shuffle(tr)
    for m in tr[:per_cat]:
        add("transfer", "transfer.0", tpl["transfer"][0].format(old=m.old_section), gold_section=m.old_id,
            gold_tag=None, gold_docs=docs(m.old_id, m.new_id), gold_governing=None, offence_date=None)
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
    real = not get_config().get("sample_mode", True)
    items = build_real(load_resources(), tpl, per_cat=a.n // 4 + 1) if real else build(load_resources(), tpl)
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
