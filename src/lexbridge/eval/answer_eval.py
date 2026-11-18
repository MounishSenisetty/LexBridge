"""Answer-level checks on saved generation results (results/*_gen.json rows).

offence_hit:      the answer names the right offence (≥60% of the gold section's official heading words).
answer_collision: it names the colliding offence instead (collision questions).
python -m lexbridge.eval.answer_eval [--tag _gen]
"""
from __future__ import annotations

import argparse
import json

from ..config import get_config
from ..ingest.statutes import StatuteStore, load_jsonl
from ..ingest.verify_mappings import _toks
from ..schemas import StatuteSection
from .run_eval import SYSTEMS, load_questions


def llm_body(answer: str) -> str:
    """Drop LexBridge's templated lines (header, dual citation) and the disclaimer: score only what the LLM wrote."""
    keep = []
    for line in answer.split("This is general legal information")[0].splitlines():
        if line.startswith(("Governing provision(s):", "Could not identify", "Dual citation:")):
            continue
        keep.append(line)
    return "\n".join(keep).strip()


def sentences(text: str) -> list[str]:
    import re
    return [x.strip() for x in re.split(r"(?<=[.;:!?])\s+|\n+", text) if len(x.strip()) > 25]


def hit(toks: set, answer: str, thr: float = 0.6) -> bool:
    a = set(_toks(answer))
    return bool(toks) and len(toks & a) / len(toks) >= thr


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_gen")
    a = ap.parse_args(argv)
    cfg = get_config()
    store = StatuteStore(load_jsonl(cfg.path("processed") / "statutes.jsonl", StatuteSection))
    qs = {q["qid"]: q for q in load_questions()}
    out = {}
    import re
    from ..generate.verifier import lexical_support
    res = None
    for name in SYSTEMS:
        p = cfg.path("results") / f"{name}{a.tag}.json"
        if not p.exists():
            continue
        rows = json.loads(p.read_text())["rows"]
        per, grounded, cite_ok = {}, [], []
        system = None
        if name != "no_rag":
            import os
            os.environ["LLM_BACKEND"] = "mock"  # retrieval only; the saved answers are what we score
            from ..baselines import get_system
            from ..resources import load_resources
            res = res or load_resources()
            system = get_system(name, res)
        for r in rows:
            q = qs[r["qid"]]
            body = llm_body(r["answer"])
            gold = q.get("gold_section") or (q.get("gold_governing_sections") or [None])[0]
            st = store.by_id.get(gold) if gold else None
            if st and st.title:
                per.setdefault(r["category"], []).append(float(hit(set(_toks(st.title)), body)))
            cite_ok.append(float(bool(re.search(r"\[[^\]]*(¶|§)[^\]]*\]", body))))
            if system is not None:
                o = system.answer(q["question"], q.get("offence_date") if q["category"] != "temporal" else None)
                ev = " ".join(c["text"] for c in o["citations"])
                if name == "lexbridge":
                    ev += " " + " ".join(store.by_id[s].text for s in o["governing_code"].get("sections", []) if s in store.by_id)
                sents = sentences(body)
                if sents:
                    grounded.append(sum(lexical_support(x, ev) >= 0.5 for x in sents) / len(sents))
        allv = [v for vs in per.values() for v in vs]
        out[name] = {"offence_hit": round(sum(allv) / len(allv), 4) if allv else None, "n": len(allv),
                     "grounded_rate": round(sum(grounded) / len(grounded), 4) if grounded else None,
                     "citation_format_rate": round(sum(cite_ok) / len(cite_ok), 4) if cite_ok else None,
                     "per_category": {c: round(sum(v) / len(v), 4) for c, v in per.items()}}
    (cfg.path("results") / f"answer_eval{a.tag}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
