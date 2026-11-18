"""C1 evaluation on real judgments without annotation (masked-code self-supervision).

Every explicit reference ("Section 302 IPC") gives a free label. We blank out that code mention (same length,
so offsets hold), re-run the resolver with the explicit rule off, and compare. Baselines: always-IPC, and the
date rule alone. This measures rules 2–6 on real text; it cannot measure references that never carry a code.

python -m lexbridge.eval.resolver_eval [--n 3000]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict

from ..config import get_config
from ..nlp.code_resolver import _family_member, dominant_code
from ..nlp.section_ner import CODE_RE, extract_sections
from ..resources import load_resources


def masked_examples(judgments, n: int, seed: int = 7):
    rnd = random.Random(seed)
    ex = []
    for j in judgments:
        for r in extract_sections(j.text):
            if r.code and r.code in ("IPC", "BNS", "CrPC", "BNSS", "IEA", "BSA"):
                ex.append((j, r))
    rnd.shuffle(ex)
    return ex[:n]


def run(n: int = 3000) -> dict:
    res = load_resources()
    cfg = get_config()
    ex = masked_examples(res.judgments, n)
    stats = Counter()
    per_rule = defaultdict(Counter)
    confusion = Counter()
    for j, r in ex:
        # Blank the code mention inside this reference's phrase only.
        phrase = j.text[r.start:r.end]
        m = list(CODE_RE.finditer(phrase))
        if not m:
            continue
        a, b = r.start + m[-1].start(), r.start + m[-1].end()
        text = j.text[:a] + " " * (b - a) + j.text[b:]
        hidden = r.model_copy(update={"code": None})
        out = res.resolver.resolve_ref(hidden, text, j.date, dominant_code(text), None, {"explicit"})
        gold = r.code
        stats["n"] += 1
        stats["correct"] += out.code == gold
        per_rule[out.rule]["n"] += 1
        per_rule[out.rule]["correct"] += out.code == gold
        stats["always_ipc"] += gold == "IPC"
        if j.date:
            fam = "old" if j.date < cfg.transition else "new"
            ctx = text[max(0, r.start - 40): r.end + 40]
            stats["date_only"] += _family_member(fam, r.section, ctx, res.lexicon) == gold
            stats["dated"] += 1
        confusion[(gold, out.code)] += 1
    N = max(1, stats["n"])
    report = {
        "n": stats["n"],
        "accuracy": round(stats["correct"] / N, 4),
        "baseline_always_IPC": round(stats["always_ipc"] / N, 4),
        "baseline_date_only": round(stats["date_only"] / max(1, stats["dated"]), 4),
        "per_rule": {k: {"n": v["n"], "acc": round(v["correct"] / max(1, v["n"]), 4)} for k, v in per_rule.items()},
        "confusion_top": [[g, p, c] for (g, p), c in confusion.most_common(12)],
    }
    out = cfg.path("results")
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolver_eval.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    print(json.dumps(run(ap.parse_args(argv).n), indent=2))


if __name__ == "__main__":
    main()
