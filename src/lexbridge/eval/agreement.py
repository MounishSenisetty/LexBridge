"""Cohen's κ between two annotators.

Usage: python -m lexbridge.eval.agreement a1.csv a2.csv --field gold_tag
   or: python -m lexbridge.eval.agreement both.csv --cols annotator_1 annotator_2
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter


def cohen_kappa(a: list[str], b: list[str]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label lists")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def _read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--field", default="gold_tag")
    ap.add_argument("--cols", nargs=2, default=None)
    a = ap.parse_args(argv)
    if len(a.files) == 1:
        rows = _read(a.files[0])
        c1, c2 = a.cols or ("annotator_1", "annotator_2")
        x, y = [r[c1] for r in rows], [r[c2] for r in rows]
    else:
        r1 = {r["qid"]: r[a.field] for r in _read(a.files[0])}
        r2 = {r["qid"]: r[a.field] for r in _read(a.files[1])}
        common = sorted(set(r1) & set(r2))
        x, y = [r1[q] for q in common], [r2[q] for q in common]
    k = cohen_kappa(x, y)
    print(f"n={len(x)} agreement={sum(i == j for i, j in zip(x, y)) / len(x):.3f} kappa={k:.3f}")
    return k


if __name__ == "__main__":
    main()
