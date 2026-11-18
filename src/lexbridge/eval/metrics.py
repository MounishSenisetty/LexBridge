"""Evaluation metrics for LexBridge-QA."""
from __future__ import annotations

import math


def ranked_docs(out: dict) -> list[str]:
    """Doc-level ranking from chunk citations (first occurrence wins)."""
    return list(dict.fromkeys(c["doc_id"] for c in out.get("citations", [])))


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return len(set(ranked[:k]) & gold) / len(gold) if gold else 0.0


def precision_at_k(ranked: list[str], gold: set[str], k: int) -> float | None:
    top = ranked[:k]
    return len(set(top) & gold) / len(top) if top and gold else (0.0 if gold else None)


def mrr(ranked: list[str], gold: set[str]) -> float:
    for i, d in enumerate(ranked, 1):
        if d in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1) if d in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def collision_error(ranked: list[str], distractors: set[str], k: int) -> float:
    """1 if any of the top-k documents is about the colliding (wrong-code) section."""
    return float(bool(set(ranked[:k]) & distractors))


def macro_f1(gold: list[str], pred: list[str]) -> float:
    labels = sorted(set(gold))
    if not labels:
        return 0.0
    f1s = []
    for l in labels:
        tp = sum(g == l and p == l for g, p in zip(gold, pred))
        fp = sum(g != l and p == l for g, p in zip(gold, pred))
        fn = sum(g == l and p != l for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


def citation_pr(out: dict, gold: set[str]) -> tuple[float | None, float | None]:
    cited_chunks = set(out.get("cited_ids", []))
    cited = {c["doc_id"] for c in out.get("citations", []) if c["id"] in cited_chunks}
    if not cited:
        return (0.0 if gold else None), (0.0 if gold else None)
    p = len(cited & gold) / len(cited)
    r = len(cited & gold) / len(gold) if gold else None
    return p, r


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None
