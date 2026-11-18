"""Reciprocal Rank Fusion."""
from __future__ import annotations


def rrf(rankings: list[list[int]], k: int = 60, weights: list[float] | None = None) -> list[tuple[int, float]]:
    weights = weights or [1.0] * len(rankings)
    scores: dict[int, float] = {}
    for w, ranking in zip(weights, rankings):
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + w / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])
