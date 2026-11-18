"""BM25. The code-aware variant appends canonical section-ID tokens (``ipc_302``) for every resolved
reference, so "302" in an IPC murder case and "302" in a BNS religious-feelings case no longer share
their most discriminative token."""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

STOP = set("the a an of and or to in for by with is was be on as at that this it its from are were has have "
           "had which shall any not under".split())
ID_BOOST = 3


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z]+|\d+[a-z]?", text.lower()) if t not in STOP]


def id_tokens(section_ids: list[str]) -> list[str]:
    return [sid.lower() for sid in section_ids for _ in range(ID_BOOST)]


class BM25Index:
    def __init__(self, texts: list[str], section_ids: list[list[str]] | None = None, code_aware: bool = False):
        self.code_aware = code_aware
        docs = []
        for i, t in enumerate(texts):
            toks = tokenize(t)
            if code_aware and section_ids:
                toks += id_tokens(section_ids[i])
            docs.append(toks or ["_empty_"])
        self.bm25 = BM25Okapi(docs)

    def search(self, query: str, k: int = 30, query_ids: list[str] | None = None) -> list[tuple[int, float]]:
        q = tokenize(query)
        if self.code_aware and query_ids:
            q += id_tokens(query_ids)
        scores = self.bm25.get_scores(q)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(i, float(scores[i])) for i in order if scores[i] > 0]
