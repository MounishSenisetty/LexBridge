"""C2 cross-code hop + collision filter."""
from __future__ import annotations

import networkx as nx

from ..index.graph import citing_judgments, counterparts
from ..schemas import Chunk

ROLE_PRIOR = {"analysis": 3, "decision": 2, "statute": 2, "facts": 1, "arguments": 1, "preamble": 1, "other": 0}


def expand_ids(g: nx.DiGraph, target_ids: list[str]) -> list[str]:
    """Targets plus their counterparts across the transition (BNS_103 → IPC_302)."""
    out = list(dict.fromkeys(target_ids))
    for sid in target_ids:
        for other, _ in counterparts(g, sid):
            if other not in out:
                out.append(other)
    return out


def graph_candidates(g: nx.DiGraph, ids: list[str], chunks: list[Chunk], k: int = 20) -> list[int]:
    """Chunk indices from judgments that CITE any of ``ids``; chunks mentioning the ids first."""
    idset = set(ids)
    docs = set()
    for sid in ids:
        docs |= set(citing_judgments(g, sid))
    scored = []
    for i, c in enumerate(chunks):
        if c.doc_id not in docs:
            continue
        direct = len(idset & set(c.section_ids))
        scored.append((-(direct * 10 + ROLE_PRIOR.get(c.role, 0)), i))
    return [i for _, i in sorted(scored)[:k]]


def _num(sid: str) -> str:
    return sid.split("_", 1)[1]


def is_collision(chunk: Chunk, allowed_ids: list[str]) -> bool:
    """True when the chunk is about a *different* offence that merely shares a section number with
    one of the allowed sections (e.g. BNS_302 religious feelings vs IPC_302 murder)."""
    allowed = set(allowed_ids)
    nums = {_num(a) for a in allowed}
    # Use chunk-local refs when present, otherwise the judgment-level refs.
    ids = set(chunk.section_ids) or set(chunk.doc_section_ids)
    wrong = {s for s in ids if _num(s) in nums and s not in allowed}
    return bool(wrong) and not (ids & allowed)


def collision_filter(chunks: list[Chunk], idxs: list[int], allowed_ids: list[str]) -> tuple[list[int], list[int]]:
    keep, dropped = [], []
    for i in idxs:
        (dropped if is_collision(chunks[i], allowed_ids) else keep).append(i)
    return keep, dropped
