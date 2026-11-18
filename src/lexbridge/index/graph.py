"""C2 graph (NetworkX DiGraph).

Nodes: ``Section`` (id ``IPC_302``) and ``Judgment`` (id = doc_id).
Edges: Judgment -CITES-> Section, Section(old) -MAPS_TO{change_type}-> Section(new),
       Section(new) -MAPPED_FROM-> Section(old)."""
from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx

from ..schemas import Chunk, SectionMapping, StatuteSection


def build_graph(statutes: list[StatuteSection], mappings: list[SectionMapping], chunks: list[Chunk],
                change_types: dict[str, str] | None = None) -> nx.DiGraph:
    g = nx.DiGraph()
    change_types = change_types or {}
    for s in statutes:
        g.add_node(s.id, kind="Section", code=s.code, section=s.section, title=s.title)
    for m in mappings:
        if m.old_id not in g:
            g.add_node(m.old_id, kind="Section", code=m.old_code, section=m.old_section, title=m.title)
        g.nodes[m.old_id].setdefault("title", m.title)
        if m.new_id:
            if m.new_id not in g:
                g.add_node(m.new_id, kind="Section", code=m.new_code, section=m.new_section, title=m.title)
            ct = change_types.get(m.old_id, m.change_type.value)
            g.add_edge(m.old_id, m.new_id, rel="MAPS_TO", change_type=ct)
            g.add_edge(m.new_id, m.old_id, rel="MAPPED_FROM", change_type=ct)
        else:
            g.nodes[m.old_id]["omitted"] = True
    for c in chunks:
        if c.doc_id not in g:
            g.add_node(c.doc_id, kind="Judgment", title=c.title, date=str(c.date) if c.date else None)
        for sid in c.section_ids:
            if sid not in g:
                code, sec = sid.split("_", 1)
                g.add_node(sid, kind="Section", code=code, section=sec, title="")
            if g.has_edge(c.doc_id, sid):
                g[c.doc_id][sid]["chunks"].append(c.chunk_id)
            else:
                g.add_edge(c.doc_id, sid, rel="CITES", chunks=[c.chunk_id])
    return g


def counterparts(g: nx.DiGraph, sid: str) -> list[tuple[str, str]]:
    """[(other_section_id, change_type)] across the transition."""
    if sid not in g:
        return []
    return [(v, d.get("change_type", "unknown")) for _, v, d in g.out_edges(sid, data=True)
            if d.get("rel") in ("MAPS_TO", "MAPPED_FROM")]


def citing_judgments(g: nx.DiGraph, sid: str) -> list[str]:
    if sid not in g:
        return []
    return sorted(u for u, _, d in g.in_edges(sid, data=True) if d.get("rel") == "CITES")


def neighbors(g: nx.DiGraph, sid: str) -> dict:
    """Section neighbourhood: counterparts and the judgments citing the section or its counterparts."""
    cps = counterparts(g, sid)
    judg = set(citing_judgments(g, sid))
    for other, _ in cps:
        judg |= set(citing_judgments(g, other))
    return {"section": sid, "counterparts": [c for c, _ in cps],
            "change_types": {c: t for c, t in cps}, "judgments": sorted(judg)}


def save_graph(g: nx.DiGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(g, f)


def load_graph(path: Path) -> nx.DiGraph:
    with open(path, "rb") as f:
        return pickle.load(f)
