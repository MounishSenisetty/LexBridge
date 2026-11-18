"""Chunking. Role-aware: paragraph → rhetorical role, consecutive same-role paragraphs merged up to
``size`` words, long paragraphs split with ``overlap``. Naive: fixed word windows (baselines/ablation).
Every chunk carries the resolved section IDs it mentions plus the judgment-level IDs."""
from __future__ import annotations

from ..ingest.judgments import paragraphs
from ..nlp.code_resolver import CodeResolver
from ..nlp.rhetorical_roles import tag_paragraph
from ..schemas import Chunk, Judgment


def _windows(words: list[str], size: int, overlap: int):
    step = max(1, size - overlap)
    for i in range(0, max(1, len(words)), step):
        yield words[i:i + size]
        if i + size >= len(words):
            break


def _family(j: Judgment, transition) -> str | None:
    if not j.date:
        return None
    return "old" if j.date < transition else "new"


def _resolved_spans(j: Judgment, resolver: CodeResolver):
    return [(r.start, r.id) for r in resolver.resolve(j.text, j.date) if r.id]


def chunk_judgment(j: Judgment, resolver: CodeResolver, size: int = 400, overlap: int = 50,
                   role_aware: bool = True) -> list[Chunk]:
    spans = _resolved_spans(j, resolver)
    doc_ids = sorted({sid for _, sid in spans})
    fam = _family(j, resolver.transition)
    # Locate paragraphs in the original text so resolved refs can be assigned by offset.
    units: list[tuple[int, int, str, str]] = []  # (start, end, text, role)
    if role_aware:
        pos = 0
        for p in paragraphs(j.text):
            s = j.text.find(p, pos)
            s = pos if s < 0 else s
            units.append((s, s + len(p), p, tag_paragraph(p)))
            pos = s + len(p)
        merged: list[list] = []
        for u in units:
            if merged and merged[-1][3] == u[3] and len((merged[-1][2] + " " + u[2]).split()) <= size:
                merged[-1] = [merged[-1][0], u[1], merged[-1][2] + "\n\n" + u[2], u[3]]
            else:
                merged.append(list(u))
        units = [tuple(m) for m in merged]
    else:
        units = [(0, len(j.text), j.text, "other")]

    chunks: list[Chunk] = []
    for para_idx, (s, e, text, role) in enumerate(units):
        words = text.split()
        pieces = [" ".join(w) for w in _windows(words, size, overlap)] if len(words) > size or not role_aware \
            else [text]
        for k, piece in enumerate(pieces):
            if not piece.strip():
                continue
            # Approximate span of this piece inside [s, e) for ref assignment.
            if len(pieces) == 1:
                ps, pe = s, e
            else:
                off = j.text.find(piece[:40], s)
                ps = off if off >= 0 else s
                pe = ps + len(piece)
            local = sorted({sid for st, sid in spans if ps <= st < pe})
            n = len(chunks)
            chunks.append(Chunk(chunk_id=f"{j.doc_id}¶{n}", doc_id=j.doc_id, text=piece, para_idx=n,
                                role=role, section_ids=local, doc_section_ids=doc_ids, date=j.date,
                                code_family=fam, title=j.title))
    return chunks


def chunk_corpus(judgments: list[Judgment], resolver: CodeResolver, size: int = 400, overlap: int = 50,
                 role_aware: bool = True) -> list[Chunk]:
    out: list[Chunk] = []
    for j in judgments:
        out += chunk_judgment(j, resolver, size, overlap, role_aware)
    return out
