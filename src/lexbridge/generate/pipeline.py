"""LexBridge online pipeline: ``LexBridge(res).answer(query, offence_date)``.

1. NER + C1 resolve query sections            5. collision filter
2. C4 temporal gate → governing code family   6. rerank → top 8
3. C2 graph hop BNS_103 → IPC_302 → judgments 7. C3 applicability tags
4. hybrid BM25 + dense (+ graph) with RRF     8. LLM answer with citations → 9. verifier
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from datetime import date

from ..config import get_config
from ..index.graph import counterparts
from ..nlp.code_resolver import _stem, _tokens
from ..resources import NON_EXPLICIT_RULES, Resources
from ..retrieve.cross_code import collision_filter, expand_ids, graph_candidates
from ..retrieve.hybrid import rrf
from ..retrieve.rerank import Reranker
from ..schemas import Applicability
from .llm import get_llm
from .prompts import ANSWER_PROMPT, ANSWER_SYSTEM, DISCLAIMER, REGENERATE_NOTE
from .temporal_gate import TemporalDecision, decide, family_of
from .verifier import verify


@dataclass
class Flags:
    c1: bool = True          # code resolver
    c2: bool = True          # cross-code graph hop
    c3: bool = True          # transferability tags
    c4: bool = True          # temporal gate
    role_aware: bool = True  # role-aware chunking
    verifier: bool = True
    collision_filter: bool = True


ABLATIONS = {
    "full": Flags(),
    "-C1 resolver": Flags(c1=False),
    "-C2 graph hop": Flags(c2=False),
    "-C3 transferability": Flags(c3=False),
    "-C4 temporal gate": Flags(c4=False),
    "-role-aware chunking": Flags(role_aware=False),
    "-verifier": Flags(verifier=False),
}


class LexBridge:
    name = "lexbridge"

    def __init__(self, res: Resources, flags: Flags | None = None, llm=None):
        self.res = res
        self.flags = flags or Flags()
        self.cfg = get_config()
        self.llm = llm or get_llm()
        self.reranker = Reranker(self.cfg)
        variant = "role" if self.flags.role_aware else "naive"
        if not self.flags.c1:
            variant = "noresolver" if self.flags.role_aware else "naive"
        self.index = res.index(variant)
        self._pw = int(self.cfg.get("llm", {}).get("prompt_chunk_words", 400))
        self._judge = self.llm if self.cfg.get("llm", {}).get("verifier_llm", True) else None

    # ------------------------------------------------------------------ step 1: C1
    def resolve_query(self, query: str, offence_date: date | None) -> list[str]:
        disable = None if self.flags.c1 else NON_EXPLICIT_RULES
        refs = self.res.resolver.resolve(query, offence_date, disable)
        ids = [r.id for r in refs if r.id and r.section != "34"]  # s.34 common intention is an auxiliary ref
        if not ids and self.flags.c1:
            ids = self.heading_lookup(query)
        return list(dict.fromkeys(ids))

    def heading_lookup(self, query: str, min_score: float = 0.5) -> list[str]:
        """'punishment for cheating' → sections whose heading words all appear in the query."""
        q = {_stem(w) for w in _tokens(query)}
        hits = []
        for sid, toks in self.res.lexicon.tok.items():
            if toks and len(toks & q) / len(toks) >= min_score and sid.split("_")[0] in ("IPC", "BNS"):
                hits.append((len(toks & q) / len(toks), sid))
        hits.sort(reverse=True)
        return [sid for s, sid in hits if s == hits[0][0]] if hits else []

    # ------------------------------------------------------------------ step 2: C4
    def temporal(self, query: str, offence_date: date | None, targets: list[str]) -> TemporalDecision:
        if self.flags.c4:
            return decide(query, offence_date, self.cfg.transition)
        fam = family_of(targets[0].split("_")[0]) if targets else None
        return TemporalDecision(fam or "both", offence_date, [], "")

    def governing_ids(self, targets: list[str], td: TemporalDecision) -> list[str]:
        """Sections that govern the answer: targets moved into the governing family where a mapping exists."""
        out = []
        for sid in targets:
            fam = family_of(sid.split("_")[0])
            if td.governing in ("both", fam) or not self.flags.c4:
                out.append(sid)
            if td.governing in ("both",) or (td.governing != fam and self.flags.c4):
                out += [o for o, _ in counterparts(self.res.graph, sid)]
        return list(dict.fromkeys(out))

    # ------------------------------------------------------------------ steps 3–6
    def retrieve(self, query: str, targets: list[str]) -> tuple[list[tuple[int, float]], dict]:
        r = self.cfg["retrieval"]
        idx = self.index
        expanded = expand_ids(self.res.graph, targets) if self.flags.c2 else list(targets)
        titles = " ".join(self.res.lexicon.raw.get(s, "") for s in expanded)
        bm = [i for i, _ in idx.bm25_code.search(query, r["bm25_k"], query_ids=expanded if self.flags.c1 else targets)]
        dn = [i for i, _ in idx.dense.search(f"{query} {titles}", r["dense_k"])]
        rankings = [bm, dn]
        gr: list[int] = []
        if self.flags.c2 and expanded:
            gr = graph_candidates(self.res.graph, expanded, idx.chunks, r["graph_k"])
            rankings.append(gr)
        fused = rrf(rankings, k=r["rrf_k"])
        order = [i for i, _ in fused]
        dropped: list[int] = []
        if self.flags.collision_filter and self.flags.c1 and expanded:
            order, dropped = collision_filter(idx.chunks, order, expanded)
        scores = dict(fused)
        top = self.reranker.rerank(query, [(i, scores[i]) for i in order], [c.text for c in idx.chunks], r["rerank_top"])
        debug = {"expanded_ids": expanded, "bm25": bm[:10], "dense": dn[:10], "graph": gr[:10],
                 "dropped_collisions": [idx.chunks[i].chunk_id for i in dropped]}
        return top, debug

    # ------------------------------------------------------------------ step 7: C3
    def tag_chunk(self, chunk, targets: list[str], governing: list[str], td: TemporalDecision) -> tuple[str, str | None]:
        if not self.flags.c3:
            return Applicability.DIRECTLY_APPLICABLE.value, None
        ids = set(chunk.section_ids) or set(chunk.doc_section_ids)
        relevant = [s for s in ids if s in set(expand_ids(self.res.graph, targets))]
        if not relevant:
            return "CONTEXT", None
        gov_fams = {family_of(g.split("_")[0]) for g in governing} or {"old", "new"}
        best = None
        for sid in relevant:
            fam = family_of(sid.split("_")[0])
            if fam in gov_fams and (td.governing != "both" or fam == "new"):
                return Applicability.DIRECTLY_APPLICABLE.value, sid
            t = self.res.transfer.get(sid) or next(
                (self.res.transfer.get(o) for o, _ in counterparts(self.res.graph, sid) if self.res.transfer.get(o)), None)
            if t:
                best = (t.tag, sid)
        return best or (Applicability.REVIEW_REQUIRED.value, relevant[0])

    # ------------------------------------------------------------------ main
    def answer(self, query: str, offence_date: date | str | None = None) -> dict:
        if isinstance(offence_date, str):
            offence_date = date.fromisoformat(offence_date) if offence_date else None
        targets = self.resolve_query(query, offence_date if self.flags.c4 else None)
        td = self.temporal(query, offence_date, targets)
        if self.flags.c4 and td.offence_date and not offence_date:  # date in the question text → re-resolve
            targets = self.resolve_query(query, td.offence_date) or targets
        governing = self.governing_ids(targets, td)
        top, debug = self.retrieve(query, targets)
        idx = self.index

        citations, ctx_chunks = [], []
        for i, score in top:
            c = idx.chunks[i]
            tag, via = self.tag_chunk(c, targets, governing, td)
            citations.append({"id": c.chunk_id, "doc_id": c.doc_id, "title": c.title,
                              "date": c.date.isoformat() if c.date else None, "role": c.role, "text": c.text,
                              "section_ids": c.section_ids or c.doc_section_ids, "tag": tag, "via": via,
                              "score": round(score, 4)})
            ctx_chunks.append({"chunk_id": c.chunk_id, "text": c.text, "tag": tag if self.flags.c3 else None})

        statutes = [self.res.store.by_id[s] for s in governing if s in self.res.store.by_id]
        applicability = []
        seen = set()
        for sid in targets + governing:
            for m in self.res.table.lookup(sid):
                if m.old_id in seen:
                    continue
                seen.add(m.old_id)
                t = self.res.transfer.get(m.old_id)
                applicability.append({"old": m.old_id, "new": m.new_id, "title": m.title,
                                      "change_type": t.label if t else "unknown",
                                      "tag": t.tag if (t and self.flags.c3) else None,
                                      "rationale": t.rationale if t else "", "verified_mapping": m.verified})

        keywords = [self.res.lexicon.raw.get(s, "") for s in governing]
        context = {"query": query, "statutes": [s.model_dump() for s in statutes], "chunks": ctx_chunks,
                   "keywords": keywords}
        prompt = ANSWER_PROMPT.format(
            query=query, governing=td.note or td.governing,
            statutes="\n".join(f"[{s.code} §{s.section}] {s.title}: {_trim(s.text, 3 * self._pw)}" for s in statutes) or "(none)",
            chunks="\n".join(f"[{c['chunk_id']}] ({c['tag']}) {_trim(c['text'], self._pw)}" for c in ctx_chunks) or "(none)")
        body = self.llm.complete(prompt, system=ANSWER_SYSTEM, task="answer", context=context)

        evidence = {c["id"]: c["text"] for c in citations}
        evidence.update({f"{s.code} §{s.section}": f"{s.title} {s.text}" for s in statutes})
        checks, faith = verify(body, evidence, self._judge)  # always measured; only acted on with the verifier
        if self.flags.verifier:
            bad = [c.claim for c in checks if not c.supported]
            if bad:  # regenerate once, then drop anything still unsupported
                body = self.llm.complete(prompt + REGENERATE_NOTE.format(claims="; ".join(bad)),
                                         system=ANSWER_SYSTEM, task="answer", context=context)
                checks, faith = verify(body, evidence, self._judge)
                bad = {c.claim for c in checks if not c.supported}
                if bad:
                    body = "\n".join(l for l in body.splitlines() if l.strip() not in bad)
                    checks, faith = verify(body, evidence, self._judge)

        answer = "\n".join(filter(None, [self._header(targets, governing, td), body, self._dual(governing), DISCLAIMER]))
        cited = [c for c in citations if f"[{c['id']}]" in answer]
        return {"answer": answer, "citations": citations, "cited_ids": [c["id"] for c in cited],
                "applicability": applicability,
                "governing_code": {**td.to_dict(), "sections": governing},
                "resolved_sections": targets, "faithfulness_score": faith,
                "debug": {**debug, "claims": [asdict(c) for c in checks], "llm": getattr(self.llm, "name", "?")}}

    def _header(self, targets, governing, td: TemporalDecision) -> str:
        if not targets:
            return "Could not identify a specific section in the question." + (f" {td.note}" if td.note else "")
        names = ", ".join(f"{s.replace('_', ' §')} ({self.title(s)})" for s in governing)
        return f"Governing provision(s): {names}. {td.note}".strip()

    def title(self, sid: str) -> str:
        st = self.res.store.by_id.get(sid)
        if st and st.title:
            return st.title
        return self.res.lexicon.raw.get(sid, "").strip() or "untitled"

    def _dual(self, governing: list[str]) -> str:
        pairs = []
        for sid in governing:
            for m in self.res.table.lookup(sid):
                pair = f"{m.new_id.replace('_', ' §') if m.new_id else '(omitted in new code)'} ⇄ {m.old_id.replace('_', ' §')}"
                if pair not in pairs:
                    pairs.append(pair)
        return ("Dual citation: " + "; ".join(pairs)) if pairs else ""


def _trim(text: str, words: int) -> str:
    w = text.split()
    return text if len(w) <= words else " ".join(w[:words]) + " …"


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json

    from ..resources import load_resources
    ap = argparse.ArgumentParser(description="Ask LexBridge a question.")
    ap.add_argument("query")
    ap.add_argument("--date", default=None, help="offence date YYYY-MM-DD")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    out = LexBridge(load_resources()).answer(a.query, a.date)
    if a.json:
        print(json.dumps(out, indent=2, default=str))
        return
    print(out["answer"])
    print("\nCitations:")
    for c in out["citations"]:
        print(f"  [{c['id']}] {c['tag']:<24} {c['title']} ({c['date']}) sections={c['section_ids']}")
    print(f"\nfaithfulness={out['faithfulness_score']}  dropped_collisions={out['debug']['dropped_collisions']}")


if __name__ == "__main__":
    main(sys.argv[1:])
