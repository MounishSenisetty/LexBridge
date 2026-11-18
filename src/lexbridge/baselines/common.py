"""Shared logic for code-unaware RAG baselines.

Baselines see section numbers but not codes. What offence they end up talking about is whatever the
top retrieved chunk is about; that is how a collision error shows up in their output."""
from __future__ import annotations

from datetime import date

from ..config import get_config
from ..generate.llm import get_llm
from ..retrieve.hybrid import diversify
from ..generate.prompts import BASELINE_PROMPT, BASELINE_SYSTEM, DISCLAIMER
from ..generate.temporal_gate import family_of
from ..generate.verifier import verify
from ..nlp.section_ner import extract_sections
from ..resources import Resources


class RAGBaseline:
    name = "rag"

    def __init__(self, res: Resources, llm=None, k: int = 8):
        self.res, self.llm, self.k = res, llm or get_llm(), k
        self.index = res.index("naive")

    def search(self, query: str) -> list[tuple[int, float]]:
        raise NotImplementedError

    def predicted_sections(self, query: str, top: list[tuple[int, float]]) -> list[str]:
        """Section numbers in the query, given whichever code the top chunk uses for that number.
        An explicit code in the query is kept (the baselines can read), else default to IPC."""
        out = []
        for ref in extract_sections(query):
            if ref.section == "34":
                continue
            if ref.code:
                out.append(f"{ref.code}_{ref.section}")
                continue
            hit = None
            for i, _ in top[:1]:
                c = self.index.chunks[i]
                hit = next((s for s in (c.section_ids or c.doc_section_ids) if s.split("_")[1] == ref.section), None)
            out.append(hit or f"IPC_{ref.section}")
        return list(dict.fromkeys(out))

    def answer(self, query: str, offence_date: date | str | None = None) -> dict:
        top = diversify(self.search(query), lambda i: self.index.chunks[i].doc_id,
                        get_config()["retrieval"].get("per_doc", 2))[: self.k]
        chunks = self.index.chunks
        cits = [{"id": chunks[i].chunk_id, "doc_id": chunks[i].doc_id, "title": chunks[i].title,
                 "date": chunks[i].date.isoformat() if chunks[i].date else None, "role": chunks[i].role,
                 "text": chunks[i].text, "section_ids": chunks[i].section_ids or chunks[i].doc_section_ids,
                 "tag": None, "score": round(s, 4)} for i, s in top]
        ctx = {"query": query, "statutes": [], "chunks": [{"chunk_id": c["id"], "text": c["text"]} for c in cits]}
        pw = int(get_config().get("llm", {}).get("prompt_chunk_words", 400))
        prompt = BASELINE_PROMPT.format(query=query, chunks="\n".join(
            f"[{c['id']}] {' '.join(c['text'].split()[:pw])}" for c in cits))
        body = self.llm.complete(prompt, system=BASELINE_SYSTEM, task="answer", context=ctx)
        _, faith = verify(body, {c["id"]: c["text"] for c in cits})
        secs = self.predicted_sections(query, top)
        fam = family_of(secs[0].split("_")[0]) if secs else None
        # Code-unaware baselines assume any precedent they retrieve applies as-is.
        applicability = [{"old": s, "new": None, "tag": "DIRECTLY_APPLICABLE"} for s in secs]
        return {"answer": f"{body}\n{DISCLAIMER}", "citations": cits,
                "cited_ids": [c["id"] for c in cits if f"[{c['id']}]" in body],
                "applicability": applicability, "governing_code": {"governing": fam or "both", "sections": secs},
                "resolved_sections": secs, "faithfulness_score": faith, "debug": {"system": self.name}}
