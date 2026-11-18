"""No-RAG baseline: the LLM answers from parametric memory. With the mock LLM this is simulated as
memory dominated by IPC-era text (section numbers read as IPC) — a stand-in for the documented
behaviour of models trained mostly on pre-2024 material. Replace with a real LLM for real numbers."""
from __future__ import annotations

from datetime import date

from ..generate.llm import get_llm
from ..generate.prompts import DISCLAIMER, NO_RAG_PROMPT
from ..nlp.section_ner import extract_sections
from ..resources import Resources


class NoRAG:
    name = "no_rag"

    def __init__(self, res: Resources, llm=None):
        self.res, self.llm = res, llm or get_llm()

    def answer(self, query: str, offence_date: date | str | None = None) -> dict:
        secs = [r.section for r in extract_sections(query) if r.section != "34"]
        ids = [f"IPC_{s}" for s in dict.fromkeys(secs)]
        parametric = [(s.split("_")[1], self.res.lexicon.raw.get(s, "")) for s in ids]
        body = self.llm.complete(NO_RAG_PROMPT.format(query=query), task="parametric", context={"parametric": parametric})
        return {"answer": f"{body}\n{DISCLAIMER}", "citations": [], "cited_ids": [],
                "applicability": [{"old": s, "new": None, "tag": "DIRECTLY_APPLICABLE"} for s in ids],
                "governing_code": {"governing": "old" if ids else "both", "sections": ids},
                "resolved_sections": ids, "faithfulness_score": None, "debug": {"system": self.name}}
