"""Claim verifier: each answer line is a claim; it must be supported by the chunk / statute it cites."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .prompts import VERIFY_PROMPT

CITE = re.compile(r"\[([^\[\]]+?)\]")
STOP = set("the a an of and or to in for by with is was be on as at that this it from are were has have had "
           "which shall any under section sections".split())


def _content(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+|\d+", s.lower()) if w not in STOP and len(w) > 1]


def lexical_support(claim: str, evidence: str) -> float:
    c = _content(CITE.sub("", claim))
    if not c:
        return 1.0
    ev = set(_content(evidence))
    return sum(w in ev for w in c) / len(c)


@dataclass
class ClaimCheck:
    claim: str
    citations: list[str]
    supported: bool
    support: float


def split_claims(answer: str) -> list[str]:
    return [l.strip() for l in answer.splitlines() if l.strip() and CITE.search(l)]


def verify(answer: str, evidence: dict[str, str], llm=None, threshold: float = 0.5) -> tuple[list[ClaimCheck], float]:
    checks = []
    for claim in split_claims(answer):
        cites = [c.strip() for c in CITE.findall(claim)]
        ev = " ".join(evidence.get(c, "") for c in cites)
        s = lexical_support(claim, ev) if ev else 0.0
        ok = s >= threshold
        if llm is not None and ev and getattr(llm, "name", "") != "mock":  # pragma: no cover - real LLM judge
            out = llm.complete_json(VERIFY_PROMPT.format(claim=claim, evidence=ev[:2000]), task="verify",
                                    context={"claim": claim, "evidence": ev})
            ok = bool(out.get("supported", ok))
        checks.append(ClaimCheck(claim, cites, ok, round(s, 3)))
    score = sum(c.supported for c in checks) / len(checks) if checks else 0.0
    return checks, round(score, 3)
