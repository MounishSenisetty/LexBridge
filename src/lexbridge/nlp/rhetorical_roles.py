"""Rule-based rhetorical role tagger for judgment paragraphs.

Roles: preamble, facts, arguments, analysis, statute, decision, other.
Good enough for role-aware chunking on CPU; swap in a trained tagger for the research version.
"""
from __future__ import annotations

import re

RULES: list[tuple[str, re.Pattern]] = [
    ("decision", re.compile(r"(?i)^\W*(held|order|result|conclusion)\s*[:.—-]|\b(appeal is (dismissed|allowed)|is (convicted|acquitted)|"
                            r"conviction .* (upheld|set aside)|sentenced to|is quashed|are quashed|bail is (granted|rejected)|charge .* is framed)\b")),
    ("facts", re.compile(r"(?i)^\W*facts?\s*[:.—-]|\b(prosecution case|on the night|the incident|fir was|it is alleged|allegedly|was recovered)\b")),
    ("arguments", re.compile(r"(?i)\b(counsel|argued|submitted|contended|submission|the defence)\b")),
    ("analysis", re.compile(r"(?i)^\W*(analysis|discussion|reasoning)\s*[:.—-]|\b(we are of the (view|opinion)|in our (view|opinion)|"
                            r"ingredients?|it is settled|the question is|presumption|can be inferred|must be proved|precedents?)\b")),
    ("statute", re.compile(r"(?i)\b(shall be punished|reads as follows|provides that)\b")),
    ("preamble", re.compile(r"(?i)\b(is charged|are charged|challenge|seeks?|petition|appellants?|the applicant)\b")),
]
ROLES = ["preamble", "facts", "arguments", "analysis", "statute", "decision", "other"]


def tag_paragraph(text: str) -> str:
    for role, rx in RULES:
        if rx.search(text):
            return role
    return "other"


def tag_paragraphs(paras: list[str]) -> list[str]:
    return [tag_paragraph(p) for p in paras]
