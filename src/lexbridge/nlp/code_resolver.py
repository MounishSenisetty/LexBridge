"""C1 — code-aware section resolver.

Rule priority (each rule returns a confidence in [0, 1]):
  1. explicit code in the same phrase            (0.98)  "Section 302 IPC"
  2. code mentioned within ±200 chars            (0.85)
  3. the document's dominant code                (0.75)  ≥2 mentions, ≥70% share
  4. document / offence date vs the transition   (0.70)  before 2024-07-01 ⇒ IPC family
  5. offence keywords vs section headings        (0.60)  "murder"+302 ⇒ IPC; "religious feelings"+302 ⇒ BNS
  6. unresolved                                  (0.20)  code=None
"""
from __future__ import annotations

import re
from collections import Counter
from datetime import date

from ..config import get_config
from ..schemas import SUCCESSOR, SectionRef, normalize_section
from .section_ner import extract_sections, find_code_mentions

WINDOW = 200
STOP = set("the of and or to a an in for by with any shall be is as on under section sections punishment "
           "whoever etc person persons".split())

# Which code a number most plausibly belongs to when only the family is known.
PROCEDURE_CUES = re.compile(r"(?i)\b(bail|anticipatory|fir|first information|investigation|cognizance|quash|"
                            r"inherent powers?|arrest|remand|charge-?sheet|summons)\b")
EVIDENCE_CUES = re.compile(r"(?i)\b(evidence|electronic record|certificate|admissib|confession|dying declaration|witness)\b")


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP and len(w) > 2}


def _stem(w: str) -> str:
    for suf in ("ings", "ing", "ed", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


class HeadingLexicon:
    """section_id -> heading tokens, built from statutes and mapping titles."""

    def __init__(self, headings: dict[str, str] | None = None):
        self.raw = dict(headings or {})
        self.tok = {k: {_stem(w) for w in _tokens(v)} for k, v in self.raw.items()}

    @classmethod
    def build(cls, statutes=None, mappings=None) -> "HeadingLexicon":
        h: dict[str, str] = {}
        for m in (mappings or []):
            h.setdefault(m.old_id, m.title)
        for s in (statutes or []):
            h[s.id] = f"{h.get(s.id, '')} {s.title}".strip()
        # New sections inherit the mapping title only if no statute heading exists.
        for m in (mappings or []):
            if m.new_id and m.new_id not in h:
                h[m.new_id] = m.title
        return cls(h)

    def score(self, sid: str, context: str) -> float:
        toks = self.tok.get(sid)
        if not toks:
            return 0.0
        ctx = {_stem(w) for w in _tokens(context)}
        return len(toks & ctx) / len(toks)


FAMILY = {"old": ("IPC", "CrPC", "IEA"), "new": ("BNS", "BNSS", "BSA")}


def _family_member(family: str, section: str, context: str, lexicon: "HeadingLexicon") -> str:
    """family 'old'|'new' -> concrete code. Heading match first; procedure/evidence cues only right next
    to the reference ("Section 438 ... anticipatory bail"); otherwise the substantive code."""
    codes = FAMILY[family]
    scored = [(lexicon.score(f"{c}_{normalize_section(section)}", context), c) for c in codes]
    best = max(scored)
    if best[0] > 0:
        return best[1]
    if PROCEDURE_CUES.search(context) and not re.search(r"(?i)\b(registered|charged|offen[cs]e)\s+under", context):
        return codes[1]
    if EVIDENCE_CUES.search(context) and not re.search(r"(?i)murder|cheat|theft", context):
        return codes[2]
    return codes[0]


def dominant_code(text: str) -> str | None:
    counts = Counter(m.code for m in find_code_mentions(text))
    # Section refs with explicit codes count too.
    total = sum(counts.values())
    if total < 2:
        return None
    code, n = counts.most_common(1)[0]
    return code if n / total >= 0.7 else None


class CodeResolver:
    def __init__(self, lexicon: HeadingLexicon | None = None, transition: date | None = None):
        self.lexicon = lexicon or HeadingLexicon()
        self.transition = transition or get_config().transition

    def resolve_ref(self, ref: SectionRef, text: str, doc_date: date | None = None,
                    dominant: str | None = None, mentions=None, disable: set[str] | None = None) -> SectionRef:
        disable = disable or set()
        r = ref.model_copy()
        if r.code and "explicit" not in disable:
            r.confidence, r.rule = 0.98, "explicit"
            return r
        r.code = None
        mentions = mentions if mentions is not None else find_code_mentions(text)
        # 2. nearest code mention within the window
        if "window" not in disable:
            near = [(min(abs(m.start - ref.end), abs(ref.start - m.end)), m) for m in mentions
                    if m.end >= ref.start - WINDOW and m.start <= ref.end + WINDOW]
            if near:
                r.code = min(near, key=lambda x: x[0])[1].code
                r.confidence, r.rule = 0.85, "window"
                return r
        # 3. dominant code in the document
        if dominant and "dominant" not in disable:
            r.code, r.confidence, r.rule = dominant, 0.75, "dominant"
            return r
        ctx = text[max(0, ref.start - 150): ref.end + 150]
        near_ctx = text[max(0, ref.start - 40): ref.end + 40]
        # 4. date
        if doc_date and "date" not in disable:
            fam = "old" if doc_date < self.transition else "new"
            r.code, r.confidence, r.rule = _family_member(fam, ref.section, near_ctx, self.lexicon), 0.70, "date"
            return r
        # 5. keyword vs heading (compare IPC/BNS, CrPC/BNSS, IEA/BSA readings of the same number)
        if "keyword" not in disable:
            best, best_s = None, 0.0
            for old, new in SUCCESSOR.items():
                for code in (old, new):
                    s = self.lexicon.score(f"{code}_{normalize_section(ref.section)}", ctx)
                    if s > best_s:
                        best, best_s = code, s
            if best and best_s > 0:
                r.code, r.confidence, r.rule = best, round(min(0.6, 0.3 + 0.3 * best_s), 3), "keyword"
                return r
        r.confidence, r.rule = 0.2, "unresolved"
        return r

    def resolve(self, text: str, doc_date: date | None = None, disable: set[str] | None = None) -> list[SectionRef]:
        refs = extract_sections(text)
        mentions = find_code_mentions(text)
        dom = dominant_code(text)
        return [self.resolve_ref(r, text, doc_date, dom, mentions, disable) for r in refs]
