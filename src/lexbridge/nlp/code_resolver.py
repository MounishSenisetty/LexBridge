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
        self.codes: dict[str, set[str]] = {}
        for k in self.raw:
            code, sec = k.split("_", 1)
            self.codes.setdefault(code, set()).add(sec)

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


def _exists(lexicon: "HeadingLexicon", code: str, section: str) -> bool | None:
    """Does ``code`` have this section? None when the lexicon knows nothing about the code (no evidence)."""
    known = lexicon.codes.get(code)
    if not known or len(known) < 100:  # partial lexicons (fixtures) are no evidence of absence
        return None
    return normalize_section(section) in known


class NumberPrior:
    """P(code | family, section number) counted from EXPLICIT citations ("Section 313 Cr.P.C.") in a corpus.
    Free supervision: no annotation, and no dependence on the inference rules."""

    def __init__(self, counts: dict | None = None):
        self.counts: dict[tuple[str, str], Counter] = counts or {}

    @classmethod
    def from_judgments(cls, judgments, transition) -> "NumberPrior":
        counts: dict[tuple[str, str], Counter] = {}
        for j in judgments:
            for r in extract_sections(j.text):
                if r.code:
                    fam = "old" if r.code in FAMILY["old"] else "new"
                    counts.setdefault((fam, normalize_section(r.section)), Counter())[r.code] += 1
        return cls(counts)

    def best(self, family: str, section: str, codes, min_count: int = 3) -> str | None:
        c = self.counts.get((family, normalize_section(section)))
        if not c:
            return None
        cand = [(c[k], k) for k in codes if c[k]]
        if not cand or sum(n for n, _ in cand) < min_count:
            return None
        return max(cand)[1]


def _family_member(family: str, section: str, context: str, lexicon: "HeadingLexicon",
                   prior: "NumberPrior | None" = None) -> str:
    """family 'old'|'new' -> concrete code. Codes that lack the section number are excluded; then heading
    match; then the corpus prior; then procedure/evidence cues; otherwise the substantive code."""
    codes = [c for c in FAMILY[family] if _exists(lexicon, c, section) is not False] or list(FAMILY[family])
    scored = [(lexicon.score(f"{c}_{normalize_section(section)}", context), c) for c in codes]
    best = max(scored)
    if best[0] > 0:
        return best[1]
    if prior is not None:
        p = prior.best(family, section, codes)
        if p:
            return p
    proc, evid = FAMILY[family][1], FAMILY[family][2]
    if proc in codes and PROCEDURE_CUES.search(context) and not re.search(r"(?i)\b(registered|charged|offen[cs]e)\s+under", context):
        return proc
    if evid in codes and EVIDENCE_CUES.search(context) and not re.search(r"(?i)murder|cheat|theft", context):
        return evid
    return codes[0]


def dominant_code(text: str, mentions=None) -> str | None:
    counts = Counter(m.code for m in (mentions if mentions is not None else find_code_mentions(text)))
    total = sum(counts.values())
    if total < 2:
        return None
    code, n = counts.most_common(1)[0]
    return code if n / total >= 0.7 else None


class Context:
    """Per-document evidence shared by all references: code mentions, which of them are attached to another
    reference ("Section 313 Cr.P.C." → that Cr.P.C. belongs to 313), and the dominant code."""

    def __init__(self, text: str):
        self.mentions = find_code_mentions(text)
        refs = extract_sections(text)
        self.attached = {m.start for m in self.mentions
                         if any(r.code and r.start <= m.start and m.end <= r.end for r in refs)}
        self.dominant = dominant_code(text, self.mentions)


class CodeResolver:
    """version "v2" (default): free mentions → date → dominant → attached mentions → keyword.
    version "v1": the original order (any mention in window → dominant → date → keyword), kept for comparison."""

    def __init__(self, lexicon: HeadingLexicon | None = None, transition: date | None = None, version: str = "v2",
                 prior: NumberPrior | None = None):
        self.lexicon = lexicon or HeadingLexicon()
        self.transition = transition or get_config().transition
        self.version = version
        self.prior = prior

    def _near(self, ref: SectionRef, mentions, keep=lambda m: True):
        sec = ref.section
        out = []
        for m in mentions:
            if m.end >= ref.start - WINDOW and m.start <= ref.end + WINDOW and keep(m):
                if self.version == "v2" and _exists(self.lexicon, m.code, sec) is False:
                    continue  # "Section 438" cannot be IPC; "Section 520" cannot be BNS
                out.append((min(abs(m.start - ref.end), abs(ref.start - m.end)), m))
        return min(out, key=lambda x: x[0])[1] if out else None

    def resolve_ref(self, ref: SectionRef, text: str, doc_date: date | None = None, ctx: Context | None = None,
                    disable: set[str] | None = None) -> SectionRef:
        disable = disable or set()
        r = ref.model_copy()
        if r.code and "explicit" not in disable:
            r.confidence, r.rule = 0.98, "explicit"
            return r
        r.code = None
        ctx = ctx or Context(text)
        wide = text[max(0, ref.start - 150): ref.end + 150]
        near_ctx = text[max(0, ref.start - 40): ref.end + 40]
        fam = None if not doc_date else ("old" if doc_date < self.transition else "new")

        def done(code, conf, rule):
            r.code, r.confidence, r.rule = code, conf, rule
            return r

        if self.version == "v1":
            m = self._near(ref, ctx.mentions) if "window" not in disable else None
            if m:
                return done(m.code, 0.85, "window")
            if ctx.dominant and "dominant" not in disable:
                return done(ctx.dominant, 0.75, "dominant")
            if fam and "date" not in disable:
                return done(_family_member(fam, ref.section, near_ctx, self.lexicon), 0.70, "date")
        else:
            # 2. a free code mention nearby ("... under the IPC, Section 302 ...")
            m = self._near(ref, ctx.mentions, lambda m: m.start not in ctx.attached) if "window" not in disable else None
            if m:
                return done(m.code, 0.85, "window")
            # 3. the date fixes the family; the member comes from section existence, headings and cues
            if fam and "date" not in disable:
                return done(_family_member(fam, ref.section, near_ctx, self.lexicon, self.prior), 0.75, "date")
            # 4. dominant code
            if ctx.dominant and "dominant" not in disable and _exists(self.lexicon, ctx.dominant, ref.section) is not False:
                return done(ctx.dominant, 0.65, "dominant")
            # 5. a code attached to a neighbouring reference
            m = self._near(ref, ctx.mentions) if "window" not in disable else None
            if m:
                return done(m.code, 0.55, "attached")
        # 6. keyword vs heading (compare IPC/BNS, CrPC/BNSS, IEA/BSA readings of the same number)
        if "keyword" not in disable:
            best, best_s = None, 0.0
            for old, new in SUCCESSOR.items():
                for code in (old, new):
                    sc = self.lexicon.score(f"{code}_{normalize_section(ref.section)}", wide)
                    if sc > best_s:
                        best, best_s = code, sc
            if best and best_s > 0:
                return done(best, round(min(0.6, 0.3 + 0.3 * best_s), 3), "keyword")
        return done(None, 0.2, "unresolved")

    def resolve(self, text: str, doc_date: date | None = None, disable: set[str] | None = None) -> list[SectionRef]:
        ctx = Context(text)
        return [self.resolve_ref(r, text, doc_date, ctx, disable) for r in extract_sections(text)]
