"""Section-reference NER (regex + rules).

Handles ``Section 302``, ``S. 302``, ``Sec. 302 IPC``, ``u/s 302``, ``302/34 IPC``,
``Sections 302 and 307``, ``498-A``, ``304B``, ``Section 103(1) of BNS`` and full code names
such as "Bharatiya Nyaya Sanhita". Attaches a code only when it appears in the same phrase;
everything else is left for the code resolver (C1).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..schemas import SectionRef, normalize_section

# Longest / most specific first. Matching is case-insensitive except bare acronyms.
CODE_NAMES: list[tuple[str, str]] = [
    (r"Bharatiya\s+Nagarik\s+Suraksha\s+Sanhita(?:,?\s*2023)?", "BNSS"),
    (r"Bharatiya\s+Nyaya\s+Sanhita(?:,?\s*2023)?", "BNS"),
    (r"Bharatiya\s+Sakshya\s+Adhiniyam(?:,?\s*2023)?", "BSA"),
    (r"Indian\s+Penal\s+Code(?:,?\s*1860)?", "IPC"),
    (r"Penal\s+Code", "IPC"),
    (r"Code\s+of\s+Criminal\s+Procedure(?:,?\s*1973)?", "CrPC"),
    (r"(?:Indian\s+)?Evidence\s+Act(?:,?\s*1872)?", "IEA"),
]
ACRONYMS: list[tuple[str, str]] = [
    (r"B\.?N\.?S\.?S\.?", "BNSS"),
    (r"B\.?N\.?S\.?", "BNS"),
    (r"B\.?S\.?A\.?", "BSA"),
    (r"I\.?P\.?C\.?", "IPC"),
    (r"Cr\.?\s?P\.?\s?C\.?", "CrPC"),
    (r"I\.?E\.?A\.?", "IEA"),
]
_code_alt = "|".join(f"(?P<n{i}>{p})" for i, (p, _) in enumerate(CODE_NAMES))
_acr_alt = "|".join(f"(?P<a{i}>{p})" for i, (p, _) in enumerate(ACRONYMS))
CODE_RE = re.compile(rf"(?i:{_code_alt})|(?<![A-Za-z])(?:{_acr_alt})(?![A-Za-z])")

NUM = r"\d{1,3}(?:\s?-\s?[A-Za-z]{1,2}(?![A-Za-z])|[A-Za-z]{1,2}(?![A-Za-z]))?(?:\s?\(\d{1,2}\))*(?:\s?\([a-z]\))?"
SEP = r"(?:\s*/\s*|\s*,\s*|\s+and\s+|\s*&\s*|\s+r/w\s+|\s+read\s+with\s+|\s+or\s+)"
LIST = rf"{NUM}(?:{SEP}{NUM})*"
TRIGGER = r"(?:\bsections?|\bsecs?\.?|\bss?\.|\bu/ss?\.?|\bunder\s+s\.)"
TRIGGER_RE = re.compile(rf"(?i){TRIGGER}\s*(?P<list>{LIST})")
# Bare "302 IPC", "302/34 IPC", "498-A I.P.C." (no trigger word).
BARE_RE = re.compile(rf"(?<![\w/(])(?P<list>{LIST})\s+(?:of\s+(?:the\s+)?)?(?=(?:{_acr_alt}|(?i:{_code_alt})))")
NUM_RE = re.compile(r"(\d{1,3})(?:\s?-\s?([A-Za-z]{1,2})|([A-Za-z]{1,2}))?(?![A-Za-z])")
# After a list: " IPC", " of the IPC", " of Bharatiya Nyaya Sanhita", ", IPC"
FOLLOW_RE = re.compile(rf"^\s*,?\s*(?:of\s+)?(?:the\s+)?(?:(?i:{_code_alt})|(?<![A-Za-z])(?:{_acr_alt})(?![A-Za-z]))")


@dataclass
class CodeMention:
    code: str
    start: int
    end: int
    raw: str


def _code_from_match(m: re.Match) -> str:
    for k, v in m.groupdict().items():
        if v is None:
            continue
        idx = int(k[1:])
        return (CODE_NAMES if k[0] == "n" else ACRONYMS)[idx][1]
    raise ValueError("no code group matched")  # pragma: no cover


def find_code_mentions(text: str) -> list[CodeMention]:
    return [CodeMention(_code_from_match(m), m.start(), m.end(), m.group(0)) for m in CODE_RE.finditer(text)]


def _split_list(s: str) -> list[str]:
    s = re.sub(r"\(\s*[0-9a-z]{1,2}\s*\)", "", s)  # drop sub-sections: 103(1) -> 103
    return [normalize_section(m.group(1) + (m.group(2) or m.group(3) or "")) for m in NUM_RE.finditer(s)]


def _explicit_code_after(text: str, end: int) -> tuple[str | None, int]:
    m = FOLLOW_RE.match(text[end:end + 60])
    if not m:
        return None, end
    cm = CODE_RE.search(m.group(0))
    return (_code_from_match(cm) if cm else None), end + m.end()


def extract_sections(text: str) -> list[SectionRef]:
    refs: list[SectionRef] = []
    covered: list[tuple[int, int]] = []
    for rx in (TRIGGER_RE, BARE_RE):
        for m in rx.finditer(text):
            s, e = m.start("list"), m.end("list")
            if any(a <= s < b for a, b in covered):
                continue
            code, phrase_end = _explicit_code_after(text, e)
            covered.append((m.start(), phrase_end))
            for sec in _split_list(m.group("list")):
                refs.append(SectionRef(section=sec, code=code, raw=text[m.start():phrase_end],
                                       start=m.start(), end=phrase_end,
                                       confidence=0.98 if code else 0.0,
                                       rule="explicit" if code else ""))
    refs.sort(key=lambda r: r.start)
    return refs
