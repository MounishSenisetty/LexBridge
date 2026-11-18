"""Date extraction and offence-date detection (used by C4 and the C1 date rule)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], 1)}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})
MON = r"(?P<mon>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"

PATTERNS = [
    ("dmy_text", re.compile(rf"(?i)\b(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?{MON}\.?,?\s+(?P<y>(?:19|20)\d\d)\b")),
    ("mdy_text", re.compile(rf"(?i)\b{MON}\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?,?\s+(?P<y>(?:19|20)\d\d)\b")),
    ("iso", re.compile(r"\b(?P<y>(?:19|20)\d\d)-(?P<m>\d{1,2})-(?P<d>\d{1,2})\b")),
    ("dmy_num", re.compile(r"\b(?P<d>\d{1,2})[./-](?P<m>\d{1,2})[./-](?P<y>(?:19|20)\d\d)\b")),
    ("my_text", re.compile(rf"(?i)\b{MON}\.?,?\s+(?P<y>(?:19|20)\d\d)\b")),
    ("year", re.compile(r"(?i)\b(?:in|during|year)\s+(?P<y>(?:19|20)\d\d)\b")),
]
OFFENCE_CUES = re.compile(
    r"(?i)(offen[cs]e|committed|incident|occurr|happened|took place|alleged|crime|on the night|"
    r"killed|murdered|assault|stole|cheated|fir\b|arrest)")


@dataclass
class ExtractedDate:
    value: date
    raw: str
    start: int
    end: int
    kind: str
    precision: str = "day"  # day | month | year


def extract_dates(text: str) -> list[ExtractedDate]:
    out: list[ExtractedDate] = []
    taken: list[tuple[int, int]] = []
    for kind, rx in PATTERNS:
        for m in rx.finditer(text):
            if any(a <= m.start() < b or a < m.end() <= b for a, b in taken):
                continue
            g = m.groupdict()
            try:
                y = int(g["y"])
                mo = MONTHS[g["mon"].lower()[:3]] if g.get("mon") else int(g.get("m") or 1)
                d = int(g.get("d") or 1)
                val = date(y, mo, d)
            except (ValueError, KeyError):
                continue
            prec = "day" if g.get("d") else ("month" if (g.get("mon") or g.get("m")) else "year")
            if prec == "year" and y == 2024:
                continue  # the transition year is ambiguous without a month
            out.append(ExtractedDate(val, m.group(0), m.start(), m.end(), kind, prec))
            taken.append((m.start(), m.end()))
    out.sort(key=lambda x: x.start)
    return out


def extract_offence_date(text: str) -> date | None:
    """Prefer a date with an offence cue in the preceding 80 chars; else the only/first date."""
    ds = extract_dates(text)
    if not ds:
        return None
    for d in ds:
        if OFFENCE_CUES.search(text[max(0, d.start - 80):d.start]):
            return d.value
    return ds[0].value
