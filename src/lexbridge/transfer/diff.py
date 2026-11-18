"""Statute text diffing for C3: whole-text ratio, ingredient vs punishment clause, best sub-section."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

PUNISH = re.compile(r"(?i)\bshall be punished\b|\bshall be liable\b")


def normalize(text: str) -> str:
    t = re.sub(r"\(\d{1,2}\)\s*", "", text)          # sub-section markers
    t = re.sub(r"[^\w\s]", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()


def ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b), autojunk=False).ratio()


def split_clause(text: str) -> tuple[str, str]:
    """(ingredients, punishment) split at the first 'shall be punished'."""
    m = PUNISH.search(text)
    return (text[:m.start()], text[m.start():]) if m else (text, "")


def subsections(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\(\d{1,2}\)", text) if p.strip()]
    return parts or [text]


@dataclass
class StatuteDiff:
    ratio: float
    ingredient_ratio: float
    punishment_ratio: float
    best_sub_ratio: float
    n_new_subsections: int
    changed_elements: list[str] = field(default_factory=list)


def diff_sections(old: str, new: str) -> StatuteDiff:
    oi, op = split_clause(old)
    subs = subsections(new)
    best_sub = max((ratio(old, s) for s in subs), default=0.0)
    # Compare the old text to the best-matching new sub-section for clause-level ratios.
    best_text = max(subs, key=lambda s: ratio(old, s)) if subs else new
    ni, np_ = split_clause(best_text)
    d = StatuteDiff(ratio=ratio(old, new), ingredient_ratio=ratio(oi, ni),
                    punishment_ratio=ratio(op, np_) if (op or np_) else 1.0,
                    best_sub_ratio=best_sub, n_new_subsections=len(subs))
    if d.ingredient_ratio < 0.97:
        d.changed_elements.append("ingredients")
    if d.punishment_ratio < 0.97:
        d.changed_elements.append("punishment")
    if len(subs) > 1 and best_sub >= 0.95:
        d.changed_elements.append(f"{len(subs) - 1} additional sub-section(s)")
    return d
