"""Automatic consistency check of IPC→BNS mappings against official India Code section text.

This is NOT legal verification. It flags rows a human should look at first:
  * heading_old / heading_new: mapping title vs the official heading of each side
  * text_sim: how much of the old section's wording survives in the mapped new section (or sub-section)
  * best_alt: for omitted rows and low-similarity rows, the BNS section whose text best matches the IPC text
A row is ``consistent`` when the official texts support the mapping; otherwise ``suspect``.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from ..schemas import SectionMapping
from ..transfer.diff import normalize, subsections
from .statutes import StatuteStore

STOP = set("the of and or to a an in for by with any shall be is as on under section whoever such that which "
           "who person not".split())


def _toks(t: str) -> list[str]:
    return [w for w in normalize(t).split() if w not in STOP and len(w) > 2]


def containment(old: str, new: str) -> float:
    """Share of the old text's content-word bigrams that also occur in the new text."""
    a, b = _toks(old), _toks(new)
    ga = set(zip(a, a[1:]))
    gb = set(zip(b, b[1:]))
    return len(ga & gb) / len(ga) if ga else 0.0


def heading_sim(a: str, b: str) -> float:
    ta, tb = set(_toks(a)), set(_toks(b))
    return len(ta & tb) / min(len(ta), len(tb)) if ta and tb else 0.0


def _strip_illustrations(t: str) -> str:
    """Drop illustration blocks but keep any later sub-sections (BNS 303(2) follows 303(1)'s illustrations)."""
    return re.sub(r"(?is)\billustrations?\b.*?(?=\(\d{1,2}\)\s*[A-Z\u201c\"]|$)", " ", t)


def _subclauses(text: str) -> dict[str, str]:
    parts = re.split(r"\((\d{1,2})\)\s*", text)
    return {f"({parts[i]})": parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def check(mappings: list[SectionMapping], store: StatuteStore) -> list[dict]:
    bns = [s for s in store.by_id.values() if s.code == "BNS"]
    bns_core = {s.id: _strip_illustrations(s.text) for s in bns}
    rows = []
    for m in mappings:
        old = store.get(m.old_code, m.old_section)
        new = store.get(m.new_code, m.new_section) if m.new_id else None
        row = {"old": m.old_id, "new": m.new_id or "", "sub": m.new_subsection, "title": m.title,
               "sources": ",".join(m.sources), "old_official": bool(old and old.verified),
               "new_official": bool(new and new.verified), "heading_old": "", "heading_new": "",
               "text_sim": "", "best_alt": "", "best_alt_sim": "", "status": "unchecked", "reason": ""}
        if not old or not old.verified:
            row["reason"] = "no official text for the old section"
            rows.append(row)
            m.auto_check = "unchecked"
            continue
        old_core = _strip_illustrations(old.text)
        omitted_old = bool(re.search(r"(?i)omitted|repealed", f"{m.title} {old.title}"))
        row["heading_old"] = "" if (not m.title or omitted_old) else round(heading_sim(m.title, old.title), 2)

        def best_alternative():
            sc = sorted(((containment(old_core, t), sid) for sid, t in bns_core.items()), reverse=True)[:1]
            return sc[0] if sc else (0.0, "")

        if m.new_id is None:
            sim, alt = best_alternative()
            row.update(best_alt=alt, best_alt_sim=round(sim, 2))
            ok = sim < 0.6
            row["status"] = "consistent" if ok else "suspect"
            row["reason"] = "no BNS section reproduces the text" if ok else f"{alt} reproduces {sim:.0%} of the IPC text"
        elif not new or not new.verified:
            row["status"], row["reason"] = "unchecked", "no official text for the new section"
        else:
            bns_title = m.notes.removeprefix("BNS heading: ") if m.notes.startswith("BNS heading:") else ""
            new_text = _strip_illustrations(new.text)
            subs = _subclauses(new_text)
            first_sub = re.match(r"\(\d{1,2}\)", m.new_subsection or "")
            target = subs.get(first_sub.group(0)) if first_sub else None
            if target is not None:  # e.g. IPC 8 → BNS 2(10): compare with that clause only
                sim = containment(old_core, target)
                head_new = max(heading_sim(bns_title, new.title), heading_sim(bns_title, target[:160]),
                               heading_sim(m.title, target[:160])) if bns_title else ""
            else:
                sim = max([containment(old_core, new_text)] + [containment(old_core, t) for t in subs.values()])
                head_new = heading_sim(bns_title, new.title) if bns_title else ""
            row["text_sim"] = round(sim, 2)
            row["heading_new"] = round(head_new, 2) if head_new != "" else ""
            head_ok = all(h == "" or h >= 0.5 for h in (row["heading_old"], row["heading_new"]))
            if sim >= 0.35 and head_ok:
                row["status"], row["reason"] = "consistent", "official texts overlap and headings agree"
            else:
                asim, alt = best_alternative()
                row.update(best_alt=alt, best_alt_sim=round(asim, 2))
                why = []
                if sim < 0.35:
                    why.append(f"only {sim:.0%} of the IPC wording appears in {m.new_id}{m.new_subsection}")
                if not head_ok:
                    why.append("heading mismatch")
                better = bool(alt and alt != m.new_id and asim > max(0.5, sim + 0.2))
                if better:
                    why.append(f"{alt} matches better ({asim:.0%})")
                if better and sim >= 0.2 and head_ok:
                    # Part of the old text moved elsewhere (498A → 85 + definition in 86): a split, not an error.
                    row["status"] = "consistent"
                    why.append("likely split")
                    m.notes = (m.notes + f"; split: part of the text is in {alt}").strip("; ")
                else:
                    # Low overlap alone is expected for rewritten offences (124A → 152): suspect only when
                    # a clearly better candidate exists or the headings disagree.
                    row["status"] = "suspect" if (better or not head_ok) else "consistent"
                row["reason"] = "; ".join(why) or "rewritten provision; no better candidate"
        m.auto_check = row["status"]
        rows.append(row)
    return rows


def write_report(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["old"])
        w.writeheader()
        w.writerows(rows)
