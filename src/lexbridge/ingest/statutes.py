"""Statute parser: bare-act text (or PDF) -> StatuteSection records.

Bare acts on India Code print sections as ``302. Punishment for murder.—Whoever ...``.
The parser splits on that heading pattern. PDFs are converted with pypdf if installed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from loguru import logger

from ..schemas import StatuteSection

CODE_FROM_FILENAME = {
    "ipc": "IPC", "indian_penal": "IPC", "crpc": "CrPC", "criminal_procedure": "CrPC",
    "iea": "IEA", "evidence": "IEA", "bns": "BNS", "nyaya": "BNS", "bnss": "BNSS",
    "nagarik": "BNSS", "bsa": "BSA", "sakshya": "BSA",
}

# Section heading at line start: number (+ optional letters), dot, title, then an em/en dash or "--".
HEADING = re.compile(r"(?m)^\s*\[?(\d{1,3}[A-Z]{0,3})\.\s+([^\n]+?(?:\n(?!\s*\[?\d{1,3}[A-Z]{0,3}\.\s)[^\n]+?)?)\s*\.?\s*(?:[—–―‑]{1,2}|--|\.-)\s*")


def code_for_file(path: Path) -> str | None:
    stem = path.stem.lower()
    # Longest key first so "bnss" wins over "bns".
    for key in sorted(CODE_FROM_FILENAME, key=len, reverse=True):
        if key in stem:
            return CODE_FROM_FILENAME[key]
    return None


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:  # pragma: no cover
            logger.warning(f"pypdf not installed; skipping {path}")
            return ""
        cache = path.with_suffix(".extracted.txt")
        if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
            return cache.read_text(encoding="utf-8")
        import logging
        logging.getLogger("pypdf").setLevel(logging.ERROR)
        text = "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
        cache.write_text(text, encoding="utf-8")
        return text
    return path.read_text(encoding="utf-8", errors="ignore")


FOOTNOTE_LINE = re.compile(r"(?m)^\s*\d{1,3}\.\s+(Subs\.|Ins\.|Omitted|Rep\.|The words|Added|Cl\.|Certain words|"
                           r"Section \d+|Renumbered|Original|For the words|Now see|See|Extended|This section|"
                           r"Explanation|Proviso|Sub-section|The original)[^\n]*$")
FOOTNOTE_MARK = re.compile(r"(?<![\w(])\d{1,2}\[")  # "1[imprisonment for life]" -> "[imprisonment for life]"


def _sec_key(sec: str) -> tuple[int, str]:
    m = re.match(r"(\d+)([A-Z]*)", sec)
    return (int(m.group(1)), m.group(2)) if m else (0, "")


def clean_pdf_text(text: str) -> str:
    text = FOOTNOTE_LINE.sub("", text)
    text = FOOTNOTE_MARK.sub("[", text)
    return re.sub(r"\s+\.(?=—)", ".", text)  # "murder .—" -> "murder.—"


def parse_statute_text(text: str, code: str, source: str = "", verified: bool = False,
                       strict_order: bool = False) -> list[StatuteSection]:
    """Split bare-act text on ``302. Heading.—Body`` headings. ``strict_order`` (PDFs) keeps a heading only if
    its number continues the sequence (same or next few numbers), which drops cross-references and footnotes."""
    if strict_order:
        text = clean_pdf_text(text)
    matches = list(HEADING.finditer(text))
    if strict_order:
        keys = [_sec_key(m.group(1).upper()) for m in matches]
        # Whole acts start at section 1; an excerpt may start anywhere (then begin at its first heading).
        start = (0, "") if any(k[0] <= 12 for k in keys) else (keys[0][0] - 1, "") if keys else (0, "")
        kept, last = [], start
        for m, k in zip(matches, keys):
            if k > last and k[0] <= last[0] + 12:
                kept.append(m)
                last = k
        matches = kept
    out: list[StatuteSection] = []
    for i, m in enumerate(matches):
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = re.sub(r"\s+", " ", text[m.end():body_end]).strip()
        out.append(StatuteSection(code=code, section=m.group(1).upper(), title=re.sub(r"\s+", " ", m.group(2)).strip(),
                                  text=body, source=source, verified=verified))
    return out


def load_indiacode_jsonl(path: Path) -> list[StatuteSection]:
    """Section-level JSONL written by scripts/fetch_indiacode.py (official India Code text)."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        sec = re.sub(r"\s+", "", str(r.get("section", ""))).upper()
        if not re.fullmatch(r"\d{1,3}[A-Z]{0,3}", sec):
            continue  # skip schedules, "1A-1Z" ranges, blank numbers
        out.append(StatuteSection(code=r["code"], section=sec, title=r.get("title", ""),
                                  text=re.sub(r"\s+", " ", r.get("text", "")).strip(),
                                  source=r.get("source", "indiacode"), verified=True))
    return out


def load_statutes(dirs: Iterable[Path], extra: Iterable[StatuteSection] = ()) -> list[StatuteSection]:
    """Earlier dirs win. Within a dir: India Code JSONL > PDF/TXT. ``extra`` (e.g. HF texts) fills gaps."""
    sections: dict[str, StatuteSection] = {}
    for d in dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.indiacode.jsonl")):
            for s in load_indiacode_jsonl(p):
                sections.setdefault(s.id, s)
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in {".txt", ".pdf"} or p.name.endswith((".pdf.txt", ".extracted.txt")):
                continue
            code = code_for_file(p)
            if not code:
                logger.warning(f"cannot infer code from filename {p.name}; skipping")
                continue
            fixture = "fixture" in str(p)
            pdf = p.suffix.lower() == ".pdf"
            for s in parse_statute_text(read_text(p), code, source=str(p.name), verified=not fixture, strict_order=pdf):
                sections.setdefault(s.id, s)
    for s in extra:
        sections.setdefault(s.id, s)
    logger.info(f"parsed {len(sections)} statute sections")
    return list(sections.values())


def save_jsonl(items, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for it in items:
            f.write(it.model_dump_json() + "\n")


def load_jsonl(path: Path, model):
    if not path.exists():
        return []
    with open(path) as f:
        return [model.model_validate(json.loads(line)) for line in f if line.strip()]


class StatuteStore:
    """Lookup by (code, section)."""

    def __init__(self, sections: list[StatuteSection]):
        self.by_id = {s.id: s for s in sections}

    def get(self, code: str, section: str) -> StatuteSection | None:
        from ..schemas import section_id
        return self.by_id.get(section_id(code, section))

    def __len__(self) -> int:
        return len(self.by_id)
