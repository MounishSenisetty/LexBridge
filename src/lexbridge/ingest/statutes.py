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
HEADING = re.compile(r"(?m)^\s*(\d{1,3}[A-Z]{0,3})\.\s+(.+?)\.?\s*(?:—|–|--|\.-)\s*")


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
        return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_statute_text(text: str, code: str, source: str = "", verified: bool = False) -> list[StatuteSection]:
    matches = list(HEADING.finditer(text))
    out: list[StatuteSection] = []
    for i, m in enumerate(matches):
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = re.sub(r"\s+", " ", text[m.end():body_end]).strip()
        out.append(StatuteSection(code=code, section=m.group(1), title=m.group(2).strip(),
                                  text=body, source=source, verified=verified))
    return out


def load_statutes(dirs: Iterable[Path]) -> list[StatuteSection]:
    sections: dict[str, StatuteSection] = {}
    for d in dirs:
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in {".txt", ".pdf"}:
                continue
            code = code_for_file(p)
            if not code:
                logger.warning(f"cannot infer code from filename {p.name}; skipping")
                continue
            fixture = "fixture" in str(p)
            for s in parse_statute_text(read_text(p), code, source=str(p.name), verified=not fixture):
                sections.setdefault(s.id, s)  # real data dirs are passed first and win
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
