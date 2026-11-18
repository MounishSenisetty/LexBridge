"""Judgment loaders: fixtures (JSONL), ILDC (CSV/JSON) and a plain folder of .txt/.pdf files."""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

from loguru import logger

from ..schemas import Judgment


def load_jsonl_judgments(path: Path) -> list[Judgment]:
    if not path.exists():
        return []
    with open(path) as f:
        return [Judgment.model_validate(json.loads(l)) for l in f if l.strip()]


def load_ildc(dir_: Path, limit: int | None = None) -> list[Judgment]:
    """ILDC ships as CSV with columns like ``name, text, label, split``. Dates are extracted later."""
    files = sorted(dir_.glob("*.csv")) + sorted(dir_.glob("*.json*"))
    if not files:
        return []
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not installed; skipping ILDC")
        return []
    out: list[Judgment] = []
    for f in files:
        df = pd.read_csv(f) if f.suffix == ".csv" else pd.read_json(f, lines=f.suffix == ".jsonl")
        for i, r in df.iterrows():
            text = str(r.get("text", ""))
            if not text.strip():
                continue
            out.append(Judgment(doc_id=f"ILDC-{r.get('name', i)}", title=str(r.get("name", "")),
                                court="Supreme Court of India", text=text, source="ildc",
                                date=_guess_date(text)))
            if limit and len(out) >= limit:
                return out
    return out


def load_folder(dir_: Path, limit: int | None = None) -> list[Judgment]:
    from .statutes import read_text
    out = []
    for p in sorted(dir_.glob("*")):
        if p.suffix.lower() not in {".txt", ".pdf"}:
            continue
        text = read_text(p)
        if text.strip():
            out.append(Judgment(doc_id=f"IK-{p.stem}", title=p.stem, text=text,
                                source="folder", date=_guess_date(text)))
        if limit and len(out) >= limit:
            break
    return out


def _guess_date(text: str) -> date | None:
    from ..nlp.date_extractor import extract_dates
    ds = extract_dates(text[:3000])
    return ds[0].value if ds else None


def paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) <= 1:  # fall back to numbered paragraphs, then sentences in groups
        paras = [p.strip() for p in re.split(r"(?m)^\s*\d{1,3}\.\s+", text) if p.strip()]
    return paras
