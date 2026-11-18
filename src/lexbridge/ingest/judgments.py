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


CRIMINAL = re.compile(r"(?i)\b(Indian Penal Code|I\.?P\.?C\.?|Penal Code|Bharatiya Nyaya Sanhita|B\.?N\.?S\.?)\b")


def load_parquet_judgments(dir_: Path, prefix: str, limit: int | None = None, criminal_only: bool = True) -> list[Judgment]:
    """HF parquet corpora (column names detected):
    * opennyaiorg/InJudgements_dataset — Titles, Court_Name, Text, Case_Type, Doc_url
    * Exploration-Lab/IL-TUR cjpe       — id/name, text, label
    ``criminal_only`` keeps judgments that cite the IPC/BNS (Case_Type == criminal when present)."""
    files = sorted(dir_.glob("*.parquet"))
    if not files:
        return []
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not installed; skipping parquet judgments")
        return []
    out: list[Judgment] = []
    for f in files:
        try:
            df = pd.read_parquet(f)
        except Exception as e:
            logger.warning(f"{f.name}: unreadable parquet ({type(e).__name__}); skipping")
            continue
        cols = {c.lower(): c for c in df.columns}
        tcol = cols.get("text") or cols.get("full_text")
        if not tcol:
            logger.warning(f"{f.name}: no text column ({list(df.columns)})")
            continue
        for i, r in df.iterrows():
            text = str(r[tcol] or "")
            if len(text) < 500:
                continue
            ctype = str(r[cols["case_type"]]).lower() if "case_type" in cols else ""
            if criminal_only and (ctype and ctype != "criminal" or not ctype) and not CRIMINAL.search(text[:20000]):
                continue
            key = next((cols[k] for k in ("cnr", "doc_url", "id", "name") if k in cols), None)
            ident = r[key] if key else i
            ident = re.sub(r"\W+", "-", str(ident).rstrip("/").split("/")[-1])[:60]
            title = str(r[cols["titles"]]) if "titles" in cols else str(r[cols["name"]]) if "name" in cols else ident
            court = str(r[cols["court_name"]]) if "court_name" in cols else ""
            d = None
            if "decision_date" in cols and r[cols["decision_date"]]:
                try:
                    d = pd.to_datetime(r[cols["decision_date"]], errors="coerce")
                    d = None if pd.isna(d) else d.date()
                except Exception:
                    d = None
            out.append(Judgment(doc_id=f"{prefix}-{ident}", title=title[:200], court=court, text=text,
                                source=f"{prefix.lower()}:{f.name}", date=d or _guess_date(text)))
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
