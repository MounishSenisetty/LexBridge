"""IPC↔BNS mapping sources -> merged SectionMapping table.

Sources, in priority order:
  1. Seed CSV (data/seed/seed_mappings.csv) — tiny, unverified, for unit tests.
  2. Hugging Face ``nandhakumarg/IPC_and_BNS_transformation`` (needs network + ``datasets``).
  3. IEEE DataPort "Indian Criminal Law Transition Mapping Dataset" xlsx in data/raw/mappings/.
Disagreements about the new section for the same old section go to mapping_conflicts.csv.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from loguru import logger

from ..schemas import ChangeType, SectionMapping, normalize_section

HF_DATASET = "nandhakumarg/IPC_and_BNS_transformation"


def _norm_sec(s) -> str | None:
    if s is None:
        return None
    s = str(s).strip()
    if not s or s.lower() in {"nan", "none", "-", "omitted", "deleted", "na", "n/a"}:
        return None
    m = re.search(r"\d{1,3}\s*-?\s*[A-Za-z]{0,3}", s)
    return normalize_section(m.group(0)) if m else None


def load_seed(path: Path) -> list[SectionMapping]:
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            new_sec = _norm_sec(r.get("new_section"))
            out.append(SectionMapping(
                old_code=r["old_code"], old_section=normalize_section(r["old_section"]),
                new_code=(r.get("new_code") or None) if new_sec else None, new_section=new_sec,
                title=r.get("title", ""),
                change_type=ChangeType(r.get("change_type") or "unknown"),
                notes=r.get("notes", ""), sources=["seed"],
                verified=str(r.get("verified", "false")).lower() == "true"))
    return out


def _rows_to_mappings(rows: list[dict], source: str) -> list[SectionMapping]:
    """Heuristic column detection; real datasets name columns differently."""
    out = []
    for r in rows:
        keys = {k.lower(): k for k in r}
        old_k = next((keys[k] for k in keys if "ipc" in k and "sec" in k), None) or \
            next((keys[k] for k in keys if "ipc" in k), None)
        new_k = next((keys[k] for k in keys if "bns" in k and "sec" in k), None) or \
            next((keys[k] for k in keys if "bns" in k), None)
        title_k = next((keys[k] for k in keys if "offen" in k or "title" in k or "desc" in k), None)
        if not old_k:
            continue
        old = _norm_sec(r[old_k])
        if not old:
            continue
        new = _norm_sec(r[new_k]) if new_k else None
        out.append(SectionMapping(old_code="IPC", old_section=old,
                                  new_code="BNS" if new else None, new_section=new,
                                  title=str(r[title_k]) if title_k else "",
                                  change_type=ChangeType.unknown if new else ChangeType.omitted,
                                  sources=[source]))
    return out


def parse_hf_records(records: list[dict]) -> tuple[list[SectionMapping], list[dict]]:
    """``nandhakumarg/IPC_and_BNS_transformation`` rows: a ``response`` column holding a dict literal with
    IPC Section/Heading/Descriptions and BNS Section/Heading/description. Returns (mappings, statute_texts)."""
    import ast
    maps, texts = [], []
    for row in records:
        r = row.get("response", row)
        if isinstance(r, str):
            try:
                r = ast.literal_eval(r)
            except (ValueError, SyntaxError):
                continue
        old = _norm_sec(r.get("IPC Section"))
        if not old:
            continue
        raw_new = str(r.get("BNS Section") or "").strip()
        repealed = bool(re.search(r"(?i)repeal|omitted|deleted", raw_new))
        m = re.match(r"^(\d{1,3}[A-Z]?)\s*(.*)$", raw_new)
        new = None if repealed or not m else normalize_section(m.group(1))
        sub = "" if not m else m.group(2).strip()
        maps.append(SectionMapping(
            old_code="IPC", old_section=old, new_code="BNS" if new else None, new_section=new, new_subsection=sub,
            title=str(r.get("IPC Heading") or "").strip(),
            change_type=ChangeType.unknown if new else ChangeType.omitted,
            notes=f"BNS heading: {str(r.get('BNS Heading') or '').strip()}" if new else "repealed in BNS",
            sources=["hf"]))
        texts.append({"code": "IPC", "section": old, "title": str(r.get("IPC Heading") or "").strip(),
                      "text": str(r.get("IPC Descriptions") or "")})
        if new and r.get("BNS description"):
            texts.append({"code": "BNS", "section": new, "title": str(r.get("BNS Heading") or "").strip(),
                          "text": str(r.get("BNS description") or "")})
    return maps, texts


def load_hf(raw_dir: Path | None = None, use_network: bool = True) -> list[SectionMapping]:
    """Local copy first (data/raw/mappings/hf/*.csv), then ``datasets.load_dataset``."""
    rows: list[dict] = []
    files = sorted((raw_dir / "hf").glob("*.csv")) if raw_dir else []
    if files:
        import csv as _csv
        for f in files:
            with open(f, newline="", encoding="utf-8") as fh:
                rows += list(_csv.DictReader(fh))
    elif use_network:
        try:
            from datasets import load_dataset
            rows = [dict(r) for r in load_dataset(HF_DATASET, split="train")]
        except ImportError:
            logger.warning("`datasets` not installed; skipping HF mapping source")
        except Exception as e:  # network blocked, dataset moved, etc.
            logger.warning(f"HF mapping download failed ({type(e).__name__}); skipping")
    if not rows:
        return []
    maps, _ = parse_hf_records(rows)
    logger.info(f"HF mappings: {len(maps)}")
    return maps


def load_ieee(raw_dir: Path) -> list[SectionMapping]:
    files = sorted(raw_dir.glob("ieee*.xlsx")) + sorted(raw_dir.glob("ieee*.csv"))
    if not files:
        return []
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not installed; skipping IEEE mapping source")
        return []
    out = []
    for f in files:
        df = pd.read_excel(f) if f.suffix == ".xlsx" else pd.read_csv(f)
        out += _rows_to_mappings(df.to_dict("records"), "ieee")
    return out


def merge(sources: list[list[SectionMapping]]) -> tuple[list[SectionMapping], list[dict]]:
    """First source wins on conflict; later sources add provenance or a conflict row."""
    merged: dict[tuple[str, str], SectionMapping] = {}
    conflicts: list[dict] = []
    for src in sources:
        for m in src:
            key = (m.old_code, m.old_section)
            if key not in merged:
                merged[key] = m.model_copy(deep=True)
                continue
            cur = merged[key]
            if cur.new_section == m.new_section:
                cur.sources = sorted(set(cur.sources) | set(m.sources))
                if cur.change_type == ChangeType.unknown and m.change_type != ChangeType.unknown:
                    cur.change_type = m.change_type
            else:
                conflicts.append({"old_code": m.old_code, "old_section": m.old_section,
                                  "kept_new": cur.new_section, "kept_source": ",".join(cur.sources),
                                  "other_new": m.new_section, "other_source": ",".join(m.sources)})
    return list(merged.values()), conflicts


def build_mappings(seed_path: Path, raw_dir: Path, use_network: bool = True, real: bool = False):
    """real=False: seed first (unit tests). real=True: HF and IEEE first; the seed is only checked against them."""
    seed = load_seed(seed_path)
    others = [load_hf(raw_dir, use_network), load_ieee(raw_dir)]
    return merge(others + [seed] if real else [seed] + others)


def write_conflicts(conflicts: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["old_code", "old_section", "kept_new", "kept_source", "other_new", "other_source"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(conflicts)


class MappingTable:
    """Bidirectional lookup over SectionMappings."""

    def __init__(self, mappings: list[SectionMapping]):
        self.mappings = mappings
        self.forward: dict[str, SectionMapping] = {m.old_id: m for m in mappings}
        self.reverse: dict[str, list[SectionMapping]] = {}
        for m in mappings:
            if m.new_id:
                self.reverse.setdefault(m.new_id, []).append(m)

    def to_new(self, code: str, section: str) -> SectionMapping | None:
        return self.forward.get(f"{code}_{normalize_section(section)}")

    def to_old(self, code: str, section: str) -> list[SectionMapping]:
        return self.reverse.get(f"{code}_{normalize_section(section)}", [])

    def counterparts(self, sid: str) -> list[str]:
        """IDs on the other side of the transition."""
        if sid in self.forward:
            nid = self.forward[sid].new_id
            return [nid] if nid else []
        return [m.old_id for m in self.reverse.get(sid, [])]

    def lookup(self, sid: str) -> list[SectionMapping]:
        if sid in self.forward:
            return [self.forward[sid]]
        return self.reverse.get(sid, [])
