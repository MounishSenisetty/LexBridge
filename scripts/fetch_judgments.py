"""Sample criminal judgments from overthelex/indian-court-decisions (CC-BY-4.0; eCourts / Supreme Court via the
AWS Open Data registry, 1950–2026). Reads a few parquet row groups over HTTP (no full 2.5 GB download).

python scripts/fetch_judgments.py [n_old] [n_new] [row_groups]
  → data/raw/judgments/overthelex/sample.parquet
Keeps English judgments citing the IPC (decided before 1 July 2024) or the BNS (decided on/after it)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

REPO = "datasets/overthelex/indian-court-decisions"
IPC = re.compile(r"Indian Penal Code|I\.\s?P\.\s?C\.|\bIPC\b")
BNS = re.compile(r"Bharatiya Nyaya Sanhita|\bB\.?N\.?S\.?\b(?!S)")
SEC = re.compile(r"(?i)\b(?:sections?|u/s|sec\.)\s*\d{1,3}")
COLS = ["cnr", "source", "court_name", "year", "full_text", "title", "decision_date", "case_type",
        "disposal_nature_normalized"]


def main(n_old: int = 1500, n_new: int = 1500, groups: str = "0,1,2,3,4,5") -> None:
    import pandas as pd
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    out = Path("data/raw/judgments/overthelex")
    out.mkdir(parents=True, exist_ok=True)
    old, new = [], []
    for split in ("high_courts/test.parquet", "supreme_court/test.parquet"):
        pf = pq.ParquetFile(fs.open(f"{REPO}/{split}", "rb", block_size=2 ** 22))
        gs = [int(g) for g in groups.split(",") if int(g) < pf.metadata.num_row_groups]
        for g in gs:
            df = pf.read_row_group(g, columns=COLS).to_pandas()
            df = df[df.full_text.str.len().between(2000, 120000) & df.full_text.str.contains(SEC)]
            dt = pd.to_datetime(df.decision_date, errors="coerce", dayfirst=True)
            after = (dt >= "2024-07-01") | (dt.isna() & (df.year >= 2025))
            o = df[~after & (df.year < 2024) & df.full_text.str.contains(IPC)]
            n = df[after & df.full_text.str.contains(BNS)]
            old.append(o)
            new.append(n)
            print(f"{split} rg{g}: {len(df)} with sections → IPC-era {len(o)}, BNS-era {len(n)}", flush=True)
            if sum(map(len, old)) >= n_old and sum(map(len, new)) >= n_new:
                break
    o = pd.concat(old).drop_duplicates("cnr").sample(frac=1, random_state=13).head(int(n_old))
    n = pd.concat(new).drop_duplicates("cnr").sample(frac=1, random_state=13).head(int(n_new))
    res = pd.concat([o, n])
    res.to_parquet(out / "sample.parquet", index=False)
    print(f"saved {len(o)} IPC-era + {len(n)} BNS-era judgments → {out/'sample.parquet'}")


if __name__ == "__main__":
    main(*[int(a) if a.isdigit() else a for a in sys.argv[1:]])
