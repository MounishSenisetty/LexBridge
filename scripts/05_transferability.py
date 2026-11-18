"""Run C3 transferability (rules + LLM) over all mappings; resumable (appends to transferability.jsonl)."""
import collections
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lexbridge.config import get_config  # noqa: E402
from lexbridge.generate.llm import get_llm  # noqa: E402
from lexbridge.ingest.statutes import StatuteStore, load_jsonl  # noqa: E402
from lexbridge.schemas import SectionMapping, StatuteSection  # noqa: E402
from lexbridge.transfer.transferability import classify_all  # noqa: E402

P = get_config().path("processed")
t = time.time()
res = classify_all(load_jsonl(P / "mappings.jsonl", SectionMapping),
                   StatuteStore(load_jsonl(P / "statutes.jsonl", StatuteSection)), get_llm(), P / "transferability.jsonl")
print(collections.Counter(r.method.split(":")[0] for r in res.values()), collections.Counter(r.label for r in res.values()))
print(f"{time.time() - t:.0f}s")
