"""C3 — precedent transferability.

Rules first (no new section ⇒ omitted; whole-text ratio > 0.97 ⇒ identical; only the punishment
clause differs ⇒ punishment_changed), otherwise an LLM JSON prompt. Results are cached in
``transferability.jsonl``."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..generate.prompts import TRANSFER_PROMPT, TRANSFER_SYSTEM
from ..ingest.statutes import StatuteStore
from ..schemas import CHANGE_TO_TAG, ChangeType, SectionMapping
from .diff import diff_sections

VALID = {c.value for c in ChangeType}


@dataclass
class TransferResult:
    old_id: str
    new_id: str | None
    label: str
    tag: str
    changed_elements: list[str] = field(default_factory=list)
    rationale: str = ""
    method: str = "rule"


def _result(m: SectionMapping, label: str, changed, rationale: str, method: str) -> TransferResult:
    ct = ChangeType(label) if label in VALID else ChangeType.unknown
    return TransferResult(m.old_id, m.new_id, ct.value, CHANGE_TO_TAG[ct].value, list(changed), rationale, method)


def classify(m: SectionMapping, statutes: StatuteStore, llm=None) -> TransferResult:
    if not m.new_id:
        return _result(m, "omitted", [], "no corresponding section in the new code", "rule")
    old = statutes.get(m.old_code, m.old_section)
    new = statutes.get(m.new_code, m.new_section)
    if not old or not new:
        if m.change_type != ChangeType.unknown:
            return _result(m, m.change_type.value, [], "taken from the mapping source", "mapping")
        return _result(m, "unknown", [], "statute text unavailable for one side", "rule")
    d = diff_sections(old.text, new.text)
    if d.ratio > 0.97:
        return _result(m, "identical", [], f"text similarity {d.ratio:.3f} > 0.97", "rule")
    if d.ingredient_ratio > 0.97 and d.punishment_ratio < 0.97:
        return _result(m, "punishment_changed", ["punishment"],
                       f"ingredients identical ({d.ingredient_ratio:.3f}); punishment differs ({d.punishment_ratio:.3f})", "rule")
    if llm is None:
        from ..generate.llm import get_llm
        llm = get_llm()
    prompt = TRANSFER_PROMPT.format(old_code=m.old_code, old_section=m.old_section, old_text=old.text[:900],
                                    new_code=m.new_code, new_section=m.new_section, new_text=new.text[:1200],
                                    labels=", ".join(sorted(VALID - {"new", "unknown"})))
    out = llm.complete_json(prompt, system=TRANSFER_SYSTEM, task="transferability", context={"diff": asdict(d)})
    label = out.get("label", "unknown")
    return _result(m, label if label in VALID else "unknown", out.get("changed_elements", d.changed_elements),
                   out.get("rationale", ""), f"llm:{getattr(llm, 'name', '?')}")


def classify_all(mappings: list[SectionMapping], statutes: StatuteStore, llm=None,
                 cache: Path | None = None, rules_only: bool = False) -> dict[str, TransferResult]:
    """Resumable: cached results are reused and new ones appended as they are computed.
    ``rules_only`` leaves mappings that need the LLM as ``unknown`` (method ``pending``) without caching them."""
    cached: dict[str, TransferResult] = {}
    if cache and cache.exists():
        for line in cache.read_text().splitlines():
            if line.strip():
                r = TransferResult(**json.loads(line))
                cached[r.old_id] = r
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        if not cache.exists():
            cache.write_text("")
    out = {}

    class _Pending:
        name = "pending"

        def complete_json(self, *a, **k):
            raise _NeedLLM

    for m in mappings:
        if m.old_id in cached:
            out[m.old_id] = cached[m.old_id]
            continue
        try:
            r = classify(m, statutes, _Pending() if rules_only else llm)
        except _NeedLLM:
            out[m.old_id] = _result(m, "unknown", [], "awaiting LLM transferability run", "pending")
            continue
        out[m.old_id] = r
        if cache:
            with open(cache, "a") as f:
                f.write(json.dumps(asdict(r)) + "\n")
    return out


class _NeedLLM(Exception):
    pass
