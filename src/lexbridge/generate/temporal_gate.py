"""C4 — temporal applicability gate.

Substantive offence law follows the date of the offence: before 1 July 2024 → IPC/CrPC/IEA,
on or after → BNS/BNSS/BSA. No date → answer under both and say so. (Procedure for proceedings
pending on the transition date is governed by the BNSS savings clause; this is flagged in the note.)"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date

from ..config import get_config
from ..nlp.date_extractor import extract_offence_date
from ..schemas import NEW_CODES, OLD_CODES


@dataclass
class TemporalDecision:
    governing: str                      # "old" | "new" | "both"
    offence_date: date | None = None
    codes: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["offence_date"] = self.offence_date.isoformat() if self.offence_date else None
        return d


def decide(query: str, offence_date: date | None = None, transition: date | None = None) -> TemporalDecision:
    transition = transition or get_config().transition
    od = offence_date or extract_offence_date(query)
    if od is None:
        return TemporalDecision("both", None, list(OLD_CODES + NEW_CODES),
                                f"No offence date given: the answer covers both codes. Offences committed before "
                                f"{transition:%d %B %Y} fall under the IPC family; offences on or after it under BNS/BNSS/BSA.")
    if od < transition:
        return TemporalDecision("old", od, list(OLD_CODES),
                                f"Offence date {od:%d %B %Y} is before {transition:%d %B %Y}: the IPC governs the offence. "
                                f"Procedure for proceedings after that date may fall under the BNSS savings clause.")
    return TemporalDecision("new", od, list(NEW_CODES),
                            f"Offence date {od:%d %B %Y} is on or after {transition:%d %B %Y}: the BNS governs the offence.")


def family_of(code: str | None) -> str | None:
    if code in OLD_CODES:
        return "old"
    if code in NEW_CODES:
        return "new"
    return None
