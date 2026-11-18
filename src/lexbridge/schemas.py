"""Core data schemas (pydantic v2)."""
from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

Code = Literal["IPC", "CrPC", "IEA", "BNS", "BNSS", "BSA"]

OLD_CODES = ("IPC", "CrPC", "IEA")
NEW_CODES = ("BNS", "BNSS", "BSA")
SUCCESSOR = {"IPC": "BNS", "CrPC": "BNSS", "IEA": "BSA"}
PREDECESSOR = {v: k for k, v in SUCCESSOR.items()}


def section_id(code: str, section: str) -> str:
    """Canonical node / token id, e.g. ('IPC', '498-a') -> 'IPC_498A'."""
    return f"{code}_{normalize_section(section)}"


def normalize_section(section: str) -> str:
    return str(section).upper().replace("-", "").replace(" ", "")


class ChangeType(str, Enum):
    identical = "identical"
    reworded = "reworded"
    punishment_changed = "punishment_changed"
    merged = "merged"
    split = "split"
    ingredients_changed = "ingredients_changed"
    omitted = "omitted"
    new = "new"
    unknown = "unknown"


class Applicability(str, Enum):
    DIRECTLY_APPLICABLE = "DIRECTLY_APPLICABLE"
    APPLICABLE_WITH_CAVEAT = "APPLICABLE_WITH_CAVEAT"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


CHANGE_TO_TAG = {
    ChangeType.identical: Applicability.DIRECTLY_APPLICABLE,
    ChangeType.reworded: Applicability.DIRECTLY_APPLICABLE,
    ChangeType.punishment_changed: Applicability.APPLICABLE_WITH_CAVEAT,
    ChangeType.merged: Applicability.APPLICABLE_WITH_CAVEAT,
    ChangeType.split: Applicability.APPLICABLE_WITH_CAVEAT,
    ChangeType.ingredients_changed: Applicability.REVIEW_REQUIRED,
    ChangeType.omitted: Applicability.NOT_APPLICABLE,
    ChangeType.new: Applicability.REVIEW_REQUIRED,
    ChangeType.unknown: Applicability.REVIEW_REQUIRED,
}


class StatuteSection(BaseModel):
    code: Code
    section: str
    title: str = ""
    text: str = ""
    chapter: Optional[str] = None
    source: str = ""
    verified: bool = False

    @property
    def id(self) -> str:
        return section_id(self.code, self.section)


class SectionMapping(BaseModel):
    old_code: Code
    old_section: str
    new_code: Optional[Code] = None
    new_section: Optional[str] = None  # None => omitted
    title: str = ""
    change_type: ChangeType = ChangeType.unknown
    notes: str = ""
    sources: list[str] = Field(default_factory=list)
    verified: bool = False

    @property
    def old_id(self) -> str:
        return section_id(self.old_code, self.old_section)

    @property
    def new_id(self) -> Optional[str]:
        if self.new_code and self.new_section:
            return section_id(self.new_code, self.new_section)
        return None


class Judgment(BaseModel):
    doc_id: str
    title: str = ""
    court: str = ""
    date: Optional[dt.date] = None
    text: str
    source: str = ""
    synthetic: bool = False
    gold_sections: list[str] = Field(default_factory=list)  # hand labels (fixtures only)


class SectionRef(BaseModel):
    section: str
    code: Optional[Code] = None
    raw: str = ""
    start: int = 0
    end: int = 0
    confidence: float = 0.0
    rule: str = ""

    @property
    def id(self) -> Optional[str]:
        return section_id(self.code, self.section) if self.code else None


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    para_idx: int = 0
    role: str = "other"
    section_ids: list[str] = Field(default_factory=list)       # resolved refs inside this chunk
    doc_section_ids: list[str] = Field(default_factory=list)   # resolved refs anywhere in the judgment
    date: Optional[dt.date] = None
    code_family: Optional[str] = None  # "old" | "new" | None
    title: str = ""
