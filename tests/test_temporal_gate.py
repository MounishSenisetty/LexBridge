from datetime import date

from lexbridge.generate.pipeline import LexBridge
from lexbridge.generate.temporal_gate import decide

T = date(2024, 7, 1)


def test_gate():
    assert decide("murder on 30 June 2024", transition=T).governing == "old"
    assert decide("murder on 1 July 2024", transition=T).governing == "new"     # boundary → new code
    assert decide("murder on 15 August 2024", transition=T).governing == "new"
    assert decide("what is the punishment for murder", transition=T).governing == "both"
    assert decide("x", offence_date=date(2019, 1, 1), transition=T).governing == "old"


def test_pipeline_moves_to_governing_code(res):
    lb = LexBridge(res)
    out = lb.answer("The accused committed an offence under Section 302 IPC on 2 January 2025. Which code governs?")
    assert out["governing_code"]["governing"] == "new"
    assert "BNS_103" in out["governing_code"]["sections"]
    out = lb.answer("What is the punishment under Section 302 for an offence committed on 12 March 2019?")
    assert out["resolved_sections"] == ["IPC_302"] and out["governing_code"]["sections"] == ["IPC_302"]
