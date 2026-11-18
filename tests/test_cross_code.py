from lexbridge.generate.pipeline import LexBridge
from lexbridge.retrieve.cross_code import expand_ids, is_collision
from lexbridge.schemas import Chunk


def test_expand(res):
    assert expand_ids(res.graph, ["BNS_103"]) == ["BNS_103", "IPC_302"]


def test_bns_103_query_returns_ipc_302_and_no_religious(res):
    out = LexBridge(res).answer("Punishment for murder under Section 103 BNS?")
    docs = {c["doc_id"] for c in out["citations"]}
    assert {"FX-001", "FX-002"} & docs                       # IPC 302 murder precedents
    religious = {"FX-003", "FX-004", "FX-006", "FX-007"}
    assert not docs & religious
    assert not any("religious" in c["text"].lower() for c in out["citations"])


def test_bns_302_query_filters_murder(res):
    out = LexBridge(res).answer("What is the punishment under Section 302 BNS?")
    docs = {c["doc_id"] for c in out["citations"]}
    assert docs and docs <= {"FX-003", "FX-004", "FX-006", "FX-007"}
    assert out["debug"]["dropped_collisions"]


def test_is_collision_unit():
    c = Chunk(chunk_id="x¶0", doc_id="x", text="", section_ids=["BNS_302"])
    assert is_collision(c, ["BNS_103", "IPC_302"])
    assert not is_collision(c, ["BNS_302", "IPC_298"])
    both = Chunk(chunk_id="y¶0", doc_id="y", text="", section_ids=["BNS_103", "IPC_302"])
    assert not is_collision(both, ["BNS_103", "IPC_302"])
