from lexbridge.index.graph import neighbors


def test_bns_103_neighbors(res):
    n = neighbors(res.graph, "BNS_103")
    assert "IPC_302" in n["counterparts"]
    cites_302 = {u for u, v, d in res.graph.in_edges("IPC_302", data=True) if d["rel"] == "CITES"}
    assert len(cites_302 & set(n["judgments"])) >= 1


def test_collision_pair_not_linked(res):
    """IPC_302 and BNS_302 share a number but must not be connected."""
    assert not res.graph.has_edge("IPC_302", "BNS_302") and not res.graph.has_edge("BNS_302", "IPC_302")
    assert "IPC_298" in neighbors(res.graph, "BNS_302")["counterparts"]


def test_chunks_are_role_tagged(res):
    roles = {c.role for c in res.index("role").chunks}
    assert {"facts", "analysis", "decision"} <= roles
