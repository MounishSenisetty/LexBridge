from fastapi.testclient import TestClient

from lexbridge.baselines import get_system
from lexbridge.generate.pipeline import ABLATIONS, LexBridge
from lexbridge.generate.prompts import DISCLAIMER


def test_answer_shape(res):
    out = LexBridge(res).answer("Punishment for murder under Section 103 BNS?")
    for k in ("answer", "citations", "applicability", "governing_code", "debug", "faithfulness_score"):
        assert k in out
    assert DISCLAIMER in out["answer"] and "Dual citation" in out["answer"]
    assert out["faithfulness_score"] == 1.0
    assert any(c["tag"] == "DIRECTLY_APPLICABLE" for c in out["citations"])
    assert out["cited_ids"]


def test_all_systems_and_ablations_run(res):
    for n in ["no_rag", "bm25", "dense", "hybrid", "lexbridge", *ABLATIONS]:
        out = get_system(n, res).answer("What is the punishment under Section 302 BNS?")
        assert "answer" in out


def test_baseline_shows_collision(res):
    """The code-unaware baseline pulls murder judgments for a BNS 302 question."""
    out = get_system("bm25", res).answer("What is the punishment under Section 302 BNS?")
    assert {c["doc_id"] for c in out["citations"][:5]} & {"FX-001", "FX-002", "FX-005"}


def test_api(res, monkeypatch):
    from lexbridge.api import server
    monkeypatch.setattr(server, "res", lambda: res)
    server.system.cache_clear()
    c = TestClient(server.app)
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/ask", json={"query": "Punishment for murder under Section 103 BNS?"}).json()
    assert r["resolved_sections"] == ["BNS_103"]
    assert c.get("/section/IPC/302").json()["title"] == "Punishment for murder"
    m = c.get("/map/BNS/302").json()
    assert m["mappings"][0]["old_id"] == "IPC_298"
    assert c.get("/map/IPC/377").json()["mappings"][0]["transferability"]["label"] == "omitted"
    assert c.get("/section/XYZ/1").status_code == 404
    server.system.cache_clear()


def test_benchmark_and_eval(res, tmp_path):
    import sys

    import yaml

    from lexbridge.config import ROOT
    from lexbridge.eval.agreement import cohen_kappa
    from lexbridge.eval.run_eval import evaluate
    sys.path.insert(0, str(ROOT / "benchmark"))
    from generate_candidates import build, stratified
    items = build(res, yaml.safe_load(open(ROOT / "benchmark" / "templates.yaml")))
    assert len(items) >= 50
    assert {i["category"] for i in items} == {"collision", "cross_code_precedent", "temporal", "transfer"}
    assert all(i["status"] == "draft" for i in items)
    sample = stratified(items, 20)
    s = evaluate(LexBridge(res), sample)["summary"]
    assert s["CER@5"] == 0.0
    assert abs(cohen_kappa(list("aabb"), list("aabb")) - 1.0) < 1e-9
    assert cohen_kappa(list("abab"), list("abba")) == 0.0
