import json

from lexbridge.generate.llm import BaseLLM
from lexbridge.ingest.mappings import MappingTable
from lexbridge.transfer.diff import diff_sections
from lexbridge.transfer.transferability import classify


class FakeLLM(BaseLLM):
    name = "fake"

    def __init__(self):
        self.calls = 0

    def complete(self, prompt, *, system="", task="answer", context=None):
        self.calls += 1
        assert task == "transferability" and "OLD (IPC" in prompt
        return json.dumps({"label": "ingredients_changed", "changed_elements": ["ingredients"], "rationale": "fake"})


def test_377_omitted(res):
    t = res.transfer["IPC_377"]
    assert (t.label, t.tag) == ("omitted", "NOT_APPLICABLE")


def test_rules_do_not_call_llm(res):
    llm = FakeLLM()
    table = MappingTable(res.mappings)
    assert classify(table.to_new("IPC", "498A"), res.store, llm).label == "identical"
    assert classify(table.to_new("IPC", "379"), res.store, llm).label == "punishment_changed"
    assert llm.calls == 0


def test_llm_path_is_used_and_parsed(res):
    llm = FakeLLM()
    t = classify(MappingTable(res.mappings).to_new("IPC", "124A"), res.store, llm)
    assert llm.calls == 1 and t.label == "ingredients_changed" and t.tag == "REVIEW_REQUIRED" and t.method == "llm:fake"


def test_diff_detects_punishment_only_change():
    d = diff_sections("Whoever commits theft shall be punished with imprisonment for three years.",
                      "Whoever commits theft shall be punished with imprisonment for five years.")
    assert d.ingredient_ratio > 0.97 and d.punishment_ratio < 0.97
