"""Real-data loaders and the mapping checker, on tiny inline samples (no data/raw needed)."""
import json

from lexbridge.ingest.mappings import parse_hf_records
from lexbridge.ingest.statutes import StatuteStore, load_indiacode_jsonl, parse_statute_text
from lexbridge.ingest.verify_mappings import check
from lexbridge.schemas import SectionMapping, StatuteSection

HF_ROWS = [
    {"response": str({"IPC Section": "420", "IPC Heading": "Cheating and dishonestly inducing delivery of property",
                      "IPC Descriptions": "420. Cheating ... Whoever cheats ...", "BNS Section": "318(4)",
                      "BNS Heading": "Cheating", "BNS description": "Whoever cheats ..."})},
    {"response": str({"IPC Section": "377", "IPC Heading": "Unnatural offences", "IPC Descriptions": "...",
                      "BNS Section": "Repealed in BNS", "BNS Heading": "Repealed in BNS", "BNS description": None})},
]


def test_parse_hf_records():
    maps, texts = parse_hf_records(HF_ROWS)
    m = {x.old_id: x for x in maps}
    assert m["IPC_420"].new_id == "BNS_318" and m["IPC_420"].new_subsection == "(4)"
    assert m["IPC_377"].new_id is None and m["IPC_377"].change_type.value == "omitted"
    assert {(t["code"], t["section"]) for t in texts} == {("IPC", "420"), ("BNS", "318"), ("IPC", "377")}


def test_indiacode_jsonl(tmp_path):
    p = tmp_path / "BNS.indiacode.jsonl"
    p.write_text(json.dumps({"code": "BNS", "section": "103", "title": "Punishment for murder",
                             "text": "(1) Whoever commits murder ...", "source": "indiacode:x"}) + "\n"
                 + json.dumps({"code": "BNS", "section": "", "title": "Schedule", "text": "..."}) + "\n")
    secs = load_indiacode_jsonl(p)
    assert [s.id for s in secs] == ["BNS_103"] and secs[0].verified


def test_pdf_parser_handles_inserted_sections_and_footnotes():
    text = ("ARRANGEMENT OF SECTIONS\n302. Punishment for murder.\n303. Punishment for murder by life-convict.\n"
            "302. Punishment for murder .—Whoever commits murder shall be punished with death, or 1[imprisonment for life].\n"
            "1. Subs. by Act 26 of 1955, s. 117, for “transportation for life”.\n"
            "2[304B. Dowry death.—(1) Where the death of a woman ...\n"
            "305. Abetment of suicide of child.—If any person under eighteen years of age commits suicide ...\n")
    secs = {s.section: s for s in parse_statute_text(text, "IPC", strict_order=True)}
    assert list(secs) == ["302", "304B", "305"]
    assert secs["302"].title == "Punishment for murder" and "[imprisonment for life]" in secs["302"].text
    assert "Subs. by" not in secs["302"].text


def _store():
    return StatuteStore([
        StatuteSection(code="IPC", section="302", title="Punishment for murder", verified=True,
                       text="Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine."),
        StatuteSection(code="BNS", section="103", title="Punishment for murder", verified=True,
                       text="(1) Whoever commits murder shall be punished with death or imprisonment for life, and shall also be liable to fine. (2) When a group of five or more persons ..."),
        StatuteSection(code="BNS", section="101", title="Murder", verified=True,
                       text="Except in the cases hereinafter excepted, culpable homicide is murder ..."),
    ])


def test_checker_consistent_and_suspect():
    good = SectionMapping(old_code="IPC", old_section="302", new_code="BNS", new_section="103",
                          title="Punishment for murder", notes="BNS heading: Punishment for murder", sources=["hf"])
    wrong = SectionMapping(old_code="IPC", old_section="302", new_code="BNS", new_section="101",
                           title="Punishment for murder", notes="BNS heading: Murder", sources=["hf"])
    omitted = SectionMapping(old_code="IPC", old_section="302", title="Punishment for murder", sources=["hf"])
    rows = check([good, wrong, omitted], _store())
    assert [r["status"] for r in rows] == ["consistent", "suspect", "suspect"]
    assert rows[1]["best_alt"] == "BNS_103" and rows[2]["best_alt"] == "BNS_103"
    assert good.auto_check == "consistent" and wrong.auto_check == "suspect"
