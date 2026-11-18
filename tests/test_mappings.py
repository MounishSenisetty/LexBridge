from lexbridge.config import get_config
from lexbridge.ingest.mappings import MappingTable, load_seed, merge
from lexbridge.ingest.statutes import parse_statute_text
from lexbridge.schemas import SectionMapping


def seed():
    return load_seed(get_config().path("seed") / "seed_mappings.csv")


def test_seed_loads_all_rows():
    assert len(seed()) == 11


def test_forward_and_reverse():
    t = MappingTable(seed())
    assert t.to_new("IPC", "302").new_id == "BNS_103"
    assert [m.old_id for m in t.to_old("BNS", "302")] == ["IPC_298"]      # collision case
    assert t.to_new("IPC", "498-A").new_id == "BNS_85"
    assert t.to_new("IPC", "377").new_id is None                          # omitted


def test_merge_records_conflicts():
    a = [SectionMapping(old_code="IPC", old_section="302", new_code="BNS", new_section="103", sources=["a"])]
    b = [SectionMapping(old_code="IPC", old_section="302", new_code="BNS", new_section="101", sources=["b"]),
         SectionMapping(old_code="IPC", old_section="307", new_code="BNS", new_section="109", sources=["b"])]
    merged, conflicts = merge([a, b])
    assert len(merged) == 2 and merged[0].new_section == "103"
    assert conflicts == [{"old_code": "IPC", "old_section": "302", "kept_new": "103", "kept_source": "a",
                          "other_new": "101", "other_source": "b"}]


def test_statute_parser_on_fixture():
    text = (get_config().path("fixtures") / "statutes" / "IPC.txt").read_text()
    secs = parse_statute_text(text, "IPC")
    assert len(secs) >= 1
    by = {s.section: s for s in secs}
    assert by["302"].title == "Punishment for murder"
    assert by["498A"].text.startswith("Whoever, being the husband")
