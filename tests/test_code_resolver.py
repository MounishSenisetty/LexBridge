from datetime import date

from lexbridge.nlp.code_resolver import CodeResolver


def one(res, text, d=None):
    refs = res.resolver.resolve(text, d)
    return refs[0]


def test_302_ipc_murder_is_ipc(res):
    r = one(res, "The accused was convicted of murder under Section 302 IPC.")
    assert (r.code, r.rule) == ("IPC", "explicit")


def test_302_bns_religious_is_bns(res):
    r = one(res, "Charged under Section 302 BNS for wounding religious feelings.")
    assert (r.code, r.rule) == ("BNS", "explicit")


def test_302_34_in_2015_judgment_is_ipc(res):
    r = one(res, "The appellants were convicted u/s 302/34 for the death of the victim.", date(2015, 8, 21))
    assert (r.code, r.rule) == ("IPC", "date")


def test_after_transition_is_bns(res):
    from lexbridge.nlp.date_extractor import extract_offence_date
    text = "Section 302 offence committed on 15 August 2024 by uttering words to wound religious feelings"
    d = extract_offence_date(text)
    assert d == date(2024, 8, 15) and d >= res.resolver.transition
    assert one(res, text, d).code == "BNS"


def test_keyword_rule_without_code_or_date(res):
    assert one(res, "Section 302 punishes murder with death").code == "IPC"
    r = one(res, "Section 302 punishes wounding religious feelings")
    assert (r.code, r.rule) == ("BNS", "keyword")


def test_window_and_unresolved(res):
    r = one(res, "Under the BNS the charge is Section 85 for cruelty.")
    assert (r.code, r.rule) == ("BNS", "window")
    bare = CodeResolver().resolve("Section 999 applies.")[0]
    assert bare.code is None and bare.rule == "unresolved" and bare.confidence < 0.5


def test_fixture_corpus_resolution_matches_hand_labels(res):
    """Doc-level resolved sections cover every hand-labelled gold section."""
    for j in res.judgments:
        got = {r.id for r in res.resolver.resolve(j.text, j.date) if r.id}
        assert set(j.gold_sections) <= got, (j.doc_id, got)
