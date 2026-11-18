import pytest

from lexbridge.nlp.date_extractor import extract_dates, extract_offence_date
from lexbridge.nlp.section_ner import extract_sections, find_code_mentions


def secs(t):
    return [(r.section, r.code) for r in extract_sections(t)]


@pytest.mark.parametrize("text,expected", [
    ("Section 302", [("302", None)]),
    ("S. 302", [("302", None)]),
    ("Sec. 302 IPC", [("302", "IPC")]),
    ("u/s 302", [("302", None)]),
    ("302/34 IPC", [("302", "IPC"), ("34", "IPC")]),
    ("Sections 302 and 307", [("302", None), ("307", None)]),
    ("Section 498-A of the Indian Penal Code", [("498A", "IPC")]),
    ("Section 304B IPC", [("304B", "IPC")]),
    ("Section 103(1) of BNS", [("103", "BNS")]),
    ("Section 103(1) of the Bharatiya Nyaya Sanhita, 2023", [("103", "BNS")]),
    ("Sections 498-A and 304-B I.P.C.", [("498A", "IPC"), ("304B", "IPC")]),
    ("Section 438 Cr.P.C.", [("438", "CrPC")]),
    ("Section 302 or 307", [("302", None), ("307", None)]),
])
def test_formats(text, expected):
    assert secs(text) == expected


def test_code_names():
    codes = [m.code for m in find_code_mentions(
        "Bharatiya Nyaya Sanhita, the BNSS, the Bharatiya Sakshya Adhiniyam, IPC, Cr.P.C. and the Evidence Act")]
    assert codes == ["BNS", "BNSS", "BSA", "IPC", "CrPC", "IEA"]


def test_no_false_positive_on_years():
    assert secs("decided in 2023 under the new law") == []


def test_dates():
    assert str(extract_dates("on 15 August 2024")[0].value) == "2024-08-15"
    assert str(extract_dates("August 15, 2024")[0].value) == "2024-08-15"
    assert str(extract_dates("dated 15.08.2024")[0].value) == "2024-08-15"
    assert str(extract_offence_date("Judgment of 3 March 2025 for an offence committed on 10 May 2023")) == "2023-05-10"
    assert extract_dates("in 2024") == []  # transition year without a month is ambiguous
