"""Streamlit UI. Run: streamlit run src/lexbridge/ui/app.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from lexbridge.baselines import get_system  # noqa: E402
from lexbridge.generate.prompts import DISCLAIMER  # noqa: E402
from lexbridge.resources import load_resources  # noqa: E402
from lexbridge.schemas import normalize_section  # noqa: E402

TAG_COLOURS = {"DIRECTLY_APPLICABLE": "#1b7f3b", "APPLICABLE_WITH_CAVEAT": "#b7791f",
               "REVIEW_REQUIRED": "#c05621", "NOT_APPLICABLE": "#c53030", "CONTEXT": "#4a5568", None: "#4a5568"}
COLLISION_QUERY = "What is the punishment under Section 302 BNS?"

st.set_page_config(page_title="LexBridge", layout="wide")


@st.cache_resource
def resources():
    return load_resources()


@st.cache_resource
def system(name):
    return get_system(name, resources())


def card(c: dict):
    colour = TAG_COLOURS.get(c.get("tag"), "#4a5568")
    st.markdown(
        f"<div style='border-left:6px solid {colour};padding:6px 12px;margin:6px 0;background:rgba(127,127,127,.06)'>"
        f"<b>[{c['id']}]</b> {c['title']} · {c['date']} · <i>{c['role']}</i> · "
        f"<span style='color:{colour};font-weight:600'>{c.get('tag') or 'untagged'}</span><br>"
        f"<small>sections: {', '.join(c['section_ids']) or '—'}</small><br>{c['text'][:500]}</div>",
        unsafe_allow_html=True)


def show(out: dict, title: str):
    st.subheader(title)
    st.text(out["answer"])
    gov = out.get("governing_code", {})
    st.caption(f"resolved: {out.get('resolved_sections')} · governing: {gov.get('governing')} {gov.get('sections')} · "
               f"faithfulness: {out.get('faithfulness_score')}")
    for c in out["citations"]:
        card(c)


st.title("LexBridge")
st.caption("Code-aware legal RAG across the 1 July 2024 IPC→BNS transition. Fixture data + mock LLM. " + DISCLAIMER)

tab_ask, tab_map = st.tabs(["Ask", "Mapping viewer"])
with tab_ask:
    q = st.text_input("Question", value=st.session_state.get("q", "Punishment for murder under Section 103 BNS?"))
    col1, col2 = st.columns(2)
    d = col1.text_input("Offence date (YYYY-MM-DD, optional)", value="")
    mode = col2.radio("System", ["LexBridge", "Baseline (hybrid RAG)", "Side by side"], horizontal=True)
    demo = st.button("Collision demo: Section 302 BNS")
    if demo:
        q, mode = COLLISION_QUERY, "Side by side"
        st.info("BNS 302 is *wounding religious feelings*; IPC 302 is *murder*. A code-unaware baseline "
                "retrieves murder cases; LexBridge resolves the code and filters the collision.")
    if st.button("Ask") or demo:
        if mode == "Side by side":
            a, b = st.columns(2)
            with a:
                show(system("hybrid").answer(q, d or None), "Baseline (hybrid RAG)")
            with b:
                show(system("lexbridge").answer(q, d or None), "LexBridge")
        else:
            name = "lexbridge" if mode == "LexBridge" else "hybrid"
            show(system(name).answer(q, d or None), mode)

with tab_map:
    r = resources()
    c1, c2 = st.columns(2)
    code = c1.selectbox("Code", ["IPC", "BNS"])
    sec = c2.text_input("Section", "302")
    sid = f"{code}_{normalize_section(sec)}"
    s = r.store.by_id.get(sid)
    if s:
        st.markdown(f"**{code} §{s.section} — {s.title}**  \n{s.text}")
    for m in r.table.lookup(sid):
        t = r.transfer.get(m.old_id)
        st.markdown(f"**{m.old_id} → {m.new_id or '(omitted)'}** · {m.title} · change: `{t.label if t else '?'}` · "
                    f"tag: `{t.tag if t else '?'}` · mapping verified: {m.verified}")
        if t:
            st.caption(t.rationale)
    st.dataframe([{"old": m.old_id, "new": m.new_id, "title": m.title, "change": r.transfer[m.old_id].label,
                   "tag": r.transfer[m.old_id].tag, "verified": m.verified} for m in r.mappings])
