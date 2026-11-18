"""FastAPI server: POST /ask, GET /section/{code}/{sec}, GET /map/{code}/{sec}, GET /health."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..baselines import get_system
from ..generate.prompts import DISCLAIMER
from ..index.graph import neighbors
from ..resources import load_resources
from ..schemas import normalize_section

app = FastAPI(title="LexBridge", description=f"Code-aware legal RAG for the IPC→BNS transition. {DISCLAIMER}")
CODES = {c.lower(): c for c in ["IPC", "CrPC", "IEA", "BNS", "BNSS", "BSA"]}


@lru_cache(maxsize=1)
def res():
    return load_resources()


@lru_cache(maxsize=8)
def system(name: str):
    return get_system(name, res())


class AskRequest(BaseModel):
    query: str
    offence_date: Optional[str] = None   # YYYY-MM-DD
    system: str = "lexbridge"            # lexbridge | no_rag | bm25 | dense | hybrid


def _code(code: str) -> str:
    c = CODES.get(code.lower())
    if not c:
        raise HTTPException(404, f"unknown code {code}")
    return c


@app.get("/health")
def health():
    r = res()
    return {"status": "ok", "sections": len(r.store), "mappings": len(r.mappings), "judgments": len(r.judgments)}


@app.post("/ask")
def ask(req: AskRequest):
    try:
        sys_ = system(req.system)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return sys_.answer(req.query, req.offence_date or None)


@app.get("/section/{code}/{sec}")
def section(code: str, sec: str):
    s = res().store.get(_code(code), sec)
    if not s:
        raise HTTPException(404, f"{code} {sec} not in the statute store")
    return s.model_dump()


@app.get("/map/{code}/{sec}")
def mapping(code: str, sec: str):
    r = res()
    sid = f"{_code(code)}_{normalize_section(sec)}"
    ms = r.table.lookup(sid)
    if not ms:
        raise HTTPException(404, f"no mapping for {sid}")
    out = []
    for m in ms:
        t = r.transfer.get(m.old_id)
        out.append({**m.model_dump(), "old_id": m.old_id, "new_id": m.new_id,
                    "transferability": t.__dict__ if t else None})
    return {"section": sid, "mappings": out, "graph": neighbors(r.graph, sid)}
