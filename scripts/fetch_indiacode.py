"""Download section-level text of the six central criminal acts from India Code.

India Code (indiacode.gov.in) is DSpace 7. Each section is an item in the SECTION collection carrying
``dc.identifier.section_number``, ``dc.title`` (heading) and ``dc.identifier.section_page_note`` (text, HTML).

python scripts/fetch_indiacode.py   → data/raw/statutes/{IPC,CrPC,IEA,BNS,BNSS,BSA}.indiacode.jsonl
Polite: one request at a time, a pause between requests."""
from __future__ import annotations

import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://indiacode.gov.in/server/api"
UA = {"User-Agent": "LexBridge-research/0.1 (statute text for academic research)"}
ACTS = {
    "BNS": r"^The Bharatiya Nyaya Sanhita, 2023$",
    "BNSS": r"^The Bharatiya Nagarik Suraksha Sanhita, 2023$",
    "BSA": r"^The Bharatiya Sakshya Adhiniyam, 2023$",
    "IPC": r"^The Indian Penal Code(, 1860)?$",
    "CrPC": r"^The Code of Criminal Procedure, 1973$",
    "IEA": r"^The Indian Evidence Act, 1872$",
}
QUERY = {"BNS": "Bharatiya Nyaya Sanhita", "BNSS": "Bharatiya Nagarik Suraksha Sanhita",
         "BSA": "Bharatiya Sakshya Adhiniyam", "IPC": "Indian Penal Code",
         "CrPC": "Code of Criminal Procedure, 1973", "IEA": "Indian Evidence Act"}
PAUSE = 0.7


def get(params: dict, path: str = "discover/search/objects"):
    time.sleep(PAUSE)
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return json.loads(r.read())
        except Exception:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))


def md(it: dict, key: str) -> str:
    v = it["metadata"].get(key)
    return v[0]["value"] if v else ""


def clean(h: str) -> str:
    t = re.sub(r"(?i)<br\s*/?>|</p>|</div>", "\n", h or "")
    t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    t = re.sub(r"[ \t\xa0]+", " ", t)
    return re.sub(r"\s*\n\s*", "\n", t).strip()


def find_act(code: str) -> dict | None:
    for page in range(5):
        d = get({"query": f'{QUERY[code]} AND dc.identifier.collection:ACT AND dc.identifier.state_name:CENTRAL',
                 "size": 40, "page": page})
        objs = d["_embedded"]["searchResult"]["_embedded"]["objects"]
        for o in objs:
            it = o["_embedded"]["indexableObject"]
            if re.match(ACTS[code], it["name"].strip(), re.I):
                return it
        if not objs:
            return None
    return None


def sections(act_id: str):
    page, total = 0, None
    while total is None or page < total:
        d = get({"query": f'dc.identifier.act_id:"{act_id}" AND dc.identifier.collection:SECTION',
                 "size": 100, "page": page})
        sr = d["_embedded"]["searchResult"]
        total = sr["page"]["totalPages"]
        for o in sr["_embedded"]["objects"]:
            yield o["_embedded"]["indexableObject"]
        page += 1


# The old codes were repealed on 1 July 2024; India Code keeps them as items with the official PDF.
REPEALED = {"IPC": '"Rep., Act 45 of 2023"', "CrPC": '"Rep., Act 46 of 2023"', "IEA": '"Rep., Act 47 of 2023"'}
REPEALED_TITLE = {"IPC": r"Indian Penal Code", "CrPC": r"Code of Criminal Procedure", "IEA": r"Indian Evidence Act"}


def fetch_repealed(code: str, out: Path) -> str:
    d = get({"query": REPEALED[code], "size": 10})
    items = [o["_embedded"]["indexableObject"] for o in d["_embedded"]["searchResult"]["_embedded"]["objects"]]
    item = next((it for it in items if re.search(REPEALED_TITLE[code], it["name"])), None)
    if not item:
        raise LookupError(f"repealed item for {code} not found")
    bundles = get({}, f"core/items/{item['uuid']}/bundles")["_embedded"]["bundles"]
    saved = []
    for b in bundles:
        if b["name"] not in ("ORIGINAL", "TEXT"):
            continue
        time.sleep(PAUSE)
        with urllib.request.urlopen(urllib.request.Request(b["_links"]["bitstreams"]["href"], headers=UA), timeout=120) as r:
            bss = json.loads(r.read())["_embedded"]["bitstreams"]
        for bs in bss:
            time.sleep(PAUSE)
            with urllib.request.urlopen(urllib.request.Request(bs["_links"]["content"]["href"], headers=UA), timeout=300) as r:
                data = r.read()
            ext = ".pdf.txt" if b["name"] == "TEXT" else ".pdf"
            path = out / f"{code.lower()}.indiacode{ext}"
            path.write_bytes(data)
            saved.append(f"{path.name} ({len(data)} B)")
    return f"{code}: {item['name']} → {', '.join(saved)}"


def main(out_dir: str = "data/raw/statutes", only: str = "") -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    failed = 0
    for code in ACTS:
        if only and code not in only.split(","):
            continue
        try:
            if code in REPEALED:
                print(fetch_repealed(code, out))
                continue
            act = find_act(code)
            if not act:
                print(f"{code}: central act item not found")
                failed += 1
                continue
            act_id = md(act, "dc.identifier.act_id")
            rows = []
            for it in sections(act_id):
                rows.append({"code": code, "section": md(it, "dc.identifier.section_number").strip(),
                             "title": it["name"].strip().rstrip("."), "text": clean(md(it, "dc.identifier.section_page_note")),
                             "order": int(md(it, "dc.identifier.order_number") or 0),
                             "source": f"indiacode:{it['uuid']}", "act_id": act_id})
            rows.sort(key=lambda r: r["order"])
            p = out / f"{code}.indiacode.jsonl"
            p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            print(f"{code}: {act['name']} → {len(rows)} sections → {p}")
        except Exception as e:
            print(f"{code}: failed ({type(e).__name__}: {e})")
            failed += 1
    return failed


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
