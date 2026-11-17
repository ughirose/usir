#!/usr/bin/env python3
"""P-2c A1/A2 pool size (metadata only, no fetching, no text). Excludes existing manifests + P-2b authors (same-person rule)."""
import sys, json, csv
from pathlib import Path
from collections import defaultdict
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/"scripts")); sys.path.insert(0, str(REPO/"scripts"/"p2b"))
import p2b_select as S
from fetch_aozora import _death_year
import re

def pool(ndc_re, ex_w, ex_a):
    km = S.person_key_map(); ex_keys = {km[a] for a in ex_a if a in km}
    by = defaultdict(list)
    with S.INDEX.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("作品著作権フラグ")!="なし" or r.get("文字遣い種別")!="新字新仮名" or r.get("役割フラグ")!="著者": continue
            ndc=r.get("分類番号","")
            if f"K{ndc_re}" in ndc and False: pass
            if not ndc_re.search(ndc): continue
            dy=_death_year(r)
            if dy is None or not (1946<=dy<=1955): continue
            u=r.get("テキストファイルURL","").strip()
            if not u.startswith("https://www.aozora.gr.jp/cards/") or not u.endswith(".zip"): continue
            wid,aid=r["作品ID"].zfill(6),r["人物ID"].zfill(6)
            if wid in ex_w or aid in ex_a or km.get(aid) in ex_keys: continue
            by[aid].append(r)
    ids=defaultdict(list)
    for a in by: ids[km.get(a,a)].append(a)
    return {min(v):[r for i in sorted(v) for r in by[i]] for v in ids.values()}

ex_w, ex_a = S.existing_ids()
m = json.loads((REPO/"data"/"aozora_p2b_manifest.json").read_text(encoding="utf-8"))
p2b_w={w["work_id"] for w in m["works"]}; p2b_a={w["author_id"] for w in m["works"]}
print("existing manifests authors", len(ex_a), "works", len(ex_w), "| P-2b authors", len(p2b_a), "works", len(p2b_w))
for label, rx in [("913", S._NDC), ("914-916", re.compile(r"(?:^|\s)(?:NDC\s*)?91[456](?:\.\d+)?(?:\s|$)"))]:
    p = pool(rx, ex_w|p2b_w, ex_a|p2b_a)
    dist=defaultdict(int)
    for a,v in p.items(): dist[min(len(v),6)]+=1
    print(label, "candidate authors", len(p), "works", sum(len(v) for v in p.values()), "authors with >=2 works", sum(len(v)>=2 for v in p.values()), "dist(1..6+)", dict(sorted(dist.items())))
