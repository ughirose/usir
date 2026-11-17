#!/usr/bin/env python3
"""P-2c: candidate-pool sizes under relaxed conditions (metadata only; counts, no text)."""
import sys, csv, re
from pathlib import Path
from collections import defaultdict
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/"scripts")); sys.path.insert(0, str(REPO/"scripts"/"p2b"))
import p2b_select as S
from fetch_aozora import _death_year
import json
km = S.person_key_map(); ex_w, ex_a = S.existing_ids()
m = json.loads((REPO/"data"/"aozora_p2b_manifest.json").read_text(encoding="utf-8"))
p2b_a={w["author_id"] for w in m["works"]}; p2b_w={w["work_id"] for w in m["works"]}
def count(lo, hi, excl_existing, ndc=S._NDC, chartype="新字新仮名"):
    exa = (ex_a if excl_existing else set())|p2b_a; exw=(ex_w if excl_existing else set())|p2b_w
    keys={km[a] for a in exa if a in km}; by=defaultdict(list)
    with S.INDEX.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("作品著作権フラグ")!="なし" or r.get("役割フラグ")!="著者": continue
            if chartype and r.get("文字遣い種別")!=chartype: continue
            n=r.get("分類番号","")
            if "K913" in n or not ndc.search(n): continue
            dy=_death_year(r)
            if dy is None or not(lo<=dy<=hi): continue
            u=r.get("テキストファイルURL","").strip()
            if not u.startswith("https://www.aozora.gr.jp/cards/") or not u.endswith(".zip"): continue
            w,a=r["作品ID"].zfill(6),r["人物ID"].zfill(6)
            if w in exw or a in exa or km.get(a) in keys: continue
            by[a].append(r)
    ids=defaultdict(list)
    for a in by: ids[km.get(a,a)].append(a)
    mg=[sum((by[i] for i in v),[]) for v in ids.values()]
    return len(mg), sum(len(x)>=2 for x in mg)
for label,args in [("baseline 1946-55, excl existing",(1946,1955,True)),
                   ("1946-55, INCLUDE existing-manifest authors (P-2b still excluded)",(1946,1955,False)),
                   ("1940-1960, excl existing",(1940,1960,True)),
                   ("1936-1965, excl existing",(1936,1965,True)),
                   ("1946-55 excl existing, any chartype",(1946,1955,True,S._NDC,None))]:
    print(label, "-> candidate authors, authors>=2 works:", count(*args))
