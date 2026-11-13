#!/usr/bin/env python3
"""Stage B-17 Task 1: _index.csv から NDC 913 新字新仮名の新規著者・候補作品を調査する."""
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"c:\dev\usir")
existing_wids = set()
existing_aids = set()
for m in ["aozora_manifest.json", "aozora_heldout_manifest.json", "aozora_heldout2_manifest.json"]:
    arr = json.loads((ROOT / "data" / m).read_text(encoding="utf-8"))
    for e in arr:
        if m == "aozora_manifest.json" and e.get("genre") != "novel":
            continue
        existing_wids.add(e["work_id"].zfill(6))
        existing_aids.add(e["author_id"].zfill(6))

print(f"Existing novel works: {len(existing_wids)}, authors: {len(existing_aids)}")

by_author = defaultdict(list)
total_913 = 0
with (ROOT / "data/aozora_cache/_index.csv").open(encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        if r.get("作品著作権フラグ") != "なし":
            continue
        if r.get("文字遣い種別") != "新字新仮名":
            continue
        ndc = r.get("分類番号", "")
        if "913" not in ndc:
            continue
        if r.get("役割フラグ") != "著者":
            continue
        death = r.get("没年月日", "").strip()
        if death:
            try:
                if int(death[:4]) > 1967:
                    continue
            except ValueError:
                pass
        wid = r["作品ID"].zfill(6)
        aid = r["人物ID"].zfill(6)
        txt_url = r.get("テキストファイルURL", "").strip()
        if not txt_url or not txt_url.endswith(".zip"):
            continue
        total_913 += 1
        if wid in existing_wids:
            continue
        by_author[aid].append({
            "work_id": wid,
            "title": r["作品名"],
            "author_id": aid,
            "author_ja": r["姓"] + r["名"],
            "ndc": ndc,
            "char_type": r["文字遣い種別"],
            "card_url": r["図書カードURL"],
            "zip_url": txt_url,
            "death_date": death,
        })

new_authors = {a: ws for a, ws in by_author.items() if a not in existing_aids}
print(f"Total NDC 913 新字新仮名 (death<=1967, zip): {total_913}")
print(f"New authors available: {len(new_authors)}, total works by new authors: {sum(len(v) for v in new_authors.values())}")
multi_work_authors = [a for a, ws in new_authors.items() if len(ws) >= 2]
print(f"New authors with >=2 works: {len(multi_work_authors)}")
top_authors = sorted(new_authors.items(), key=lambda kv: -len(kv[1]))[:25]
for aid, ws in top_authors:
    print(f"  {aid} {ws[0]['author_ja']} (death={ws[0]['death_date'][:4]}): {len(ws)} works")
