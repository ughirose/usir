#!/usr/bin/env python3
"""FAW-IMPL supplement fetch: download planned works (primary, then backup) with heldout3 filters, cache as data/aozora_cache/fawsup_{aid}_{wid}.txt,
write data/aozora_faw_supplement_manifest.json. Existing scripts/fetch_aozora.py is reused unchanged."""
import csv, hashlib, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from fetch_aozora import fetch_clean_text  # noqa
D = REPO / "data"; C = D / "aozora_cache"
MINC, MAXC, MINP, MINPAIR, MINPC, PER = 6000, 200000, 30, 25, 30, 2
plan = json.loads((Path(__file__).with_name("supplement_plan.json")).read_text(encoding="utf-8"))
idx = {}
with (C / "_index.csv").open(encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f): idx[(r["人物ID"].zfill(6), r["作品ID"].zfill(6))] = r
man, skipped = [], []
for au in plan["authors"]:
    got = 0
    for w in au["works"]:
        if got >= PER: break
        aid, wid = au["author_id"], w["work_id"]; row = idx[(aid, wid)]
        p = C / f"fawsup_{aid}_{wid}.txt"
        try:
            if p.is_file(): text, zu = p.read_text(encoding="utf-8"), w["zip_url"]
            else:
                res = fetch_clean_text(row); text, zu = res["text"], res["zip_url"]; p.write_text(text, encoding="utf-8")
        except Exception as ex:
            skipped.append(dict(author_id=aid, work_id=wid, title=w["title"], reason=f"fetch_error: {ex}")); continue
        n = len(text); paras = [l for l in text.split("\n") if len(l.strip()) >= MINPC]
        why = None
        if n < MINC: why = f"too_short<{MINC}"
        elif n > MAXC: why = f"too_long>{MAXC}"
        elif len(paras) < MINP or len(paras) - 1 < MINPAIR: why = "insufficient_paragraphs"
        if why:
            p.unlink(missing_ok=True); skipped.append(dict(author_id=aid, work_id=wid, title=w["title"], char_count=n, reason=why)); continue
        got += 1
        man.append(dict(author_id=aid, author_ja=au["author_ja"], title=w["title"], work_id=wid, zip_url=zu, card_url=w["card_url"],
                        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(), char_count=n, genre="faw_supplement"))
        print(aid, au["author_ja"], w["title"], wid, n, flush=True)
(D / "aozora_faw_supplement_manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
(Path(__file__).with_name("supplement_skipped.json")).write_text(json.dumps(skipped, ensure_ascii=False, indent=1), encoding="utf-8")
import collections
c = collections.Counter(e["author_id"] for e in man)
print("works", len(man), "authors", len(c), "authors>=2", sum(v >= 2 for v in c.values()), "skipped", len(skipped))
