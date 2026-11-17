#!/usr/bin/env python3
"""FAW-IMPL supplement plan (deterministic; seed 20261114 = heldout3 rule). Builds the candidate list of 2nd/3rd works
for heldout3 authors absent from FAW/p2cE/p2cA2. Network: HEAD requests only (Content-Length) with --head. No text download here."""
import csv, json, random, re, sys, time, urllib.request
from collections import defaultdict
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from fetch_aozora import _death_year, USER_AGENT  # noqa
D = REPO / "data"
SEED, N_AUTHORS, PER_AUTHOR, BACKUP = 20261114, 12, 2, 2
NDC = re.compile(r"(?:^|\s)(?:NDC\s*)?913(?:\.\d+)?(?:\s|$)")
SPLIT = re.compile(r"[（(](?:上|中|下|前|後|続|[一二三四五六七八九十0-9]+)[）)]$")
def ids(obj, key, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key and isinstance(v, str): out.add(v.zfill(6))
            else: ids(v, key, out)
    elif isinstance(obj, list):
        for v in obj: ids(v, key, out)
used_w, used_a = set(), set()
for p in sorted(D.glob("aozora*manifest*.json")):
    j = json.loads(p.read_text(encoding="utf-8")); ids(j, "work_id", used_w)
cs = json.loads((REPO / "scripts/faw_impl/cohort_check_results.json").read_text(encoding="utf-8"))
excl_a = {r["author_id"] for r in cs["p2cE_rows"]} | {r["author_id"] for r in cs["p2cA2_rows"]}
faw_a = set()
for p in ("aozora_manifest.json", "aozora_p2b_manifest.json", "aozora_p2c_manifest.json"):
    j = json.loads((D / p).read_text(encoding="utf-8")); ids(j, "author_id", faw_a)
h3 = json.loads((D / "aozora_heldout3_manifest.json").read_text(encoding="utf-8"))
h3_by_a = {e["author_id"].zfill(6): e for e in h3}
eligible = sorted(a for a in h3_by_a if a not in faw_a and a not in {x.zfill(6) for x in excl_a})
pool = defaultdict(list)
with (D / "aozora_cache/_index.csv").open(encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        if r.get("作品著作権フラグ") != "なし" or r.get("文字遣い種別") != "新字新仮名" or r.get("役割フラグ") != "著者": continue
        n = r.get("分類番号", "")
        if "K913" in n or not NDC.search(n): continue
        dy = _death_year(r)
        if dy is None or dy < 1868 or dy > 1967: continue
        u = r.get("テキストファイルURL", "").strip()
        if not u.startswith("https://www.aozora.gr.jp/cards/") or not u.endswith(".zip"): continue
        w, a = r["作品ID"].zfill(6), r["人物ID"].zfill(6)
        if w in used_w or a not in eligible: continue
        pool[a].append(r)
rng = random.Random(SEED)
authors = sorted(eligible); rng.shuffle(authors)
plan = []
for a in authors:
    rows = sorted({r["作品ID"]: r for r in pool[a]}.values(), key=lambda r: r["作品ID"]); rng.shuffle(rows)
    base = {SPLIT.sub("", h3_by_a[a]["title"]).strip()}; picked = []
    for r in rows:
        b = SPLIT.sub("", r["作品名"]).strip()
        if b in base: continue
        base.add(b); picked.append(r)
    if len(picked) >= PER_AUTHOR: plan.append((a, picked))
sel = plan[:N_AUTHORS]
out = dict(seed=SEED, n_eligible_authors=len(eligible), n_authors_with_ge2_candidates=len(plan), authors=[])
for a, picked in sel:
    e = h3_by_a[a]
    out["authors"].append(dict(author_id=a, author_ja=f"{picked[0]['姓']}{picked[0]['名']}", existing_heldout3=dict(work_id=e["work_id"], title=e["title"]),
        works=[dict(work_id=r["作品ID"].zfill(6), title=r["作品名"], zip_url=r["テキストファイルURL"].strip(), card_url=r["図書カードURL"], role="primary" if i < PER_AUTHOR else "backup")
               for i, r in enumerate(picked[:PER_AUTHOR + BACKUP])]))
if "--head" in sys.argv:
    for au in out["authors"]:
        for w in au["works"]:
            req = urllib.request.Request(w["zip_url"], method="HEAD", headers={"User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(req, timeout=30) as rs: w["zip_bytes"] = int(rs.headers.get("Content-Length", -1))
            except Exception as ex: w["zip_bytes"] = f"err:{ex}"
            time.sleep(1.0)
(Path(__file__).with_name("supplement_plan.json")).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(out["n_eligible_authors"], out["n_authors_with_ge2_candidates"])
for au in out["authors"]:
    print(au["author_id"], au["author_ja"], "|", au["existing_heldout3"]["title"], "->", [(w["work_id"], w["title"], w["role"], w.get("zip_bytes")) for w in au["works"]])
