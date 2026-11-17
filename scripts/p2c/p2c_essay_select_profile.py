#!/usr/bin/env python3
"""P-2c Task A2: essay register trial (NDC 914-916) + technical-document availability scan (metadata only). Frozen in STATUS_p2c.md 登録 A2."""
from __future__ import annotations
import csv
import hashlib
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in ("python", "scripts", "experiments", "scripts/p2b"):
    sys.path.insert(0, str(REPO / p))
import p2b_select as S  # noqa: E402
from fetch_aozora import _death_year, fetch_clean_text  # noqa: E402
from scale_separated_level3 import extract_ortho24_kasane8 as FN  # noqa: E402
from usir_py.v5 import split_work_into_scenes  # noqa: E402

SEED, TARGET, MAXW, MINSC = 20261417, 8, 3, 12
NDC_ESSAY = re.compile(r"(?:^|\s)(?:NDC\s*)?91[456](?:\.\d+)?(?:\s|$)")
NDC_TECH = re.compile(r"(?:^|\s)(?:NDC\s*)?[45]\d\d(?:\.\d+)?(?:\s|$)")
OUT_M = REPO / "data" / "aozora_p2c_essay_manifest.json"
OUT_N = REPO / "data" / "p2c_essay_scene_profiles.npz"
OUT_R = REPO / "scripts" / "p2c" / "p2c_essay_results.json"
km = S.person_key_map()


def key(a):
    return km.get(a, a)


def pool(rx):
    by = defaultdict(list)
    with S.INDEX.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("作品著作権フラグ") != "なし" or r.get("文字遣い種別") != "新字新仮名" or r.get("役割フラグ") != "著者":
                continue
            n = r.get("分類番号", "")
            if not rx.search(n) or "K" + n.split()[-1] in n and False:
                continue
            dy = _death_year(r)
            if dy is None or not (1946 <= dy <= 1955):
                continue
            u = r.get("テキストファイルURL", "").strip()
            if not u.startswith("https://www.aozora.gr.jp/cards/") or not u.endswith(".zip"):
                continue
            by[r["人物ID"].zfill(6)].append(r)
    g = defaultdict(list)
    for a in by:
        g[key(a)].append(a)
    return {min(v): [r for i in sorted(v) for r in by[i]] for v in g.values()}


tech = pool(NDC_TECH)
tech_res = {"authors": len(tech), "authors_ge2_works": sum(len(v) >= 2 for v in tech.values()), "works": sum(len(v) for v in tech.values())}
ess = {a: v for a, v in pool(NDC_ESSAY).items() if len(v) >= 2}
fic_keys = set()
for f in ("aozora_p2b_manifest", "aozora_p2c_manifest"):
    for w in json.loads((REPO / "data" / f"{f}.json").read_text(encoding="utf-8"))["works"]:
        fic_keys.add(key(w["author_id"]))
rng = random.Random(SEED)
authors = sorted(ess)
rng.shuffle(authors)
for a in authors:
    ess[a].sort(key=lambda r: r["作品ID"])
    rng.shuffle(ess[a])
acc, rej, nauth = [], [], 0
for aid in authors:
    if nauth >= TARGET:
        break
    got, titles, sc = [], set(), 0
    for r in ess[aid]:
        if len(got) >= MAXW:
            break
        wid, title = r["作品ID"].zfill(6), r["作品名"]
        base = S._SPLIT_SUFFIX.sub("", title).strip()
        if base in titles:
            continue
        cp = REPO / "data" / "aozora_cache" / f"p2cA2_{aid}_{wid}.txt"
        try:
            if cp.is_file():
                text = cp.read_text(encoding="utf-8")
                zurl = r["テキストファイルURL"]
            else:
                res = fetch_clean_text(r)
                text, zurl = res["text"], res["zip_url"]
                cp.write_text(text, encoding="utf-8")
                time.sleep(0.4)
        except Exception:  # noqa: BLE001
            continue
        paras = [x for x in (y.strip() for y in text.split("\n")) if len(x) >= S.MIN_PARA_CHARS]
        if not (S.MIN_CHARS <= len(text) <= S.MAX_CHARS) or len(paras) < S.MIN_PARAS:
            continue
        ns = len(split_work_into_scenes(text))
        got.append({"author_id": aid, "author_ja": f"{r['姓']}{r['名']}", "work_id": wid, "title": title, "ndc": r["分類番号"],
                    "card_url": r["図書カードURL"], "zip_url": zurl, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "char_count": len(text), "n_scenes": ns, "also_in_fiction_corpus": key(aid) in fic_keys})
        titles.add(base)
        sc += ns
    if len(got) >= 2 and sc >= MINSC:
        acc.extend(got)
        nauth += 1
    else:
        rej.append({"author_id": aid, "works_ok": len(got), "scenes": sc})
OUT_M.write_text(json.dumps({"stage": "P-2c A2", "seed": SEED, "n_authors": nauth, "n_works": len(acc), "works": acc, "rejected": rej,
                             "candidate_authors_ge2_works": len(authors), "technical_scan_ndc_4xx_5xx": tech_res},
                            indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("essay authors", nauth, "works", len(acc), "scenes", sum(w["n_scenes"] for w in acc), "tech", tech_res, flush=True)
if nauth == 0:
    sys.exit(0)

A, W, X = [], [], []
for w in acc:
    t = (REPO / "data" / "aozora_cache" / f"p2cA2_{w['author_id']}_{w['work_id']}.txt").read_text(encoding="utf-8")
    for s in split_work_into_scenes(t):
        A.append(w["author_id"])
        W.append(f"{w['author_id']}_{w['work_id']}")
        X.append(FN(s))
X = np.stack(X)
A = np.array(A)
W = np.array(W)
np.savez_compressed(OUT_N, author=A, work=W, X=X)
XF = np.load(REPO / "data" / "p2c_scene_profiles.npz")["X"]


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


muF, sdF = XF.mean(0), np.maximum(XF.std(0, ddof=1), 1e-6)
muE, sdE = X.mean(0), np.maximum(X.std(0, ddof=1), 1e-6)


def split_rxx(Xa, Wa, mu, sd):
    Z = (Xa - mu) / sd
    cnt = [0, 0]
    sm = [0, 0]
    for w, k in sorted(((w, int((Wa == w).sum())) for w in sorted(set(Wa))), key=lambda t: (-t[1], t[0])):
        g = 0 if cnt[0] <= cnt[1] else 1
        cnt[g] += k
        sm[g] = sm[g] + Z[Wa == w].sum(0)
    return cos(sm[0] / cnt[0], sm[1] / cnt[1])


rx, rx_f = {}, {}
for a in sorted(set(A)):
    m = A == a
    if len(set(W[m])) >= 2:
        rx[a] = split_rxx(X[m], W[m], muE, sdE)
        rx_f[a] = split_rxx(X[m], W[m], muF, sdF)
dsd = (muE - muF) / sdF
res = {"n_scenes": int(len(X)), "n_authors": nauth, "n_works": len(acc), "technical_scan": tech_res,
       "R_xx_work_split_essay_register_reference": {a: float(v) for a, v in rx.items()},
       "R_xx_work_split_median_essay_ref": float(np.median(list(rx.values()))),
       "R_xx_work_split_fiction_reference": {a: float(v) for a, v in rx_f.items()},
       "R_xx_work_split_median_fiction_ref": float(np.median(list(rx_f.values()))),
       "essay_minus_fiction_mean_shift_sd_rms": float(np.sqrt((dsd ** 2).mean())),
       "essay_minus_fiction_mean_shift_sd_max_abs": float(np.abs(dsd).max()),
       "n_essay_authors_also_in_fiction_corpus": len({w["author_id"] for w in acc if w["also_in_fiction_corpus"]}),
       "note": "descriptive only; author effects are not separated from register effects"}
OUT_R.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if not k.startswith("R_xx_work_split_essay") and not k.startswith("R_xx_work_split_fiction")}, ensure_ascii=False, indent=1))
print("rxx essay ref", {a: round(v, 3) for a, v in rx.items()})
