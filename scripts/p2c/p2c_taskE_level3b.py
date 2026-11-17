#!/usr/bin/env python3
"""P-2c Task E: Level 3b (cos(dz, u_spec)) with the centroid reference corpus extended toward the 15 v1.1 authors.
Parameters = STATUS_p2c.md 登録 E (frozen). Measurement only; Gate 3b freeze unchanged. Raw text is cached in data/aozora_cache/ (gitignored)
and never written out; numbers only. Step 0 re-implements the P-1 3b computation and must reproduce P-1 (n=49, p=0.390 / 0.361) before the extension is run."""
from __future__ import annotations
import csv
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
for p in ("python", "scripts", "experiments", "scripts/p2b"):
    sys.path.insert(0, str(REPO / p))
from scale_separated_level3 import extract_ortho24_kasane8 as FN, cosine_sim  # noqa: E402
from fable_v5_grand_shootout import load_works  # noqa: E402
from usir_py.v5 import AozoraCorpusSplitter, split_work_into_scenes  # noqa: E402
from fetch_aozora import fetch_clean_text  # noqa: E402
import p2b_select as S  # noqa: E402

DS = REPO / "data/usir_v5_eval/real_revision_pairs_v1_1.jsonl"
OUT = REPO / "scripts/p2c/p2c_taskE_results.json"
MAN_OUT = REPO / "data/aozora_p2c_taskE_centroid_manifest.json"
SEED_SEL = 20261617
MAX_W, MIN_C, MAX_C, MIN_SCENES = 4, 3000, 300000, 6

rows = [json.loads(line) for line in DS.read_text(encoding="utf-8").splitlines() if line.strip()]
P = [r for r in rows if r["same_orthography"]]
assert len(rows) == 439 and len(P) == 405
authors15 = sorted({r["author"] for r in rows})
v11_work_ids = set()
ctypes = defaultdict(set)
for r in P:
    for ed in ("edition_a", "edition_b"):
        v11_work_ids.add(r["provenance"][ed]["work_id"].zfill(6))
        ctypes[r["author"]].add(r["provenance"][ed]["char_type"])
for r in rows:
    for ed in ("edition_a", "edition_b"):
        v11_work_ids.add(r["provenance"][ed]["work_id"].zfill(6))

# ---------- reference corpus (P-1): 106 works
works = load_works()
splitter = AozoraCorpusSplitter()
name2id = {}
for f in ("aozora_manifest", "aozora_heldout_manifest", "aozora_heldout2_manifest"):
    for e in json.loads((REPO / "data" / f"{f}.json").read_text(encoding="utf-8")):
        name2id[e["author_ja"]] = e["author_id"]
id2name = {v: k for k, v in name2id.items()}
scenes_by_aid = {}
for w in works:
    scenes_by_aid.setdefault(w["author_id"], []).extend(split_work_into_scenes(splitter.get_cached_text(w))[:3])
cent_p1, allv = {}, []
for aid, scs in scenes_by_aid.items():
    vs = [FN(s) for s in scs]
    cent_p1[aid] = np.mean(vs, 0)
    allv.extend(vs)
cmean = np.mean(allv, 0)
dim = len(cmean)
cent_p1_name = {id2name[a]: c for a, c in cent_p1.items() if a in id2name and id2name[a] in authors15}
overlap_106 = sorted({w["work_id"].zfill(6) for w in works} & v11_work_ids)

# ---------- extension: fetch additional works for the other authors
def person_rows(name):
    out = []
    with S.INDEX.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("役割フラグ") == "著者" and f"{r['姓']}{r['名']}" == name and r.get("作品著作権フラグ") == "なし":
                u = r.get("テキストファイルURL", "").strip()
                if u.startswith("https://www.aozora.gr.jp/cards/") and u.endswith(".zip"):
                    out.append(r)
    return out


def select_and_load():
    rng = random.Random(SEED_SEL)
    sel, log = {}, {}
    for name in authors15:
        if name in cent_p1_name:
            log[name] = {"status": "P-1 centroid retained (106-work corpus)"}
            continue
        cand = [r for r in person_rows(name) if r["作品ID"].zfill(6) not in v11_work_ids and r.get("文字遣い種別") in ctypes[name]]
        cand.sort(key=lambda r: r["作品ID"])
        rng.shuffle(cand)
        got, titles, skipped = [], set(), []
        for r in cand:
            if len(got) >= MAX_W:
                break
            wid, title = r["作品ID"].zfill(6), r["作品名"]
            base = S._SPLIT_SUFFIX.sub("", title).strip()
            if base in titles:
                skipped.append((wid, "series_duplicate"))
                continue
            cp = REPO / "data" / "aozora_cache" / f"p2cE_{r['人物ID'].zfill(6)}_{wid}.txt"
            try:
                if cp.is_file():
                    text = cp.read_text(encoding="utf-8")
                else:
                    text = fetch_clean_text(r)["text"]
                    cp.write_text(text, encoding="utf-8")
                    time.sleep(0.4)
            except Exception as exc:  # noqa: BLE001
                skipped.append((wid, f"fetch_error:{type(exc).__name__}"))
                continue
            if not (MIN_C <= len(text) <= MAX_C):
                skipped.append((wid, f"length:{len(text)}"))
                continue
            got.append({"work_id": wid, "author_id": r["人物ID"].zfill(6), "title": title, "char_type": r["文字遣い種別"], "char_count": len(text),
                        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "card_url": r["図書カードURL"]})
            titles.add(base)
        sel[name] = got
        log[name] = {"n_candidates": len(cand), "n_selected": len(got), "skipped": skipped[:10], "char_types_allowed": sorted(ctypes[name])}
    return sel, log


sel, sel_log = select_and_load()
cent_ext = dict(cent_p1_name)  # v1.1 authors with a centroid (name-keyed)
cent_pool_p1 = {("id", a): c for a, c in cent_p1.items()}  # P-1: ALL 106-corpus authors form the 'other authors' pool
excluded = {}
new_cent = {}
for name, got in sel.items():
    vs = []
    for g in got:
        text = (REPO / "data" / "aozora_cache" / f"p2cE_{g['author_id']}_{g['work_id']}.txt").read_text(encoding="utf-8")
        vs.extend(FN(s) for s in split_work_into_scenes(text)[:3])
    if len(vs) >= MIN_SCENES:
        cent_ext[name] = np.mean(vs, 0)
        new_cent[name] = cent_ext[name]
        sel_log[name]["n_centroid_scenes"] = len(vs)
    else:
        excluded[name] = {"reason": f"centroid scenes {len(vs)} < {MIN_SCENES}"}
        sel_log[name]["n_centroid_scenes"] = len(vs)
MAN_OUT.write_text(json.dumps({"stage": "P-2c Task E", "seed": SEED_SEL, "selection": {k: v for k, v in sel.items()}, "log": sel_log, "excluded_authors": excluded,
                               "v11_overlap_with_106_corpus_work_ids": overlap_106}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------- Level 3b (same definition as scripts/p1/p1_level3_v11.py)
def wil(x, y):
    dd = np.asarray(x) - np.asarray(y)
    s = wilcoxon(dd, alternative="two-sided")
    return {"W": float(s.statistic), "p_two": float(s.pvalue), "p_bonf_k4": min(1.0, 4 * float(s.pvalue)), "n": len(dd)}


def cluster_ci(rs, ka, kb, B=10000, seed=7):
    g = {}
    for r in rs:
        g.setdefault(r["wg"], []).append(r[ka] - r[kb])
    ks = list(g)
    b = np.random.default_rng(seed)
    out = [np.concatenate([g[ks[i]] for i in b.integers(0, len(ks), len(ks))]).mean() for _ in range(B)]
    return {"mean_diff": float(np.mean(np.concatenate(list(g.values())))), "ci95_lo": float(np.percentile(out, 2.5)), "ci95_hi": float(np.percentile(out, 97.5)), "n_clusters": len(ks)}


def run_3b(pool, own):
    """pool: key -> centroid (all authors; 'other authors' = pool minus the pair's own key). own: author name -> pool key."""
    rng = np.random.default_rng(42)
    recs = []
    for r in P:
        zb, za = FN(r["text_before"]), FN(r["text_after"])
        dz = za - zb
        ndz = float(np.linalg.norm(dz))
        vc = cmean - zb
        nvc = np.linalg.norm(vc)
        ucorp = vc / nvc if nvc > 1e-12 else np.zeros(dim)
        vr = rng.normal(0, 1, dim)  # keep the P-1 draw order
        _ = float(np.mean([cosine_sim(rng.normal(0, 1, dim), ucorp) for _ in range(200)]))
        rec = {"wg": r["work_group_id"], "author": r["author"]}
        name = r["author"]
        if name in own:
            vt = pool[own[name]] - zb
            nvt = np.linalg.norm(vt)
            if nvt > 1e-12:
                vp = vt - float(vt @ ucorp) * ucorp
                nvp = np.linalg.norm(vp)
                us = vp / nvp if nvp > 1e-12 else np.zeros(dim)
                rec["rho_perp"] = float(nvp / nvt)
                rec["cos_spec"] = cosine_sim(dz, us)
                vr2 = rng.normal(0, 1, dim)
                vr2 = vr2 / np.linalg.norm(vr2) * ndz
                rec["cos_spec_rand"] = cosine_sim(vr2, us)
                _ = float(np.mean([cosine_sim(rng.normal(0, 1, dim), us) for _ in range(200)]))
                oc = []
                for on, c in pool.items():
                    if on == own[name]:
                        continue
                    vo = c - zb
                    nn = np.linalg.norm(vo)
                    if nn > 1e-12:
                        q = vo - float(vo @ ucorp) * ucorp
                        nq = np.linalg.norm(q)
                        oc.append(cosine_sim(dz, q / nq if nq > 1e-12 else np.zeros(dim)))
                rec["cos_spec_other"] = float(np.mean(oc)) if oc else 0.0
        recs.append(rec)
    S_ = [r for r in recs if "cos_spec" in r]
    out = {"n": len(S_), "n_work_groups": len({r["wg"] for r in S_}), "authors_with_centroid": sorted({r["author"] for r in S_}),
           "authors_without_centroid": sorted({r["author"] for r in recs} - {r["author"] for r in S_}), "n_excluded_pairs": len(recs) - len(S_),
           "median_rho_perp": float(np.median([r["rho_perp"] for r in S_])), "n_quasi_parallel_rho_lt_0.10": int(sum(r["rho_perp"] < 0.10 for r in S_)),
           "mean_cos_spec": float(np.mean([r["cos_spec"] for r in S_])), "mean_cos_rand": float(np.mean([r["cos_spec_rand"] for r in S_])),
           "mean_cos_other": float(np.mean([r["cos_spec_other"] for r in S_])),
           "vs_random": wil([r["cos_spec"] for r in S_], [r["cos_spec_rand"] for r in S_]), "vs_other": wil([r["cos_spec"] for r in S_], [r["cos_spec_other"] for r in S_]),
           "cluster_ci_vs_random": cluster_ci(S_, "cos_spec", "cos_spec_rand"), "cluster_ci_vs_other": cluster_ci(S_, "cos_spec", "cos_spec_other")}
    out["by_author"] = {a: {"n": sum(r["author"] == a for r in S_), "mean_cos_spec": float(np.mean([r["cos_spec"] for r in S_ if r["author"] == a])),
                            "mean_cos_other": float(np.mean([r["cos_spec_other"] for r in S_ if r["author"] == a]))} for a in sorted({r["author"] for r in S_})}
    def lvl(key, ka, kb):
        g = {}
        for r in S_:
            g.setdefault(r[key], []).append(r[ka] - r[kb])
        m = np.array([np.mean(v) for v in g.values()])
        w = wilcoxon(m, alternative="two-sided")
        return {"n_units": len(m), "W": float(w.statistic), "p_two": float(w.pvalue), "p_bonf_k4": min(1.0, 4 * float(w.pvalue)), "n_pos": int((m > 0).sum()), "mean_of_unit_means": float(m.mean())}
    out["sens_cluster_level_vs_random"] = lvl("wg", "cos_spec", "cos_spec_rand")
    out["sens_cluster_level_vs_other"] = lvl("wg", "cos_spec", "cos_spec_other")
    out["sens_author_level_vs_random"] = lvl("author", "cos_spec", "cos_spec_rand")
    out["sens_author_level_vs_other"] = lvl("author", "cos_spec", "cos_spec_other")
    out["leave_one_author_out_vs_other"] = {}
    for a in sorted({r["author"] for r in S_}):
        sub = [r for r in S_ if r["author"] != a]
        if len(sub) >= 10:
            out["leave_one_author_out_vs_other"][a] = wil([r["cos_spec"] for r in sub], [r["cos_spec_other"] for r in sub])
    return out


own_p1 = {nm: ("id", name2id[nm]) for nm in cent_p1_name}
pool_ext = dict(cent_pool_p1)
own_ext = dict(own_p1)
for nm, c in new_cent.items():
    pool_ext[("name", nm)] = c
    own_ext[nm] = ("name", nm)
res = {"step0_reproduce_P1": run_3b(cent_pool_p1, own_p1), "extended": run_3b(pool_ext, own_ext), "n_other_pool_p1": len(cent_pool_p1), "n_other_pool_ext": len(pool_ext),
       "v11_work_overlap_with_106_corpus": {"n_overlap_work_ids": len(overlap_106), "ids": overlap_106, "n_v11_work_ids": len(v11_work_ids)},
       "centroid_selection": sel_log, "excluded_authors": excluded, "n_authors_with_centroid": len(cent_ext)}
OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
for k in ("step0_reproduce_P1", "extended"):
    r = res[k]
    print(k, "n", r["n"], "clusters", r["n_work_groups"], "authors", len(r["authors_with_centroid"]), "rho_perp_med", round(r["median_rho_perp"], 3),
          "vs_rand p", round(r["vs_random"]["p_two"], 4), "bonf", round(r["vs_random"]["p_bonf_k4"], 4), "vs_other p", round(r["vs_other"]["p_two"], 4), "bonf", round(r["vs_other"]["p_bonf_k4"], 4))
    print("   ci_rand", r["cluster_ci_vs_random"], "\n   ci_other", r["cluster_ci_vs_other"])
print("overlap 106 vs v1.1 work ids:", overlap_106)
print({k: (v.get("n_selected"), v.get("n_centroid_scenes")) for k, v in sel_log.items()}, excluded)
