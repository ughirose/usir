#!/usr/bin/env python
"""Stage P-1 Task 1: Level 3a/3b on real_revision_pairs v1.1 Primary (same_orthography, n=405), 32D micro core
(ortho24_cont + kasane8_active). Reuses the B-8/B-12 feature extractor, centroid construction and cosine definitions.
Read-only for all frozen files. Not a re-rating of the frozen Level 3b grade."""
import json, sys, hashlib
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
REPO = Path(__file__).resolve().parents[2]
for p in ("python", "scripts", "experiments"): sys.path.insert(0, str(REPO / p))
import scale_separated_level3 as L3
from scale_separated_level3 import extract_ortho24_kasane8 as FN, cosine_sim
from fable_v5_grand_shootout import load_works
from usir_py.v5 import AozoraCorpusSplitter, split_work_into_scenes

DS = REPO / "data/usir_v5_eval/real_revision_pairs_v1_1.jsonl"
OUT = Path(__file__).resolve().parent / "p1_level3_v11.json"
rng = np.random.default_rng(42)
rows = [json.loads(l) for l in DS.read_text(encoding="utf-8").splitlines() if l.strip()]
P = [r for r in rows if r["same_orthography"]]
assert len(rows) == 439 and len(P) == 405
name2id = {}
for f in ("aozora_manifest", "aozora_heldout_manifest", "aozora_heldout2_manifest"):
    for e in json.loads((REPO / "data" / f"{f}.json").read_text(encoding="utf-8")): name2id[e["author_ja"]] = e["author_id"]

works = load_works(); splitter = AozoraCorpusSplitter()
scenes_by_aid = {}
for w in works: scenes_by_aid.setdefault(w["author_id"], []).extend(split_work_into_scenes(splitter.get_cached_text(w))[:3])
cent, allv = {}, []
for aid, scs in scenes_by_aid.items():
    vs = [FN(s) for s in scs]; cent[aid] = np.mean(vs, 0); allv.extend(vs)
cmean = np.mean(allv, 0); dim = len(cmean)

recs = []
for r in P:
    zb, za = FN(r["text_before"]), FN(r["text_after"]); dz = za - zb
    aid = name2id.get(r["author"]); ndz = float(np.linalg.norm(dz))
    vc = cmean - zb; nvc = np.linalg.norm(vc); ucorp = vc / nvc if nvc > 1e-12 else np.zeros(dim)
    rec = {"pair_id": r["pair_id"], "wg": r["work_group_id"], "author": r["author"], "aid": aid, "band": r["band"], "dz": dz,
           "cos_corp": cosine_sim(dz, ucorp)}
    vr = rng.normal(0, 1, dim); vr = vr / np.linalg.norm(vr) * ndz; rec["cos_corp_rand"] = cosine_sim(vr, ucorp)
    rec["cos_corp_rand_avg"] = float(np.mean([cosine_sim(rng.normal(0, 1, dim), ucorp) for _ in range(200)]))
    if aid in cent:
        vt = cent[aid] - zb; nvt = np.linalg.norm(vt)
        if nvt > 1e-12:
            vp = vt - float(vt @ ucorp) * ucorp; nvp = np.linalg.norm(vp)
            us = vp / nvp if nvp > 1e-12 else np.zeros(dim)
            rec["rho_perp"] = float(nvp / nvt); rec["cos_spec"] = cosine_sim(dz, us)
            vr2 = rng.normal(0, 1, dim); vr2 = vr2 / np.linalg.norm(vr2) * ndz; rec["cos_spec_rand"] = cosine_sim(vr2, us)
            rec["cos_spec_rand_avg"] = float(np.mean([cosine_sim(rng.normal(0, 1, dim), us) for _ in range(200)]))
            oc = []
            for oa, c in cent.items():
                if oa == aid: continue
                vo = c - zb; n = np.linalg.norm(vo)
                if n > 1e-12:
                    q = vo - float(vo @ ucorp) * ucorp; nq = np.linalg.norm(q)
                    oc.append(cosine_sim(dz, q / nq if nq > 1e-12 else np.zeros(dim)))
            rec["cos_spec_other"] = float(np.mean(oc)) if oc else 0.0
    recs.append(rec)

def wil(x, y):
    d = np.asarray(x) - np.asarray(y); nz = d[np.abs(d) > 1e-12]
    s = wilcoxon(d, alternative="two-sided"); return {"W": float(s.statistic), "p_two": float(s.pvalue), "p_bonf_k4": min(1.0, 4 * float(s.pvalue)), "n": len(d), "n_nonzero": int(len(nz))}

def cluster_ci(rs, key_a, key_b, B=10000, seed=7):
    """Percentile bootstrap CI of mean(a-b) resampling work_group_id clusters."""
    g = {}
    for r in rs: g.setdefault(r["wg"], []).append(r[key_a] - r[key_b])
    ks = list(g); b = np.random.default_rng(seed); out = []
    for _ in range(B):
        pick = b.integers(0, len(ks), len(ks)); vals = np.concatenate([g[ks[i]] for i in pick]); out.append(vals.mean())
    mean_all = float(np.mean(np.concatenate(list(g.values()))))
    return {"mean_diff": mean_all, "ci95_lo": float(np.percentile(out, 2.5)), "ci95_hi": float(np.percentile(out, 97.5)), "n_clusters": len(ks), "B": B}

def cluster_level(rs, ka, kb):
    g = {}
    for r in rs: g.setdefault(r["wg"], []).append(r[ka] - r[kb])
    m = np.array([np.mean(v) for v in g.values()]); s = wilcoxon(m, alternative="two-sided")
    return {"n_clusters": len(m), "W": float(s.statistic), "p_two": float(s.pvalue), "n_pos": int((m > 0).sum())}

res = {"dataset_sha256": hashlib.sha256(DS.read_bytes()).hexdigest().upper(), "n_primary": len(recs),
       "n_work_groups": len({r["wg"] for r in recs}), "n_authors": len({r["author"] for r in recs}),
       "bonferroni_family": "k=4 (repository convention, B-10 s0); raw two-sided p and p*4 reported; family = {3a vs random, 3b vs random, 3b vs other-authors, 3b vs random on rho_perp>=0.10 subset}"}
res["3a"] = {"n": len(recs), "mean_cos_corp": float(np.mean([r["cos_corp"] for r in recs])), "mean_cos_rand": float(np.mean([r["cos_corp_rand"] for r in recs])),
             "test_vs_random": wil([r["cos_corp"] for r in recs], [r["cos_corp_rand"] for r in recs]),
             "cluster_ci_vs_random": cluster_ci(recs, "cos_corp", "cos_corp_rand"),
             "sens_avg200_random": wil([r["cos_corp"] for r in recs], [r["cos_corp_rand_avg"] for r in recs]),
             "sens_cluster_level_wilcoxon": cluster_level(recs, "cos_corp", "cos_corp_rand_avg")}
S = [r for r in recs if "cos_spec" in r]
res["3b"] = {"n": len(S), "n_work_groups": len({r["wg"] for r in S}), "authors_with_centroid": sorted({r["author"] for r in S}),
             "authors_without_centroid_excluded": sorted({r["author"] for r in recs if "cos_spec" not in r}), "n_excluded_pairs": len(recs) - len(S),
             "median_rho_perp": float(np.median([r["rho_perp"] for r in S])), "n_quasi_parallel_rho_lt_0.10": int(sum(r["rho_perp"] < 0.10 for r in S)),
             "mean_cos_spec": float(np.mean([r["cos_spec"] for r in S])), "mean_cos_rand": float(np.mean([r["cos_spec_rand"] for r in S])),
             "mean_cos_other": float(np.mean([r["cos_spec_other"] for r in S])),
             "vs_random": wil([r["cos_spec"] for r in S], [r["cos_spec_rand"] for r in S]),
             "vs_other": wil([r["cos_spec"] for r in S], [r["cos_spec_other"] for r in S]),
             "cluster_ci_vs_random": cluster_ci(S, "cos_spec", "cos_spec_rand"), "cluster_ci_vs_other": cluster_ci(S, "cos_spec", "cos_spec_other"),
             "sens_avg200_random": wil([r["cos_spec"] for r in S], [r["cos_spec_rand_avg"] for r in S]),
             "sens_cluster_level_vs_other": cluster_level(S, "cos_spec", "cos_spec_other"),
             "sens_cluster_level_vs_random": cluster_level(S, "cos_spec", "cos_spec_rand_avg")}
NQ = [r for r in S if r["rho_perp"] >= 0.10]
res["3b_non_quasi_parallel"] = {"n": len(NQ), "vs_random": wil([r["cos_spec"] for r in NQ], [r["cos_spec_rand"] for r in NQ]),
                                "vs_other": wil([r["cos_spec"] for r in NQ], [r["cos_spec_other"] for r in NQ])}
res["3b_leave_one_author_out_vs_other"] = {}
for a in sorted({r["author"] for r in S}):
    sub = [r for r in S if r["author"] != a]
    if len(sub) >= 10: res["3b_leave_one_author_out_vs_other"][a] = {"n": len(sub), **wil([r["cos_spec"] for r in sub], [r["cos_spec_other"] for r in sub])}
res["by_author_3a_3b"] = {a: {"n": sum(r["author"] == a for r in recs), "mean_cos_corp": float(np.mean([r["cos_corp"] for r in recs if r["author"] == a])),
        "mean_cos_spec": (float(np.mean([r["cos_spec"] for r in S if r["author"] == a])) if any(r["author"] == a for r in S) else None)} for a in sorted({r["author"] for r in recs})}

# ---- Delta z_author split-half consistency (D11 Step 1)
sd = np.std(np.stack([r["dz"] for r in recs]), 0); sd[sd < 1e-12] = 1.0
def split_half(rs, scale, nsplit=2000, seed=11):
    g = np.random.default_rng(seed); X = np.stack([r["dz"] / scale for r in rs]); n = len(X); cs = []; null = []
    for _ in range(nsplit):
        perm = g.permutation(n); a, b = perm[: n // 2], perm[n // 2 :]
        cs.append(cosine_sim(X[a].mean(0), X[b].mean(0)))
        sg = g.choice([-1.0, 1.0], n)[:, None]; Y = X * sg
        null.append(cosine_sim(Y[a].mean(0), Y[b].mean(0)))
    cs, null = np.array(cs), np.array(null)
    return {"n_pairs": n, "mean_cos": float(cs.mean()), "median_cos": float(np.median(cs)), "ci95_lo": float(np.percentile(cs, 2.5)), "ci95_hi": float(np.percentile(cs, 97.5)),
            "null_signflip_mean_cos": float(null.mean()), "null_signflip_p95": float(np.percentile(null, 95)), "frac_splits_above_null_p95": float((cs > np.percentile(null, 95)).mean())}
auth = {}
for r in recs: auth.setdefault(r["author"], []).append(r)
res["split_half"] = {"min_pairs_required": 6, "n_resamples": 2000, "seed": 11, "per_author": {}, "per_author_zscored": {}}
for a, rs in sorted(auth.items(), key=lambda kv: -len(kv[1])):
    if len(rs) >= 6:
        res["split_half"]["per_author"][a] = {"n_work_groups": len({r["wg"] for r in rs}), **split_half(rs, np.ones(dim))}
        res["split_half"]["per_author_zscored"][a] = {"n_work_groups": len({r["wg"] for r in rs}), **split_half(rs, sd)}
res["split_half"]["authors_below_min"] = {a: len(rs) for a, rs in auth.items() if len(rs) < 6}
wgs = {}
for a, rs in auth.items():
    gs = sorted({r["wg"] for r in rs})
    if len(gs) >= 2 and len(rs) >= 6:
        X = {g_: np.stack([r["dz"] for r in rs if r["wg"] == g_]).mean(0) for g_ in gs}; ks = list(X); g = np.random.default_rng(5); cs = []
        for _ in range(500):
            p = g.permutation(len(ks)); h = len(ks) // 2; cs.append(cosine_sim(np.mean([X[ks[i]] for i in p[:h]], 0), np.mean([X[ks[i]] for i in p[h:]], 0)))
        wgs[a] = {"n_work_groups": len(gs), "mean_cos": float(np.mean(cs))}
res["split_half"]["by_work_group_halves"] = wgs
OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({k: res[k] for k in ("n_primary", "n_work_groups", "n_authors", "3a", "3b", "3b_non_quasi_parallel")}, indent=1, ensure_ascii=False))
print(json.dumps(res["split_half"], indent=1, ensure_ascii=False))
