#!/usr/bin/env python3
"""Stage FAW task A/C pilot (descriptive, read-only): zero rates and author-level work-split reliability of legacy vs candidate
Awase/Nagare indicators. NOT a gate and NOT a selection test: the same data (AOZORA106, POSTWAR) is later reused for design
comments, so any final validation must use data not used here (see design doc section C).
Work-split: greedy by (n_scenes desc, work_id asc) into two groups, group with fewer scenes first (tie -> A), as in P-2c registration B.
Reliability proxy: across authors (>=2 works), Pearson r and Spearman rho between group-A and group-B indicator values.
Run: python -I scripts/faw/faw_pilot.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / "python"))
import faw_measure as M  # noqa: E402
from usir_py.v5 import split_work_into_scenes  # noqa: E402
from usir_py.v5 import aozora_scene_splitter as S  # noqa: E402
from usir_py.v5 import six_pole_atlas as A  # noqa: E402

OUT = HERE / "faw_pilot_results.json"
THETA = 0.25  # legacy still threshold in six_pole_atlas.compute_nagare_profile (:521)


def slope(y):
    y = np.asarray(y, float)
    if len(y) < 3: return None
    x = np.linspace(-0.5, 0.5, len(y))
    return float(np.sum(x * (y - y.mean())) / np.sum(x * x))


def lag1(y):
    y = np.asarray(y, float)
    if len(y) < 4: return None
    a, b = y[:-1] - y.mean(), y[1:] - y.mean()
    d = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / d) if d > 1e-12 else None


def work_record(text):
    scenes = split_work_into_scenes(text)
    aw, ve_l, vs_l, recs = [], [], [], []
    for i, sc in enumerate(scenes):
        a, nag, _, _, _ = S.compute_heuristic_signals(sc, i, len(scenes))
        ve, ae, vs, as_ = M.raw_signals(sc)
        aw.append(a); ve_l.append(ve); vs_l.append(vs); recs.append({"nagare": nag, "awase": a})
    VF = np.array([M.norm(a["f_event_valence"]) for a in aw]); VS = np.array([M.norm(a["f_style_valence"]) for a in aw])
    AF = np.array([M.norm(a["f_event_arousal"]) for a in aw]); AS = np.array([M.norm(a["f_style_arousal"]) for a in aw])
    nag = A.compute_nagare_profile(recs)
    es, ms = np.array(nag["event"]["speed"]), np.array(nag["mood"]["speed"])
    return dict(n=len(scenes), VF=VF, VS=VS, AF=AF, AS=AS, rve=np.array(ve_l), rvs=np.array(vs_l), es=es, ms=ms)


def awase_vec(recs):
    """legacy and candidate Awase indicators from a list of work records (pooled scenes)."""
    VF = np.concatenate([r["VF"] for r in recs]); VS = np.concatenate([r["VS"] for r in recs])
    AF = np.concatenate([r["AF"] for r in recs]); AS = np.concatenate([r["AS"] for r in recs])
    rve = np.concatenate([r["rve"] for r in recs]); rvs = np.concatenate([r["rvs"] for r in recs])
    ir = np.maximum(0, -VF * VS); de = np.abs(VF) * np.maximum(0, AF - AS)
    # legacy irony_grad pooled per work (weighted by scene count)
    def grad(r):
        per = np.maximum(0, -r["VF"] * r["VS"])
        if len(per) >= 5:
            k = max(1, len(per) // 5); return float(per[-k:].mean() - per[:k].mean())
        if len(per) >= 2: return float(per[-1] - per[0])
        return 0.0
    w = np.array([r["n"] for r in recs], float)
    lg = float(np.sum(w * np.array([grad(r) for r in recs])) / w.sum())
    sl = [slope(-r["VF"] * r["VS"]) for r in recs]
    slw = [(r["n"], s) for r, s in zip(recs, sl) if s is not None]
    sl_v = float(sum(n * s for n, s in slw) / sum(n for n, _ in slw)) if slw else None
    return {
        "L_irony_mean": float(ir.mean()), "L_detachment_mean": float(de.mean()), "L_irony_active": float(np.mean(ir > 0)), "L_irony_grad": lg,
        "A1_signed_irony_mean": float(np.mean(-VF * VS)),
        "A1d_signed_detach_mean": float(np.mean(np.abs(VF) * (AF - AS))),
        "A2_raw_signed_irony_mean": float(np.mean(-rve * rvs)),
        "A3_neg_centered_cov_int": float(-np.mean((VF - VF.mean()) * (VS - VS.mean()))),
        "A3r_neg_centered_cov_raw": float(-np.mean((rve - rve.mean()) * (rvs - rvs.mean()))),
        "A5_signed_irony_slope": sl_v,
    }


def nagare_vec(recs):
    ms_all = np.concatenate([r["ms"] for r in recs]); es_all = np.concatenate([r["es"] for r in recs])
    joint = np.maximum(es_all, ms_all)
    ph = float(np.sum(ms_all) / (np.sum(es_all) + np.sum(ms_all) + 1e-12))
    # legacy still_density: per-work run count / n_scenes, pooled by scene weight
    def still(r):
        runs, cur = 0, -1
        for t in range(len(r["es"])):
            if r["es"][t] < 0.25 and r["ms"][t] < 0.25:
                if cur == -1: cur = t
            else:
                if cur != -1 and t - cur >= 2: runs += 1
                cur = -1
        if cur != -1 and len(r["es"]) - cur >= 2: runs += 1
        return runs
    L_still = float(sum(still(r) for r in recs) / sum(r["n"] for r in recs))
    def ratio(r):
        m = r["ms"]
        if len(m) >= 4:
            h = len(m) // 2; return (m[h:].mean() + 1e-4) / (m[:h].mean() + 1e-4)
        return 1.0
    wn = np.array([r["n"] for r in recs], float)
    L_ratio = float(np.sum(wn * np.array([ratio(r) for r in recs])) / wn.sum())
    def wmean(fn):
        v = [(len(r["ms"]), fn(r)) for r in recs]; v = [(n, x) for n, x in v if x is not None]
        return float(sum(n * x for n, x in v) / sum(n for n, _ in v)) if v else None
    def logratio_slope(r):  # N3: OLS slope of mood speed over normalized position
        return slope(r["ms"])
    return {
        "L_phi_zeami": ph, "L_mood_speed_mean": float(ms_all.mean()), "L_still_density": L_still, "L_speed_ratio": L_ratio,
        "N2a_p10_joint_speed": float(np.quantile(joint, 0.10)),
        "N2b_soft_still_mass": float(np.mean(np.maximum(0.0, 1.0 - joint / THETA))),
        "N3_mood_speed_slope": wmean(logratio_slope),
        "N4_mood_speed_lag1ac": wmean(lambda r: lag1(r["ms"])),
    }


def greedy(works):
    order = sorted(works, key=lambda w: (-w[1], w[0])); cnt = [0, 0]; g = [[], []]
    for wid, k, _ in order:
        i = 0 if cnt[0] <= cnt[1] else 1
        cnt[i] += k; g[i].append(wid)
    return g


def corr(x, y):
    x, y = np.array(x, float), np.array(y, float)
    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0: return None, None
    rx, ry = np.argsort(np.argsort(x)).astype(float), np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(x, y)[0, 1]), float(np.corrcoef(rx, ry)[0, 1])


def run(cohort):
    recs = {}
    for key, text in M.load_works(cohort):
        recs[key] = work_record(text)
    by_author = {}
    for k in recs: by_author.setdefault(k.split(":")[0], []).append(k)
    A_vals, B_vals = {}, {}
    for a, ws in sorted(by_author.items()):
        if len(ws) < 2: continue
        gA, gB = greedy([(w, recs[w]["n"], 0) for w in ws])
        if not gA or not gB: continue
        for tag, grp in (("A", gA), ("B", gB)):
            rr = [recs[w] for w in grp]
            v = {**awase_vec(rr), **nagare_vec(rr)}
            (A_vals if tag == "A" else B_vals)[a] = v
    out = {"n_authors_ge2_works": len(A_vals), "n_works": len(recs), "indicators": {}}
    for name in next(iter(A_vals.values())).keys():
        xs = [(A_vals[a][name], B_vals[a][name]) for a in A_vals if A_vals[a][name] is not None and B_vals[a][name] is not None]
        r, rho = corr([x for x, _ in xs], [y for _, y in xs])
        allv = [A_vals[a][name] for a in A_vals] + [B_vals[a][name] for a in B_vals]
        allv = [v for v in allv if v is not None]
        out["indicators"][name] = {"n_pairs": len(xs), "pearson": r, "spearman": rho,
                                   "zero_rate_group": float(np.mean(np.array(allv) == 0)), "distinct_group_values": len({round(v, 9) for v in allv})}
    return out


if __name__ == "__main__":
    res = {c: run(c) for c in ("POSTWAR", "AOZORA106")}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for c, r in res.items():
        print(c, "authors>=2 works:", r["n_authors_ge2_works"], "works:", r["n_works"])
        for k, v in r["indicators"].items():
            f = lambda x: "  n/a" if x is None else f"{x:+.3f}"
            print(f"  {k:28s} n={v['n_pairs']:3d} r={f(v['pearson'])} rho={f(v['spearman'])} zero={v['zero_rate_group']:.2f}")
