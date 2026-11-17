#!/usr/bin/env python3
"""Stage FAW-IMPL task B: measurement of FAW continuous metrics vs legacy metrics on the confirmation cohort.

Deterministic. The only RNG use is the author-level bootstrap (numpy default_rng, SEED_BOOT, N_BOOT fixed below)
and the P-2b perturbation RNG (SEED_PERT + offset, identical to scripts/p2b/p2b_analysis.py).

Pre-registered choices (fixed before the first run on confirmation data):
- Cohort: p2cE (manifest works, cached, minus work IDs shared with the FAW design cohorts) = primary (novel);
  p2cA2 = secondary (essay; degeneracy / missing-rate check only, not used for the (a) judgement).
- Grouping: P-2c registration B (scene count descending, work ID ascending, greedy assignment to the group
  with fewer scenes, ties -> A). Authors with >= 2 works only.
- Group value of a metric = mean over the group's works whose metric is not None.
- R_xx = Pearson r (primary) and Spearman rho (secondary) across authors between group A and group B values.
- (a): delta = R_xx(new) - R_xx(legacy); 95% CI by author-level bootstrap (percentile), N_BOOT=2000, SEED_BOOT=20261117;
  draws where either r is undefined are dropped and counted.
- Perturbations: P-2b registration 4 (ending 100%, comma 50%, sensory 50%), scene level, one RNG stream per kind in
  work-ID-ascending, scene order. Response = |m_pert - m_orig| / sigma_scene,r, sigma = SD over cohort scenes (or
  transitions) of the per-scene quantity underlying the metric (defined in UNDERLYING below). Ratio = new / legacy.
- Truncation: first K scenes, K in {6, 10, 20}; works with T >= K; drift = mean |m_K - m_full| / SD_between_works(m_full).
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[2]
for p in ("python", "scripts", "scripts/p2b"):
    sys.path.insert(0, str(REPO / p))
from usir_py.v5 import split_work_into_scenes  # noqa: E402
from usir_py.v5.aozora_scene_splitter import build_scene_annotation, compute_heuristic_signals  # noqa: E402
from usir_py.v5.faw_continuous import compute_faw_continuous, awase_scene_terms, _speeds  # noqa: E402
from usir_py.v5.six_pole_atlas import (  # noqa: E402
    NarrativeRGNormalizer, aggregate_work_profile_from_scenes, compute_awase_profile, compute_nagare_profile, normalized_hsic)
import p2b_analysis as P2B  # noqa: E402  (perturb, constants; main() is guarded)

SEED_BOOT, N_BOOT = 20261117, 2000
K_TRUNC = (6, 10, 20)
OUT = Path(__file__).with_name("faw_impl_measure_results.json")

LEGACY = ["irony_mean", "detachment_mean", "irony_active", "irony_grad", "phi_zeami", "mood_speed_mean", "still_density", "speed_ratio"]
NEW = ["aw1_c_irony", "aw1_c_det", "aw4_g", "ng0_phi_zeami", "ng0_mood_speed_mean", "ng1_s_still", "ng2_speed_slope",
       "ng1_s_still_th020", "ng1_s_still_th035"]
# (new, legacy replaced) pairs for criteria (a)-(c). NG-0 pairs have identical values by construction (R_xx diff = 0).
PAIRS = [("aw1_c_irony", "irony_mean"), ("aw1_c_det", "detachment_mean"), ("aw4_g", "irony_grad"),
         ("ng1_s_still", "still_density"), ("ng2_speed_slope", "speed_ratio"),
         ("ng0_phi_zeami", "phi_zeami"), ("ng0_mood_speed_mean", "mood_speed_mean")]
# Case A: new candidates vs adopted canonical dimensions.
CASE_A = [("aw4_g", "aw1_c_irony"), ("aw4_g", "aw1_c_det"), ("ng2_speed_slope", "ng0_mood_speed_mean"),
          ("ng1_s_still_th020", "ng1_s_still"), ("ng1_s_still_th035", "ng1_s_still")]
ALL_M = LEGACY + NEW


def legacy_metrics(scenes):
    """Exact replica of NarrativeRGNormalizer.extract_work_features (Awase/Nagare entries) + default-substitution flags."""
    aw = compute_awase_profile(scenes)
    nag = compute_nagare_profile(scenes)
    T = len(scenes)
    tw = max(T, 1)
    ir = aw["irony"]["per_scene"]
    fb = {}
    if len(ir) >= 5:
        n = max(1, len(ir) // 5)
        grad = float(np.mean(ir[-n:]) - np.mean(ir[:n]))
        fb["irony_grad"] = False
    elif len(ir) >= 2:
        grad = float(ir[-1] - ir[0]); fb["irony_grad"] = False
    else:
        grad = 0.0; fb["irony_grad"] = True
    active = float(np.mean(np.array(ir) > 0.0)) if len(ir) > 0 else 0.0
    ms = nag["mood"]["speed"]
    if len(ms) >= 4:
        mid = len(ms) // 2
        ratio = (float(np.mean(ms[mid:])) + 1e-4) / (float(np.mean(ms[:mid])) + 1e-4)
        smean = float(np.mean(ms)); fb["speed_ratio"] = False
    elif len(ms) > 0:
        smean = float(np.mean(ms)); ratio = 1.0; fb["speed_ratio"] = True
    else:
        smean = 0.5; ratio = 1.0; fb["speed_ratio"] = True
    fb["mood_speed_mean"] = len(ms) == 0
    sev, smo = float(np.sum(nag["event"]["speed"])), float(np.sum(ms))
    fb["phi_zeami"] = not (sev + smo) > 0
    fb["irony_mean"] = T == 0; fb["detachment_mean"] = T == 0; fb["irony_active"] = T == 0; fb["still_density"] = False
    vals = {"irony_mean": aw["irony"]["mean"], "detachment_mean": aw["detachment"]["mean"], "irony_active": active,
            "irony_grad": grad, "phi_zeami": nag["phi_zeami"], "mood_speed_mean": smean,
            "still_density": len(nag["still_runs"]) / tw, "speed_ratio": ratio}
    return {k: float(v) for k, v in vals.items()}, fb


def underlying(scenes):
    """Per-scene / per-transition quantities used as sigma_scene,r for each metric."""
    aw = compute_awase_profile(scenes)
    st = awase_scene_terms(scenes)
    ev, mo = _speeds(scenes)
    ev, mo = np.asarray(ev, float), np.asarray(mo, float)
    J = np.maximum(ev, mo) if len(ev) else ev
    out = {"irony_mean": aw["irony"]["per_scene"], "detachment_mean": aw["detachment"]["per_scene"],
           "irony_active": [float(x > 0) for x in aw["irony"]["per_scene"]], "irony_grad": aw["irony"]["per_scene"],
           "aw1_c_irony": st["c_irony"], "aw1_c_det": st["c_det"], "aw4_g": st["c_irony"],
           "phi_zeami": list(mo / np.where(ev + mo > 0, ev + mo, 1.0)) if len(ev) else [],
           "ng0_phi_zeami": list(mo / np.where(ev + mo > 0, ev + mo, 1.0)) if len(ev) else [],
           "mood_speed_mean": list(mo), "ng0_mood_speed_mean": list(mo), "speed_ratio": list(mo), "ng2_speed_slope": list(mo),
           "still_density": [float(a < 0.25 and b < 0.25) for a, b in zip(ev, mo)],
           "ng1_s_still": list(np.exp(-((J / 0.25) ** 2))) if len(J) else [],
           "ng1_s_still_th020": list(np.exp(-((J / 0.20) ** 2))) if len(J) else [],
           "ng1_s_still_th035": list(np.exp(-((J / 0.35) ** 2))) if len(J) else []}
    return out


def work_metrics(scenes):
    leg, fb = legacy_metrics(scenes)
    new = compute_faw_continuous(scenes)
    return {**leg, **new}, fb


def annotate(texts, aid, wid, full=False):
    n = len(texts)
    out = []
    for i, t in enumerate(texts):
        if full:
            out.append(build_scene_annotation(t, aid, wid, i, n))
        else:
            aw, nag, _, _, _ = compute_heuristic_signals(t, i, n)
            out.append({"awase": aw, "nagare": nag})
    return out


def split_groups(works):
    """P-2c registration B. works: list of dicts with work_id, n_scenes."""
    order = sorted(works, key=lambda w: (-w["n_scenes"], w["work_id"]))
    A, B, sa, sb = [], [], 0, 0
    for w in order:
        if sa <= sb:
            A.append(w); sa += w["n_scenes"]
        else:
            B.append(w); sb += w["n_scenes"]
    return A, B


def gmean(ws, m):
    v = [w["m"][m] for w in ws if w["m"][m] is not None]
    return (float(np.mean(v)) if v else None), len(v)


def rxx_table(works, metrics):
    by_author = collections.defaultdict(list)
    for w in works:
        by_author[w["author_id"]].append(w)
    authors = sorted(a for a, ws in by_author.items() if len(ws) >= 2)
    split = {a: split_groups(by_author[a]) for a in authors}
    res, vec = {}, {}
    for m in metrics:
        xa, xb, used = [], [], []
        for a in authors:
            A, B = split[a]
            ga, _ = gmean(A, m); gb, _ = gmean(B, m)
            if ga is None or gb is None:
                continue
            xa.append(ga); xb.append(gb); used.append(a)
        xa, xb = np.array(xa), np.array(xb)
        r = rho = None
        if len(xa) >= 3 and np.ptp(xa) > 0 and np.ptp(xb) > 0:
            r = float(pearsonr(xa, xb)[0]); rho = float(spearmanr(xa, xb)[0])
        res[m] = {"n_authors": len(used), "authors_dropped_no_valid": len(authors) - len(used), "pearson_r": r, "spearman_rho": rho}
        vec[m] = (used, xa, xb)
    return res, vec, authors


def _r(xa, xb):
    if len(xa) < 3 or np.ptp(xa) == 0 or np.ptp(xb) == 0:
        return np.nan
    return pearsonr(xa, xb)[0]


def bootstrap_diff(vec, new, old):
    un, na, nb = vec[new]; uo, oa, ob = vec[old]
    common = [a for a in un if a in set(uo)]
    ia = {a: i for i, a in enumerate(un)}; ib = {a: i for i, a in enumerate(uo)}
    NA = np.array([na[ia[a]] for a in common]); NB = np.array([nb[ia[a]] for a in common])
    OA = np.array([oa[ib[a]] for a in common]); OB = np.array([ob[ib[a]] for a in common])
    n = len(common)
    if n < 3:
        return {"n_common_authors": n, "note": "too few authors"}
    rng = np.random.default_rng(SEED_BOOT)
    diffs, dropped = [], 0
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        d = _r(NA[idx], NB[idx]) - _r(OA[idx], OB[idx])
        if np.isnan(d):
            dropped += 1
        else:
            diffs.append(d)
    point = _r(NA, NB) - _r(OA, OB)
    out = {"n_common_authors": n, "point_diff_pearson": None if np.isnan(point) else float(point), "n_boot": N_BOOT,
           "n_boot_dropped_undefined": dropped}
    if diffs:
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        out.update(ci95_lo=float(lo), ci95_hi=float(hi), ci_lo_gt_0=bool(lo > 0))
    return out


def load_cohort(prefix, rows):
    D = REPO / "data" / "aozora_cache"
    works = []
    for r in sorted(rows, key=lambda r: (r["author_id"], r["work_id"])):
        if not r["cached"] or r["dup_work"]:
            continue
        text = (D / f"{r.get('prefix', prefix)}_{r['author_id']}_{r['work_id']}.txt").read_text(encoding="utf-8")
        texts = split_work_into_scenes(text)
        works.append({"author_id": r["author_id"], "work_id": r["work_id"], "author_seen": r["author_seen"], "chars": len(text),
                      "texts": texts, "n_scenes": len(texts)})
    return works


def analyse(name, works, deep=True):
    res = {"cohort": name, "n_works": len(works), "n_authors": len({w["author_id"] for w in works}),
           "n_scenes": int(sum(w["n_scenes"] for w in works))}
    # --- metrics (original text); full annotation for 'other pole' features
    legacy_fb = collections.defaultdict(int)
    und_pool = collections.defaultdict(list)
    profiles = []
    for w in works:
        sc = annotate(w["texts"], w["author_id"], w["work_id"], full=True)
        w["scenes"] = sc
        w["m"], fb = work_metrics(sc)
        w["fb"] = fb
        for k, v in fb.items():
            legacy_fb[k] += int(v)
        for k, v in underlying(sc).items():
            und_pool[k].extend(v)
        prof = aggregate_work_profile_from_scenes(sc)
        prof["work_id"] = w["work_id"]; prof["author_id"] = w["author_id"]
        profiles.append(prof)
    # legacy parity check against NarrativeRGNormalizer
    nr = NarrativeRGNormalizer({"works": profiles})
    mism = 0
    idx = {"phi_zeami": 7, "mood_speed_mean": 8, "still_density": 9, "speed_ratio": 10, "irony_mean": 11, "detachment_mean": 12,
           "irony_active": 13, "irony_grad": 14}
    for w, prof in zip(works, profiles):
        f = nr.extract_work_features(f"{w['author_id']}:{w['work_id']}")
        for k, i in idx.items():
            if abs(f[i] - w["m"][k]) > 1e-12:
                mism += 1
    res["legacy_replica_mismatch_cells"] = mism
    # --- census
    census = {}
    for m in ALL_M:
        nn = [f"{w['author_id']}:{w['work_id']}" for w in works if w["m"][m] is None]
        census[m] = {"none_count": len(nn), "none_rate": len(nn) / len(works), "none_works": nn}
    for m in LEGACY:
        census[m]["legacy_default_substitution_count"] = int(legacy_fb[m])
        census[m]["legacy_default_substitution_rate"] = legacy_fb[m] / len(works)
    res["census"] = census
    res["value_summary"] = {m: ({"min": float(np.min(v)), "median": float(np.median(v)), "max": float(np.max(v)),
                                 "zero_rate": float(np.mean(np.array(v) == 0)), "n_distinct": int(len(set(v)))} if (v := [w["m"][m] for w in works if w["m"][m] is not None]) else None)
                            for m in ALL_M}
    # --- R_xx
    rx, vec, authors = rxx_table(works, ALL_M)
    res["rxx"] = {"n_authors_ge2": len(authors), "metrics": rx}
    res["rxx_diff"] = {f"{n}__vs__{o}": bootstrap_diff(vec, n, o) for n, o in PAIRS}
    # unknown-author subset (authors not in FAW design cohorts)
    unk = [w for w in works if not w["author_seen"]]
    rxu, vecu, au = rxx_table(unk, ALL_M)
    res["rxx_unknown_authors_only"] = {"n_works": len(unk), "n_authors_ge2": len(au), "metrics": rxu}
    # --- relationships
    rel = {}
    for n, o in PAIRS + CASE_A:
        pairs = [(w["m"][n], w["m"][o]) for w in works if w["m"][n] is not None and w["m"][o] is not None]
        if len(pairs) >= 3 and np.ptp([p[0] for p in pairs]) > 0 and np.ptp([p[1] for p in pairs]) > 0:
            rel[f"{n}__vs__{o}"] = {"n": len(pairs), "spearman_rho": float(spearmanr(*zip(*pairs))[0])}
        else:
            rel[f"{n}__vs__{o}"] = {"n": len(pairs), "spearman_rho": None}
    res["spearman_new_vs_ref"] = rel
    # nHSIC vs log(chars) and vs 'other poles' (Kamae+Sadame, features 0..6, z-scored)
    F = np.array([nr.extract_work_features(f"{w['author_id']}:{w['work_id']}") for w in works])
    other = F[:, :7]
    sd = other.std(axis=0, ddof=1); sd[sd < 1e-12] = 1.0
    other = (other - other.mean(axis=0)) / sd
    lc = np.log(np.array([w["chars"] for w in works], float))[:, None]
    nh = {}
    for m in ALL_M:
        ok = [i for i, w in enumerate(works) if w["m"][m] is not None]
        if len(ok) < 5 or np.ptp([works[i]["m"][m] for i in ok]) == 0:
            nh[m] = None
            continue
        x = np.array([works[i]["m"][m] for i in ok], float)[:, None]
        nh[m] = {"n": len(ok), "nhsic_log_chars": normalized_hsic(x, lc[ok]), "nhsic_other_poles_kamae_sadame": normalized_hsic(x, other[ok])}
    res["nhsic"] = nh
    res["nhsic_delta_new_minus_legacy"] = {
        f"{n}__vs__{o}": ({k: nh[n][k] - nh[o][k] for k in ("nhsic_log_chars", "nhsic_other_poles_kamae_sadame")} if nh[n] and nh[o] else None)
        for n, o in PAIRS}
    # --- sigma_scene,r
    sig = {m: float(np.std(und_pool[m], ddof=1)) if len(und_pool[m]) > 2 else None for m in ALL_M}
    res["sigma_scene_r"] = sig
    if not deep:
        return res, works
    # --- perturbation
    flat = [(wi, si) for wi, w in enumerate(works) for si in range(w["n_scenes"])]
    texts_all = [works[wi]["texts"][si] for wi, si in flat]
    resp = {m: {} for m in ALL_M}
    for kind, rate in P2B.PRIMARY.items():
        pt = P2B.perturb(kind, texts_all, rate)
        per_work = collections.defaultdict(list)
        for (wi, si), t in zip(flat, pt):
            per_work[wi].append(t)
        for wi, w in enumerate(works):
            pm, _ = work_metrics(annotate(per_work[wi], w["author_id"], w["work_id"]))
            for m in ALL_M:
                a, b = w["m"][m], pm[m]
                if a is None or b is None or sig[m] in (None, 0):
                    continue
                resp[m].setdefault(kind, []).append((b - a) / sig[m])
    pert = {}
    for m in ALL_M:
        pert[m] = {k: {"n": len(v), "mean_abs_response": float(np.mean(np.abs(v))), "sd_response": float(np.std(v, ddof=1)) if len(v) > 1 else None}
                   for k, v in resp[m].items()}
    res["perturbation"] = pert
    ratio = {}
    for n, o in PAIRS:
        ratio[f"{n}__vs__{o}"] = {k: (pert[n][k]["mean_abs_response"] / pert[o][k]["mean_abs_response"]
                                      if k in pert[n] and k in pert[o] and pert[o][k]["mean_abs_response"] > 0 else None) for k in P2B.PRIMARY}
        vals = [v for v in ratio[f"{n}__vs__{o}"].values() if v is not None]
        ratio[f"{n}__vs__{o}"]["max_over_kinds"] = max(vals) if vals else None
        ratio[f"{n}__vs__{o}"]["le_2_0"] = (max(vals) <= 2.0) if vals else None
    res["perturbation_ratio_new_over_legacy"] = ratio
    # --- truncation drift
    drift = {}
    for K in K_TRUNC:
        sub = [w for w in works if w["n_scenes"] >= K]
        d = {"n_works": len(sub)}
        for m in ALL_M:
            full = [w["m"][m] for w in sub]
            trunc = []
            for w in sub:
                mt, _ = work_metrics(w["scenes"][:K])
                trunc.append(mt[m])
            ok = [(f, t) for f, t in zip(full, trunc) if f is not None and t is not None]
            sdb = float(np.std([f for f, _ in ok], ddof=1)) if len(ok) > 2 else None
            d[m] = {"n": len(ok), "n_none_trunc": sum(t is None for t in trunc),
                    "mean_abs_diff": float(np.mean([abs(f - t) for f, t in ok])) if ok else None,
                    "mean_abs_diff_over_between_sd": (float(np.mean([abs(f - t) for f, t in ok])) / sdb) if ok and sdb else None}
        drift[f"K{K}"] = d
    res["truncation_drift"] = drift
    return res, works


def main():
    sel = json.loads((Path(__file__).with_name("cohort_check_results.json")).read_text(encoding="utf-8"))
    out = {"seed_boot": SEED_BOOT, "n_boot": N_BOOT, "k_trunc": list(K_TRUNC)}
    prim, _ = analyse("p2cE", load_cohort("p2cE", sel["p2cE_rows"]))
    out["primary_p2cE"] = prim
    sec, _ = analyse("p2cA2", load_cohort("p2cA2", sel["p2cA2_rows"]), deep=False)
    out["secondary_p2cA2"] = sec
    # extended cohorts (supplement from heldout3 authors absent from FAW / p2cE / p2cA2; work-ID dedup done in supplement_plan.py)
    D = REPO / "data"
    sup = json.loads((D / "aozora_faw_supplement_manifest.json").read_text(encoding="utf-8"))
    sup_a = {e["author_id"] for e in sup}
    h3 = json.loads((D / "aozora_heldout3_manifest.json").read_text(encoding="utf-8"))
    mk = lambda e, pre: {"author_id": e["author_id"], "work_id": e["work_id"], "cached": True, "dup_work": False, "author_seen": False, "prefix": pre}
    rows_new = [mk(e, "fawsup") for e in sup]
    rows_h3 = [mk(e, "heldout3") for e in h3 if e["author_id"] in sup_a]
    ext, _ = analyse("p2cE+supplement(new+heldout3-first)", load_cohort("p2cE", sel["p2cE_rows"] + rows_new + rows_h3))
    out["extended_with_h3first"] = ext
    ext2, _ = analyse("p2cE+supplement(new only)", load_cohort("p2cE", sel["p2cE_rows"] + rows_new))
    out["extended_new_only"] = ext2
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
