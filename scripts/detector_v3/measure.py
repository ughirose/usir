#!/usr/bin/env python3
"""DETV3 T3: measurement of detector v3 + ablations under the P-2c protocol (5-fold by work, seed 11; OP-A / OP-B; 4 splits).
Input: data/detv3_scene_signals.npz (from signals.py). Output: scripts/detector_v3/detv3_results.json. Deterministic (seeds fixed)."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import detector_v3 as D  # noqa: E402

OUT = HERE / "detv3_results.json"
KINDS = ["ending", "comma", "sensory"]
DOSE = ["ending@0.5", "comma@0.25", "comma@1.0", "sensory@0.25", "sensory@1.0"]
CFGS = [D.Cfg("full"), D.Cfg("surface_only", sig=D.SURF), D.Cfg("S1_only", sig=D.S1), D.Cfg("S2_only", sig=D.S2),
        D.Cfg("no_work_norm", work_norm=False), D.Cfg("no_eb_shrink", m0=0.0)]
CI_SUBSETS_FOR = {"full"}
BOOT_B, BOOT_SEED, FOLD_SEED = 2000, 7, 11

d = np.load(REPO / "data" / "detv3_scene_signals.npz")
p2c = np.load(REPO / "data" / "p2c_scene_profiles.npz")
AID, WID, COH = d["author"], d["work"], p2c["cohort"]
RAW = {k: d[k] for k in ["orig"] + KINDS + DOSE}
n = len(AID)
WKEY = np.array([f"{a}_{w}" for a, w in zip(AID, WID)])
uw = sorted(set(WKEY))
WI = np.array([uw.index(w) for w in WKEY])
W_COH = np.array([COH[WI == i][0] for i in range(len(uw))])


def auc(pos, neg):
    n1, n0 = len(pos), len(neg)
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def workset_folds(work_ids, seed):
    perm = np.random.default_rng(seed).permutation(len(work_ids))
    return {work_ids[p]: i % 5 for i, p in enumerate(perm)}


def thresholds(train_works, f_seed, cfg):
    tr = np.where(np.isin(WI, train_works))[0]
    m = D.fit(RAW["orig"][tr], WI[tr], cfg)
    thrA = float(np.percentile(D.score(m, RAW["orig"][tr], RAW["orig"][tr], WI[tr]), 90))
    fo = workset_folds(list(train_works), f_seed)
    inner = []
    for h in range(5):
        a = np.where(np.isin(WI, [w for w in train_works if fo[w] != h]))[0]
        b = np.where(np.isin(WI, [w for w in train_works if fo[w] == h]))[0]
        mi = D.fit(RAW["orig"][a], WI[a], cfg)
        inner.append(D.score(mi, RAW["orig"][b], RAW["orig"][b], WI[b]))
    return m, thrA, float(np.percentile(np.concatenate(inner), 90))


def evaluate(test_idx, m, thrA, thrB, S, scen):
    """Write scores/hits for scenes test_idx (complete works). scen 'A': baseline of positives = other ORIGINAL scenes; 'Bp': perturbed."""
    w = WI[test_idx]
    ro = RAW["orig"][test_idx]
    S["neg"][test_idx] = D.score(m, ro, ro, w)
    S["negA"][test_idx] = S["neg"][test_idx] >= thrA
    S["negB"][test_idx] = S["neg"][test_idx] >= thrB
    for k in KINDS + DOSE:
        rk = RAW[k][test_idx]
        s = D.score(m, rk, ro if scen == "A" else rk, w)
        S["pos"][k][test_idx] = s
        S["hitA"][k][test_idx] = s >= thrA
        S["hitB"][k][test_idx] = s >= thrB


def new_store():
    return {"neg": np.full(n, np.nan), "negA": np.zeros(n, bool), "negB": np.zeros(n, bool),
            "pos": {k: np.full(n, np.nan) for k in KINDS + DOSE}, "hitA": {k: np.zeros(n, bool) for k in KINDS + DOSE},
            "hitB": {k: np.zeros(n, bool) for k in KINDS + DOSE}}


def report(mask, S):
    idx = np.where(mask)[0]
    r = {"n_scenes": int(len(idx)), "n_works": int(len(set(WI[idx]))), "fpr_OP_A": float(S["negA"][idx].mean()),
         "fpr_OP_B": float(S["negB"][idx].mean()), "per_type": {}}
    for k in KINDS:
        r["per_type"][k] = {"auc": auc(S["pos"][k][idx], S["neg"][idx]), "tpr_OP_A": float(S["hitA"][k][idx].mean()),
                            "tpr_OP_B": float(S["hitB"][k][idx].mean())}
    r["pooled"] = {"auc": auc(np.concatenate([S["pos"][k][idx] for k in KINDS]), S["neg"][idx]),
                   "tpr_OP_A": float(np.mean([S["hitA"][k][idx].mean() for k in KINDS])),
                   "tpr_OP_B": float(np.mean([S["hitB"][k][idx].mean() for k in KINDS]))}
    r["dose_response"] = {lv: {"auc": auc(S["pos"][lv][idx], S["neg"][idx]), "tpr_OP_B": float(S["hitB"][lv][idx].mean())} for lv in DOSE}
    return r


def cluster_boot(mask, S):
    idx = np.where(mask)[0]
    wks = sorted(set(WI[idx]))
    mem = [idx[WI[idx] == w] for w in wks]
    rb = np.random.default_rng(BOOT_SEED)
    acc = {k: [] for k in KINDS + ["pooled"]}
    for _ in range(BOOT_B):
        ii = np.concatenate([mem[p] for p in rb.integers(0, len(wks), len(wks))])
        for k in KINDS:
            acc[k].append(auc(S["pos"][k][ii], S["neg"][ii]))
        acc["pooled"].append(auc(np.concatenate([S["pos"][k][ii] for k in KINDS]), S["neg"][ii]))
    return {k: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] for k, a in acc.items()}


def run(cfg, scen="A"):
    t0 = time.time()
    works = list(range(len(uw)))
    fo = workset_folds(works, FOLD_SEED)
    fos = np.array([fo[w] for w in WI])
    S = new_store()
    per_fold = []
    for f in range(5):
        trw = [w for w in works if fo[w] != f]
        m, thrA, thrB = thresholds(trw, FOLD_SEED + f, cfg)
        te = np.where(fos == f)[0]
        evaluate(te, m, thrA, thrB, S, scen)
        per_fold.append({"fold": f, "n_test_works": int(sum(fo[w] == f for w in works)), "thr_OP_A": thrA, "thr_OP_B": thrB})
    allm = np.ones(n, bool)
    res = {"cfg": cfg.name, "scenario": scen, "per_fold": per_fold}
    for fr in per_fold:
        idx = np.where(fos == fr["fold"])[0]
        fr["fpr_OP_B"] = float(S["negB"][idx].mean())
        fr["fpr_OP_A"] = float(S["negA"][idx].mean())
        fr["pooled_auc"] = auc(np.concatenate([S["pos"][k][idx] for k in KINDS]), S["neg"][idx])
    res["all"] = report(allm, S)
    res["all"]["auc_ci95_work_cluster"] = cluster_boot(allm, S)
    for nm, mk in (("P-2b_existing", COH == 0), ("new", COH == 1)):
        res[nm] = report(mk, S)
        if cfg.name in CI_SUBSETS_FOR and scen == "A":
            res[nm]["auc_ci95_work_cluster"] = cluster_boot(mk, S)
    # external: train on P-2b cohort works, test on new cohort
    trw = [w for w in works if W_COH[w] == 0]
    m, thrA, thrB = thresholds(trw, FOLD_SEED + 99, cfg)
    S2 = new_store()
    evaluate(np.where(COH == 1)[0], m, thrA, thrB, S2, scen)
    res["external"] = {"thr_OP_A": thrA, "thr_OP_B": thrB, **report(COH == 1, S2)}
    if cfg.name in CI_SUBSETS_FOR and scen == "A":
        res["external"]["auc_ci95_work_cluster"] = cluster_boot(COH == 1, S2)
    res["seconds"] = round(time.time() - t0, 1)
    print(f"{cfg.name}/{scen}: pooled AUC {res['all']['pooled']['auc']:.4f} FPR_B {res['all']['fpr_OP_B']:.4f} ({res['seconds']}s)", flush=True)
    return res


def main():
    out = {"n_scenes": int(n), "n_works": len(uw), "seeds": {"fold": FOLD_SEED, "boot": BOOT_SEED, "boot_B": BOOT_B},
           "passline": {"auc_min": 0.85, "fpr_max": 0.10}, "configs": {}, "scenario_Bprime": {}}
    for c in CFGS:
        out["configs"][c.name] = run(c, "A")
    for c in (CFGS[0], CFGS[1]):
        out["scenario_Bprime"][c.name] = run(c, "Bp")
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("saved", OUT)


if __name__ == "__main__":
    main()
