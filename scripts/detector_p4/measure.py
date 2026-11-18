#!/usr/bin/env python3
"""P-4a measurement. Protocol identical to DETV4 (5-fold by work, fold seed 11, inner seed 11+fold, bootstrap B=2000 seed 7, work-cluster 95% CI).
Usage:
  measure.py repro  -> v4 configuration re-computed in this framework (max-p rule, hurdle + composite q3') on data/detv4_scene_signals.npz; compare to detv4_results.json
  measure.py pop    -> registered p_OP per v13 config (minimum-p rule) -> pop_registered.json
  measure.py run    -> full measurement of all v13 configs + pre-registered A/B decision -> detp4_results.json
The A/B decision rule (decide()) is pre-registered in the stage prompt (section 2) and committed before `run` is executed."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import detector_p4 as DP  # noqa: E402
import legacy_v4 as LG  # noqa: E402

KINDS = ["ending", "comma", "sensory"]
DOSE = ["ending@0.5", "comma@0.25", "comma@1.0", "sensory@0.25", "sensory@1.0"]
OPS = ["B", "P", "N"]  # B: inner-CV p90; P: registered whole-corpus p_OP; N: nested (train-works-only) p_OP
BOOT_B, BOOT_SEED, FOLD_SEED = 2000, 7, 11
SMALL_N = [1, 2, 3, 4, 6, 8]
SMALL_REPS, SMALL_SEED = 20, 11
N_BINS = [(8, 10), (11, 15), (16, 20), (21, 30), (31, 60), (61, 200)]

V13 = [DP.Cfg("A_poisson_noshrink", "poisson", shrink=False), DP.Cfg("A_poisson_shrink", "poisson", shrink=True),
       DP.Cfg("B_anscombe_noshrink", "anscombe", shrink=False), DP.Cfg("B_anscombe_shrink", "anscombe", shrink=True),
       DP.Cfg("D_plain_rate", "rate")]  # D: diagnostic only, not eligible for the decision


class Ctx:
    def __init__(self, npz, det):
        d = np.load(npz)
        p2c = np.load(REPO / "data" / "p2c_scene_profiles.npz")
        self.det = det
        self.AID, self.WID, self.COH = d["author"], d["work"], p2c["cohort"]
        self.RAW = {k: d[k] for k in ["orig"] + KINDS + DOSE}
        self.n = len(self.AID)
        WKEY = np.array([f"{a}_{w}" for a, w in zip(self.AID, self.WID)])
        self.uw = sorted(set(WKEY))
        self.WI = np.array([self.uw.index(w) for w in WKEY])
        self.W_COH = np.array([self.COH[self.WI == i][0] for i in range(len(self.uw))])


class Failed(Exception):
    pass


def auc(pos, neg):
    n1, n0 = len(pos), len(neg)
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def workset_folds(work_ids, seed):
    perm = np.random.default_rng(seed).permutation(len(work_ids))
    return {work_ids[p]: i % 5 for i, p in enumerate(perm)}


def pop_for(c, cfg):
    d = c.det
    m = d.fit(c.RAW["orig"], c.WI, cfg)
    return d.pop(d.score(m, c.RAW["orig"], c.RAW["orig"], c.WI), c.WI, cfg)


def thresholds(c, train_works, f_seed, cfg, p_global):
    d, R, WI = c.det, c.RAW["orig"], c.WI
    tr = np.where(np.isin(WI, train_works))[0]
    m = d.fit(R[tr], WI[tr], cfg)
    s_tr = d.score(m, R[tr], R[tr], WI[tr])
    fo = workset_folds(list(train_works), f_seed)
    inner = []
    for h in range(5):
        a = np.where(np.isin(WI, [w for w in train_works if fo[w] != h]))[0]
        b = np.where(np.isin(WI, [w for w in train_works if fo[w] == h]))[0]
        mi = d.fit(R[a], WI[a], cfg)
        inner.append(d.score(mi, R[b], R[b], WI[b]))
    proc = d.pop(s_tr, WI[tr], cfg)
    if proc["failed"]:
        raise Failed("nested p_OP: no feasible p in a training split")
    thr = {"B": float(np.percentile(np.concatenate(inner), 90)), "P": float(np.percentile(s_tr, p_global)),
           "N": float(np.percentile(s_tr, proc["p_op"]))}
    return m, thr, proc["p_op"]


def new_store(n):
    return {"neg": np.full(n, np.nan), "negH": {o: np.zeros(n, bool) for o in OPS},
            "pos": {k: np.full(n, np.nan) for k in KINDS + DOSE},
            "hit": {o: {k: np.zeros(n, bool) for k in KINDS + DOSE} for o in OPS}}


def evaluate(c, test_idx, m, thr, S, scen):
    d, w, ro = c.det, c.WI[test_idx], c.RAW["orig"][test_idx]
    S["neg"][test_idx] = d.score(m, ro, ro, w)
    for o in OPS:
        S["negH"][o][test_idx] = S["neg"][test_idx] >= thr[o]
    for k in KINDS + DOSE:
        rk = c.RAW[k][test_idx]
        s = d.score(m, rk, ro if scen == "A" else rk, w)
        S["pos"][k][test_idx] = s
        for o in OPS:
            S["hit"][o][k][test_idx] = s >= thr[o]


def report(c, mask, S):
    idx = np.where(mask)[0]
    r = {"n_scenes": int(len(idx)), "n_works": int(len(set(c.WI[idx]))), "per_op": {}, "per_type_auc": {}}
    for k in KINDS:
        r["per_type_auc"][k] = auc(S["pos"][k][idx], S["neg"][idx])
    r["pooled_auc"] = auc(np.concatenate([S["pos"][k][idx] for k in KINDS]), S["neg"][idx])
    for o in OPS:
        r["per_op"][o] = {"fpr": float(S["negH"][o][idx].mean()),
                          "tpr": {k: float(S["hit"][o][k][idx].mean()) for k in KINDS},
                          "tpr_pooled": float(np.mean([S["hit"][o][k][idx].mean() for k in KINDS]))}
    r["dose_response"] = {lv: {"auc": auc(S["pos"][lv][idx], S["neg"][idx]), "tpr_B": float(S["hit"]["B"][lv][idx].mean()),
                               "tpr_P": float(S["hit"]["P"][lv][idx].mean())} for lv in DOSE}
    return r


def cluster_boot(c, mask, S):
    idx = np.where(mask)[0]
    wks = sorted(set(c.WI[idx]))
    mem = [idx[c.WI[idx] == w] for w in wks]
    rb = np.random.default_rng(BOOT_SEED)
    acc = {k: [] for k in KINDS + ["pooled"]}
    for _ in range(BOOT_B):
        ii = np.concatenate([mem[p] for p in rb.integers(0, len(wks), len(wks))])
        for k in KINDS:
            acc[k].append(auc(S["pos"][k][ii], S["neg"][ii]))
        acc["pooled"].append(auc(np.concatenate([S["pos"][k][ii] for k in KINDS]), S["neg"][ii]))
    return {k: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] for k, a in acc.items()}


def j9_bins(c, S):
    """Scene-count dependence on the full corpus (works have 8..168 scenes): FPR / score distribution / AUC per scene-count bin."""
    nsc = np.bincount(c.WI)
    out = []
    for lo, hi in N_BINS:
        ws = [w for w in range(len(nsc)) if lo <= nsc[w] <= hi]
        if not ws:
            continue
        idx = np.where(np.isin(c.WI, ws))[0]
        r = {"n_scenes_range": [lo, hi], "n_works": len(ws), "n_scenes": int(len(idx)),
             "mean_neg_score": float(S["neg"][idx].mean()), "p95_neg_score": float(np.percentile(S["neg"][idx], 95)),
             "fpr": {o: float(S["negH"][o][idx].mean()) for o in OPS},
             "fpr_per_work_max_P": float(max(S["negH"]["P"][c.WI == w].mean() for w in ws)),
             "pooled_auc": auc(np.concatenate([S["pos"][k][idx] for k in KINDS]), S["neg"][idx])}
        out.append(r)
    return out


def small_n_eval(c, cfg, folds_info, fos):
    """Direct small-n test: each held-out work is subsampled to m scenes (SMALL_REPS random draws); the work's normalization reference is the subsample only.
    Models/thresholds are those of the full-corpus fold evaluation (so FPR/AUC are comparable). m=1 exercises the k=0 fallback."""
    d, WI = c.det, c.WI
    res = {}
    rng = np.random.default_rng(SMALL_SEED)
    for m_ in SMALL_N:
        neg_s, pos_s, negP, negN = [], {k: [] for k in KINDS}, [], []
        for fi in folds_info:
            model, thr = fi["model"], fi["thr"]
            for w in sorted(set(WI[fos == fi["fold"]])):
                iw = np.where(WI == w)[0]
                if len(iw) < m_:
                    continue
                for _ in range(SMALL_REPS):
                    ii = np.sort(rng.choice(iw, m_, replace=False))
                    wk = np.zeros(m_, int)
                    ro = c.RAW["orig"][ii]
                    s0 = d.score(model, ro, ro, wk)
                    neg_s.append(s0); negP.append(s0 >= thr["P"]); negN.append(s0 >= thr["N"])
                    for k in KINDS:
                        pos_s[k].append(d.score(model, c.RAW[k][ii], ro, wk))
        neg = np.concatenate(neg_s)
        r = {"m": m_, "n_neg": int(len(neg)), "mean_neg_score": float(neg.mean()), "p95_neg_score": float(np.percentile(neg, 95)),
             "fpr_P": float(np.concatenate(negP).mean()), "fpr_N": float(np.concatenate(negN).mean()),
             "per_type_auc": {k: auc(np.concatenate(pos_s[k]), neg) for k in KINDS}}
        r["pooled_auc"] = auc(np.concatenate([np.concatenate(pos_s[k]) for k in KINDS]), neg)
        res[str(m_)] = r
    return res


def run(c, cfg, p_global, scen="A", ci=True, small=False):
    t0 = time.time()
    works = list(range(len(c.uw)))
    fo = workset_folds(works, FOLD_SEED)
    fos = np.array([fo[w] for w in c.WI])
    S, per_fold, finfo = new_store(c.n), [], []
    for f in range(5):
        trw = [w for w in works if fo[w] != f]
        m, thr, p_nested = thresholds(c, trw, FOLD_SEED + f, cfg, p_global)
        evaluate(c, np.where(fos == f)[0], m, thr, S, scen)
        per_fold.append({"fold": f, "n_test_works": int(sum(fo[w] == f for w in works)), "thr": thr, "p_op_nested": p_nested})
        finfo.append({"fold": f, "model": m, "thr": thr})
    res = {"cfg": cfg.name, "scenario": scen, "p_op_registered": p_global, "per_fold": per_fold}
    for fr in per_fold:
        idx = np.where(fos == fr["fold"])[0]
        fr["fpr"] = {o: float(S["negH"][o][idx].mean()) for o in OPS}
        fr["pooled_auc"] = auc(np.concatenate([S["pos"][k][idx] for k in KINDS]), S["neg"][idx])
    res["all"] = report(c, np.ones(c.n, bool), S)
    if ci and scen == "A":
        res["all"]["auc_ci95_work_cluster"] = cluster_boot(c, np.ones(c.n, bool), S)
    for nm, mk in (("P-2b_existing", c.COH == 0), ("new", c.COH == 1)):
        res[nm] = report(c, mk, S)
        if ci and scen == "A":
            res[nm]["auc_ci95_work_cluster"] = cluster_boot(c, mk, S)
    trw = [w for w in works if c.W_COH[w] == 0]
    m, thr, p_nested = thresholds(c, trw, FOLD_SEED + 99, cfg, p_global)
    S2 = new_store(c.n)
    evaluate(c, np.where(c.COH == 1)[0], m, thr, S2, scen)
    res["external"] = {"thr": thr, "p_op_nested": p_nested, **report(c, c.COH == 1, S2)}
    if ci and scen == "A":
        res["external"]["auc_ci95_work_cluster"] = cluster_boot(c, c.COH == 1, S2)
    if scen == "A":
        res["j9_scene_count_bins"] = j9_bins(c, S)
        if small:
            res["j9_small_n"] = small_n_eval(c, cfg, finfo, fos)
    res["seconds"] = round(time.time() - t0, 1)
    a = res["all"]
    print(f"{cfg.name}/{scen}: AUC {a['pooled_auc']:.4f} FPR B {a['per_op']['B']['fpr']:.4f} P {a['per_op']['P']['fpr']:.4f} "
          f"N {a['per_op']['N']['fpr']:.4f} (p_OP {p_global}) ({res['seconds']}s)", flush=True)
    return res


# ---- pre-registered A/B decision (stage prompt section 2) ----
def ci_lo(r):
    return r["all"]["auc_ci95_work_cluster"]["pooled"][0]


def eligible(r):
    if r.get("failed"):
        return False
    po = r["all"]["per_op"]
    return po["P"]["fpr"] <= 0.10 and po["N"]["fpr"] <= 0.10


def pick(a, b):
    """Higher CI lower bound wins; both must satisfy the FPR constraint to be eligible; tie at 4 decimals -> a."""
    ea, eb = eligible(a), eligible(b)
    if ea and not eb: return a
    if eb and not ea: return b
    if not ea and not eb: return None
    return a if round(ci_lo(a), 4) >= round(ci_lo(b), 4) else b


def decide(cfgs):
    """Stage 1 (inside A): mu_hat shrinkage; tie -> no shrink. Stage 2: A_best vs B with the same mu_hat; tie -> A (Poisson)."""
    out = {"stage1": None, "stage2": None, "adopted": None}
    a1 = pick(cfgs["A_poisson_noshrink"], cfgs["A_poisson_shrink"])
    if a1 is None:
        return out
    out["stage1"] = a1["cfg"]
    b = cfgs["B_anscombe_" + a1["cfg"].split("_")[-1]]
    w = pick(a1, b)
    out["stage2"] = None if w is None else w["cfg"]
    out["adopted"] = out["stage2"]
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "repro":
        c = Ctx(REPO / "data" / "detv4_scene_signals.npz", LG)
        ref = json.loads((REPO / "scripts" / "detector_v4" / "detv4_results.json").read_text(encoding="utf-8"))
        cfg = LG.Cfg("v4_repro")
        pr = pop_for(c, cfg)
        r = run(c, cfg, pr["p_op"], "A", ci=True)
        r["p_op_rule_max"] = {"p_op": pr["p_op"], "fallback_to_floor": pr["fallback_to_floor"]}
        rv = ref["configs"]["v4"]
        r["reference_detv4_results"] = {"pooled_auc": rv["all"]["pooled_auc"], "fpr_P": rv["all"]["per_op"]["P"]["fpr"],
                                        "fpr_N": rv["all"]["per_op"]["N"]["fpr"], "fpr_B": rv["all"]["per_op"]["B"]["fpr"],
                                        "ci": rv["all"]["auc_ci95_work_cluster"]["pooled"], "p_op": rv["p_op_registered"]}
        rr = r["reference_detv4_results"]
        r["diff_vs_reference"] = {"pooled_auc": r["all"]["pooled_auc"] - rr["pooled_auc"],
                                  "fpr_P": r["all"]["per_op"]["P"]["fpr"] - rr["fpr_P"],
                                  "fpr_N": r["all"]["per_op"]["N"]["fpr"] - rr["fpr_N"],
                                  "fpr_B": r["all"]["per_op"]["B"]["fpr"] - rr["fpr_B"],
                                  "ci_lo": r["all"]["auc_ci95_work_cluster"]["pooled"][0] - rr["ci"][0],
                                  "ci_hi": r["all"]["auc_ci95_work_cluster"]["pooled"][1] - rr["ci"][1],
                                  "p_op": pr["p_op"] - rr["p_op"]}
        (HERE / "detv4_repro_results.json").write_text(json.dumps(r, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("diff", r["diff_vs_reference"])
        return
    c = Ctx(REPO / "data" / "detp4_scene_signals.npz", DP)
    if mode == "pop":
        out = {}
        for cfg in V13:
            r = pop_for(c, cfg)
            out[cfg.name] = r
            print(cfg.name, "p_OP", r["p_op"], "FAILED" if r["failed"] else "")
        (HERE / "pop_registered.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        return
    pop = json.loads((HERE / "pop_registered.json").read_text(encoding="utf-8"))
    out = {"n_scenes": int(c.n), "n_works": len(c.uw), "seeds": {"fold": FOLD_SEED, "boot": BOOT_SEED, "boot_B": BOOT_B},
           "passline": {"auc_ci_lo_min": 0.85, "fpr_max": 0.10}, "configs": {}, "scenario_Bprime": {}}
    for cfg in V13:
        pr = pop[cfg.name]
        if pr["failed"]:
            out["configs"][cfg.name] = {"cfg": cfg.name, "failed": "registered p_OP: no feasible p in grid (minimum-p rule)"}
            print(cfg.name, "FAILED (registered p_OP)")
            continue
        try:
            out["configs"][cfg.name] = run(c, cfg, pr["p_op"], "A", ci=True, small=True)
        except Failed as e:
            out["configs"][cfg.name] = {"cfg": cfg.name, "failed": str(e)}
            print(cfg.name, "FAILED", e)
    out["decision"] = decide(out["configs"])
    print("decision", out["decision"])
    ad = out["decision"]["adopted"]
    if ad:
        cfg = next(x for x in V13 if x.name == ad)
        out["scenario_Bprime"][ad] = run(c, cfg, pop[ad]["p_op"], "Bp", ci=False)
        a = out["configs"][ad]["all"]
        out["branch"] = {"adopted": ad, "ci_lo": a["auc_ci95_work_cluster"]["pooled"][0], "fpr_P": a["per_op"]["P"]["fpr"],
                         "fpr_N": a["per_op"]["N"]["fpr"],
                         "met": bool(a["auc_ci95_work_cluster"]["pooled"][0] >= 0.85 and a["per_op"]["P"]["fpr"] <= 0.10 and a["per_op"]["N"]["fpr"] <= 0.10)}
        print("branch", out["branch"])
    (HERE / "detp4_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("saved")


if __name__ == "__main__":
    main()
