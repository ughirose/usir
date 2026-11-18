#!/usr/bin/env python3
"""DETV4 T3: measurement. Usage: measure.py pop   -> computes registered p_OP per config (pop_registered.json, design §5)
                                    measure.py run   -> full measurement (reads pop_registered.json) -> detv4_results.json
Protocol identical to DETV3 (5-fold by work, seed 11, inner seed 11+fold, boot B=2000 seed 7)."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import detector_v4 as D  # noqa: E402

OUT, POP = HERE / "detv4_results.json", HERE / "pop_registered.json"
KINDS = ["ending", "comma", "sensory"]
DOSE = ["ending@0.5", "comma@0.25", "comma@1.0", "sensory@0.25", "sensory@1.0"]
OPS = ["B", "P", "N"]  # B: inner-CV p90 (v3 OP-B); P: registered whole-corpus p_OP; N: nested (train-works-only) p_OP
CFGS = [D.Cfg("v4"), D.Cfg("abl_i_no_hurdle", hurdle=False), D.Cfg("abl_ii_v3q2", q3mode="v3q2"),
        D.Cfg("abl_iii_tau1", tau=1.0), D.Cfg("abl_iii_tau4", tau=4.0)]
BOOT_B, BOOT_SEED, FOLD_SEED = 2000, 7, 11

d = np.load(REPO / "data" / "detv4_scene_signals.npz")
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


def pop_for(cfg):
    m = D.fit(RAW["orig"], WI, cfg)
    s = D.score(m, RAW["orig"], RAW["orig"], WI)
    return D.p_op_procedure(s, WI)


def thresholds(train_works, f_seed, cfg, p_global):
    tr = np.where(np.isin(WI, train_works))[0]
    m = D.fit(RAW["orig"][tr], WI[tr], cfg)
    s_tr = D.score(m, RAW["orig"][tr], RAW["orig"][tr], WI[tr])
    fo = workset_folds(list(train_works), f_seed)
    inner = []
    for h in range(5):
        a = np.where(np.isin(WI, [w for w in train_works if fo[w] != h]))[0]
        b = np.where(np.isin(WI, [w for w in train_works if fo[w] == h]))[0]
        mi = D.fit(RAW["orig"][a], WI[a], cfg)
        inner.append(D.score(mi, RAW["orig"][b], RAW["orig"][b], WI[b]))
    proc = D.p_op_procedure(s_tr, WI[tr])
    thr = {"B": float(np.percentile(np.concatenate(inner), 90)), "P": float(np.percentile(s_tr, p_global)),
           "N": float(np.percentile(s_tr, proc["p_op"]))}
    return m, thr, proc["p_op"]


def new_store():
    return {"neg": np.full(n, np.nan), "negH": {o: np.zeros(n, bool) for o in OPS},
            "pos": {k: np.full(n, np.nan) for k in KINDS + DOSE},
            "hit": {o: {k: np.zeros(n, bool) for k in KINDS + DOSE} for o in OPS}}


def evaluate(test_idx, m, thr, S, scen):
    w, ro = WI[test_idx], RAW["orig"][test_idx]
    S["neg"][test_idx] = D.score(m, ro, ro, w)
    for o in OPS:
        S["negH"][o][test_idx] = S["neg"][test_idx] >= thr[o]
    for k in KINDS + DOSE:
        rk = RAW[k][test_idx]
        s = D.score(m, rk, ro if scen == "A" else rk, w)
        S["pos"][k][test_idx] = s
        for o in OPS:
            S["hit"][o][k][test_idx] = s >= thr[o]


def report(mask, S):
    idx = np.where(mask)[0]
    r = {"n_scenes": int(len(idx)), "n_works": int(len(set(WI[idx]))), "per_op": {}, "per_type_auc": {}}
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


def run(cfg, p_global, scen="A", ci=True):
    t0 = time.time()
    works = list(range(len(uw)))
    fo = workset_folds(works, FOLD_SEED)
    fos = np.array([fo[w] for w in WI])
    S, per_fold = new_store(), []
    for f in range(5):
        trw = [w for w in works if fo[w] != f]
        m, thr, p_nested = thresholds(trw, FOLD_SEED + f, cfg, p_global)
        evaluate(np.where(fos == f)[0], m, thr, S, scen)
        per_fold.append({"fold": f, "n_test_works": int(sum(fo[w] == f for w in works)), "thr": thr, "p_op_nested": p_nested})
    res = {"cfg": cfg.name, "scenario": scen, "p_op_registered": p_global, "per_fold": per_fold}
    for fr in per_fold:
        idx = np.where(fos == fr["fold"])[0]
        fr["fpr"] = {o: float(S["negH"][o][idx].mean()) for o in OPS}
        fr["pooled_auc"] = auc(np.concatenate([S["pos"][k][idx] for k in KINDS]), S["neg"][idx])
    res["all"] = report(np.ones(n, bool), S)
    if ci and scen == "A":
        res["all"]["auc_ci95_work_cluster"] = cluster_boot(np.ones(n, bool), S)
    for nm, mk in (("P-2b_existing", COH == 0), ("new", COH == 1)):
        res[nm] = report(mk, S)
        if ci and scen == "A":
            res[nm]["auc_ci95_work_cluster"] = cluster_boot(mk, S)
    trw = [w for w in works if W_COH[w] == 0]
    m, thr, p_nested = thresholds(trw, FOLD_SEED + 99, cfg, p_global)
    S2 = new_store()
    evaluate(np.where(COH == 1)[0], m, thr, S2, scen)
    res["external"] = {"thr": thr, "p_op_nested": p_nested, **report(COH == 1, S2)}
    if ci and scen == "A":
        res["external"]["auc_ci95_work_cluster"] = cluster_boot(COH == 1, S2)
    res["seconds"] = round(time.time() - t0, 1)
    a = res["all"]
    print(f"{cfg.name}/{scen}: AUC {a['pooled_auc']:.4f} FPR B {a['per_op']['B']['fpr']:.4f} P {a['per_op']['P']['fpr']:.4f} "
          f"N {a['per_op']['N']['fpr']:.4f} (p_OP {p_global}) ({res['seconds']}s)", flush=True)
    return res


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "pop":
        out = {}
        for c in CFGS:
            r = pop_for(c)
            out[c.name] = r
            print(c.name, "p_OP", r["p_op"], "floor" if r["fallback_to_floor"] else "")
        POP.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        return
    pop = json.loads(POP.read_text(encoding="utf-8"))
    out = {"n_scenes": int(n), "n_works": len(uw), "seeds": {"fold": FOLD_SEED, "boot": BOOT_SEED, "boot_B": BOOT_B},
           "passline": {"auc_min": 0.85, "fpr_max": 0.10}, "configs": {}, "scenario_Bprime": {}}
    for c in CFGS:
        out["configs"][c.name] = run(c, pop[c.name]["p_op"], "A", ci=(c.name == "v4"))
    out["scenario_Bprime"]["v4"] = run(CFGS[0], pop["v4"]["p_op"], "Bp", ci=False)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("saved", OUT)


if __name__ == "__main__":
    main()
