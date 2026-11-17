#!/usr/bin/env python3
"""DETV3 T0: verbatim copy of scripts/p2c/p2c_analysis.py with only OUT redirected (reproduces v2 without touching frozen p2c_results.json)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[2]
SIGMA_FLOOR = 1e-6
OUT = REPO / "scripts" / "detector_v3" / "v2_repro_results.json"
KINDS = ["ending", "comma", "sensory"]
DOSE_LEVELS = ["ending@0.5", "comma@0.25", "comma@1.0", "sensory@0.25", "sensory@1.0"]
VARIANTS = ["composite", "sub", "surf", "ortho", "kasane", "q1", "q2", "q3"]
ORTHO, KASANE = slice(0, 24), slice(24, 32)

d = np.load(REPO / "data" / "p2c_scene_profiles.npz")
AID, WID, COH, X, Q = d["author"], d["work"], d["cohort"], d["X"], d["Q"]
XP = {k: d[f"X_{k}"] for k in KINDS}
QP = {k: d[f"Q_{k}"] for k in KINDS}
n = len(X)
WKEY = np.array([f"{a}_{w}" for a, w in zip(AID, WID)])
uw = sorted(set(WKEY))
wi = {w: i for i, w in enumerate(uw)}
WI = np.array([wi[w] for w in WKEY])
W_COH = np.array([COH[WI == i][0] for i in range(len(uw))])


def auc(pos, neg):
    n1, n0 = len(pos), len(neg)
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0


def summ(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return {"n": 0}
    q = np.percentile(v, [25, 50, 75])
    return {"n": len(v), "mean": float(v.mean()), "median": float(q[1]), "q25": float(q[0]), "q75": float(q[2]),
            "iqr": float(q[2] - q[0]), "min": float(v.min()), "max": float(v.max())}


def stats(x):
    return x.mean(0), np.maximum(x.std(0, ddof=1), SIGMA_FLOOR)


# ======================= Task B =======================
def greedy_split_rxx(works):
    """works: list of (work_id, n_scenes, sumZ). 登録 B: n desc, id asc; assign to the group with fewer scenes (tie -> A)."""
    order = sorted(works, key=lambda w: (-w[1], w[0]))
    cnt = [0, 0]
    sm = [0.0, 0.0]
    for _, k, s in order:
        g = 0 if cnt[0] <= cnt[1] else 1
        cnt[g] += k
        sm[g] = sm[g] + s
    return cos(sm[0] / cnt[0], sm[1] / cnt[1]) if min(cnt) > 0 else float("nan")


def rxx_all(mu, sd, loo=False):
    out = {}
    for a in sorted(set(AID)):
        idx = np.where(AID == a)[0]
        wl = sorted(set(WKEY[idx]))
        if len(wl) < 2:
            continue
        m_, s_ = stats(X[AID != a]) if loo else (mu, sd)
        Z = (X[idx] - m_) / s_
        works = [(w, int((WKEY[idx] == w).sum()), Z[WKEY[idx] == w].sum(0)) for w in wl]
        out[a] = greedy_split_rxx(works)
    return out


def parity_rxx(mu, sd):
    out = {}
    for a in sorted(set(AID)):
        idx = np.where(AID == a)[0]
        Z = (X[idx] - mu) / sd
        out[a] = cos(Z[0::2].mean(0), Z[1::2].mean(0))
    return out


mu_all, sd_all = stats(X)
mu_b, sd_b = stats(X[COH == 0])
auth_name = {}
auth_src = {}
for f in ("aozora_p2b_manifest", "aozora_p2c_manifest"):
    for w in json.loads((REPO / "data" / f"{f}.json").read_text(encoding="utf-8"))["works"]:
        auth_name[w["author_id"]] = w["author_ja"]
        if "source_manifests" in w:
            auth_src[w["author_id"]] = w["source_manifests"]
auth_coh = {a: int(COH[AID == a][0]) for a in set(AID)}

R_main = rxx_all(mu_all, sd_all)
R_p2bref = rxx_all(mu_b, sd_b)
R_loo = rxx_all(None, None, loo=True)
R_par = parity_rxx(mu_all, sd_all)
nw = {a: len(set(WKEY[AID == a])) for a in R_main}
per_author = {a: {"author": auth_name[a], "cohort": "P-2b" if auth_coh[a] == 0 else "P-2c-new", "n_works": nw[a],
                  "n_scenes": int((AID == a).sum()), "R_xx_work_split": R_main[a], "R_xx_work_split_p2b_reference": R_p2bref[a],
                  "R_xx_work_split_leave_author_out": R_loo[a], "R_xx_parity_reference": R_par[a],
                  "source_manifests": auth_src.get(a)} for a in sorted(R_main)}
Zall = (X - mu_all) / sd_all
wl_all = [w for a in sorted(R_main) for w in sorted(set(WKEY[AID == a]))]
wdat = {w: (int((WKEY == w).sum()), Zall[WKEY == w].sum(0)) for w in wl_all}
sizes = [nw[a] for a in sorted(R_main)]
g = np.random.default_rng(11)
null_vals, null_med = [], []
for _ in range(2000):
    perm = g.permutation(len(wl_all))
    pos_ = 0
    vv = []
    for k in sizes:
        ws = [(wl_all[i], wdat[wl_all[i]][0], wdat[wl_all[i]][1]) for i in perm[pos_:pos_ + k]]
        pos_ += k
        vv.append(greedy_split_rxx(ws))
    null_vals.extend(vv)
    null_med.append(float(np.median(vv)))
null_vals = np.array(null_vals)
vals = {"all": [v["R_xx_work_split"] for v in per_author.values()],
        "P-2b": [v["R_xx_work_split"] for v in per_author.values() if v["cohort"] == "P-2b"],
        "new": [v["R_xx_work_split"] for v in per_author.values() if v["cohort"] != "P-2b"]}
B = {"per_author": per_author, "n_authors_ge2_works": len(R_main),
     "summary_work_split": {k: summ(v) for k, v in vals.items()},
     "summary_work_split_p2b_reference": summ(list(R_p2bref.values())),
     "summary_work_split_leave_author_out": summ(list(R_loo.values())),
     "summary_parity_reference_ge2_work_authors": summ([R_par[a] for a in R_main]),
     "summary_parity_reference_all_authors": summ(list(R_par.values())),
     "null_work_reassignment": {"n_perm": 2000, "seed": 11, "per_author_p50": float(np.percentile(null_vals, 50)),
                                "p95": float(np.percentile(null_vals, 95)), "p99": float(np.percentile(null_vals, 99)),
                                "max": float(null_vals.max()), "median_over_authors_p95": float(np.percentile(null_med, 95)),
                                "median_over_authors_p50": float(np.percentile(null_med, 50))},
     "n_authors_passing_candidate_lines": {str(L): {"all": int(sum(v >= L for v in vals["all"])), "P-2b": int(sum(v >= L for v in vals["P-2b"])),
                                                    "new": int(sum(v >= L for v in vals["new"]))} for L in (0.3, 0.4, 0.5, 0.6)}}


# ======================= Task C =======================
def fit(train_idx):
    mu, sd = stats(X[train_idx])
    Z = (X[train_idx] - mu) / sd
    dg = [np.linalg.norm(Z[:, ORTHO], axis=1), np.linalg.norm(Z[:, KASANE], axis=1)]
    gm = [v.mean() for v in dg]
    gs = [max(v.std(ddof=1), SIGMA_FLOOR) for v in dg]
    qm = Q[train_idx].mean(0)
    qs = np.maximum(Q[train_idx].std(0, ddof=1), SIGMA_FLOOR)
    return mu, sd, gm, gs, qm, qs


def score(fm, Xs, Qs):
    mu, sd, gm, gs, qm, qs = fm
    Z = (Xs - mu) / sd
    s_o = (np.linalg.norm(Z[:, ORTHO], axis=1) - gm[0]) / gs[0]
    s_k = (np.linalg.norm(Z[:, KASANE], axis=1) - gm[1]) / gs[1]
    qz = np.abs(Qs - qm) / qs
    sub = np.maximum(s_o, s_k)
    surf = qz.max(1)
    return {"composite": np.maximum(sub, surf), "sub": sub, "surf": surf, "ortho": s_o, "kasane": s_k,
            "q1": qz[:, 0], "q2": qz[:, 1], "q3": qz[:, 2]}


def workset_folds(work_ids, seed):
    perm = np.random.default_rng(seed).permutation(len(work_ids))
    fo = {}
    for i, p in enumerate(perm):
        fo[work_ids[p]] = i % 5
    return fo


def thresholds(train_works, f_seed):
    """A: p90 of training negatives. B: p90 of nested inner-held-out negatives (inner 5-fold by work)."""
    tr = np.isin(WI, train_works)
    fm = fit(np.where(tr)[0])
    sc = score(fm, X[tr], Q[tr])
    thrA = {v: float(np.percentile(sc[v], 90)) for v in VARIANTS}
    fo = workset_folds(list(train_works), f_seed)
    inner = {v: [] for v in VARIANTS}
    for h in range(5):
        trw = [w for w in train_works if fo[w] != h]
        tew = [w for w in train_works if fo[w] == h]
        m_tr = np.isin(WI, trw)
        m_te = np.isin(WI, tew)
        s2 = score(fit(np.where(m_tr)[0]), X[m_te], Q[m_te])
        for v in VARIANTS:
            inner[v].append(s2[v])
    thrB = {v: float(np.percentile(np.concatenate(inner[v]), 90)) for v in VARIANTS}
    return fm, thrA, thrB


def alloc():
    mk = lambda: {v: np.full(n, np.nan) for v in VARIANTS}  # noqa: E731
    mb = lambda: {v: np.zeros(n, bool) for v in VARIANTS}  # noqa: E731
    neg = mk()
    pos = {k: mk() for k in KINDS}
    dose = {lv: {"s": np.full(n, np.nan), "hitB": np.zeros(n, bool)} for lv in DOSE_LEVELS}
    tp = {"negA": mb(), "negB": mb(), "posA": {k: mb() for k in KINDS}, "posB": {k: mb() for k in KINDS}}
    return neg, pos, dose, tp


def evaluate(test_mask, fm, thrA, thrB, neg_s, pos_s, dose_s, tp):
    idx = np.where(test_mask)[0]
    s = score(fm, X[idx], Q[idx])
    for v in VARIANTS:
        neg_s[v][idx] = s[v]
        tp["negA"][v][idx] = s[v] >= thrA[v]
        tp["negB"][v][idx] = s[v] >= thrB[v]
    for k in KINDS:
        sk = score(fm, XP[k][idx], QP[k][idx])
        for v in VARIANTS:
            pos_s[k][v][idx] = sk[v]
            tp["posA"][k][v][idx] = sk[v] >= thrA[v]
            tp["posB"][k][v][idx] = sk[v] >= thrB[v]
    for lv in DOSE_LEVELS:
        sk = score(fm, d[f"Xd_{lv}"][idx], d[f"Qd_{lv}"][idx])["composite"]
        dose_s[lv]["s"][idx] = sk
        dose_s[lv]["hitB"][idx] = sk >= thrB["composite"]


def report(mask, neg, pos, dose, tp):
    idx = np.where(mask)[0]
    r = {"n_scenes": int(len(idx)), "n_works": int(len(set(WI[idx]))), "variants": {}}
    for v in VARIANTS:
        e = {"fpr_OP_A": float(tp["negA"][v][idx].mean()), "fpr_OP_B": float(tp["negB"][v][idx].mean()), "per_type": {}}
        allp = []
        for k in KINDS:
            e["per_type"][k] = {"auc": auc(pos[k][v][idx], neg[v][idx]), "tpr_OP_A": float(tp["posA"][k][v][idx].mean()),
                                "tpr_OP_B": float(tp["posB"][k][v][idx].mean())}
            allp.append(pos[k][v][idx])
        e["pooled"] = {"auc": auc(np.concatenate(allp), neg[v][idx]),
                       "tpr_OP_B": float(np.mean([tp["posB"][k][v][idx].mean() for k in KINDS])),
                       "tpr_OP_A": float(np.mean([tp["posA"][k][v][idx].mean() for k in KINDS]))}
        e["min_type_auc"] = float(min(x["auc"] for x in e["per_type"].values()))
        r["variants"][v] = e
    r["dose_response_composite"] = {lv: {"auc": auc(dose[lv]["s"][idx], neg["composite"][idx]), "tpr_OP_B": float(dose[lv]["hitB"][idx].mean())}
                                    for lv in DOSE_LEVELS}
    return r


def cluster_boot(mask, neg, pos, variants=("composite",), B_=2000, seed=7):
    idx = np.where(mask)[0]
    wks = sorted(set(WI[idx]))
    mem = [idx[WI[idx] == w] for w in wks]
    rb = np.random.default_rng(seed)
    out = {v: {k: [] for k in KINDS + ["pooled"]} for v in variants}
    for _ in range(B_):
        pick = rb.integers(0, len(wks), len(wks))
        ii = np.concatenate([mem[p] for p in pick])
        for v in variants:
            for k in KINDS:
                out[v][k].append(auc(pos[k][v][ii], neg[v][ii]))
            out[v]["pooled"].append(auc(np.concatenate([pos[k][v][ii] for k in KINDS]), neg[v][ii]))
    return {v: {k: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] for k, a in out[v].items()} for v in variants}


all_works = list(range(len(uw)))
fo = workset_folds(all_works, 11)
neg, pos, dose, tp = alloc()
fold_of_scene = np.array([fo[w] for w in WI])
fold_rows = []
for f in range(5):
    trw = [w for w in all_works if fo[w] != f]
    fm, thrA, thrB = thresholds(trw, 11 + f)
    evaluate(fold_of_scene == f, fm, thrA, thrB, neg, pos, dose, tp)
    fold_rows.append({"fold": f, "n_test_works": int(sum(fo[w] == f for w in all_works)),
                      "thr_OP_A_composite": thrA["composite"], "thr_OP_B_composite": thrB["composite"]})
allm = np.ones(n, bool)
C_cf = {"folds": 5, "seed": 11, "per_fold": fold_rows, "all": report(allm, neg, pos, dose, tp),
        "P-2b_existing": report(COH == 0, neg, pos, dose, tp), "new": report(COH == 1, neg, pos, dose, tp)}
C_cf["all"]["auc_ci95_work_cluster"] = cluster_boot(allm, neg, pos, ("composite", "sub", "surf"))
C_cf["P-2b_existing"]["auc_ci95_work_cluster"] = cluster_boot(COH == 0, neg, pos)
C_cf["new"]["auc_ci95_work_cluster"] = cluster_boot(COH == 1, neg, pos)
for fr in C_cf["per_fold"]:
    idx = np.where(fold_of_scene == fr["fold"])[0]
    fr["fpr_OP_B"] = float(tp["negB"]["composite"][idx].mean())
    fr["fpr_OP_A"] = float(tp["negA"]["composite"][idx].mean())
    fr["pooled_auc_composite"] = auc(np.concatenate([pos[k]["composite"][idx] for k in KINDS]), neg["composite"][idx])

trw_ext = [w for w in all_works if W_COH[w] == 0]
fm, thrA, thrB = thresholds(trw_ext, 11 + 99)
neg2, pos2, dose2, tp2 = alloc()
evaluate(COH == 1, fm, thrA, thrB, neg2, pos2, dose2, tp2)
C_ext = {"train": "P-2b cohort (all works), nested OP-B inside it", "test": "new cohort",
         "thr_OP_A_composite": thrA["composite"], "thr_OP_B_composite": thrB["composite"], **report(COH == 1, neg2, pos2, dose2, tp2)}
C_ext["auc_ci95_work_cluster"] = cluster_boot(COH == 1, neg2, pos2)

# ---- diagnostics (descriptive; not used for selection)
Dg = {}
for k in KINDS:
    ms = ((XP[k] - X) / sd_all).mean(0)
    top = np.argsort(-np.abs(ms))[:5]
    Dg[k] = {"top5_dims_mean_z_shift": [{"dim": int(j), "group": "ortho" if j < 24 else "kasane", "mean_shift_sd": float(ms[j])} for j in top],
             "rms_mean_shift_ortho": float(np.sqrt((ms[ORTHO] ** 2).mean())), "rms_mean_shift_kasane": float(np.sqrt((ms[KASANE] ** 2).mean())),
             "frac_scenes_changed_feature_vector": float((np.abs(XP[k] - X).max(1) > 1e-12).mean())}
qsd = Q.std(0, ddof=1)
Dg["surface_quantities"] = {"names": ["comma_density", "sensory_rate", "ending_diversity"], "mean_all": Q.mean(0).tolist(),
                            "between_scene_sd": qsd.tolist(), "cv_between_scene": (qsd / Q.mean(0)).tolist(),
                            "expected_effect_size_sd_units": {k: ((QP[k] - Q).mean(0) / qsd).tolist() for k in KINDS},
                            "scene_level_effect_size_sd_units_sd": {k: ((QP[k] - Q) / qsd).std(0).tolist() for k in KINDS}}
Dg["share_of_scenes_with_zero_comma"] = float((Q[:, 0] == 0).mean())
Dg["share_of_scenes_with_zero_sensory"] = float((Q[:, 1] == 0).mean())

res = {"n_scenes": int(n), "n_works": len(uw), "n_authors": len(set(AID)),
       "cohorts": {"P-2b_existing": {"scenes": int((COH == 0).sum()), "works": int((W_COH == 0).sum()), "authors": len(set(AID[COH == 0]))},
                   "new": {"scenes": int((COH == 1).sum()), "works": int((W_COH == 1).sum()), "authors": len(set(AID[COH == 1]))}},
       "B": B, "C_crossfit": C_cf, "C_external_P2b_train_new_test": C_ext, "C_diagnostics": Dg, "passline": {"auc_min": 0.85, "fpr_max": 0.10}}
OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("saved", OUT)
