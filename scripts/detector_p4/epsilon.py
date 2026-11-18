#!/usr/bin/env python3
"""P-4a epsilon_py: Python-vs-Python reproducibility floor (design 3.4-3) for the adopted config.
Three alternatives to the canonical float64 pipeline (fixtures/detp4_fixture.json is the canonical output):
  (1) summation order changed: scene rows randomly permuted before fit / LOO / scoring (seed 11), results mapped back to the original scene order
  (2) np.longdouble: NOTE on this platform (Windows, MSVC numpy) longdouble is binary64, so (2) is identical to canonical by construction (recorded, not independent)
  (3) 50-digit decimal reference of the full-corpus model on the orig scenes (independent high-precision check of float64 error; added because (2) is not independent here)
Compared quantities: mu_g, sig_w, cal, Z, tail_p, u, score, thresholds (B/P/N), p_OP rows; plus discrete flips (tail_p, flag = score >= threshold)."""
from __future__ import annotations
import json, sys
from decimal import Decimal, getcontext
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import measure as M  # noqa: E402
import detector_p4 as DP  # noqa: E402

getcontext().prec = 50
VERS = ["orig"] + M.KINDS


def run_pipeline(c, cfg, p_global, dtype=None):
    """Returns dict of quantities in c's own scene order."""
    if dtype is not None:
        c.RAW = {k: v.astype(dtype) for k, v in c.RAW.items()}
    works = list(range(len(c.uw)))
    fo = M.workset_folds(works, M.FOLD_SEED)
    fos = np.array([fo[w] for w in c.WI])
    full = DP.fit(c.RAW["orig"], c.WI, cfg)
    sfull = DP.score_parts(full, c.RAW["orig"], c.RAW["orig"], c.WI)
    pop = DP.p_op_procedure(sfull["score"], c.WI, rule=cfg.rule)
    q = {"full.mu_g": full.mu_g, "full.sig_w": full.sig_w, "full.cal": np.concatenate(full.cal), "full.Z": sfull["Z"], "full.tail_p": sfull["tail_p"],
         "full.u": sfull["u"], "full.score": sfull["score"], "full.pop.mean_fpr": [r["mean_fpr"] for r in pop["rows"]],
         "full.pop.se": [r["se"] for r in pop["rows"]], "full.pop.bound_value": [r["bound_value"] for r in pop["rows"]],
         "full.pop.p_op": [pop["p_op"]]}
    n = c.n
    for v in VERS:
        for key, shp in (("Z", 3), ("tail_p", 3), ("u", 3), ("score", 1)):
            q[f"folds.{v}.{key}"] = np.full((n, shp), np.nan)
    thr = {}
    for f in range(5):
        trw = [w for w in works if fo[w] != f]
        m, th, pn = M.thresholds(c, trw, M.FOLD_SEED + f, cfg, p_global)
        thr[f] = th
        q[f"fold{f}.mu_g"], q[f"fold{f}.sig_w"], q[f"fold{f}.cal"] = m.mu_g, m.sig_w, np.concatenate(m.cal)
        q[f"fold{f}.thr"] = [th["B"], th["P"], th["N"]]
        te = np.where(fos == f)[0]
        ro = c.RAW["orig"][te]
        for v in VERS:
            sp = DP.score_parts(m, c.RAW[v][te], ro, c.WI[te])
            for key in ("Z", "tail_p", "u"):
                q[f"folds.{v}.{key}"][te] = sp[key]
            q[f"folds.{v}.score"][te, 0] = sp["score"]
    q["_fos"], q["_thr"] = fos, thr
    return q


def to64(q):
    return {k: (np.asarray(v, dtype=np.float64) if not k.startswith("_") else v) for k, v in q.items()}


def compare(base, alt, perm=None):
    """Max abs diff per quantity group. perm: alt rows are in permuted order (alt_row i == original row perm[i]) for per-scene quantities."""
    out, flips = {}, {}
    for k, b in base.items():
        if k.startswith("_"):
            continue
        a = alt[k]
        if perm is not None and (k.startswith("folds.") or k in ("full.Z", "full.tail_p", "full.u", "full.score")):
            aa = np.empty_like(a)
            aa[perm] = a
            a = aa
        out[k] = float(np.nanmax(np.abs(a - b))) if np.size(b) else 0.0
        if k.endswith(".tail_p"):
            flips[k] = int(np.sum(a != b))
    # discrete flag flips
    fl = 0
    for v in VERS:
        for key in ("score",):
            s_b, s_a = base[f"folds.{v}.score"][:, 0], (alt[f"folds.{v}.score"][np.argsort(perm)][:, 0] if perm is not None else alt[f"folds.{v}.score"][:, 0])
            for f in range(5):
                te = base["_fos"] == f
                for i, o in enumerate(("B", "P", "N")):
                    fl += int(np.sum((s_b[te] >= base["_thr"][f][o]) != (s_a[te] >= alt["_thr"][f][o])))
    return out, flips, fl


def group_max(d):
    g = {}
    for k, v in d.items():
        key = k.split(".")[-1] if k.startswith(("folds.", "full.")) else k.split(".", 1)[-1]
        key = key.replace("pop.", "pop_")
        g[key] = max(g.get(key, 0.0), v)
    return g


# ---- (3) 50-digit decimal reference (full-corpus model, orig scenes, anscombe_shrink only) ----
def decimal_reference(c, cfg):
    assert cfg.q3mode == "anscombe" and cfg.shrink
    R = c.RAW["orig"]
    n = len(R)
    Dm = [[Decimal(float(R[i, j])) for j in range(3)] for i in range(n)]
    mu_g = [sum(Dm[i][j] for i in range(n)) / n for j in range(3)]
    num, den = [Decimal(0)] * 3, 0
    byw = [np.where(c.WI == w)[0] for w in np.unique(c.WI)]
    for ii in byw:
        k = len(ii)
        if k >= 2:
            for j in range(3):
                m = sum(Dm[i][j] for i in ii) / k
                num[j] += sum((Dm[i][j] - m) ** 2 for i in ii)  # = (k-1) * var(ddof=1)
            den += k - 1
    floor2 = Decimal(DP.FLOOR) ** 2
    sig_w = [max(num[j] / den, floor2).sqrt() for j in range(3)]
    m0 = Decimal(cfg.m0)
    Z = [[Decimal(0)] * 3 for _ in range(n)]
    for ii in byw:
        k = len(ii) - 1
        for i in ii:
            for j in range(2):
                if k <= 0:
                    mu, var = mu_g[j], sig_w[j] ** 2
                else:
                    others = [Dm[t][j] for t in ii if t != i]
                    m = sum(others) / k
                    s2 = sum((x - m) ** 2 for x in others) / (k - 1) if k >= 2 else Decimal(0)
                    mu = (k * m + m0 * mu_g[j]) / (k + m0)
                    w = max(k - 1, 0)
                    var = (w * s2 + m0 * sig_w[j] ** 2) / (w + m0)
                sd = max(var.sqrt(), Decimal(DP.FLOOR))
                Z[i][j] = -(Dm[i][j] - mu) / sd
    cnt = np.rint(R[:, 2] * np.maximum(R[:, 3], 1.0))
    for ii in byw:
        k = len(ii) - 1
        for i in ii:
            if k <= 0:
                muh = mu_g[2]
            else:
                m = sum(Dm[t][2] for t in ii if t != i) / k
                muh = (k * m + m0 * mu_g[2]) / (k + m0)
            lam = max(muh * Decimal(float(max(R[i, 3], 1.0))), Decimal(DP.LAM_FLOOR))
            Z[i][2] = 2 * (lam + Decimal("0.375")).sqrt() - 2 * (Decimal(float(cnt[i])) + Decimal("0.375")).sqrt()
    cal = [sorted(Z[i][j] for i in range(n)) for j in range(3)]
    import bisect
    P = [[Decimal(0)] * 3 for _ in range(n)]
    for j in range(3):
        for i in range(n):
            left = bisect.bisect_left(cal[j], Z[i][j])
            P[i][j] = (1 + (n - left)) / Decimal(n + 1)
    U = [[-p.ln() for p in row] for row in P]
    S = [max(r) for r in U]
    return {"full.mu_g": mu_g, "full.sig_w": sig_w, "full.Z": Z, "full.tail_p": P, "full.u": U, "full.score": S}


def main():
    res = json.loads((HERE / "detp4_results.json").read_text(encoding="utf-8"))
    pop = json.loads((HERE / "pop_registered.json").read_text(encoding="utf-8"))
    name = res["decision"]["adopted"]
    cfg = next(x for x in M.V13 if x.name == name)
    p_global = pop[name]["p_op"]
    path = M.REPO / "data" / "detp4_scene_signals.npz"
    base = to64(run_pipeline(M.Ctx(path, DP), cfg, p_global))
    report = {"cfg": name, "longdouble_bits": int(np.finfo(np.longdouble).bits), "methods": {}}

    perm = np.random.default_rng(11).permutation(M.Ctx(path, DP).n)
    c2 = M.Ctx(path, DP)
    c2.RAW = {k: v[perm] for k, v in c2.RAW.items()}
    c2.WI, c2.COH = c2.WI[perm], c2.COH[perm]
    q2 = to64(run_pipeline(c2, cfg, p_global))
    # alt per-scene arrays are in permuted row order; fold assignment arrays differ in order -> compare through mapping
    q2["_fos"] = q2["_fos"]
    d1, f1, fl1 = compare_perm(base, q2, perm)
    report["methods"]["reorder_rows_seed11"] = {"max_abs_diff": d1, "group_max": group_max(d1), "tail_p_flips": f1, "flag_flips": fl1}

    c3 = M.Ctx(path, DP)
    q3 = to64(run_pipeline(c3, cfg, p_global, dtype=np.longdouble))
    d2, f2, fl2 = compare(base, q3)
    report["methods"]["longdouble"] = {"max_abs_diff": d2, "group_max": group_max(d2), "tail_p_flips": f2, "flag_flips": fl2,
                                       "note": "longdouble is binary64 on this platform; not an independent check"}

    ref = decimal_reference(M.Ctx(path, DP), cfg)
    dd = {}
    for k, v in ref.items():
        a = np.array([[float(x) for x in row] if isinstance(row, list) else float(row) for row in v], dtype=float)
        b = base[k]
        a = a.reshape(b.shape) if a.size == b.size else a
        dd[k] = float(np.abs(a - b).max())
        if k == "full.tail_p":
            dd["tail_p_flips"] = int(np.sum(a != b))
    sdv = np.array([float(x) for x in ref["full.score"]])
    sf = base["full.score"]
    dd["flag_flips_at_p_global_float_vs_decimal"] = int(np.sum((sf >= np.percentile(sf, p_global)) != (sdv >= np.percentile(sdv, p_global))))
    report["methods"]["decimal50_full_model_orig"] = {"max_abs_diff": dd}

    # eps_py is defined on the CONTINUOUS quantities before the discrete calibration step: mu_g, sig_w, cal, Z.
    # tail_p / u / score / thresholds depend on exact comparisons against cal (ties and near-ties) and are reported separately (they can jump by one calibration rank).
    cont_keys = ("mu_g", "sig_w", "cal", "Z")
    cont = [max(v for k, v in report["methods"][m]["group_max"].items() if k in cont_keys) for m in ("reorder_rows_seed11", "longdouble")]
    dec = max(v for k, v in dd.items() if k in ("full.mu_g", "full.sig_w", "full.Z"))
    eps = max(cont + [dec])
    report["epsilon_py"] = {"continuous_quantities": list(cont_keys), "max_abs_diff_over_methods": eps,
                            "per_method": {"reorder": cont[0], "longdouble": cont[1], "decimal50": dec},
                            "rust_tolerance_rule": "max(100*eps_py, 1e-12)", "rust_tolerance": max(100 * eps, 1e-12),
                            "discrete_sensitivity": {"decimal50_vs_float64_tail_p_flips_full_model_orig": dd.get("tail_p_flips"),
                                                     "decimal50_vs_float64_max_abs_u_diff": dd.get("full.u"),
                                                     "reorder_tail_p_flips": sum(report["methods"]["reorder_rows_seed11"]["tail_p_flips"].values()),
                                                     "reorder_flag_flips": report["methods"]["reorder_rows_seed11"]["flag_flips"]},
                            "note": "tail_p / flags / p_OP are decided by exact comparison, not by tolerance (design 3.4-3); near-ties in Z (esp. q1 ending diversity, many exact ties) can flip tail_p by one rank under 1e-13 noise"}
    (HERE / "epsilon_py.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "methods"}, indent=1))
    for m, v in report["methods"].items():
        print(m, {k: (x if not isinstance(x, dict) else None) for k, x in v.items() if k != "max_abs_diff"}, v.get("group_max"))
    print("decimal", report["methods"]["decimal50_full_model_orig"]["max_abs_diff"])


def compare_perm(base, alt, perm):
    """alt computed on rows permuted by perm: alt row i is original row perm[i]. Fold-level (non per-scene) quantities compare directly."""
    # fold test selection in alt is by scene identity (fold determined by work), so map per-scene arrays back by perm.
    out, flips, fl = compare(base, alt, perm=perm)
    return out, flips, fl


if __name__ == "__main__":
    main()
