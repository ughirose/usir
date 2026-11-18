#!/usr/bin/env python3
"""P-4a fixtures for the Rust port (P-4b): numbers only, no raw text (design 3.4).
Usage: fixtures.py [cfg_name]  (default: the adopted config in detp4_results.json)
Writes fixtures/detp4_fixture.json:
  meta, scenes (work id, fold, raw [q1,q2,q3,L_total] for orig + 3 perturbations),
  full_model (fit on all 56 works: mu_g, sig_w, cal, p_OP rows),
  folds (per fold: train works, mu_g, sig_w, cal, thresholds B/P/N, nested p_OP; per test scene and version: Z, tail_p, u, score),
  final (pooled / per-type AUC + work-cluster CI from detp4_results.json).
Floats are written with repr (shortest round-trip), so they are bit-exact when read back as binary64."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import measure as M  # noqa: E402
import detector_p4 as DP  # noqa: E402

OUT = HERE / "fixtures" / "detp4_fixture.json"


def fl(a):
    return np.asarray(a, float).tolist()


def build(cfg_name=None):
    res = json.loads((HERE / "detp4_results.json").read_text(encoding="utf-8"))
    pop = json.loads((HERE / "pop_registered.json").read_text(encoding="utf-8"))
    cfg_name = cfg_name or res["decision"]["adopted"]
    cfg = next(x for x in M.V13 if x.name == cfg_name)
    c = M.Ctx(M.REPO / "data" / "detp4_scene_signals.npz", DP)
    p_global = pop[cfg.name]["p_op"]
    works = list(range(len(c.uw)))
    fo = M.workset_folds(works, M.FOLD_SEED)
    fos = np.array([fo[w] for w in c.WI])
    versions = ["orig"] + M.KINDS

    full = DP.fit(c.RAW["orig"], c.WI, cfg)
    s_full = DP.score_parts(full, c.RAW["orig"], c.RAW["orig"], c.WI)
    out = {"meta": {"schema": "detp4_fixture/1", "cfg": cfg.name, "q3mode": cfg.q3mode, "shrink": cfg.shrink, "m0": cfg.m0,
                    "FLOOR": DP.FLOOR, "LAM_FLOOR": DP.LAM_FLOOR, "Z_SE": DP.Z_SE, "FPR_BOUND": DP.FPR_BOUND, "grid": fl(DP.GRID),
                    "raw_columns": ["q1_ending_diversity", "q2_comma_density", "q3_sensory_rate", "L_total"], "versions": versions,
                    "seeds": res["seeds"], "p_op_registered": p_global,
                    "notes": ["q1,q2,q3 all lower=anomalous; Z columns are anomaly-oriented (higher = more anomalous)",
                              "q3 count c = rint(q3 * max(L_total,1)); anscombe z3 = 2*sqrt(lam+3/8) - 2*sqrt(c+3/8), lam = max(mu_hat*L_total, LAM_FLOOR)",
                              "tail_p = (1 + #{cal >= z}) / (n_cal + 1); u = -ln(tail_p); score = max_k u_k",
                              "percentile = numpy linear interpolation; AUC = average-rank (scipy rankdata)"]},
           "scenes": {"work_id": [f"{a}_{w}" for a, w in zip(c.AID, c.WID)], "fold": fos.tolist(), "cohort": c.COH.tolist(),
                      "raw": {v: c.RAW[v].tolist() for v in versions}},
           "full_model": {"mu_g": fl(full.mu_g), "sig_w": fl(full.sig_w), "cal": [fl(x) for x in full.cal],
                          "orig_Z": s_full["Z"].tolist(), "orig_tail_p": s_full["tail_p"].tolist(), "orig_u": s_full["u"].tolist(),
                          "orig_score": fl(s_full["score"]), "p_op_procedure": M.DP.p_op_procedure(s_full["score"], c.WI, rule=cfg.rule)},
           "folds": []}
    for f in range(5):
        trw = [w for w in works if fo[w] != f]
        m, thr, p_nested = M.thresholds(c, trw, M.FOLD_SEED + f, cfg, p_global)
        te = np.where(fos == f)[0]
        ro = c.RAW["orig"][te]
        fr = {"fold": f, "train_works": [c.uw[w] for w in trw], "test_scene_index": te.tolist(), "mu_g": fl(m.mu_g), "sig_w": fl(m.sig_w),
              "cal": [fl(x) for x in m.cal], "thresholds": thr, "p_op_nested": p_nested, "test": {}}
        for v in versions:
            sp = DP.score_parts(m, c.RAW[v][te], ro, c.WI[te])  # scenario A: reference = orig scenes of the same work
            fr["test"][v] = {"Z": sp["Z"].tolist(), "tail_p": sp["tail_p"].tolist(), "u": sp["u"].tolist(), "score": fl(sp["score"])}
        out["folds"].append(fr)
    a = res["configs"][cfg.name]["all"]
    out["final"] = {"pooled_auc": a["pooled_auc"], "per_type_auc": a["per_type_auc"], "auc_ci95_work_cluster": a["auc_ci95_work_cluster"],
                    "fpr": {o: a["per_op"][o]["fpr"] for o in a["per_op"]},
                    "note": "CI is the Python bootstrap value (canonical). Rust is required to match the AUC point estimate only (design 3.5)."}
    return out


def main():
    out = build(sys.argv[1] if len(sys.argv) > 1 else None)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print("saved", OUT, f"{OUT.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
