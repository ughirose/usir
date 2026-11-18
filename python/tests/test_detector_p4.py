"""P-4a unit tests: v13 reference detector (Poisson / Anscombe q3, LOO EB, minimum-p rule), v4 reproduction, fixture self-consistency."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import poisson

REPO = Path(__file__).resolve().parents[2]
P4 = REPO / "scripts" / "detector_p4"
sys.path.insert(0, str(P4))
import detector_p4 as DP  # noqa: E402

RNG = np.random.default_rng(0)


def make_raw(n_works=6, per=12):
    rows, w = [], []
    for i in range(n_works):
        for _ in range(per):
            L = float(RNG.integers(200, 1600))
            rate = RNG.uniform(0.005, 0.05)
            rows.append([RNG.uniform(0.3, 1), RNG.uniform(0, 0.1), np.rint(rate * L) / L, L])
            w.append(i)
    return np.array(rows, float), np.array(w)


def test_loo_normalize_matches_v4_formula():
    x = RNG.uniform(0, 1, (9, 3))
    mu_g, sig_w = x.mean(0), np.array([0.2, 0.3, 0.1])
    a = DP.loo_normalize(x, x, mu_g, sig_w, 8.0)
    k = 8
    i = 3
    others = np.delete(x, i, 0)
    m, s2 = others.mean(0), others.var(0, ddof=1)
    mu = (k * m + 8.0 * mu_g) / (k + 8.0)
    var = ((k - 1) * s2 + 8.0 * sig_w ** 2) / ((k - 1) + 8.0)
    np.testing.assert_allclose(a[i], -(x[i] - mu) / np.sqrt(var), rtol=1e-9, atol=1e-12)


def test_single_scene_work_uses_global_fallback():
    x = np.array([[0.5, 0.02, 0.01]])
    mu_g, sig_w = np.array([0.6, 0.03, 0.02]), np.array([0.1, 0.01, 0.01])
    z = DP.loo_normalize(x, x, mu_g, sig_w, 8.0)
    np.testing.assert_allclose(z[0], -(x[0] - mu_g) / sig_w)


def test_poisson_logcdf_matches_scipy():
    lam = RNG.uniform(0.01, 90, 400)
    c = RNG.integers(0, 160, 400).astype(float)
    np.testing.assert_allclose(DP.poisson_lower_logp(c, lam), poisson.logcdf(c, lam), atol=1e-11)


def test_tail_p_and_score_are_max_of_u():
    raw, w = make_raw()
    for mode in ("poisson", "anscombe", "rate"):
        m = DP.fit(raw, w, DP.Cfg("t", mode))
        sp = DP.score_parts(m, raw, raw, w)
        np.testing.assert_allclose(sp["score"], sp["u"].max(1))
        assert np.all(sp["tail_p"] > 0) and np.all(sp["tail_p"] <= 1) and np.all(np.isfinite(sp["score"]))


def test_anomaly_direction_lower_sensory_is_higher_score():
    raw, w = make_raw()
    for mode in ("poisson", "anscombe"):
        m = DP.fit(raw, w, DP.Cfg("t", mode))
        base = DP.score_parts(m, raw, raw, w)["u"][:, 2]
        pert = raw.copy()
        pert[:, 2] = 0.0  # sensory characters removed
        assert DP.score_parts(m, pert, raw, w)["u"][:, 2].mean() > base.mean()


def test_p_op_min_rule_and_failure():
    scores = RNG.normal(size=600)
    widx = np.repeat(np.arange(12), 50)
    r = DP.p_op_procedure(scores, widx)
    ok = [x["p"] for x in r["rows"] if x["ok"]]
    assert not r["failed"] and r["p_op"] == min(ok)
    r_max = DP.p_op_procedure(scores, widx, rule="max")
    assert r_max["p_op"] == max(ok)
    # infeasible bound -> failure (never a silent floor)
    bad = DP.p_op_procedure(scores, widx, bound=-1.0)
    assert bad["failed"] and bad["p_op"] is None


def test_grid_is_80_to_97_step_half():
    assert DP.GRID[0] == 80.0 and DP.GRID[-1] == 97.0 and len(DP.GRID) == 35


def test_signals_npz_has_l_total_and_integral_counts():
    d = np.load(REPO / "data" / "detp4_scene_signals.npz")
    for v in ("orig", "ending", "comma", "sensory"):
        a = d[v]
        assert a.shape[1] == 4 and a[:, 3].min() >= 1
        cnt = a[:, 2] * a[:, 3]
        assert np.abs(cnt - np.rint(cnt)).max() < 1e-9
    v4 = np.load(REPO / "data" / "detv4_scene_signals.npz")
    np.testing.assert_array_equal(d["orig"][:, :3], v4["orig"][:, [2, 0, 1]])


def test_fixture_self_consistency():
    fx = json.loads((P4 / "fixtures" / "detp4_fixture.json").read_text(encoding="utf-8"))
    assert "text" not in fx["scenes"]
    cfg = DP.Cfg(fx["meta"]["cfg"], fx["meta"]["q3mode"], shrink=fx["meta"]["shrink"])
    raw = np.array(fx["scenes"]["raw"]["orig"])
    keys = sorted(set(fx["scenes"]["work_id"]))
    widx = np.array([keys.index(k) for k in fx["scenes"]["work_id"]])
    m = DP.fit(raw, widx, cfg)
    np.testing.assert_array_equal(m.mu_g, np.array(fx["full_model"]["mu_g"]))
    np.testing.assert_array_equal(np.concatenate(m.cal), np.concatenate([np.array(c) for c in fx["full_model"]["cal"]]))
    f0 = fx["folds"][0]
    te = np.array(f0["test_scene_index"])
    sp = DP.score_parts(m, raw[te], raw[te], widx[te])  # full-model scoring only used for shape check
    assert sp["Z"].shape == np.array(f0["test"]["orig"]["Z"]).shape
    # per-fold stored scores satisfy score = max(u) and u = -ln(tail_p)
    for v in fx["meta"]["versions"]:
        t = f0["test"][v]
        np.testing.assert_allclose(np.array(t["score"]), np.array(t["u"]).max(1), atol=0)
        np.testing.assert_allclose(np.array(t["u"]), -np.log(np.array(t["tail_p"])), atol=1e-15)


def test_v4_reproduction_headline_values():
    r = json.loads((P4 / "detv4_repro_results.json").read_text(encoding="utf-8"))
    assert abs(r["all"]["pooled_auc"] - 0.8602617635251695) < 1e-12
    assert abs(r["all"]["per_op"]["P"]["fpr"] - 0.030367607884922748) < 1e-12
    assert r["diff_vs_reference"]["pooled_auc"] == 0.0


def test_preregistered_decision_rule():
    sys.path.insert(0, str(P4))
    import measure as M

    def mk(name, lo, fpr=0.07):
        return {"cfg": name, "all": {"auc_ci95_work_cluster": {"pooled": [lo, 0.9]}, "per_op": {"P": {"fpr": fpr}, "N": {"fpr": fpr}}}}

    # A wins on a tie at 4 decimals; stage-1 tie -> no shrink
    cf = {"A_poisson_noshrink": mk("A_poisson_noshrink", 0.86801), "A_poisson_shrink": mk("A_poisson_shrink", 0.86804),
          "B_anscombe_noshrink": mk("B_anscombe_noshrink", 0.86803), "B_anscombe_shrink": mk("B_anscombe_shrink", 0.86804)}
    assert M.decide(cf)["adopted"] == "A_poisson_noshrink"
    # a clearly higher CI lower bound wins; an FPR violation makes it ineligible
    cf["B_anscombe_noshrink"] = mk("B_anscombe_noshrink", 0.90)
    assert M.decide(cf)["adopted"] == "B_anscombe_noshrink"
    cf["B_anscombe_noshrink"] = mk("B_anscombe_noshrink", 0.90, fpr=0.2)
    assert M.decide(cf)["adopted"] == "A_poisson_noshrink"
    cf["A_poisson_noshrink"] = mk("A_poisson_noshrink", 0.99, fpr=0.2)
    cf["A_poisson_shrink"] = mk("A_poisson_shrink", 0.99, fpr=0.2)
    assert M.decide(cf)["adopted"] is None
