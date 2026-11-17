"""DETV3 unit tests: signal definitions, LOO EB normalization, leakage guarantees (design doc §6)."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "detector_v3"))
sys.path.insert(0, str(REPO / "python"))
import detector_v3 as D  # noqa: E402

RNG = np.random.default_rng(0)


def make_raw(n_works=6, per=12):
    rows, w = [], []
    for i in range(n_works):
        base = RNG.normal(0, 0.01, 1) + 0.04
        for _ in range(per):
            ls = RNG.uniform(20, 60)
            dr = RNG.uniform(0, 0.5)
            q1 = base + 0.001 * ls + RNG.normal(0, 0.005)
            rows.append([q1[0], RNG.uniform(0, 0.1), RNG.uniform(0.3, 1), ls, dr, RNG.uniform(0, 1), RNG.uniform(0, 0.1), RNG.integers(0, 6)])
            w.append(i)
    return np.array(rows, float), np.array(w)


def test_loo_excludes_self_and_shrinks():
    ref = RNG.normal(size=(10, 7))
    mu_g, sig_w = np.zeros(7), np.ones(7)
    x = ref.copy()
    z = D.loo_normalize(x, ref, mu_g, sig_w, 8.0)
    i = 3
    k = 9
    others = np.delete(ref, i, 0)
    m, s2 = others.mean(0), others.var(0, ddof=1)
    mu = (k * m + 8 * mu_g) / (k + 8)
    var = ((k - 1) * s2 + 8 * sig_w ** 2) / (k - 1 + 8)
    np.testing.assert_allclose(z[i], D.SGN * (x[i] - mu) / np.sqrt(var), rtol=1e-10)


def test_loo_row_value_does_not_affect_own_baseline():
    ref = RNG.normal(size=(10, 7))
    a = D.loo_normalize(ref, ref, np.zeros(7), np.ones(7), 8.0)
    ref2 = ref.copy()
    ref2[4] += 100.0  # change only scene 4's reference value
    b = D.loo_normalize(ref, ref2, np.zeros(7), np.ones(7), 8.0)
    np.testing.assert_allclose(a[4], b[4])  # scene 4's own baseline excludes row 4
    assert not np.allclose(a[5], b[5])      # others' baselines include row 4


def test_single_scene_work_falls_back_to_prior():
    x = np.ones((1, 7))
    z = D.loo_normalize(x, x, np.zeros(7), np.full(7, 2.0), 8.0)
    np.testing.assert_allclose(z[0], D.SGN * 0.5)


def test_no_leak_from_test_fold_into_fit_and_score():
    raw, w = make_raw()
    tr, te = w < 4, w >= 4
    cfg = D.Cfg()
    m1 = D.fit(raw[tr], w[tr], cfg)
    raw2 = raw.copy()
    raw2[te] += 50.0  # arbitrary change of test-fold data must not alter the fitted model
    m2 = D.fit(raw2[tr], w[tr], cfg)
    np.testing.assert_allclose(m1.coef, m2.coef)
    for a, b in zip(m1.cal, m2.cal):
        np.testing.assert_array_equal(a, b)
    # scoring work 4 depends only on work 4's own rows
    idx4 = np.where(w == 4)[0]
    s_a = D.score(m1, raw[idx4], raw[idx4], w[idx4])
    r5 = raw.copy()
    r5[w == 5] += 50.0
    s_b = D.score(m1, r5[idx4], r5[idx4], w[idx4])
    np.testing.assert_allclose(s_a, s_b)


def test_score_monotone_in_anomaly_direction():
    raw, w = make_raw()
    m = D.fit(raw, w, D.Cfg(sig=[0]))
    idx = np.where(w == 0)[0]
    lo = raw[idx].copy()
    lo[:, 0] -= 0.05  # comma density drop (anomalous direction)
    hi = raw[idx].copy()
    hi[:, 0] += 0.05
    s_lo = D.score(m, lo, raw[idx], w[idx])
    s_hi = D.score(m, hi, raw[idx], w[idx])
    assert s_lo.mean() > D.score(m, raw[idx], raw[idx], w[idx]).mean() > s_hi.mean()


def test_signals_definitions():
    import re
    sys.path.insert(0, str(REPO / "scripts" / "detector_v3"))
    import signals as S
    t = "彼は静かに歩いた。「こんにちは」と言った。香りが漂う、風が吹く。" + "あ" * 35 + "。"
    v = dict(zip(S.NAMES, S.signals(t)))
    assert 0 <= v["DR"] <= 1
    assert v["S1b"] > 0  # the 35-char comma-less sentence counts as long comma-less
    assert v["S2b"] >= 0
    no_comma = S.signals("あ" * 40 + "。")
    assert dict(zip(S.NAMES, no_comma))["S1b"] == 1.0
    assert dict(zip(S.NAMES, S.signals("。")))["S1b"] in (0.0, 1.0)
