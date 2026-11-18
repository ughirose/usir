"""DETV4 unit tests: q3' components A/B, Hurdle weight, p_OP decision rule, leakage (design doc §4, §5, §8)."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "detector_v4"))
sys.path.insert(0, str(REPO / "scripts" / "detector_v3"))
import importlib.util  # noqa: E402
import detector_v4 as D  # noqa: E402

_spec = importlib.util.spec_from_file_location("detv4_signals", REPO / "scripts" / "detector_v4" / "signals.py")
S4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S4)  # loaded under a unique name (v3 also has a module called "signals")

RNG = np.random.default_rng(0)


def make_raw(n_works=6, per=12):
    rows, w = [], []
    for i in range(n_works):
        for _ in range(per):
            ls = RNG.uniform(20, 60)
            lnarr = RNG.uniform(200, 1500)
            rows.append([RNG.uniform(0.02, 0.08), RNG.uniform(0, 0.1), RNG.uniform(0.3, 1), ls, RNG.uniform(0, 0.5),
                         RNG.uniform(0, 1), RNG.uniform(0.005, 0.05), RNG.integers(0, 6), lnarr])
            w.append(i)
    return np.array(rows, float), np.array(w)


# --- q3' components A and B (signal definitions) ---
def test_component_a_density_excludes_dialogue():
    sense = next(iter(S4.S3._SENSE))
    t = "彼は歩いた。" + sense + "が見えた。「" + sense * 20 + "」と言った。"
    v = S4.signals(t)
    nar = S4.S3._DLG.sub("", S4.S3._WS.sub("", t))
    expected = sum(c in S4.S3._SENSE for c in nar) / len(nar)
    assert abs(v[6] - expected) < 1e-12
    assert v[8] == len(nar)
    assert v[6] < sum(c in S4.S3._SENSE for c in t) / len(t)  # dialogue sensory chars not counted


def test_component_b_longest_nonsensory_run():
    sense = next(iter(S4.S3._SENSE))
    plain = "彼は歩いた。"
    assert not any(c in S4.S3._SENSE for c in plain)
    t = plain * 3 + sense + "。" + plain * 5 + sense + "。" + plain
    assert S4.signals(t)[7] == 5.0
    assert S4.signals(sense + "。" + sense + "。")[7] == 0.0


# --- Hurdle weight ---
def test_hurdle_boundary_and_monotone():
    assert D.hurdle_weight(0.0, 2.0) == 0.0
    e = np.linspace(0, 50, 200)
    w = D.hurdle_weight(e, 2.0)
    assert np.all(np.diff(w) > 0) and w[-1] < 1.0 + 1e-12 and np.all(w >= 0)
    assert abs(D.hurdle_weight(2.0, 2.0) - (1 - np.exp(-1))) < 1e-12
    assert D.hurdle_weight(3.0, 1.0) > D.hurdle_weight(3.0, 4.0)  # smaller tau -> larger weight


def test_zero_sensory_work_gets_zero_weight_and_zero_q3():
    raw, widx = make_raw()
    m = D.fit(raw, widx, D.Cfg())
    x = raw.copy()
    x[widx == 0, D.RATE_COL] = 0.0  # work 0 has no sensory chars anywhere -> E_count = 0
    p = D.score_parts(m, x, x, widx)
    assert np.all(p["e_count"][widx == 0] == 0.0) and np.all(p["w"][widx == 0] == 0.0)
    assert np.all(p["u3"][widx == 0] == 0.0)


def test_mu_loo_excludes_self():
    rate = np.array([1.0, 2.0, 3.0, 10.0])
    out = D.loo_mean_rate(rate, 99.0)
    assert np.allclose(out, [5.0, 14 / 3, 13 / 3, 2.0])  # mean of the other three values
    rate2 = rate.copy(); rate2[0] = 1000.0
    assert D.loo_mean_rate(rate2, 99.0)[0] == out[0]  # own value does not enter own mu_LOO
    assert D.loo_mean_rate(np.array([4.0]), 99.0)[0] == 99.0


# --- p_OP decision rule ---
def test_p_op_rule_picks_largest_satisfying():
    widx = np.repeat(np.arange(10), 20)
    scores = np.tile(np.arange(20, dtype=float), 10)  # identical works -> FPR_w(p) deterministic
    r = D.p_op_procedure(scores, widx, grid=np.array([80.0, 90.0, 95.0]))
    ok = {x["p"]: x["ok"] for x in r["rows"]}
    sat = [p for p, v in ok.items() if v]
    assert r["p_op"] == (max(sat) if sat else 80.0)
    assert r["p_op"] == 95.0  # FPR falls as p rises, so the largest grid p satisfies the bound here
    for x in r["rows"]:
        assert abs(x["bound_value"] - (x["mean_fpr"] + 1.28 * x["se"])) < 1e-12


def test_p_op_falls_back_to_floor_when_nothing_satisfies():
    widx = np.repeat(np.arange(4), 5)
    scores = np.zeros(20)  # all ties -> every scene >= threshold -> FPR = 1
    r = D.p_op_procedure(scores, widx, grid=np.array([80.0, 85.0]))
    assert r["p_op"] == 80.0 and r["fallback_to_floor"]


def test_p_op_threshold_uses_only_other_works():
    widx = np.repeat(np.arange(3), 10)
    s = np.concatenate([np.arange(10.0), np.arange(10.0) + 100, np.arange(10.0) + 50])
    r = D.p_op_procedure(s, widx, grid=np.array([90.0]))
    f = [float(np.mean(s[widx == w] >= np.percentile(s[widx != w], 90.0))) for w in range(3)]
    assert abs(r["rows"][0]["mean_fpr"] - np.mean(f)) < 1e-12
    assert f[0] == 0.0 and f[1] > 0.0  # work 1 (highest scores) exceeds the threshold built from the others


# --- leakage ---
def test_other_works_unaffected_by_a_scene_change():
    raw, widx = make_raw()
    m = D.fit(raw, widx, D.Cfg())
    i = 3
    x2 = raw.copy()
    x2[i, [0, 2, 6, 7, 8]] += 5.0
    a = D.score(m, raw, raw, widx)
    b = D.score(m, x2, x2, widx)
    other_works = np.where(widx != widx[i])[0]
    assert np.allclose(a[other_works], b[other_works])  # scenes of other works never use scene i
    same_work_others = [j for j in np.where(widx == widx[i])[0] if j != i]
    assert any(abs(a[j] - b[j]) > 0 for j in same_work_others)  # same-work scenes use i as reference (by design)


def test_own_ref_value_excluded_from_own_score():
    raw, widx = make_raw()
    m = D.fit(*make_raw(), D.Cfg())  # fitted on other data: no exact ties between scored values and calibration values
    i = 5
    ref2 = raw.copy()
    ref2[i, [0, 2, 6, 7, 8]] += 7.0  # scene i's own reference row changes; x_i itself fixed
    a = D.score_parts(m, raw, raw, widx)
    b = D.score_parts(m, raw, ref2, widx)
    assert abs(a["score"][i] - b["score"][i]) < 1e-12  # LOO: own row never enters own baseline
    assert abs(a["e_count"][i] - b["e_count"][i]) < 1e-12


def test_training_data_changes_calibration_only_via_fit():
    raw, widx = make_raw()
    m1 = D.fit(raw, widx, D.Cfg())
    tr2 = raw.copy()
    tr2[:, 0] = tr2[:, 0] ** 3 * 1e3 + RNG.normal(0, 1, len(tr2))  # changes the comma-density distribution shape
    m2 = D.fit(tr2, widx, D.Cfg())
    assert not np.allclose(m1.cal[D.IQ2], m2.cal[D.IQ2])
    assert np.allclose(m1.cal[D.IA], m2.cal[D.IA])  # unrelated signal calibration unchanged
