"""Unit tests for usir_py.v5.faw_continuous (Stage FAW-IMPL)."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from usir_py.v5 import faw_continuous as F
from usir_py.v5.six_pole_atlas import compute_nagare_profile, compute_awase_profile

REPO = Path(__file__).resolve().parents[3]


def aw_scene(vf, vs, af=0, as_=0):
    return {"awase": {"f_event_valence": vf, "f_style_valence": vs, "f_event_arousal": af, "f_style_arousal": as_}}


def nag_scene(e_logp, m_logp):
    return {"nagare": {"event_logp": list(e_logp), "mood_logp": list(m_logp)}}


def onehot(i, hi=8.0):
    v = [0.0] * 16
    v[i] = hi
    return v


def test_aw1_empty_is_none_single_scene_defined():
    assert F.aw1_signed_counterpoint([]) is None
    r = F.aw1_signed_counterpoint([aw_scene(2, -2)])
    assert r["c_irony"] == pytest.approx(1.0)  # -(1)(-1)
    assert r["c_det"] == pytest.approx(0.0)


def test_aw1_keeps_sign_unlike_legacy_relu():
    sc = [aw_scene(2, 2), aw_scene(2, -2)]  # s = -1, +1
    r = F.aw1_signed_counterpoint(sc)
    assert r["c_irony"] == pytest.approx(0.0)
    assert compute_awase_profile(sc)["irony"]["mean"] == pytest.approx(0.5)  # legacy unchanged


def test_aw1_c_det_signed_range():
    sc = [aw_scene(2, 0, af=-2, as_=2)]  # |1| * (-1 - 1) = -2
    assert F.aw1_signed_counterpoint(sc)["c_det"] == pytest.approx(-2.0)


def test_aw1_zero_signal_is_zero_not_none():
    r = F.aw1_signed_counterpoint([aw_scene(0, 0)] * 4)
    assert r == {"c_irony": 0.0, "c_det": 0.0}


def test_aw4_none_below_three_scenes():
    assert F.aw4_irony_slope([]) is None
    assert F.aw4_irony_slope([aw_scene(2, -2)] * 2) is None


def test_aw4_slope_linear_series():
    # s_t = -V_F V_f ; V_F fixed 1.0 (event=2); V_f = -s_t. Want s = [-1,0,1] -> V_f = [1,0,-1] -> style = [2,0,-2]
    sc = [aw_scene(2, 2), aw_scene(2, 0), aw_scene(2, -2)]
    assert F.aw4_irony_slope(sc) == pytest.approx(2.0)  # rise of 2 over normalized span 1


def test_aw4_zero_signal_is_zero():
    assert F.aw4_irony_slope([aw_scene(0, 0)] * 5) == 0.0


def test_aw4_no_scene_count_branching():
    # slope for T=3..40 on constant series is exactly 0 (one formula for all T)
    for t in range(3, 41):
        assert F.aw4_irony_slope([aw_scene(2, 2)] * t) == pytest.approx(0.0, abs=1e-12)


def test_ng_none_contracts():
    one = [nag_scene(onehot(0), onehot(0))]
    assert F.ng1_smooth_still(one) is None
    assert F.mood_speed_mean_ii_a(one) is None
    assert F.phi_zeami_ii_a(one) is None
    assert F.ng2_speed_slope(one) is None
    two = [nag_scene(onehot(0), onehot(0)), nag_scene(onehot(1), onehot(1))]  # 1 transition
    assert F.ng1_smooth_still(two) is not None
    assert F.ng2_speed_slope(two) is None  # < 3 transitions
    assert F.phi_zeami_ii_a(two) == pytest.approx(0.5)  # equal event and mood speeds


def test_phi_zero_denominator_is_none_not_default():
    same = [nag_scene(onehot(2), onehot(2))] * 4
    assert compute_nagare_profile(same)["phi_zeami"] == 0.5  # legacy substitutes 0.5 (unchanged)
    assert F.phi_zeami_ii_a(same) is None
    assert F.mood_speed_mean_ii_a(same) == pytest.approx(0.0, abs=1e-6)


def test_ng1_static_is_one_and_decreases_with_motion():
    static = [nag_scene(onehot(3), onehot(3))] * 5
    assert F.ng1_smooth_still(static) == pytest.approx(1.0, abs=1e-6)
    moving = [nag_scene(onehot(i % 16), onehot((i * 5) % 16)) for i in range(6)]
    v = F.ng1_smooth_still(moving)
    assert 0.0 < v < 0.01  # large jumps: exp(-(J/0.25)^2) ~ 0
    assert v > 0.0  # strictly positive (no hard zero)


def test_ng1_theta_monotone():
    ev = [onehot(0, 1.0), onehot(1, 1.0), onehot(2, 1.0), onehot(3, 1.0)]
    sc = [nag_scene(e, e) for e in ev]
    a, b, c = (F.ng1_smooth_still(sc, th) for th in (0.20, 0.25, 0.35))
    assert a < b < c


def test_ng1_matches_manual_formula():
    sc = [nag_scene(onehot(0, 1.0), onehot(0, 1.0)), nag_scene(onehot(1, 1.0), onehot(0, 1.0)), nag_scene(onehot(1, 1.0), onehot(0, 1.0))]
    prof = compute_nagare_profile(sc)
    j = [max(e, m) for e, m in zip(prof["event"]["speed"], prof["mood"]["speed"])]
    exp = float(np.mean([math.exp(-((x / 0.25) ** 2)) for x in j]))
    assert F.ng1_smooth_still(sc) == pytest.approx(exp)


def test_ng2_slope_sign():
    # mood speeds increasing: scenes drift with growing step
    mood = [onehot(0, 6.0)]
    seq = []
    base = np.zeros(16)
    for k in range(6):
        v = np.zeros(16)
        v[0] = 6.0
        v[1] = 0.6 * k * k  # increasing steps in logit space
        seq.append(nag_scene(onehot(0), list(v)))
    assert F.ng2_speed_slope(seq) > 0
    assert F.ng2_speed_slope(seq[::-1]) < 0


def test_ng0_equals_legacy_on_frozen_annotations():
    path = REPO / "data" / "v5" / "aozora_106works_scenes.jsonl"
    if not path.exists():
        pytest.skip("frozen annotations not available")
    works = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            works.setdefault((r["author_id"], r["work_id"]), []).append(r)
    maxd_phi = maxd_mood = 0.0
    n = 0
    for key, sc in list(works.items())[:40]:
        sc.sort(key=lambda r: r["scene_index"])
        leg = compute_nagare_profile(sc)
        mo = leg["mood"]["speed"]
        if not mo:
            continue
        n += 1
        maxd_phi = max(maxd_phi, abs(F.phi_zeami_ii_a(sc) - leg["phi_zeami"]))
        maxd_mood = max(maxd_mood, abs(F.mood_speed_mean_ii_a(sc) - float(np.mean(mo))))
    assert n > 0
    assert maxd_phi < 1e-9 and maxd_mood == 0.0


def test_compute_faw_continuous_keys_and_none_for_empty():
    out = F.compute_faw_continuous([])
    assert set(out) == {"aw1_c_irony", "aw1_c_det", "aw4_g", "ng0_phi_zeami", "ng0_mood_speed_mean", "ng1_s_still",
                        "ng2_speed_slope", "ng1_s_still_th020", "ng1_s_still_th035"}
    assert all(v is None for v in out.values())
