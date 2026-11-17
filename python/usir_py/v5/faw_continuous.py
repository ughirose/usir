"""Stage FAW-IMPL: continuous auxiliary metrics for Awase / Nagare (Tier 1), run alongside legacy metrics.

Definitions follow docs/handoff/2026-11-17_usir_v5_faw_design.md (section B). Legacy metrics in six_pole_atlas.py
are not modified; Nagare speeds are taken from compute_nagare_profile so that the sphere lift is shared.

Degenerate-input contract (ii-a): None is returned only when the input is missing
(too few scenes / transitions). No default value is substituted and no epsilon is added to a denominator.
A work whose signal is 0 keeps the value 0 (it is not dropped).

Metrics
- AW-1  aw1_signed_counterpoint: C_irony = mean(-V_F V_f) in [-1, 1]; C_det = mean(|V_F| (A_F - A_f)) in [-2, 2].
        None only for 0 scenes.
- AW-4  aw4_irony_slope: OLS slope of s_t = -V_F V_f on x_t = t/(T-1) - 1/2. None for T < 3.
- NG-0  phi_zeami_ii_a, mood_speed_mean_ii_a: legacy values, but None for 0 transitions (and for phi, None
        when the denominator is exactly 0; legacy returned 0.5).
- NG-1  ng1_smooth_still: mean_t exp(-(J_t/theta)^2), J_t = max(event speed, mood speed), theta = 0.25
        (sensitivity: 0.20, 0.35). None for 0 transitions.
- NG-2  ng2_speed_slope: OLS slope of mood speed on x_k = k/(K-1) - 1/2 (K transitions). None for K < 3.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from usir_py.v5.six_pole_atlas import compute_nagare_profile, norm_val

NG1_THETA = 0.25
NG1_THETA_SENSITIVITY = (0.20, 0.35)
SLOPE_MIN_POINTS = 3


def _ols_slope_centered(y: Sequence[float]) -> Optional[float]:
    """OLS slope of y on x_i = i/(n-1) - 1/2 (n >= 3). The denominator depends only on n."""
    n = len(y)
    if n < SLOPE_MIN_POINTS:
        return None
    x = np.arange(n, dtype=float) / (n - 1) - 0.5
    yv = np.asarray(y, dtype=float)
    sxx = float(np.sum(x * x))  # x has zero mean by construction
    return float(np.sum(x * (yv - yv.mean())) / sxx)


def awase_scene_terms(scenes: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """Per-scene signed terms s_t = -V_F V_f and d_t = |V_F| (A_F - A_f) (no clipping)."""
    s_t: List[float] = []
    d_t: List[float] = []
    for s in scenes:
        a = s.get("awase", {})
        vf = norm_val(a.get("f_event_valence", 0))
        vs = norm_val(a.get("f_style_valence", 0))
        af = norm_val(a.get("f_event_arousal", 0))
        as_ = norm_val(a.get("f_style_arousal", 0))
        s_t.append(float(-vf * vs))
        d_t.append(float(abs(vf) * (af - as_)))
    return {"c_irony": s_t, "c_det": d_t}


def aw1_signed_counterpoint(scenes: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    if len(scenes) == 0:
        return None
    t = awase_scene_terms(scenes)
    return {"c_irony": float(np.mean(t["c_irony"])), "c_det": float(np.mean(t["c_det"]))}


def aw4_irony_slope(scenes: List[Dict[str, Any]]) -> Optional[float]:
    if len(scenes) < SLOPE_MIN_POINTS:
        return None
    return _ols_slope_centered(awase_scene_terms(scenes)["c_irony"])


def _speeds(scenes: List[Dict[str, Any]]):
    prof = compute_nagare_profile(scenes)
    return prof["event"]["speed"], prof["mood"]["speed"]


def phi_zeami_ii_a(scenes: List[Dict[str, Any]]) -> Optional[float]:
    ev, mo = _speeds(scenes)
    if len(ev) == 0:
        return None
    den = float(np.sum(ev)) + float(np.sum(mo))
    if den == 0.0:
        return None  # 0/0: input carries no motion; legacy substituted 0.5
    return float(np.sum(mo)) / den


def mood_speed_mean_ii_a(scenes: List[Dict[str, Any]]) -> Optional[float]:
    _, mo = _speeds(scenes)
    if len(mo) == 0:
        return None
    return float(np.mean(mo))


def ng1_smooth_still(scenes: List[Dict[str, Any]], theta: float = NG1_THETA) -> Optional[float]:
    ev, mo = _speeds(scenes)
    if len(ev) == 0:
        return None
    j = np.maximum(np.asarray(ev, dtype=float), np.asarray(mo, dtype=float))
    return float(np.mean(np.exp(-((j / theta) ** 2))))


def ng2_speed_slope(scenes: List[Dict[str, Any]]) -> Optional[float]:
    _, mo = _speeds(scenes)
    return _ols_slope_centered(mo)


def compute_faw_continuous(scenes: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """All FAW continuous metrics for one work (list of scene annotation dicts). Missing input -> None."""
    aw1 = aw1_signed_counterpoint(scenes)
    out: Dict[str, Optional[float]] = {
        "aw1_c_irony": None if aw1 is None else aw1["c_irony"],
        "aw1_c_det": None if aw1 is None else aw1["c_det"],
        "aw4_g": aw4_irony_slope(scenes),
        "ng0_phi_zeami": phi_zeami_ii_a(scenes),
        "ng0_mood_speed_mean": mood_speed_mean_ii_a(scenes),
        "ng1_s_still": ng1_smooth_still(scenes, NG1_THETA),
        "ng2_speed_slope": ng2_speed_slope(scenes),
    }
    for th in NG1_THETA_SENSITIVITY:
        out[f"ng1_s_still_th{int(round(th * 100)):03d}"] = ng1_smooth_still(scenes, th)
    return out
