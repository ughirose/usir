"""run_v4_bidirectional_experiment._direction_metrics: Climax paragraphs are scored with or without a discharge."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "colab"))

from run_v4_bidirectional_experiment import _CLIMAX_REF, _direction_metrics  # noqa: E402
from usir_py.extractor_24d import extract_usir_24d  # noqa: E402

TEXTS = [
    "雨上がりの古寺に朝が来た。苔の匂いが濃くなり、鐘の音が遠くへ消えた。庭には誰もいなかった。",
    "山門の前に石段が続いている。霧が低く流れ、杉の梢だけが明るかった。",
    "禅堂の障子が白んだ。炉の灰は冷たく、茶碗が一つ置かれていた。",
    "鐘が鳴った。割れた。叫びのように。まるで雷鳴だった。だが誰も動かない。",
    "縁側に風が通る。柱の影が長く伸びて、床の木目が見えた。",
    "雷が落ちた。鋭く、硬く、突然に。違う、これは夢ではない。",
]
WELLS = ["Narrative", "Narrative", "Narrative", "Climax", "Narrative", "Climax"]


def _q(text):
    return np.asarray(extract_usir_24d(text)["q"], dtype=float)[:12]


def _paras(discharge):
    out = []
    for i, (t, w) in enumerate(zip(TEXTS, WELLS)):
        dm = discharge.get(i)
        out.append({"index": i, "well": w, "text": t, "discharged_momentum": dm})
    return out


def _sd(v):
    return float(np.mean(v[[3, 7, 11]])), float(np.mean(v[[0, 4, 8]]))


def test_climax_ref_is_unit_vector_shock_plus_flux_minus():
    assert np.linalg.norm(_CLIMAX_REF) == pytest.approx(1.0)
    assert np.allclose(_CLIMAX_REF[[3, 7, 11]], 1 / np.sqrt(6)) and np.allclose(_CLIMAX_REF[[0, 4, 8]], -1 / np.sqrt(6))
    assert np.count_nonzero(_CLIMAX_REF) == 6


def test_undischarged_climax_gets_text_based_metrics_and_cos_ref():
    # only paragraph 3 (Climax) is discharged; paragraph 5 (Climax) has no discharge
    dv = [0.0] * 12
    dv[3] = dv[7] = 0.8
    dv[0] = 0.1
    res = _direction_metrics(_paras({3: dv}))
    qs = [_q(t) for t in TEXTS]
    q_base = np.mean([qs[i] for i in (0, 1, 2, 5)], axis=0)       # steered = {3, 4}
    r5 = res[5]
    assert r5["well"] == "Climax" and r5["steered"] is False
    dq = qs[5] - q_base
    ds, df = _sd(dq)
    assert r5["dS"] == pytest.approx(ds, abs=1e-4) and r5["dF"] == pytest.approx(df, abs=1e-4)
    assert r5["delta_sf"] == pytest.approx(ds - df, abs=1e-4)
    assert r5["d_realised_reg"] == pytest.approx(float(np.tanh((ds - df) / 0.10)), abs=1e-4)
    assert r5["cos_ref"] == pytest.approx(float(dq @ _CLIMAX_REF / np.linalg.norm(dq)), abs=1e-4)
    assert "cos_p_dq" not in r5 and "d_target" not in r5          # discharge-dependent fields stay discharge-only
    # discharged Climax: previous fields unchanged, cos_ref added
    r3 = res[3]
    dq3 = qs[3] - q_base
    dvn = np.asarray(dv)
    assert r3["cos_p_dq"] == pytest.approx(float(dvn @ dq3 / (np.linalg.norm(dvn) * np.linalg.norm(dq3))), abs=1e-4)
    assert r3["delta_sf"] == pytest.approx(_sd(dq3)[0] - _sd(dq3)[1], abs=1e-4)
    assert "d_target" in r3 and "cos_ref" in r3
    # non-Climax paragraphs without discharge: no direction fields
    for i in (0, 1, 2):
        assert "delta_sf" not in res[i] and "cos_ref" not in res[i]


def test_no_discharge_at_all_scores_every_climax():
    res = _direction_metrics(_paras({}))
    qs = [_q(t) for t in TEXTS]
    q_base = np.mean(qs, axis=0)
    for i in (3, 5):
        dq = qs[i] - q_base
        ds, df = _sd(dq)
        assert res[i]["delta_sf"] == pytest.approx(ds - df, abs=1e-4)
        assert "cos_ref" in res[i] and "cos_p_dq" not in res[i]
    cl = [d["delta_sf"] for d in res if d.get("well") == "Climax" and d.get("delta_sf") is not None]
    assert len(cl) == 2                                             # aggregated list includes undischarged Climax
    assert all("delta_sf" not in res[i] for i in (0, 1, 2, 4))
