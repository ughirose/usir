"""Phase 36-1: narrative-channel orthographic statistics."""
from __future__ import annotations

import math

from usir_py.orthographic_prior import NARR_DIM, narrative_sentence_lengths, ortho_matrix_narr, ortho_vector_narr


def test_dialogue_replaced_and_boundary_fragments_dropped():
    t = "かれは。彼は長い道を歩き、坂を上り、門の前で立ち止まった。「行くぞ」と彼は言った。「はい」短い。途中で切"
    # first fragment (window start) and unterminated tail dropped; each dialogue span counts as one character
    assert narrative_sentence_lengths(t) == [25, 8, 4]
    v = ortho_vector_narr(t)
    assert v.shape == (NARR_DIM,)
    assert math.isclose(v[0], math.log((25 + 8 + 4) / 3))
    assert math.isclose(v[2], 2 / 3)                       # commas only inside kept narrative sentences
    assert math.isclose(v[3], len("「行くぞ」「はい」") / len(t))


def test_no_terminal_falls_back_and_matrix_shape():
    v = ortho_vector_narr("終止符のない断片だけ")
    assert all(math.isfinite(x) for x in v)
    assert ortho_matrix_narr(["あ。い。", "う。え。"]).shape == (2, NARR_DIM)


# ---- Phase 36-2: proposal band / biases and the year hard mask -------------------------------------------------
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from usir_py.orthographic_prior import proposal_from_prior  # noqa: E402
from usir_py.realtime_decoder import ClauseTwistedSMCDecoder, ForbiddenWordsLogits  # noqa: E402

_ART = Path(__file__).resolve().parents[2] / "artifacts" / "style_atlas" / "fisher_v1.json"


def test_proposal_defaults_reproduce_phase31_and_overrides_apply():
    ob = json.loads(_ART.read_text(encoding="utf-8"))["ortho_by_author"]
    ref = list(ob.values())
    a, b = proposal_from_prior(ob["000879"], ref), proposal_from_prior(ob["001383"], ref)
    assert (a["steady_min_chars"], a["steady_max_chars"], a["steady_max_commas"]) == (23, 68, 4)   # Phase 31 values
    assert (b["steady_min_chars"], b["steady_max_chars"], b["steady_max_commas"]) == (35, 101, 3)
    assert a["hold_bias"] == -12.0 and a["push_bias"] == 3.0
    b2 = proposal_from_prior(ob["001383"], ref, band=(0.8, 1.3), comma_slack=2.0, hold_bias=-18.0, push_bias=8.0)
    mu = b2["target_sent_len"]
    assert b2["steady_min_chars"] == round(0.8 * mu) and b2["steady_max_chars"] == round(1.3 * mu)
    assert b2["steady_max_commas"] == round(b2["comma_target"] + 2.0)


def test_steady_hold_and_push_follow_proposal():
    gp = {"steady_min_chars": 20, "steady_max_chars": 30, "steady_max_commas": 9, "comma_target": 0.0,
          "hold_bias": -18.0, "push_bias": 8.0}
    d = ClauseTwistedSMCDecoder(dim=12, seed=0, rhythm=True, rhythm_profile="steady", genome_proposal=gp)
    assert d._rhythm_params("短い節")[0] == -18.0
    assert d._rhythm_params("あ" * 31)[0] == 8.0
    assert d._rhythm_params("あ" * 25)[0] == 0.0


def test_year_digit_penalty_separate_from_word_penalty():
    class Tok:
        vocab = ["一", "九", "三", "〇", "雨"]

        def decode(self, ids, **k):
            return "".join(self.vocab[i] for i in ids)

        def batch_decode(self, rows, **k):
            return [self.decode(r) for r in rows]
    tk = Tok()
    digits = {i: s for i, s in enumerate(tk.vocab) if s in "一九三〇"}
    proc = ForbiddenWordsLogits(tk, 0, [], {}, ctx_tail="", digit_ids=digits, digit_penalty=1e4)
    sc = proc(np.array([[0, 1, 2]]), np.zeros((1, 5)))     # 一九三 + 〇 -> 一九三〇
    assert sc[0, 3] == -1e4 and sc[0, 4] == 0.0


# ---- Phase 36-3: unit steering direction and the shaped GRPO reward ----------------------------------------------
from colab import run_phase33_phase_c as drv33  # noqa: E402
from usir_py.activation_steerer import SymplecticActivationSteerer  # noqa: E402


def test_unit_direction_sets_injection_to_alpha_times_norm():
    rng = np.random.RandomState(0)
    v = rng.randn(16, 12).astype(np.float32)
    v /= np.linalg.norm(v, axis=0)
    v[:, 1] = v[:, 0]                                   # collinear columns: ||V dq|| > 1 for dq along both
    st = SymplecticActivationSteerer(d_model=16, inject_positions="last", unit_direction=True)
    st._v_steer = v
    dq = np.zeros(12, dtype=np.float32)
    dq[0] = dq[1] = 1.0
    assert np.linalg.norm(v @ dq) > 1.5
    assert abs(np.linalg.norm(st.steer_direction(dq)) - 1.0) < 1e-6
    h = rng.randn(1, 3, 16).astype(np.float32)
    out = st.steer_tensor(h, dq, 0.1)
    delta = np.linalg.norm(out[0, -1] - h[0, -1])
    assert abs(delta - 0.1 * (np.linalg.norm(h[0, -1]) + 1e-8)) < 1e-4
    st.unit_direction = False
    assert abs(np.linalg.norm(st.steer_direction(dq)) - np.linalg.norm(v @ dq)) < 1e-6


def test_shaped_reward_is_continuous_and_ranks_by_cv():
    fn = drv33.make_shaped_reward(ppl_thresh=2.0, alpha_ppl=1.0)
    flat = "今日は晴れです。明日も晴れです。昨日も晴れでした。週末も晴れです。来週も晴れです。"
    varied = "晴れ。明日の朝は早く起きて、駅まで歩き、そこから電車に乗って海の見える町へ向かう予定です。雨なら中止。"
    rf, rv = fn(flat, 1.5), fn(varied, 1.5)
    assert 0.0 < rf < rv
    assert fn(varied, 6.0) < rv                          # PPL barrier still applies
    assert fn("", 1.0) == 0.0
