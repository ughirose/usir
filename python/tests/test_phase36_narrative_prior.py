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


# ---- Phase 37: byte-split "〇" in the year mask, in-band ramp, incomplete-tail stripping ---------------------------
from usir_py.lora_grpo_forge import strip_incomplete_tail  # noqa: E402
from usir_py.realtime_decoder import byte_split_digit_pairs  # noqa: E402


class _ByteTok:
    """Mock of a byte-level BPE vocabulary in which "〇" has no token: it is ids (5, 6), each decoding to U+FFFD."""
    vocab = ["二", "一", "年", "人", "雨", "<b0>", "<b1>"]

    def encode(self, s, add_special_tokens=False):
        out = []
        for ch in s:
            out += [5, 6] if ch == "〇" else [self.vocab.index(ch)]
        return out

    def decode(self, ids, **k):
        s, i = "", 0
        while i < len(ids):
            if ids[i] == 5 and i + 1 < len(ids) and ids[i + 1] == 6:
                s, i = s + "〇", i + 2
            elif ids[i] in (5, 6):
                s, i = s + "�", i + 1
            else:
                s, i = s + self.vocab[ids[i]], i + 1
        return s

    def batch_decode(self, rows, **k):
        return [self.decode(list(r)) for r in rows]


def _year_proc(tk, ctx_ids):
    digits = {i: s for i, s in enumerate(tk.vocab) if s in "一二"}
    pairs = byte_split_digit_pairs(tk)
    assert pairs == [(5, 6, "〇")]
    proc = ForbiddenWordsLogits(tk, 0, [], {}, ctx_tail="", digit_ids=digits, digit_penalty=1e4, digit_pairs=pairs)
    return proc(np.array([ctx_ids]), np.zeros((1, len(tk.vocab))))[0]


def test_byte_split_zero_cannot_complete_a_year():
    tk = _ByteTok()
    # 二〇一 + 〇: the second byte of 〇 after [二, 〇, 一, b0] is masked; other tokens are not
    sc = _year_proc(tk, [0, 5, 6, 1, 5])
    assert sc[6] == -1e4 and sc[2] == 0.0 and sc[3] == 0.0
    # 二〇一 + (start of 〇): the first byte is masked too, so the year cannot be started
    sc = _year_proc(tk, [0, 5, 6, 1])
    assert sc[5] == -1e4 and sc[1] == -1e4 and sc[4] == 0.0
    # a single digit before 〇 is not a year: 二〇 stays possible (e.g. counts), nothing is masked
    sc = _year_proc(tk, [0, 5])
    assert sc[6] == 0.0
    # without the pair table (Phase 36 behaviour) the completion was not masked
    digits = {i: s for i, s in enumerate(tk.vocab) if s in "一二"}
    p36 = ForbiddenWordsLogits(tk, 0, [], {}, ctx_tail="", digit_ids=digits, digit_penalty=1e4)
    assert p36(np.array([[0, 5, 6, 1, 5]]), np.zeros((1, 7)))[0, 6] == 0.0


def test_in_band_ramp_defaults_off_and_grows_inside_band():
    gp = {"steady_min_chars": 20, "steady_max_chars": 40, "steady_max_commas": 9, "comma_target": 0.0,
          "hold_bias": -18.0, "push_bias": 8.0}
    d = ClauseTwistedSMCDecoder(dim=12, seed=0, rhythm=True, rhythm_profile="steady", genome_proposal=gp)
    assert d._rhythm_params("あ" * 30)[0] == 0.0                       # Phase 36 behaviour without band_ramp
    d.genome_proposal = dict(gp, band_ramp=4.0)
    assert d._rhythm_params("あ" * 20)[0] == 0.0
    assert abs(d._rhythm_params("あ" * 30)[0] - 2.0) < 1e-9
    assert abs(d._rhythm_params("あ" * 40)[0] - 4.0) < 1e-9
    assert d._rhythm_params("あ" * 41)[0] == 8.0 and d._rhythm_params("あ" * 5)[0] == -18.0
    ob = json.loads(_ART.read_text(encoding="utf-8"))["ortho_by_author"]
    assert proposal_from_prior(ob["000879"], list(ob.values()))["band_ramp"] == 0.0


def test_strip_incomplete_tail_and_shaped_reward_option():
    assert strip_incomplete_tail("晴れです。明日も晴れ") == "晴れです。"
    assert strip_incomplete_tail("晴れです。") == "晴れです。"
    assert strip_incomplete_tail("終止なし") == "終止なし"
    # flat complete sentences + a long fragment cut by the token limit: the fragment alone inflates CV_L
    cut = "今日は晴れです。明日も晴れです。昨日も晴れでした。週末は家でゆっくりと本を読みながら過ごしたいと考えていて、そのために"
    fn0 = drv33.make_shaped_reward(ppl_thresh=2.0, alpha_ppl=1.0)
    fn1 = drv33.make_shaped_reward(ppl_thresh=2.0, alpha_ppl=1.0, strip_tail=True)
    assert fn1(cut, 1.5) == fn0(strip_incomplete_tail(cut), 1.5)
    assert fn1(cut, 1.5) != fn0(cut, 1.5)
