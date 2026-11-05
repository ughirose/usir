"""Phase 31: genome -> proposal coupling, contrastive Fisher term, forbidden words."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from usir_py.fisher_atlas import delta_fisher, fit_fisher, load_fisher_artifact
from usir_py.orthographic_prior import proposal_from_prior
from usir_py.realtime_decoder import (ClauseTwistedSMCDecoder, ForbiddenWordsLogits, KanjiBiasLogits,
                                      forbidden_completion_table)

ART = Path(__file__).resolve().parents[2] / "artifacts" / "style_atlas" / "fisher_v1.json"


def test_proposal_is_monotone_in_author_statistics():
    ref = [{"log_mean_sent_len": 3.4, "commas_per_sent": 1.5, "kanji_ratio": 0.27}] * 3
    long_ = proposal_from_prior({"log_mean_sent_len": 3.8, "commas_per_sent": 1.0, "kanji_ratio": 0.30}, ref)
    short = proposal_from_prior({"log_mean_sent_len": 3.2, "commas_per_sent": 2.5, "kanji_ratio": 0.24}, ref)
    assert long_["steady_min_chars"] > short["steady_min_chars"]
    assert long_["steady_max_chars"] > short["steady_max_chars"]
    assert short["steady_max_commas"] > long_["steady_max_commas"]
    assert long_["kanji_bias"] > 0 > short["kanji_bias"]
    same = proposal_from_prior(ref[0], ref)          # an author at the atlas mean keeps the mean
    assert same["target_sent_len"] == pytest.approx(np.exp(3.4))


@pytest.mark.skipif(not ART.exists(), reason="fisher_v1.json not built")
def test_artifact_roundtrip_and_measured_directions():
    art = json.loads(ART.read_text(encoding="utf-8"))
    fm = load_fisher_artifact(art)
    assert fm.W.shape[0] == 24 + 27
    ob = art["ortho_by_author"]
    ref = list(ob.values())
    pa, pb = proposal_from_prior(ob["000879"], ref), proposal_from_prior(ob["001383"], ref)
    # measured: the second author writes longer sentences with fewer commas per sentence
    assert pb["target_sent_len"] > pa["target_sent_len"]
    assert pa["comma_target"] > pb["comma_target"]


class FakeTok:
    """Character-level tokenizer over a tiny vocabulary."""
    vocab = ["<pad>", "少", "年", "女", "少年", "は", "一", "九", "六", "八", "1968", "時", "、", "。", "雨"]
    pad_token_id = 0
    eos_token_id = 0

    def __len__(self):
        return len(self.vocab)

    def batch_decode(self, seqs, skip_special_tokens=True):
        return ["".join(self.vocab[i] for i in s if i) for s in seqs]


def test_forbidden_word_completion_banned_but_allowed_word_kept():
    tk = FakeTok()
    tab = forbidden_completion_table(tk.vocab, ["少年"])
    proc = ForbiddenWordsLogits(tk, 0, ["少年"], tab, ctx_tail="")
    ids = np.array([[1], [5]])                    # row 0 ends with 少, row 1 ends with は
    sc = proc(ids, np.zeros((2, len(tk.vocab))))
    assert sc[0, 2] == -15.0                      # 年 after 少 -> 少年
    assert sc[0, 3] == 0.0                        # 女 after 少 -> 少女 allowed
    assert sc[1, 2] == 0.0                        # 年 after は is fine
    assert sc[0, 4] == -15.0 and sc[1, 4] == -15.0  # 少年 as one token always banned


def test_year_runs_banned():
    tk = FakeTok()
    digits = {i: s for i, s in enumerate(tk.vocab) if any(c in "0123456789一九六八" for c in s)}
    proc = ForbiddenWordsLogits(tk, 0, [], {}, ctx_tail="", digit_ids=digits)
    sc = proc(np.array([[6, 7, 8]]), np.zeros((1, len(tk.vocab))))   # 一九六 + 八 -> 一九六八
    assert sc[0, 9] == -15.0 and sc[0, 10] == -15.0
    sc2 = proc(np.array([[14]]), np.zeros((1, len(tk.vocab))))        # 雨 + 一 is fine
    assert sc2[0, 6] == 0.0


def test_kanji_bias_adds_constant():
    sc = KanjiBiasLogits([1, 2], 0.4)(None, np.zeros((2, 5)))
    assert sc[:, 1].tolist() == [0.4, 0.4] and sc[:, 0].tolist() == [0.0, 0.0]


def test_genome_proposal_overrides_steady_band_and_pushes_commas():
    d = ClauseTwistedSMCDecoder(dim=12, seed=0, rhythm=True, rhythm_profile="steady",
                                genome_proposal={"steady_min_chars": 30, "steady_max_chars": 90,
                                                 "steady_max_commas": 5, "comma_target": 3.5, "comma_push": 1.5})
    assert (d.steady_min_chars, d.steady_max_chars, d.steady_max_commas) == (30, 90, 5)
    pb, cb, _, _ = d._rhythm_params("雨が降る夜、古い工房の奥で青年は")   # 16 chars, 1 comma < 3.5 - 1
    assert pb == -12.0 and cb == 1.5


def test_fisher_term_prefers_candidate_toward_author():
    rng = np.random.default_rng(0)
    # 51-dim toy metric whose class "a" lies along the first orthographic dimension (log mean sentence length)
    X = {"a": rng.normal(0, 1, (200, 51)), "b": rng.normal(0, 1, (200, 51))}
    X["a"][:, 24] += 3.0
    fm = fit_fisher(X, reg=1e-2)
    d = ClauseTwistedSMCDecoder(dim=12, seed=0, fisher_metric=fm, fisher_author="a", fisher_gamma=1.0)
    raw = d._fisher_term("雨が降った。", ["風が吹いた。", "古い工房の奥で、青年は長いあいだ黙って時計の針を見つめていたのだった。"])
    assert raw is not None and raw[1] > raw[0]    # the long-sentence candidate moves toward "a"
    w = d.compute_particle_weights("雨が降った。", ["風が吹いた。", "古い工房の奥で、青年は長いあいだ黙って時計の針を見つめていたのだった。"])
    assert w["term5_fisher"][1] > w["term5_fisher"][0]


def test_delta_fisher_sign():
    rng = np.random.default_rng(1)
    X = {"a": rng.normal(0, 1, (200, 51)), "b": rng.normal(0, 1, (200, 51))}
    X["a"][:, 24] += 3.0
    fm = fit_fisher(X, reg=1e-2)
    long_text = "古い工房の奥で、青年は長いあいだ黙って時計の針を見つめていたのだった。" * 20
    short_text = "雨だ。風だ。夜だ。" * 60
    assert delta_fisher(long_text, fm, "a", "b") > delta_fisher(short_text, fm, "a", "b")
