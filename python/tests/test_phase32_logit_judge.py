"""Phase 32: logit-debiased pairwise judge."""
from __future__ import annotations

import re

import pytest

from usir_py.pairwise_judge import (build_logit_prompt, logit_preference, logit_preference_paragraphs,
                                    logit_tournament)


def _first(prompt: str) -> str:
    return re.search(r'作品1:\n"""\n(.*?)\n"""', prompt, re.S).group(1)


def _second(prompt: str) -> str:
    return re.search(r'作品2:\n"""\n(.*?)\n"""', prompt, re.S).group(1)


def biased_judge(quality, bias):
    """s = quality(first) - quality(second) + bias, split into two log-probs."""
    def fn(prompt):
        s = quality(_first(prompt)) - quality(_second(prompt)) + bias
        return (s / 2.0, -s / 2.0)
    return fn


def test_constant_position_bias_cancels_exactly():
    q = {"良い作品": 1.3, "普通の作品": 0.2}
    for bias in (0.0, 2.0, -5.0, 40.0):
        r = logit_preference("良い作品", "普通の作品", biased_judge(q.get, bias))
        assert r["preference"] == pytest.approx(1.1)
        assert r["position_bias"] == pytest.approx(bias)


def test_antisymmetry_and_prompt_contents():
    fn = biased_judge(len, 3.0)
    ab = logit_preference("長い長い作品", "短い", fn)
    ba = logit_preference("短い", "長い長い作品", fn)
    assert ab["preference"] == pytest.approx(-ba["preference"])
    p = build_logit_prompt("X", "Y")
    assert _first(p) == "X" and _second(p) == "Y"


def test_content_dependent_position_effect_is_not_removed():
    # interaction: the first text gets a bonus proportional to its own length -> not a constant bias
    def fn(prompt):
        s = 0.1 * len(_first(prompt))
        return (s, 0.0)
    r = logit_preference("あああ", "い", fn)
    assert r["preference"] == pytest.approx((0.3 - 0.1) / 2)   # residual from the interaction, documented


def test_paragraph_mode_and_tournament():
    q = {"a1": 1.0, "a2": 2.0, "b1": 0.0, "b2": 0.5}
    fn = biased_judge(lambda t: q.get(t, 0.0), 4.0)
    r = logit_preference_paragraphs(["a1", "a2"], ["b1", "b2"], fn)
    assert r["n"] == 2 and r["n_wins_a"] == 2
    assert r["preference_mean"] == pytest.approx((1.0 + 1.5) / 2)
    t = logit_tournament({"A": "a1", "B": "b1"}, [("A", "B")], fn, paragraphs={"A": ["a1"], "B": ["b1"]})
    assert t["matches"][0]["full"]["preference"] == pytest.approx(1.0)
    assert t["matches"][0]["paragraph"]["preference_mean"] == pytest.approx(1.0)
