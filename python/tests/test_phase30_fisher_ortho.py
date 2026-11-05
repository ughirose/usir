"""Phase 30: orthographic prior and class-balanced Fisher metric."""
from __future__ import annotations

import math

import numpy as np
import pytest

from usir_py.fisher_atlas import fit_fisher
from usir_py.orthographic_prior import (ORTHO_DIM, ORTHO_EXT_DIM, ORTHO_EXT_NAMES, ortho_vector,
                                        ortho_vector_ext, sentence_lengths)


def test_ortho_vector_values():
    t = "雨が降る。私は、傘を持って、外へ出た。"
    assert sentence_lengths(t) == [5, 14]
    v = ortho_vector(t)
    assert v.shape == (ORTHO_DIM,)
    assert v[0] == pytest.approx(math.log(9.5))
    assert v[2] == pytest.approx(100 * 2 / 19)
    assert v[3] == pytest.approx(1.0)
    assert v[6] == 0.0


def test_ortho_ext_layout_and_final_fractions_sum_to_one():
    v = ortho_vector_ext("雨が降った。彼は笑う。「行く」")
    assert v.shape == (ORTHO_EXT_DIM,) and len(ORTHO_EXT_NAMES) == ORTHO_EXT_DIM
    fin = v[ORTHO_DIM + 10:]
    assert fin.sum() == pytest.approx(1.0)


def test_no_target_vocabulary_in_prior():
    from usir_py.orthographic_prior import FINAL_CHARS, FUNCTION_KANA
    assert all(len(c) == 1 for c in FUNCTION_KANA + FINAL_CHARS)


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    means = {"a": [0, 0, 0], "b": [3, 0, 0], "c": [0, 3, 0]}
    sizes = {"a": 2000, "b": 30, "c": 30}       # strongly imbalanced window counts
    return {k: rng.normal(m, 1.0, size=(sizes[k], 3)) for k, m in means.items()}, means, rng


def test_fisher_separates_and_is_class_balanced():
    X, means, rng = _toy()
    fm = fit_fisher(X, reg=1e-3)
    assert fm.W.shape == (3, 2)
    for k, m in means.items():
        test = rng.normal(m, 1.0, size=(20, 3))
        assert fm.predict(test) == k
    # a single window exactly between a and b must not be pulled to the large class
    s = fm.scores(np.array([[1.5, 0, 0]]))
    assert abs(s["a"] - s["b"]) < 0.5


def test_fisher_direction_points_to_class():
    X, means, _ = _toy(1)
    fm = fit_fisher(X, reg=1e-3)
    v = fm.direction("b")
    assert np.linalg.norm(v) == pytest.approx(1.0)
    zb = (np.array(means["b"], float) - fm.center) / fm.scale
    za = (np.array(means["a"], float) - fm.center) / fm.scale
    assert v @ zb > v @ za
