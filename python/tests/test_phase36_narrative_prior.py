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
