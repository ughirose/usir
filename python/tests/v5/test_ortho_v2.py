"""Unit tests for orthographic prior v2."""
import numpy as np
import pytest
from usir_py.orthographic_prior import (
    NARR_V2_NAMES,
    ORTHO_EXT_V2_NAMES,
    content_char_count,
    is_content_char,
    ortho_vector_ext_v2,
    ortho_vector_narr_v2,
)


def test_content_char_detection():
    assert is_content_char("あ") is True
    assert is_content_char("漢") is True
    assert is_content_char("A") is True
    assert is_content_char("1") is True
    assert is_content_char("■") is True
    assert is_content_char("□") is True
    assert is_content_char("、") is False
    assert is_content_char("。") is False
    assert is_content_char("！") is False
    assert is_content_char("「") is False


def test_content_char_count():
    text = "吾輩は猫である。名前はまだ無い。"
    # 吾輩は猫である(7) + 名前はまだ無い(7) = 14 content chars
    assert content_char_count(text) == 14


def test_v2_m1_non_comma_invariance():
    # Deleting commas must leave all 28 non-comma dimensions completely unchanged
    text_orig = "私は、昨日、山へ行った。天気が、とても良かった。"
    text_m1 = text_orig.replace("、", "")

    v_ext_orig = ortho_vector_ext_v2(text_orig)
    v_ext_m1 = ortho_vector_ext_v2(text_m1)

    v_narr_orig = ortho_vector_narr_v2(text_orig)
    v_narr_m1 = ortho_vector_narr_v2(text_m1)

    v_orig = np.concatenate([v_ext_orig, v_narr_orig])
    v_m1 = np.concatenate([v_ext_m1, v_narr_m1])

    # 3 comma dims: [2, 3, 29]
    comma_dims = [2, 3, 29]
    keep28 = [i for i in range(31) if i not in comma_dims]

    # Non-comma dimensions must be bitwise identical or within 1e-12
    max_diff_keep28 = np.max(np.abs(v_orig[keep28] - v_m1[keep28]))
    assert max_diff_keep28 < 1e-12

    # Comma dimensions must change
    assert v_orig[2] > 0 and v_m1[2] == 0.0
    assert v_orig[3] > 0 and v_m1[3] == 0.0
