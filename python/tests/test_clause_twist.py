"""Clause-local subject-predicate twist (主述ねじれ) detection (M15.1)."""

import pytest

from usir_py.clause_twist import detect_clause_twists, split_clauses
from usir_py.nishiki_unified_field import evaluate_toulmin_components

NEGATIVES = [
    "彼は走ったが、私は疲れた。",
    "問題は残ったが、私は考えた。",
    "風の音が止んだので、私は耳を澄ました。",
    "音が消えたが私は耳を澄ました。",
    "私の目標は走ることだ。",
    "私は走った。",
    "彼が考えた問題は難しい。",
    "この結果は有効だと考えます。",
    "「風の音が耳を澄ました」と彼は言った。",
    "風が走った。",
]
POSITIVES = [
    ("風の音が耳を澄ました。", "stimulus_x_perceiver"),
    ("問題が走った。", "abstract_x_motion"),
    ("時間が考えた。", "abstract_x_perceiver"),
    ("昨日、静かな声が、耳を澄ました。", "stimulus_x_perceiver"),
]


@pytest.mark.parametrize("s", NEGATIVES)
def test_no_twist(s):
    assert detect_clause_twists(s) == []


@pytest.mark.parametrize("s,kind", POSITIVES)
def test_twist_detected(s, kind):
    res = detect_clause_twists(s)
    assert len(res) == 1 and res[0].kind == kind


def test_subject_scope_resets_per_clause():
    assert len(split_clauses("彼は走ったが、私は疲れた。")) == 2
    # Same words in one clause are a twist; across a clause boundary they are not.
    assert detect_clause_twists("音が耳を澄ました。")
    assert not detect_clause_twists("音が消えたが、耳を澄ました。")


def test_toulmin_integration_and_flag():
    on = evaluate_toulmin_components("風の音が耳を澄ました。")
    off = evaluate_toulmin_components("風の音が耳を澄ました。", clause_twist=False)
    assert on.clause_twist_count == 1 and on.twist_count >= 1
    assert off.clause_twist_count == 0
    neg = evaluate_toulmin_components("彼は走ったが、私は疲れた。")
    assert neg.clause_twist_count == 0
