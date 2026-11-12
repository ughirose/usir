"""Tests for U6 Numeral and Negation Guards."""
import pytest
from usir_py.v5.u6_guards import (
    check_numeral_guard,
    check_negation_guard,
    extract_numerals,
    parse_japanese_numeral,
    u6_classify,
)


def test_numeral_parser():
    assert parse_japanese_numeral("三年") is None or parse_japanese_numeral("三") == 3
    assert parse_japanese_numeral("3") == 3
    assert parse_japanese_numeral("３") == 3
    assert parse_japanese_numeral("十") == 10
    assert parse_japanese_numeral("百五十") == 150
    assert parse_japanese_numeral("一万二千") == 12000


def test_numeral_guard_identical_value_style_polish():
    # "三年" -> "3年": Numeral values are identical ([3] == [3]) -> passes guard
    passes, reason, nb, na = check_numeral_guard("三年が経過した。", "3年が経過した。")
    assert passes is True
    assert reason == "NUMERAL_MATCH"

    res = u6_classify("三年が経過した。", "3年が経過した。")
    assert res["label"] == "STYLE_POLISH"
    assert res["guard_passed"] is True


def test_numeral_guard_value_change_content_rewrite():
    # "三年前" -> "十年前": Values change ([3] != [10]) -> rejected
    passes, reason, nb, na = check_numeral_guard("三年前の事件であった。", "十年前の事件であった。")
    assert passes is False
    assert "NUMERAL_MISMATCH" in reason

    res = u6_classify("三年前の事件であった。", "十年前の事件であった。")
    assert res["label"] == "CONTENT_REWRITE"
    assert res["guard_passed"] is False


def test_negation_guard_polarity_flip_content_rewrite():
    # "泣いた" -> "泣かなかった": Affirmative -> Negative (flip)
    passes, reason, pb, pa = check_negation_guard("彼は泣いた。", "彼は泣かなかった。")
    assert passes is False
    assert "NEGATION_FLIP" in reason

    res = u6_classify("彼は泣いた。", "彼は泣かなかった。")
    assert res["label"] == "CONTENT_REWRITE"
    assert res["guard_passed"] is False


def test_negation_guard_double_negation_exemption():
    # Double negation "ないわけではない" cancels flip
    passes, reason, pb, pa = check_negation_guard("彼は知っている。", "彼は知らないわけではない。")
    assert passes is True
    assert reason == "NEGATION_PRESERVED"


def test_negation_guard_lexical_prefix_ignored():
    # Lexical prefix "不" alone does not flip auxiliary polarity
    passes, reason, pb, pa = check_negation_guard("可能な計画である。", "不可能な計画である。")
    assert passes is True  # Lexical change handled by content Jaccard, not negation guard
