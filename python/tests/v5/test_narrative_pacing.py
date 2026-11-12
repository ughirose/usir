"""Unit tests for narrative pacing and metaphor prototype metrics."""
import pytest
from usir_py.v5.narrative_pacing_and_metaphor import (
    extract_all_narrative_pacing,
    extract_hw22_metaphor_leap,
    extract_hw24_cliffhanger,
    extract_hw25_uid_ratio,
)


def test_hw22_metaphor_leap():
    # Concrete + Abstract novel simile
    text_vivid = "彼女の悲哀は、暗い水底の冷たい石のように沈んでいた。"
    leap_vivid = extract_hw22_metaphor_leap(text_vivid)
    assert leap_vivid > 0.0

    # Dead metaphor cliche
    text_dead = "彼の心は夢のようであった。"
    leap_dead = extract_hw22_metaphor_leap(text_dead)
    # Dead metaphor should be penalized relative to vivid novel metaphor
    assert leap_vivid >= leap_dead

    # Pure literal sentence
    text_literal = "机の上に本と鉛筆が置かれている。"
    leap_lit = extract_hw22_metaphor_leap(text_literal)
    assert leap_lit == 0.0


def test_hw24_cliffhanger():
    # Strong cliffhanger: open question at tail
    text_cliff = "夜の闇が迫っていた。彼は扉を開けた。そこにいたのは誰だったのだろうか？"
    c_cliff = extract_hw24_cliffhanger(text_cliff)
    assert c_cliff > 0.5

    # Tidy closure: damping
    text_closed = "夜の闇が迫っていた。こうしてすべては平穏のうちに終わった。"
    c_closed = extract_hw24_cliffhanger(text_closed)
    assert c_closed < c_cliff


def test_hw25_uid_ratio():
    # Heavy exposition and kanji compounds with low carrier
    text_heavy = "超弦理論構造幾何学体系の定義とは、高次元多様体空間の解析力学的記述に由来する。"
    gamma_heavy = extract_hw25_uid_ratio(text_heavy)

    # Action and dialogue carrier
    text_kinetic = "「逃げろ！」と叫んで、彼は全力で走った。「追ってくるぞ！」と息を呑んだ。"
    gamma_kinetic = extract_hw25_uid_ratio(text_kinetic)

    assert gamma_heavy > gamma_kinetic


def test_extract_all():
    text = "「誰だ？」と叫んだ。魂が凍るような冷気であった。"
    metrics = extract_all_narrative_pacing(text)
    assert "hw22_metaphor_leap" in metrics
    assert "hw24_cliffhanger" in metrics
    assert "hw25_uid_ratio" in metrics
