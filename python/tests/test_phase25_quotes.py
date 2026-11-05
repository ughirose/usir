"""Phase 25.4: format_novel quote normalisation."""

from __future__ import annotations

from usir_py.text_format import format_novel

I = "　"


def test_half_width_quotes_become_corner_brackets():
    out = format_novel(['彼は笑った。"何がおかしいか見てみろ。"空が暗い。'], 120, seed=0)
    assert out.split("\n") == [I + "彼は笑った。", "「何がおかしいか見てみろ。」", I + "空が暗い。"]


def test_smart_quotes():
    out = format_novel(["彼は言った。“行こう。”風が吹いた。"], 120, seed=0)
    assert out.split("\n") == [I + "彼は言った。", "「行こう。」", I + "風が吹いた。"]


def test_smart_quote_multi_sentence_stays_one_dialogue():
    out = format_novel(["“行こう。急げ。”風が吹いた。"], 120, seed=0)
    assert out.split("\n")[0] == "「行こう。急げ。」"


def test_top_level_double_corner_is_dialogue():
    out = format_novel(["彼は笑った。『行こう。』風が吹いた。"], 120, seed=0)
    assert out.split("\n") == [I + "彼は笑った。", "「行こう。」", I + "風が吹いた。"]


def test_nested_double_corner_preserved():
    out = format_novel(["「彼は『行こう』と言った。」風が吹いた。"], 120, seed=0)
    assert out.split("\n") == ["「彼は『行こう』と言った。」", I + "風が吹いた。"]


def test_double_corner_in_narration_not_dialogue():
    out = format_novel(["彼は『吾輩は猫である』を読んだ。"], 120, seed=0)
    assert out == I + "彼は『吾輩は猫である』を読んだ。"


def test_unclosed_quote_closed_at_sentence_end():
    out = format_novel(['"何がおかしいか見てみろ。空が暗い。'], 120, seed=0)
    assert out.split("\n") == ["「何がおかしいか見てみろ。」", I + "空が暗い。"]


def test_unclosed_quote_closed_at_paragraph_end():
    out = format_novel(["彼は言った。“行こう"], 120, seed=0)
    assert out.split("\n") == [I + "彼は言った。", "「行こう」"]


def test_no_quote_text_unchanged():
    par = "朝が来た。「行こう。」と彼は言った。風が吹いた。"
    assert "「行こう。」と彼は言った。" in format_novel([par], 120, seed=3).split("\n")
    assert format_novel(["あ。"], 120) == I + "あ。"
