"""Tests for U6 Numeral and Negation Guards with Stage B-6 closed-class exclusions."""
import pytest
from usir_py.v5.u6_guards import (
    check_numeral_guard,
    check_negation_guard,
    extract_negation_polarity,
    extract_numerals,
    extract_numbers,
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


def test_numeral_closed_class_idiom_exclusions():
    # 7 specific idioms starting with 一: 一通り, 一旦, 一応, 一緒, 一層, 一向, 一斉
    idiom_sentences = [
        "一通りの確認を終えた。",
        "一旦停止して確認した。",
        "一応の合意に達した。",
        "一緒に出かけることにした。",
        "一層の注意を払った。",
        "一向に気にする様子はない。",
        "一斉に歓声が上がった。",
    ]
    for s in idiom_sentences:
        nums = extract_numerals(s)
        assert nums == [], f"Expected empty numerals for idiom '{s}', got {nums}"

    # Style polish replacement of idiom without numeral change
    res = u6_classify("一通りの確認を終えた。", "ひととおりの確認を終えた。")
    assert res["label"] == "STYLE_POLISH"
    assert res["guard_passed"] is True


def test_numeral_closed_class_man_ga_ichi_exclusion():
    # 万が一 / 万一 should extract [] (not [10000, 1] or [10000])
    assert extract_numerals("万が一の事態に備える") == []
    assert extract_numbers("万が一の事態に備える") == []
    assert extract_numerals("万が一の場合") == []
    assert extract_numbers("万が一の場合") == []
    assert extract_numerals("万一の事態に備える") == []
    assert extract_numbers("万一の事態に備える") == []

    # Idiom polish: 万が一 -> 万一
    res = u6_classify("万が一の事態に備えておく。", "万一の事態に備えておく。")
    assert res["label"] == "STYLE_POLISH"
    assert res["guard_passed"] is True

    # Real numeral vs 万が一: 万が一 vs 百件 -> mismatch
    res_mismatch = u6_classify("万が一の事故に備える。", "百件の事故に備える。")
    assert res_mismatch["label"] == "CONTENT_REWRITE"
    assert res_mismatch["guard_passed"] is False


def test_numeral_adverbial_juuppun_exclusion():
    # Adverbial 十分 (meaning 'sufficiently') should NOT extract 10
    assert extract_numerals("十分に休んだ。") == []
    assert extract_numerals("十分な休養を取った。") == []
    assert extract_numerals("テストカバレッジは十分である。") == []

    # Polishing 十分 -> 充分 should stay STYLE_POLISH
    res = u6_classify("十分に休息を取った。", "充分に休息を取った。")
    assert res["label"] == "STYLE_POLISH"
    assert res["guard_passed"] is True

    # Temporal 十分 (10 minutes, followed by 間/後/前) SHOULD extract 10
    assert extract_numerals("十分前に出発した。") == [10]
    assert extract_numerals("十分間待った。") == [10]
    assert extract_numerals("十分後に連絡する。") == [10]

    # Temporal change 10分前 -> 3分前 should be rejected as CONTENT_REWRITE
    res_temporal = u6_classify("十分前に出発した。", "三分前に出発した。")
    assert res_temporal["label"] == "CONTENT_REWRITE"
    assert res_temporal["guard_passed"] is False


def test_negation_adverb_mazu_exclusion():
    # "先ず" / "まず" is an adverb, not verbal negation
    assert extract_negation_polarity("先ず、深呼吸をした。") == 0
    assert extract_negation_polarity("まず、彼に連絡した。") == 0

    # Polishing 先ず -> まず should stay STYLE_POLISH
    passes, reason, pb, pa = check_negation_guard("先ず、彼に挨拶した。", "まず、彼に挨拶した。")
    assert passes is True
    assert pb == 0 and pa == 0
    res = u6_classify("先ず、彼に挨拶した。", "まず、彼に挨拶した。")
    assert res["label"] == "STYLE_POLISH"


def test_negation_conjecture_modality_exclusion():
    # 違いない / ちがいない / かもしれない / かも知れない do not add verbal negation
    assert extract_negation_polarity("彼が犯人に違いない。") == 0
    assert extract_negation_polarity("彼が犯人にちがいない。") == 0
    assert extract_negation_polarity("彼が犯人かもしれない。") == 0
    assert extract_negation_polarity("彼が犯人かも知れない。") == 0

    # Polishing 違いない -> ちがいない should stay STYLE_POLISH
    passes, reason, pb, pa = check_negation_guard("彼が来たに違いない。", "彼が来たちがいない。")
    assert passes is True
    assert pb == 0 and pa == 0
    res = u6_classify("彼が来たに違いない。", "彼が来たちがいない。")
    assert res["label"] == "STYLE_POLISH"


def test_negation_adjective_stems_exclusion():
    # Adjectives with stem (少/切/危/汚/儚) + ない are state descriptions, not verbal negations
    assert extract_negation_polarity("予算が少ない。") == 0
    assert extract_negation_polarity("切ない恋心である。") == 0
    assert extract_negation_polarity("危ない道を進む。") == 0
    assert extract_negation_polarity("汚い手口を使う。") == 0
    assert extract_negation_polarity("儚い命であった。") == 0

    # Also inflections (少なかった, 切なく, 危なければ)
    assert extract_negation_polarity("経験が少なかった。") == 0
    assert extract_negation_polarity("切なく響いた。") == 0
    assert extract_negation_polarity("危なければ逃げる。") == 0

    # Polishing 少ない -> 僅少 should not trigger NEGATION_FLIP
    passes, reason, pb, pa = check_negation_guard("残された時間が少ない。", "残された時間が僅少である。")
    assert passes is True
    assert pb == 0 and pa == 0


def test_numeral_placeholder_orthographic_protection():
    # "一万二千円" -> "12000円": Numeral normalization protects against Jaccard/length false positives
    res_short = u6_classify("一万二千円", "12000円")
    assert res_short["label"] == "STYLE_POLISH"
    assert res_short["guard_passed"] is True
    assert res_short["nums_before"] == [12000]
    assert res_short["nums_after"] == [12000]

    res_sent = u6_classify("費用は一万二千円だった。", "費用は12000円だった。")
    assert res_sent["label"] == "STYLE_POLISH"
    assert res_sent["guard_passed"] is True

    # Real semantic rewrite with matching numerals must still be rejected by Jaccard
    res_diff = u6_classify("一万二千円の費用をかけた。", "12000円の利益を得た。")
    assert res_diff["label"] == "CONTENT_REWRITE"
    assert "SURFACE_DIVERGENCE" in res_diff["rejection_reason"]


def test_numeral_fraction_ratio_jubun_no_ichi():
    # 割合・分数表現: "十分の一" extracts [10, 1] without being swallowed by adverbial exclusion
    nums = extract_numerals("被害は十分の一になった。")
    assert nums == [10, 1]

    # Matching orthographic polish: 十分の一 -> 10分の1 is STYLE_POLISH
    res = u6_classify("被害は十分の一になった。", "被害は10分の1になった。")
    assert res["label"] == "STYLE_POLISH"
    assert res["guard_passed"] is True

    # Fraction value change: 十分の一 -> 五分の一 is CONTENT_REWRITE
    res_change = u6_classify("被害は十分の一になった。", "被害は五分の一になった。")
    assert res_change["label"] == "CONTENT_REWRITE"
    assert res_change["guard_passed"] is False


def test_numeral_single_unit_kanji_hyaku():
    # 単独位取り漢数字: "百の兵がいた" extracts [100]
    nums = extract_numerals("城門の前には百の兵がいた。")
    assert nums == [100]

    # Matching polish: 百 -> 100 is STYLE_POLISH
    res = u6_classify("城門の前には百の兵がいた。", "城門の前には100の兵がいた。")
    assert res["label"] == "STYLE_POLISH"
    assert res["guard_passed"] is True

    # Value change: 百 -> 千 is CONTENT_REWRITE
    res_change = u6_classify("城門の前には百の兵がいた。", "城門の前には千の兵がいた。")
    assert res_change["label"] == "CONTENT_REWRITE"
    assert res_change["guard_passed"] is False


def test_u6_test_vectors_fixture_pass():
    import json
    from pathlib import Path

    # cwd 非依存: tests/v5/ から 3 階層上がリポジトリルート
    repo_root = Path(__file__).resolve().parents[3]
    fixture_path = repo_root / "docs/handoff/v10/u6_guards_test_vectors.json"
    assert fixture_path.exists(), f"Missing fixture file: {fixture_path}"

    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    n_vecs = data["metadata"]["n_vectors"]
    assert n_vecs == 38, f"Expected 38 vectors, got {n_vecs}"
    assert data["metadata"]["calibrated"] is False
    assert len(data["test_vectors"]) == 38

    for vec in data["test_vectors"]:
        res = u6_classify(vec["text_before"], vec["text_after"])
        assert res["label"] == vec["expected_label"], (
            f"Vector {vec['id']} ({vec['category']}): expected {vec['expected_label']}, got {res['label']}"
        )


def test_u6_profiles():
    from usir_py.v5.u6_guards import U6_PROFILES, PROVISIONAL_CALIBRATED_THRESHOLDS

    assert "uncalibrated_v1" in U6_PROFILES
    assert "provisional_micro_v2" in U6_PROFILES
    assert U6_PROFILES["provisional_micro_v2"]["min_jaccard"] == 0.25
    assert U6_PROFILES["provisional_micro_v2"]["max_char_change"] == 0.85
    assert PROVISIONAL_CALIBRATED_THRESHOLDS["min_jaccard"] == 0.25
    assert PROVISIONAL_CALIBRATED_THRESHOLDS["max_char_change"] == 0.85

    # Test classifying with profile keyword
    # A pair with char_change = 0.4167 (passes provisional_micro_v2 max_c=0.85, fails uncalibrated_v1 max_c=0.25)
    tb = "空は青く澄み渡っていた。"
    ta = "空は青くどこまでも澄み渡っていた。"
    res_def = u6_classify(tb, ta)
    assert res_def["label"] == "CONTENT_REWRITE"  # len_change > 0.25

    res_micro = u6_classify(tb, ta, profile="provisional_micro_v2")
    assert res_micro["label"] == "STYLE_POLISH"  # len_change <= 0.85 and jaccard >= 0.25

