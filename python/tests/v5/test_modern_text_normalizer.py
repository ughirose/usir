"""Unit tests for Modern Text Normalizer and Tier-C Clean-Room Extractor."""
import math
import pytest

from usir_py.v5.modern_text_normalizer import (
    compute_punctuation_normalization_diff,
    extract_tier_c_irreversible_profile,
    mask_entities_to_black_square,
    normalize_modern_text,
    tokenize_closed_class_skeleton,
    Order3PPMModel,
)


def test_normalize_modern_punctuation():
    # 1. Ellipsis normalization
    raw_ellipsis = "彼は黙り込んだ…そして言った……本当なのか....."
    norm = normalize_modern_text(raw_ellipsis)
    assert "……" in norm
    assert "…" not in norm.replace("……", "")
    assert "....." not in norm

    # 2. Dash normalization
    raw_dash = "その時―雷鳴が轟いた――"
    norm_dash = normalize_modern_text(raw_dash)
    assert "――" in norm_dash
    assert "―" not in norm_dash.replace("――", "")

    # 3. Exclamation & Question marks
    raw_excl = "何だって!? 本当なのか?! まさか!! 嘘だろう??"
    norm_excl = normalize_modern_text(raw_excl)
    assert "！？" in norm_excl
    assert "！！" in norm_excl
    assert "？？" in norm_excl
    assert "!?" not in norm_excl


def test_web_novel_rubi_removal():
    # Kakuyomu/Narou pipe rubi
    t1 = "|竜騎士《ドラグーン》の少年が走る。"
    assert normalize_modern_text(t1) == "竜騎士の少年が走る。"

    # Non-pipe double bracket rubi
    t2 = "親文字《おやもじ》を正しく認識する。"
    assert normalize_modern_text(t2) == "親文字を正しく認識する。"

    # Parentheses rubi (pure kana inside)
    t3 = "結界(けっかい)が破られた。"
    assert normalize_modern_text(t3) == "結界が破られた。"

    t4 = "魔術師（まじゅつし）の弟子。"
    assert normalize_modern_text(t4) == "魔術師の弟子。"

    # Non-rubi parentheticals should be preserved
    t5 = "彼は笑った（注：笑ってはいけない）。（笑）"
    norm5 = normalize_modern_text(t5)
    assert "（注：笑ってはいけない）" in norm5
    assert "（笑）" in norm5


def test_dialogue_dominant_layout_and_nan_guards():
    # Dialogue-heavy layout (50% to 80% dialogue ratio)
    dialogue_text = (
        "「こんにちは、今日はいい天気ですね」と彼女は微笑んだ。\n"
        "「ええ、本当に散歩日和です」と私は答えた。\n"
        "「どこへ行くのですか？」「図書館まで本を返しにね」\n"
        "「奇遇ですね、私も同じ方向です」"
    )
    norm = normalize_modern_text(dialogue_text)
    diff = compute_punctuation_normalization_diff(dialogue_text, norm)

    ratio = diff["dialogue_ratio"]
    assert 0.50 <= ratio <= 0.85, f"Expected dialogue ratio between 0.5 and 0.85, got {ratio}"
    assert not math.isnan(ratio)
    assert not math.isinf(ratio)

    # Empty text guard
    diff_empty = compute_punctuation_normalization_diff("", "")
    assert diff_empty["dialogue_ratio"] == 0.0
    assert not math.isnan(diff_empty["char_delta_ratio"])


def test_cleanroom_irreversible_profile_invariant():
    # Raw input text containing confidential/proprietary story entities
    proprietary_text = (
        "私立探偵の赤城神楽は、深夜の新宿アルタ前で秘密同盟の怪しい密談を目撃した。"
        "「コードネーム・プロメテウスの計画は最終段階だ」と男は囁いた。"
    )

    prof = extract_tier_c_irreversible_profile(proprietary_text)

    # Invariant 1: Status must be verified
    assert prof["status"] == "CLEAN_ROOM_TIER_C_VERIFIED"

    # Invariant 2: Ortho v2 must be 31-dimensional finite floats
    ortho = prof["ortho_v2_31d"]
    assert len(ortho) == 31
    assert all(isinstance(x, (float, int)) and not math.isnan(x) and not math.isinf(x) for x in ortho)

    # Invariant 3: PPM bits/token and coverage must be valid finite floats
    assert prof["ppm_bits_per_token"] > 0.0
    assert not math.isnan(prof["ppm_bits_per_token"])
    assert 0.0 <= prof["closed_class_coverage_pct"] <= 100.0

    # Invariant 4: NO RAW TEXT LEAKAGE into the output dict
    # None of the proper nouns / confidential content strings can appear anywhere in keys or values
    secret_substrings = [
        "赤城神楽", "新宿アルタ", "秘密同盟", "怪しい密談", "プロメテウス",
        "私立探偵", "最終段階",
    ]

    def _assert_no_leak(obj):
        if isinstance(obj, str):
            for secret in secret_substrings:
                assert secret not in obj, f"Clean-room violation: found secret '{secret}' in string '{obj}'"
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _assert_no_leak(k)
                _assert_no_leak(v)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _assert_no_leak(item)

    _assert_no_leak(prof)


def test_mask_entities_to_black_square():
    raw = "吾輩は猫である。名前はまだ無い。"
    masked = mask_entities_to_black_square(raw)
    # Closed class particles and auxiliaries are preserved: は, で, ある, 。, は, まだ, ない
    assert "は" in masked
    assert "。" in masked
    assert "■" in masked
    # Content words should be masked to ■
    assert "吾輩" not in masked
    assert "猫" not in masked
    assert "名前" not in masked
