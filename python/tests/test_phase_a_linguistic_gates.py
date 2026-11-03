"""Tests for Phase A Hexagonal SyntaxTreeFrame and Linguistic Gates (OPEN-01..04).

Verifies:
1. OPEN-01: 「冷たい氷」（Physical Case 1） vs 「冷たい視線」（Abstract Case 3 Attenuation）
2. OPEN-02: 「紅蓮の炎が燃え盛る」（肯定） vs 「紅蓮の炎などなかった」（否定知覚実在ゼロ＋残像コントラスト）
3. OPEN-03: 未知オノマトペ「ぎらぎら」「しんと」「ぷつり」の音韻合成と Mimetic Immunity
4. OPEN-04: 地の文と会話文のバフチン分離と同一話者 Dialogic Collapse 検知
5. SyntaxDriver & SyntaxTreeFrame: Cleanroom bunketsu tree parsing and data structure roundtrip
"""

from __future__ import annotations

import numpy as np
import pytest

from usir_py.drivers.syntax_driver import SyntaxDriver
from usir_py.hth_lexicon import HTHPolyLexicon
from usir_py.linguistic_gates import (
    apply_open01_polysemy_damping,
    compute_concreteness,
    compute_dual_sensory_tensor,
    evaluate_modality_alpha,
    extract_dual_channel_bakhtinian,
    is_mimetic_word,
    synthesize_mimetic_8d,
)
from usir_py.syntax_frame import SyntaxEdge, SyntaxNode, SyntaxTreeFrame


@pytest.fixture(scope="module")
def lexicon() -> HTHPolyLexicon:
    lex = HTHPolyLexicon()
    yield lex
    lex.close()


@pytest.fixture(scope="module")
def driver(lexicon: HTHPolyLexicon) -> SyntaxDriver:
    return SyntaxDriver(lexicon=lexicon, force_cleanroom=True)


# ==============================================================================
# 1. OPEN-01: 「冷たい氷」 vs 「冷たい視線」
# ==============================================================================

def test_open01_tsumetai_koori_vs_tsumetai_shisen(lexicon: HTHPolyLexicon):
    """Verify OPEN-01 tri-state resolution and attenuation on dependency edge (u -> v).

    - 「冷たい氷」: 氷 is concrete (C(u) >= 0.45) -> Physical Case 1, no damping.
    - 「冷たい視線」: 視線 is abstract (C(u) < 0.30) -> Abstract Case 3, secondary sense damped by C(u).
    """
    entry_koori = lexicon.lookup("氷")
    entry_shisen = lexicon.lookup("視線")
    entry_cold = lexicon.lookup("冷たい")

    assert entry_koori is not None, "氷 must exist in HTH lexicon"
    assert entry_shisen is not None, "視線 must exist in HTH lexicon"
    assert entry_cold is not None, "冷たい must exist in HTH lexicon"

    # Concreteness computation
    c_koori = compute_concreteness("氷", lemma="氷", sensory_vec=entry_koori.primary_sensory_vector)
    c_shisen = compute_concreteness("視線", lemma="視線", sensory_vec=entry_shisen.primary_sensory_vector)

    assert c_koori >= 0.45, f"氷 must have C(u) >= 0.45, got {c_koori:.3f}"
    assert c_shisen < 0.30, f"視線 must have C(u) < 0.30, got {c_shisen:.3f}"

    # Apply OPEN-01 to 「冷たい氷」
    res_koori = apply_open01_polysemy_damping(
        u_concreteness=c_koori,
        v_primary_vec=entry_cold.primary_sensory_vector,
        v_secondary_vec=entry_cold.secondary_sensory_vector,
        u_sensory_vec=entry_koori.primary_sensory_vector,
    )
    assert res_koori.case == "case1_physical"
    assert res_koori.slot == "primary_physical"
    assert res_koori.damping_factor == 1.0
    assert np.allclose(res_koori.effective_sensory, entry_cold.primary_sensory_vector)

    # Apply OPEN-01 to 「冷たい視線」
    res_shisen = apply_open01_polysemy_damping(
        u_concreteness=c_shisen,
        v_primary_vec=entry_cold.primary_sensory_vector,
        v_secondary_vec=entry_cold.secondary_sensory_vector,
        u_sensory_vec=entry_shisen.primary_sensory_vector,
    )
    assert res_shisen.case == "case3_abstract_transfer"
    assert res_shisen.slot == "secondary_abstract"
    assert res_shisen.damping_factor == c_shisen

    # Verify that physical sensory energy in 「冷たい氷」 is substantially larger than in 「冷たい視線」
    norm_koori = float(np.linalg.norm(res_koori.effective_sensory))
    norm_shisen = float(np.linalg.norm(res_shisen.effective_sensory))
    assert norm_koori > norm_shisen, (
        f"Physical energy of 冷たい氷 ({norm_koori:.3f}) must exceed damped 冷たい視線 ({norm_shisen:.3f})"
    )
    assert norm_shisen < 0.35 * norm_koori, (
        f"冷たい視線 must be damped by at least ~65% compared to literal, got {norm_shisen / norm_koori:.3f}"
    )


# ==============================================================================
# 2. OPEN-02: 「紅蓮の炎が燃え盛る」 vs 「紅蓮の炎などなかった」
# ==============================================================================

def test_open02_affirmative_vs_negation(driver: SyntaxDriver):
    """Verify OPEN-02 Dual Sensory Tensor (s+, s-).

    - 肯定「紅蓮の炎が燃え盛る」:
      alpha_mod = +1.0 -> s+ = s_eff > 0, s- = 0.0 (no afterimage spike).
    - 否定「紅蓮の炎などなかった」:
      alpha_mod = -0.5 -> s+ = 0.0 (perceptual reality strictly zero!),
                          s- = |-0.5| * ||s_eff||_2 > 0 (cognitive afterimage contrast spike).
    """
    # Baseline sensory vector for fire / flame
    s_flame = np.array([0.9, 0.2, 0.8, 0.1, 0.7, 0.95, 0.85, 0.4], dtype=float)

    # Affirmative case: 「燃え盛る」
    alpha_aff = evaluate_modality_alpha(tail="る")
    assert alpha_aff == 1.0

    res_aff = compute_dual_sensory_tensor(s_flame, alpha_aff)
    assert not res_aff.is_negated
    assert np.allclose(res_aff.s_plus, s_flame)
    assert res_aff.s_minus == 0.0
    assert np.all(res_aff.s_minus_vec == 0.0)

    # Negated case: 「なかった」
    alpha_neg = evaluate_modality_alpha(tail="なかった")
    assert alpha_neg == -0.5

    res_neg = compute_dual_sensory_tensor(s_flame, alpha_neg)
    assert res_neg.is_negated
    # Perceptual reality must be strictly zero:
    assert np.all(res_neg.s_plus == 0.0), "Negated concept must have strictly zero perceptual reality s+"
    # Cognitive afterimage contrast must be non-zero:
    expected_contrast = 0.5 * float(np.linalg.norm(s_flame))
    assert res_neg.s_minus == pytest.approx(expected_contrast, rel=1e-5)
    assert res_neg.s_minus > 0.5, "Cognitive afterimage contrast s- must be significant"

    # End-to-end dependency path negation propagation:
    # 1. Affirmative: 「紅蓮の炎が燃え盛る」
    frame_aff = driver.parse_sentence("紅蓮の炎が燃え盛る")
    flame_nodes_aff = [n for n in frame_aff.nodes if "炎" in n.surface]
    assert len(flame_nodes_aff) >= 1
    assert not flame_nodes_aff[0].is_negation_scope
    alpha_aff_node = evaluate_modality_alpha("", is_negation_scope=flame_nodes_aff[0].is_negation_scope)
    res_flame_aff = compute_dual_sensory_tensor(s_flame, alpha_aff_node)
    assert np.allclose(res_flame_aff.s_plus, s_flame)

    # 2. Negated: 「紅蓮の炎などなかった」
    frame_neg = driver.parse_sentence("紅蓮の炎などなかった")
    flame_nodes_neg = [n for n in frame_neg.nodes if "炎" in n.surface]
    assert len(flame_nodes_neg) >= 1
    assert flame_nodes_neg[0].is_negation_scope, "炎 node must inherit negation scope from なかった along dependency path"
    alpha_neg_node = evaluate_modality_alpha("", is_negation_scope=flame_nodes_neg[0].is_negation_scope)
    assert alpha_neg_node == -0.5
    res_flame_neg = compute_dual_sensory_tensor(s_flame, alpha_neg_node)
    assert np.all(res_flame_neg.s_plus == 0.0), "Perceptual reality must be strictly zero for negated flame"
    assert res_flame_neg.s_minus > 0.0, "Cognitive afterimage contrast must be non-zero for negated flame"


# ==============================================================================
# 3. OPEN-03: 未知オノマトペ「ぎらぎら」「しんと」「ぷつり」の音韻合成
# ==============================================================================

def test_open03_mimetic_sound_symbolism_and_immunity():
    """Verify OPEN-03 sound symbolism synthesis and Mimetic Immunity Protocol."""
    # 1. ぎらぎら: Voiced consonant 'g' + ABAB reduplication
    assert is_mimetic_word("ぎらぎら")
    mim_gira = synthesize_mimetic_8d("ぎらぎら")
    assert mim_gira.is_mimetic
    assert mim_gira.mimetic_immunity
    # Voiced consonant imparts kinematic weight (d5) and tactile roughness (d3) and arousal (d7)
    assert mim_gira.sensory_vec[4] >= 0.45  # Kinematic
    assert mim_gira.sensory_vec[2] >= 0.35  # Tactile
    assert mim_gira.sensory_vec[6] >= 0.45  # Arousal (0.35 + 0.15 from ABAB)
    assert mim_gira.rhythm_bonus == 0.35

    # 2. しんと: Voiceless fricative 's' + hatsuon 'n'
    assert is_mimetic_word("しんと")
    mim_shin = synthesize_mimetic_8d("しんと")
    assert mim_shin.is_mimetic
    assert mim_shin.mimetic_immunity
    # Voiceless fricative gives auditory silence/friction (d2) and coolness (d6)
    assert mim_shin.sensory_vec[1] >= 0.40  # Auditory
    assert mim_shin.sensory_vec[5] >= 0.30  # Temperature
    # Hatsuon gives visceral afterglow/reverberation (d8)
    assert mim_shin.sensory_vec[7] >= 0.45  # Reverberation

    # 3. ぷつり: Semi-voiced 'p' + sokuon/ri impulse
    assert is_mimetic_word("ぷつり")
    mim_putsu = synthesize_mimetic_8d("ぷつり")
    assert mim_putsu.is_mimetic
    assert mim_putsu.mimetic_immunity
    # Semi-voiced gives visual brightness (d1) and kinematic abruptness (d5)
    assert mim_putsu.sensory_vec[0] >= 0.40  # Visual
    assert mim_putsu.sensory_vec[4] >= 0.70  # Kinematic (0.45 + 0.35 from ri)
    assert mim_putsu.impulse_bonus == 0.30

    # 4. Mimetic Immunity Protocol (Section 25 / Spec 2.1)
    # Even if dependent context has low concreteness C(u) = 0.10, mimetic is never damped!
    damped_res = apply_open01_polysemy_damping(
        u_concreteness=0.10,
        v_primary_vec=mim_gira.sensory_vec,
        v_is_mimetic=True,
    )
    assert damped_res.case == "mimetic_immunity"
    assert damped_res.damping_factor == 1.0
    assert np.allclose(damped_res.effective_sensory, mim_gira.sensory_vec)


# ==============================================================================
# 4. OPEN-04: 地の文と会話文のバフチン分離と Dialogic Collapse 検知
# ==============================================================================

def test_open04_bakhtinian_separation_and_dialogic_collapse():
    """Verify OPEN-04 Bakhtinian 2-channel separation and Dialogic Collapse threshold."""
    # Text with distinct polyphonic voices:
    text_distinct = (
        "夕暮れの路地裏で太郎は「おい、てめえ何やってんだよ！ぶっ飛ばすぞ！」と怒鳴った。"
        "しかし花子は「大変申し訳ございません。直ちに確認いたしますわ」と静かに頭を下げた。"
    )
    res_distinct = extract_dual_channel_bakhtinian(text_distinct, collapse_threshold=0.15)

    # Narration channel must have dialogues cleanly stripped
    assert "「" not in res_distinct.narration_text
    assert "」" not in res_distinct.narration_text
    assert "夕暮れの路地裏で太郎は" in res_distinct.narration_text
    assert "しかし花子は" in res_distinct.narration_text

    # Dialogue utterances must be separated per character
    assert "太郎" in res_distinct.speaker_utterances
    assert "花子" in res_distinct.speaker_utterances
    assert len(res_distinct.speaker_utterances["太郎"]) == 1
    assert len(res_distinct.speaker_utterances["花子"]) == 1

    # Stylometric voice distance between rough slang and formal keigo
    dist = res_distinct.pairwise_distances[("太郎", "花子")]
    assert dist >= 0.15, f"Expected distinct voices (Delta >= 0.15), got {dist:.3f}"
    assert not res_distinct.dialogic_collapse_detected

    # Text with homogeneous voices (Dialogic Collapse):
    text_collapse = (
        "会議室で太郎は「本日の進捗状況を報告いたします」と言った。"
        "次郎も「本日の進捗状況を報告いたします」と答えた。"
    )
    res_collapse = extract_dual_channel_bakhtinian(text_collapse, collapse_threshold=0.15)
    assert "太郎" in res_collapse.speaker_utterances
    assert "次郎" in res_collapse.speaker_utterances
    dist_collapse = res_collapse.pairwise_distances[("太郎", "次郎")]
    assert dist_collapse < 0.15, f"Expected collapse (Delta < 0.15), got {dist_collapse:.3f}"
    assert res_collapse.dialogic_collapse_detected, "Identical speech patterns must trigger dialogic collapse"


# ==============================================================================
# 5. SyntaxDriver & SyntaxTreeFrame End-to-End
# ==============================================================================

def test_syntax_driver_tree_building(driver: SyntaxDriver):
    """Verify that SyntaxDriver parses Japanese sentences into complete SyntaxTreeFrames."""
    # Test sentence: 「冷たい氷」
    frame_koori = driver.parse_sentence("冷たい氷")
    assert isinstance(frame_koori, SyntaxTreeFrame)
    assert len(frame_koori.nodes) >= 2
    assert frame_koori.root_id == len(frame_koori.nodes) - 1
    assert frame_koori.nodes[0].surface.startswith("冷た")
    assert "氷" in frame_koori.nodes[-1].surface

    # Test sentence: 「紅蓮の炎などなかった」
    frame_neg = driver.parse_sentence("紅蓮の炎などなかった")
    assert isinstance(frame_neg, SyntaxTreeFrame)
    # The final node should be in negation scope
    assert frame_neg.nodes[-1].is_negation_scope

    # Test dictionary serialization roundtrip
    frame_dict = frame_neg.to_dict()
    restored = SyntaxTreeFrame.from_dict(frame_dict)
    assert restored.sentence_id == frame_neg.sentence_id
    assert restored.text == frame_neg.text
    assert len(restored.nodes) == len(frame_neg.nodes)
    assert len(restored.edges) == len(frame_neg.edges)
    assert restored.max_depth == frame_neg.max_depth
    assert restored.depth_distribution == frame_neg.depth_distribution

    # Empty text edge case
    empty_frame = driver.parse_sentence("")
    assert len(empty_frame.nodes) == 0
    assert empty_frame.root_id == -1


def test_edge_cases_and_boundaries(driver: SyntaxDriver, lexicon: HTHPolyLexicon):
    """Verify extreme edge cases, non-mimetic exclusions, and boundary thresholds."""
    # 1. Non-mimetic exclusion check (preventing false positives on inflections)
    for word in ("なかった", "あった", "行った", "言った", "知った", "先生", "人間", "問題"):
        assert not is_mimetic_word(word), f"'{word}' must NOT be classified as mimetic!"

    # 2. Known mimetics and adverbial particle forms must match
    for mim in ("ぎらぎら", "ぎらぎらと", "きらきら", "ざわざわ", "しんと", "ぷつり", "ぷつりと", "ぴたり", "ふわり", "どっしり"):
        assert is_mimetic_word(mim), f"'{mim}' must be classified as mimetic!"

    # Voiceless velar stops (K-line: きらきら) must have visual and auditory activation
    syn_kira = synthesize_mimetic_8d("きらきら")
    assert syn_kira.sensory_vec[0] >= 0.35  # Visual sparkle
    assert syn_kira.sensory_vec[1] >= 0.35  # Auditory crispness

    # 3. OPEN-01 Synesthetic Metaphor (Case 2: C(u) < 0.45 but <s_u, s_v> > 0.25)
    u_vec = np.array([0.8, 0.1, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    v_pri = np.array([0.7, 0.2, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    res_metaphor = apply_open01_polysemy_damping(
        u_concreteness=0.20,
        v_primary_vec=v_pri,
        u_sensory_vec=u_vec,
    )
    assert res_metaphor.case == "case2_synesthetic_metaphor"
    assert res_metaphor.slot == "primary_physical"
    assert res_metaphor.metaphor_bonus == 0.35
    assert res_metaphor.damping_factor == 1.0

    # 4. Speculative Modality (alpha = 0.6)
    alpha_spec = evaluate_modality_alpha("だろう")
    assert alpha_spec == 0.6
    res_spec = compute_dual_sensory_tensor(v_pri, alpha_spec)
    assert res_spec.is_speculative
    assert not res_spec.is_negated
    assert np.allclose(res_spec.s_plus, 0.6 * v_pri)
    assert res_spec.s_minus == 0.0

    # 5. Multi-sentence parsing & radial entropy via SyntaxDriver
    multi_text = "冷たい氷が解けた。紅蓮の炎が消え去った。"
    frames = driver.parse_text(multi_text)
    assert len(frames) == 2
    assert frames[0].sentence_id == 0
    assert frames[1].sentence_id == 1
    assert frames[0].radial_entropy >= 0.0
    assert frames[0].nodes[-1].pos == "動詞"  # Cleanroom punctuation-stripped verb classification

    # 6. External morphological analyzer mode (Janome)
    janome_driver = SyntaxDriver(lexicon=lexicon, force_cleanroom=False)
    frame_janome = janome_driver.parse_sentence("冷たい氷が解けた。")
    assert frame_janome.nodes[0].pos == "形容詞"
    assert "氷" in frame_janome.nodes[1].surface
    assert frame_janome.nodes[-1].pos == "動詞"
    assert frame_janome.radial_entropy >= 0.0

    # 7. Speaker extraction robustness
    from usir_py.linguistic_gates import extract_speaker
    assert extract_speaker("太郎は笑って", "と言った") == "太郎"
    assert extract_speaker("私は", "と叫んだ") == "私"
    assert extract_speaker("花子は", "と答えた") == "花子"
