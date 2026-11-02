"""Unit & Integration Tests for USIR v4.0 Bidirectional Schrödinger Bridge.

Verifies:
1. Boundary pinning across all 3 domains (novel, article, cot_reasoning).
2. Bidirectional rollout & meet-in-the-middle collision solver.
3. Midpoint action integral minimization and infilling welding.
4. Universal domain robustness without hardcoded outputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from usir_py.bidirectional_bridge import (
    BoundaryAnchors,
    CollisionBridgeResult,
    TensorBridgeResult,
    compute_collision_action,
    compute_trajectory_repetition_penalty,
    pin_boundary_anchors,
    sanitize_wave_steps,
    solve_bidirectional_collision_bridge,
    solve_tensor_collision_bridge,
    derive_endogenous_goal_hypotheses,
    execute_closed_loop_rewrite,
)



TEST_FWD_NOVEL = [
    ["雨夜の湯治場と老陶工の気配が漂う。", "冷え切った空気が作業場を満たしていた。", "古い轆轤が低く軋み始める。"],
    ["雨夜の湯治場と老陶工の気配が漂う。", "軒先からの水滴が踏み石を打つ。", "静まり返った谷底に火の色が差す。"],
    ["雨夜の湯治場と老陶工の気配が漂う。", "湿った土の匂いが立ち込めていた。", "老人は無言で粘土の重みを測る。"],
    ["雨夜の湯治場と老陶工の気配が漂う。", "遠い夜風が木立を揺らしていた。", "削り屑が床に静かに積もる。"],
]
TEST_BWD_NOVEL = [
    ["窯の焚口から微かな熱気が漏れる。", "言葉のないまま時間は過ぎていった。", "微かな軋みとともにすべての輪郭が沈む。"],
    ["指先が土の冷たさに慣れていく。", "暗がりのなかに炎の残像が浮かぶ。", "微かな軋みとともにすべての輪郭が沈む。"],
    ["夜雨の音がすべてを包み隠していく。", "静寂がふたたび作業場を満たした。", "微かな軋みとともにすべての輪郭が沈む。"],
    ["形を得た器が静かに冷えていく。", "暗闇の底で沈黙が定着していった。", "微かな軋みとともにすべての輪郭が沈む。"],
]

TEST_FWD_ARTICLE = [
    ["組織の非同期コミュニケーションと生産性に関する観測。", "即時チャットは認知的過負荷を招く。", "文脈の永続化が意思決定を加速する。"],
    ["組織の非同期コミュニケーションと生産性に関する観測。", "同期会議の削減が集中時間を生む。", "情報ハブの分散がボトルネックを防ぐ。"],
    ["組織の非同期コミュニケーションと生産性に関する観測。", "非対称なコミュニケーションを許容する。", "ドキュメントによる合意形成を徹底する。"],
    ["組織の非同期コミュニケーションと生産性に関する観測。", "業務ログの透明性が信頼を担保する。", "心理的安全性と実行速度を両立する。"],
]
TEST_BWD_ARTICLE = [
    ["自律的な意思決定フレームワークが浸透する。", "組織スケールに伴う情報劣化を遮断する。", "したがって本質的な競争優位は非同期プロトコルに帰着する。"],
    ["知識資産が複利的にレバレッジされる。", "属人性に依存しない組織基盤が完成する。", "したがって本質的な競争優位は非同期プロトコルに帰着する。"],
    ["調整コストの線形爆発を未然に防止する。", "高密度な知識創造へ集中可能となる。", "したがって本質的な競争優位は非同期プロトコルに帰着する。"],
    ["オープンな文脈共有が格差を解消する。", "強靭な自律分散エコシステムが確立される。", "したがって本質的な競争優位は非同期プロトコルに帰着する。"],
]

TEST_FWD_COT = [
    ["前提確定：コンパクト群上の不変測度の存在証明。", "ステップ1：ハール測度の左不変性を仮定する。", "ステップ2：畳み込み代数の性質を精査する。"],
    ["前提確定：コンパクト群上の不変測度の存在証明。", "ステップ1：可換代数の極大イデアルを構成する。", "ステップ2：ゲルファント変換を適用する。"],
    ["前提確定：コンパクト群上の不変測度の存在証明。", "ステップ1：一様連続関数族の等程度連続性を示す。", "ステップ2：アスコリ・アルツェラの定理を準備する。"],
    ["前提確定：コンパクト群上の不変測度の存在証明。", "ステップ1：ヒルベルト空間の凸閉集合を定義する。", "ステップ2：直交射影定理から最小元を特定する。"],
]
TEST_BWD_COT = [
    ["ステップN-2：核空間が自明に縮退する。", "ステップN-1：全単射性と等長性が示される。", "結論：不変測度の一意存在が証明された Q.E.D. 。"],
    ["ステップN-2：背理法仮定の矛盾が導かれる。", "ステップN-1：境界条件の整合性が完結する。", "結論：不変測度の一意存在が証明された Q.E.D. 。"],
    ["ステップN-2：ルベーグ収束定理が適用される。", "ステップN-1：極限の一意性が確定する。", "結論：不変測度の一意存在が証明された Q.E.D. 。"],
    ["ステップN-2：位相幾何学的不変量が保存される。", "ステップN-1：証明の導出鎖が完全に閉じる。", "結論：不変測度の一意存在が証明された Q.E.D. 。"],
]

TEST_GOALS = [
    "やがて微細な熱量の移動も完全に途絶え、空間のすべての輪郭が冷たく沈み込み、あとに不可逆な物理的平衡と静寂だけが残された。",
    "夜の底へと光の粒子の最後の余韻が吸い込まれ、空間を満たしていた微かな揺らぎは消滅し、世界は確固たる静止へと溶け落ちていった。",
    "遠くの物音も大気と同化して完全に消え去り、閉ざされた視界の奥には冷えた空気の重みと不動の輪郭だけが静かに定着していた。",
    "吹き抜けていた風も完全に凪ぎ、冷え切った大地と影の交錯する地点に、外界から隔絶された絶対的な沈黙が満ちていた。",
]


def test_pin_boundary_anchors_3_domains():
    """Verify boundary pinning across novel, article, and CoT reasoning domains."""
    for domain in ["novel", "article", "cot_reasoning"]:
        anchors = pin_boundary_anchors(theme="夜明けの雪山", domain=domain)
        assert anchors.domain == domain
        assert len(anchors.x_0) > 10
        assert len(anchors.x_T) > 10
        assert anchors.q_start.shape == (24,)
        assert anchors.q_end.shape == (24,)
        assert anchors.kamae_coords.shape == (3,)


def test_pin_boundary_anchors_custom_override():
    """Verify custom boundary anchors override default templates."""
    custom_start = "凍てつく湖の岸辺に、旅人は馬を止めた。"
    custom_end = "薄氷が静かに割れ、水底へと青い光が吸い込まれていった。"
    anchors = pin_boundary_anchors(
        theme="氷湖",
        domain="novel",
        custom_start=custom_start,
        custom_end=custom_end,
    )
    assert anchors.x_0 == custom_start
    assert anchors.x_T == custom_end


def test_solve_bidirectional_collision_bridge_novel():
    """Verify bidirectional meet-in-the-middle collision on a novel theme at authentic paragraph scale."""
    anchors = pin_boundary_anchors(theme="雨夜の湯治場と老陶工", domain="novel")
    res = solve_bidirectional_collision_bridge(anchors, forward_candidates=TEST_FWD_NOVEL, backward_candidates=TEST_BWD_NOVEL, k_candidates=4)

    assert isinstance(res, CollisionBridgeResult)
    assert res.domain == "novel"
    assert len(res.forward_waves) == 4
    assert len(res.backward_waves) == 4
    assert res.collision_matrix.shape == (4, 4)
    assert res.min_action > 0.0
    assert 0.0 <= res.m_bridge <= 1.0

    # Optimal pair must have the minimum action in the matrix
    opt_i, opt_j = res.optimal_pair
    assert res.collision_matrix[opt_i, opt_j] == res.min_action
    assert res.min_action == np.min(res.collision_matrix)

    # Welded trajectory contains both waves and meets authentic paragraph scale (400-800 chars)
    assert len(res.welded_trajectory) >= 6
    assert len(res.welded_text) >= 80


def test_solve_bidirectional_collision_bridge_article():
    """Verify bidirectional meet-in-the-middle collision on an analytical article."""
    anchors = pin_boundary_anchors(theme="組織の非同期コミュニケーションと生産性", domain="article")
    res = solve_bidirectional_collision_bridge(anchors, forward_candidates=TEST_FWD_ARTICLE, backward_candidates=TEST_BWD_ARTICLE, k_candidates=4)

    assert res.domain == "article"
    assert res.optimal_pair[0] in range(4)
    assert res.optimal_pair[1] in range(4)
    assert res.m_bridge > 0.0
    assert "コミュニケーション" in res.welded_text or "組織" in res.welded_text
    assert len(res.welded_text) >= 80


def test_solve_bidirectional_collision_bridge_cot_reasoning():
    """Verify bidirectional meet-in-the-middle collision on a CoT reasoning problem."""
    anchors = pin_boundary_anchors(theme="コンパクト群上の不変測度の存在証明", domain="cot_reasoning")
    res = solve_bidirectional_collision_bridge(anchors, forward_candidates=TEST_FWD_COT, backward_candidates=TEST_BWD_COT, k_candidates=4)

    assert res.domain == "cot_reasoning"
    assert res.min_action < 10.0
    assert res.m_bridge > 0.0
    # Must preserve step structure and authentic proof paragraph scale
    assert any("ステップ" in step for step in res.welded_trajectory)
    assert len(res.welded_text) >= 80


def test_solve_bidirectional_collision_bridge_empty_candidates_validation():
    """Verify that empty or invalid candidate lists raise clean ValueErrors."""
    anchors = pin_boundary_anchors(theme="テスト", domain="novel")

    with pytest.raises(ValueError, match="non-empty lists"):
        solve_bidirectional_collision_bridge(anchors, forward_candidates=[], backward_candidates=[])

    with pytest.raises(ValueError, match="at least one non-empty candidate"):
        solve_bidirectional_collision_bridge(anchors, forward_candidates=[[]], backward_candidates=[[]])

    with pytest.raises(ValueError, match="at least one non-empty candidate"):
        solve_bidirectional_collision_bridge(anchors, forward_candidates=[["   "]], backward_candidates=[[""]])


def test_execute_closed_loop_rewrite_single_sentence_no_self_loop():
    """Verify execute_closed_loop_rewrite on single sentence generates distinct x_T without self-loop."""
    short_draft = "冷たい雨が古びた瓦屋根を激しく打ち鳴らしていた。"
    res = execute_closed_loop_rewrite(short_draft, domain="novel", candidate_infillings_fwd=TEST_FWD_NOVEL, candidate_infillings_bwd=TEST_BWD_NOVEL, candidate_goals=TEST_GOALS)

    assert isinstance(res, CollisionBridgeResult)
    assert res.anchors.x_0 == short_draft
    # x_T must not be identical to x_0
    assert res.anchors.x_T != short_draft
    assert len(res.anchors.x_T) > 10
    assert len(res.welded_text) >= 80


def test_derive_endogenous_goal_hypotheses_degenerate_input():
    """Verify derive_endogenous_goal_hypotheses handles empty or symbol-only inputs safely."""
    for bad_input in ["", "   \n\t  ", "!@#$%^&*()"]:
        optimal_x_T, scored = derive_endogenous_goal_hypotheses(bad_input, domain="article", candidate_goals=TEST_GOALS, k_candidates=3)
        assert isinstance(optimal_x_T, str)
        assert len(optimal_x_T) > 10
        assert len(scored) == 3


def test_repetition_penalty_in_collision_action():
    """Verify compute_collision_action penalizes duplicate verbatim sentences."""
    anchors = pin_boundary_anchors(theme="反復検知", domain="novel")

    fwd_normal = [["冷え切った空気に足音が消える。"]]
    bwd_normal = [["凍りついた小枝が細い音を立てた。"]]
    # Verbatim duplicate step
    bwd_duplicate = [["冷え切った空気に足音が消える。"]]

    res_normal = solve_bidirectional_collision_bridge(
        anchors,
        forward_candidates=fwd_normal,
        backward_candidates=bwd_normal,
    )
    res_duplicate = solve_bidirectional_collision_bridge(
        anchors,
        forward_candidates=fwd_normal,
        backward_candidates=bwd_duplicate,
    )

    assert res_normal.min_action < res_duplicate.min_action


def test_collision_action_sensitivity():
    """Action integral must penalize disconnected trajectories more than coherent ones."""
    anchors = pin_boundary_anchors(theme="テスト", domain="novel")

    # Coherent waves
    fwd_coherent = [
        ["山道を登る。", "杉の木立から木漏れ日が差した。"],
    ]
    bwd_coherent = [
        ["木漏れ日が土を温める。", "頂上の社にたどり着いた。"],
    ]

    # Incoherent waves (topic leap)
    bwd_incoherent = [
        ["宇宙ステーションの軌道が修正された。", "火星の探査機が着陸した。"],
    ]

    res_good = solve_bidirectional_collision_bridge(
        anchors,
        forward_candidates=fwd_coherent,
        backward_candidates=bwd_coherent,
    )
    res_bad = solve_bidirectional_collision_bridge(
        anchors,
        forward_candidates=fwd_coherent,
        backward_candidates=bwd_incoherent,
    )

    assert res_good.min_action < res_bad.min_action
    assert res_good.m_bridge > res_bad.m_bridge


def test_solve_tensor_collision_bridge():
    """Verify abstract tensor Schrödinger bridge collision across candidate latent trajectories."""
    q_start = np.full(24, 0.1)
    q_end = np.full(24, 0.9)

    fwd_aligned = [
        np.array([np.full(24, 0.1), np.full(24, 0.3), np.full(24, 0.5)]),
    ]
    bwd_aligned = [
        np.array([np.full(24, 0.52), np.full(24, 0.7), np.full(24, 0.9)]),
    ]
    bwd_divergent = [
        np.array([np.full(24, -0.8), np.full(24, 0.0), np.full(24, 0.9)]),
    ]

    res_aligned = solve_tensor_collision_bridge(q_start, q_end, fwd_aligned, bwd_aligned)
    res_divergent = solve_tensor_collision_bridge(q_start, q_end, fwd_aligned, bwd_divergent)

    assert isinstance(res_aligned, TensorBridgeResult)
    assert res_aligned.min_action > 0.0
    assert res_aligned.min_action < res_divergent.min_action
    assert res_aligned.m_bridge > res_divergent.m_bridge
    assert res_aligned.action_matrix.shape == (1, 1)
    assert res_aligned.welded_tensor.shape == (6, 24)


def test_derive_endogenous_goal_hypotheses():
    """Verify Mode B endogenous goal hypothesis derivation without pre-pinned end sentences."""
    x_0 = "未明の静寂のなかで、谷あいの水車が微かに軋む音を立てていた。"
    optimal_x_T, scored = derive_endogenous_goal_hypotheses(x_0, domain="novel", candidate_goals=TEST_GOALS, k_candidates=3)

    assert isinstance(optimal_x_T, str)
    assert len(optimal_x_T) > 10
    assert len(scored) == 3
    assert scored[0]["score"] <= scored[1]["score"]
    assert "kamae_diff" in scored[0]
    assert "d_3phase" in scored[0]


def test_execute_closed_loop_rewrite():
    """Verify Mode A closed-loop rewrite on an unconstrained drifting draft."""
    draft = (
        "電車の事故で怪我を負った私は、温泉宿で療養することにした。"
        "部屋の窓から外を眺めると、雲がゆっくりと流れていた。"
        "夕方になって鳥が鳴き、やがて空は暗くなっていった。"
        "微かな軋みとともに輪郭が沈み、あとに冷たい沈黙だけが残った。"
    )
    res = execute_closed_loop_rewrite(draft, domain="novel")

    assert isinstance(res, CollisionBridgeResult)
    assert "怪我を負った" in res.anchors.x_0
    assert "冷たい沈黙だけが残った" in res.anchors.x_T
    assert res.min_action > 0.0
    assert len(res.welded_text) > 50


def test_repetition_penalty_in_collision_action():
    """Verify compute_collision_action penalizes duplicate verbatim sentences."""
    anchors = pin_boundary_anchors(theme="反復検知", domain="novel")

    fwd_normal = [["冷え切った空気に足音が消える。"]]
    bwd_normal = [["凍りついた小枝が細い音を立てた。"]]
    # Verbatim duplicate step
    bwd_duplicate = [["冷え切った空気に足音が消える。"]]

    res_normal = solve_bidirectional_collision_bridge(
        anchors,
        forward_candidates=fwd_normal,
        backward_candidates=bwd_normal,
    )
    res_duplicate = solve_bidirectional_collision_bridge(
        anchors,
        forward_candidates=fwd_normal,
        backward_candidates=bwd_duplicate,
    )

    assert res_normal.min_action < res_duplicate.min_action

def test_solve_bidirectional_collision_bridge_missing_candidates_raises():
    """Verify that calling solve_bidirectional_collision_bridge without candidates raises ValueError."""
    anchors = pin_boundary_anchors(theme="未指定テスト", domain="novel")
    with pytest.raises(ValueError, match="must be explicitly provided"):
        solve_bidirectional_collision_bridge(anchors)


def test_derive_endogenous_goal_hypotheses_missing_candidates_raises():
    """Verify that calling derive_endogenous_goal_hypotheses without candidates raises ValueError."""
    with pytest.raises(ValueError, match="must be explicitly provided"):
        derive_endogenous_goal_hypotheses("冒頭文", domain="novel")


def test_parrot_infill_deduplication_gate():
    """オウム返し Infill の完全除去テスト: infill_generator returning duplicate/echo sentences is stripped."""
    anchors = pin_boundary_anchors(theme="重複遮断検証", domain="article")
    fwd = [["都市空間の歩行速度は過去五十年で一割上昇した。", "スマートフォンの普及により視線は下方に固定され、受動的な群衆流動が常態化している。"]]
    bwd = [["身体的な余白を取り戻す建築設計が求められている。", "広場における偶発的な滞留こそが都市の呼吸を再生する。"]]

    # Parrot infill repeating fwd[-1] and bwd[0] verbatim
    def parrot_infill(f_last: str, b_first: str) -> str:
        return f"{f_last}{b_first}"

    res = solve_bidirectional_collision_bridge(
        anchors,
        forward_candidates=fwd,
        backward_candidates=bwd,
        infill_generator=parrot_infill,
    )
    assert res.infilling_step is None
    assert len(res.welded_trajectory) == 4
    assert res.welded_trajectory.count(fwd[0][-1]) == 1


def test_prefix_repetition_penalty_and_5gram_gate():
    """Prefix 反復ペナルティテスト: leading 5-char echo and 5-gram overlap incur trajectory penalties."""
    steps_clean = [
        "夜明け前の山道に白い霧が立ち込めていた。",
        "遠くの沢から冷たい水音が微かに響いてくる。",
        "石畳の階段を踏みしめるたびに草の匂いが立った。",
    ]
    steps_prefix_echo = [
        "朝日が昇る頃には、山道に白い霧が立ち込めていた。",
        "遠くの沢から冷たい水音が微かに響いてくる。",
        "朝日が昇る頃には、谷底の集落に淡い光が差し込んだ。",
    ]
    pen_clean = compute_trajectory_repetition_penalty(steps_clean)
    pen_echo = compute_trajectory_repetition_penalty(steps_prefix_echo)
    assert pen_clean == 0.0
    assert pen_echo >= 4.0


def test_cot_meta_poem_conclusion_rejection():
    """CoT メタポエム結論の棄却テスト: mathematical derivation is selected over meta-explanation poem."""
    x0 = "実数列 $a_n$ が有界かつ単調増加であるとき、極限 $\\lim_{n\\to\\infty} a_n = \\sup_n a_n$ の存在を示せ。"
    candidates = [
        "この定理は解析学における重要な基盤であり、社会や科学の様々な応用モデル化において不可欠な役割を果たしている。",
        "上限 $M = \\sup_n a_n$ に対し $\\forall \\varepsilon > 0, \\exists N \\in \\mathbb{N}$ で $M - \\varepsilon < a_n \\le M$ となり収束が一意に示される。Q.E.D.",
    ]
    optimal_x_T, rankings = derive_endogenous_goal_hypotheses(
        x0,
        domain="cot_reasoning",
        candidate_goals=candidates,
        k_candidates=2,
    )
    assert "Q.E.D." in optimal_x_T
    assert "重要な基盤" not in optimal_x_T
    assert rankings[0]["score"] < rankings[1]["score"]


def test_sanitize_wave_steps_and_kanji_stem_jaccard():
    """Verify sanitize_wave_steps strips kanji stem paraphrases (Jaccard >= 0.50), meta-leaks, and CJK tokens."""
    x0 = "風雪の峠道と廃屋の灯りの気配が漂い、張り詰めた冷気が指先をかすかに凍らせていた。"
    xT = "微かな軋みとともにすべての輪郭が沈み、あとに冷たい沈黙だけが残った。"
    raw_steps = [
        x0,
        "それは雪国の伝統であり、またそれは現代の世界でも、我々の生活を支える力となる。",
        "それらは、雪国の伝統であり、現代の世界にもその力を支えている。",  # Jaccard >= 0.50 paraphrase
        "この展開文は、冒頭文で示された状況を補強する。",  # Meta-leak
        "函数列是恒正且有界である。",  # CJK artifact
        "囲炉裏の残り火が暗がりで静かに揺れていた。",
        xT,
    ]
    cleaned = sanitize_wave_steps(raw_steps, x_0=x0, x_T=xT)
    assert len(cleaned) == 4
    assert cleaned[0] == x0
    assert "それは雪国の伝統であり" in cleaned[1]
    assert "囲炉裏の残り火" in cleaned[2]
    assert cleaned[3] == xT

    # Also verify compute_trajectory_repetition_penalty penalizes stem Jaccard >= 0.50 by +6.0
    pen_paraphrase = compute_trajectory_repetition_penalty([
        "それは雪国の伝統であり、またそれは現代の世界でも、我々の生活を支える力となる。",
        "それらは、雪国の伝統であり、現代の世界にもその力を支えている。",
    ])
    assert pen_paraphrase >= 6.0


def test_sanitize_wave_steps_non_kanji_and_zero_kana_cjk():
    """Verify sanitize_wave_steps deduplicates non-kanji sentences and strips zero-kana Chinese sentences."""
    x0 = "あかりがしずかにゆれていた。"
    xT = "そしてすべてがしずまりかえった。"
    raw_steps = [
        x0,
        "そよかぜがまどをかすかにゆらしていく。",
        "そよかぜがまどをかすかにゆらしていく。",  # Non-kanji verbatim duplicate
        "假设在希尔伯特空间 (H) 中，(M) 是一个闭部分空间。",  # Simplified Chinese leak
        "晨光漸漸穿透，濡濕石板映射黎明前寂寥。",  # Zero-kana Traditional Chinese leak
        x0,  # Intermediate copy of x0 must not bypass deduplication as boundary
        xT,
    ]
    cleaned = sanitize_wave_steps(raw_steps, x_0=x0, x_T=xT)
    assert cleaned == [x0, "そよかぜがまどをかすかにゆらしていく。", xT]


def test_sanitize_wave_steps_preserves_japanese_words_and_prunes_chinese_grammar():
    """Verify standard Japanese with 若者, 若い, 若干, 寄与, 与える is preserved while Chinese grammar is pruned."""
    from usir_py.bidirectional_bridge import _has_meta_or_cjk_leak

    # Valid Japanese sentences must NEVER be flagged as leaks
    valid_japanese = [
        "若者たちは雪の中を静かに歩き始めた。",
        "若い頃の記憶が鮮やかに蘇る。",
        "若干の修正を論文に加え、精度を向上させた。",
        "この研究の成果は今後の科学技術の発展に大きく寄与する。",
        "彼に新しい役割を与えることで組織の活性化を図る。",
        "給与明細を確認し、今後の計画を立てる。",
        "老若男女が集まる広場に温かい光が差した。",
    ]
    for s in valid_japanese:
        assert not _has_meta_or_cjk_leak(s), f"False positive leak detected on valid Japanese: {s}"

    # Chinese grammar leak sentences MUST be flagged as leaks
    chinese_grammar_leaks = [
        r"若 $x \in S$ 则 $f(x) > 0$ である。",
        r"\text{若} \quad f_{N(x)}(x) \to f(x) \text{ 则必然成立}。",
        r"若且仅当 $n \to \infty$ の極限が存在する。",
        r"空間 $V$ 与 $W$ の直和分解を考える。",
        r"与其相反的是、別の境界条件が課される。",
    ]
    for s in chinese_grammar_leaks:
        assert _has_meta_or_cjk_leak(s), f"Failed to detect Chinese grammar leak: {s}"


def test_sanitize_wave_steps_boundary_whitespace_and_x0_echo_dedup():
    """Verify sanitize_wave_steps is resilient to whitespace and prunes intermediate steps echoing x_0."""
    x0 = "国境の長いトンネルを抜けると雪国であった。"
    xT = "夜の底が白くなった。"

    # 1. Trailing and leading whitespace resilience
    steps_with_ws = ["  ", x0, "鳥が静かに歌っていた。", xT, "   "]
    cleaned_ws = sanitize_wave_steps(steps_with_ws, x_0=x0, x_T=xT)
    assert cleaned_ws == [x0, "鳥が静かに歌っていた。", xT], f"Unexpected cleaned steps: {cleaned_ws}"

    # 2. Intermediate step echoing x_0 must be pruned against x_0 anchor
    trajectory_with_x0_echo = [
        x0,
        "信号所の手前で列車が止まった。",
        "長いトンネルを抜けると雪国であった。",  # 5-gram and stem match with x0 at idx > 0
        xT,
    ]
    cleaned_echo = sanitize_wave_steps(trajectory_with_x0_echo, x_0=x0, x_T=xT)
    assert len(cleaned_echo) == 3
    assert cleaned_echo[0] == x0
    assert cleaned_echo[1] == "信号所の手前で列車が止まった。"
    assert cleaned_echo[2] == xT


def test_sanitize_wave_steps_prunes_euc_jp_simplified_cjk_and_chinese_grammar():
    """Verify non-Japanese CJK (隐, 测, 步) and Chinese grammar (令 $f_n$, 在...上, 第一步) are pruned with 0 false positives."""
    from usir_py.bidirectional_bridge import _has_meta_or_cjk_leak, sanitize_wave_steps
    from usir_py.realtime_decoder import is_non_japanese_cjk

    # 1. Direct character / string EUC-JP whitelist check
    assert is_non_japanese_cjk("隐")
    assert is_non_japanese_cjk("测")
    assert is_non_japanese_cjk("这")
    assert is_non_japanese_cjk("们")
    assert is_non_japanese_cjk("为")
    assert is_non_japanese_cjk("设")
    assert is_non_japanese_cjk("证")
    assert is_non_japanese_cjk("满")
    assert is_non_japanese_cjk("导")
    assert is_non_japanese_cjk("边")
    assert is_non_japanese_cjk("步")  # U+6B65 (simplified/old)
    assert not is_non_japanese_cjk("歩")  # U+6B69 (standard Japanese)
    assert not is_non_japanese_cjk("希")
    assert not is_non_japanese_cjk("著")
    assert not is_non_japanese_cjk("据")
    assert not is_non_japanese_cjk("与")
    assert not is_non_japanese_cjk("若")

    # 2. Chinese mathematical & spatial grammar patterns
    leaks = [
        "令 $f_n$ 為一実関数列とする。",
        "したがって、令 $x \\in \\Omega$ とする。",
        "在閉区間 $[a, b]$ 上で連続関数を仮定する。",
        "第一步：ヒルベルト空間の完備性を確認する。",
        "第二步：コーシー列の収束を評価する。",
        "霧の中に隐れた輪郭を凝視する。",
        "温度の微細な変化を测定した。",
    ]
    for leak_sent in leaks:
        assert _has_meta_or_cjk_leak(leak_sent), f"Failed to detect leak: {leak_sent}"

    # 3. Standard Japanese Joyo kanji preservation (zero false positives)
    clean_japanese = [
        "希望に満ちた夜明けの光が梢を照らし出した。",
        "著しい科学技術の進展が理論の妥当性を支えている。",
        "旅人は道端の巨石に腰を据え、乾いた喉を潤した。",
        "自律分散的なプロトコルが組織の生産性に寄与する。",
        "若き哲学者たちは深夜まで激論を交わした。",
        "訴訟の手続きが滞りなく完了した。",
        "命題の前提条件を簡潔に説明する。",
        "湿った空気が草むらに深く沈殿していく。",
        "一歩一歩、雪道を踏みしめて前進する。",
    ]
    for j_sent in clean_japanese:
        assert not _has_meta_or_cjk_leak(j_sent), f"False positive leak detected: {j_sent}"


def test_tail_clause_local_jaccard_novel_04_motif_pruning():
    """Verify novel-04 tail-clause motif repetition (湿地...渡り鳥) is pruned by _is_kanji_stem_repetition."""
    from usir_py.bidirectional_bridge import _is_kanji_stem_repetition, sanitize_wave_steps

    s_a = "遠くの湿地では渡り鳥の姿すら消え…。"
    s_b = "寒風に揺れる干拓地の光跡に、遠くの湿地でさえも渡り鳥の群れが消えていった。"

    # Must trigger repetition check
    assert _is_kanji_stem_repetition(s_a, s_b), "Failed to detect novel-04 tail clause repetition!"

    # In sanitize_wave_steps, s_b must be pruned when s_a preceded it
    x0 = "薄暮の干拓地に冷たい風が吹き抜けていた。"
    xT = "やがて冷たい夜の静寂が湿地全体を覆い尽くした。"
    cleaned = sanitize_wave_steps([x0, s_a, s_b, xT], x_0=x0, x_T=xT)
    assert s_a in cleaned
    assert s_b not in cleaned
    assert len(cleaned) == 3


def test_paragraph_scale_barrier_penalty():
    """Verify compute_collision_action adds underflow penalty when max_possible_len >= 250 and L < 320."""
    from usir_py.bidirectional_bridge import TrajectoryWave, compute_collision_action
    import numpy as np

    tip = np.full(24, 0.5)
    fwd = TrajectoryWave(wave_id=0, direction="forward", steps=["短い文です。"], tip_vector=tip, direction_vector=tip)
    bwd = TrajectoryWave(wave_id=0, direction="backward", steps=["終わりの文です。"], tip_vector=tip, direction_vector=tip)

    # When max_possible_len is None or short (< 250), no underflow penalty
    s_normal, _, _, _ = compute_collision_action(fwd, bwd, max_possible_len=None)
    # When max_possible_len >= 250 (e.g. from realistic candidate pool), underflow penalty is added
    s_penalized, _, _, _ = compute_collision_action(fwd, bwd, max_possible_len=350)

    assert s_penalized > s_normal + 10.0, f"Expected underflow barrier penalty: {s_penalized} vs {s_normal}"


def test_continuous_rollout_3point_skeleton_selection():
    """Verify USIR v4.0 continuous rollout selection via 3-point geodesic skeleton (x_0 -> x_mid* -> x_T*).

    Tests that:
    1. Intermediate Midpoint Pivot x_mid* is deterministically identified from collision waves.
    2. single-pass continuous rollout candidates (>= 400 chars) are evaluated with S_full.
    3. Candidate with minimum S_full is chosen intact as welded_text without sentence trimming.
    """
    anchors = pin_boundary_anchors(theme="雨夜の湯治場と老陶工", domain="novel")

    # Construct 3 complete paragraph candidates (>= 400 chars each)
    cand_repetitive = anchors.x_0 + " " + ("冷気が指先と肌を凍らせていた。" * 25) + anchors.x_T
    assert len(cand_repetitive) >= 400

    cand_optimal = (
        anchors.x_0 + " "
        + "軒先からの雨垂れが規則正しく踏み石を打ち鳴らし、暗がりの奥で老陶工の静かな呼吸だけが湿った空気に混ざり合っていた。"
        + "作業台に積もった陶土の匂いが立ち込め、使い込まれた古い轆轤が低く軋みながら規則的な回転を刻み始める。"
        + "削り屑が濡れた板張りの床へと静かに積もり、窯の焚口から漏れ出す微かな熱気が作業場全体の冷気を徐々に押し広げていく。"
        + "手の中で滑らかに立ち上がる器の輪郭を確かめながら、言葉のないまま沈黙の時間がゆっくりと沈殿していった。"
        + "やがて暗闇の底で炎の残像が静かに溶け落ち、冷気の奥に張り詰めていた研ぎ澄まされた緊張が静かに解き放たれていく。"
        + "火勢の衰えとともに影が伸び、土と炎の対話が終わりを告げる頃、冷気は再び澄み渡り、作業場の隅々まで穏やかな静けさが満ちていった。"
        + anchors.x_T
    )
    assert len(cand_optimal) >= 400

    cand_divergent = (
        "急激な相転移と超伝導状態の発現により、量子ゆらぎが巨視的な結晶格子の欠陥をことごとく覆い尽くしていった。"
        "電子密度の特異点近傍において対称性の自発的破れが生じ、空間の位相的欠陥が指数関数的な速度で急速に増殖していった。"
        "ゲージ場の微細なゆらぎは長距離の非線形相互作用を媒介し、局所的なエネルギー散逸とエントロピー生成を不可逆的に加速させた。"
        "したがって臨界温度近傍における秩序パラメータの動的振る舞いは、従来の平均場理論や準粒子描像による漸近的予測を完全に逸脱している。"
        "最終的にすべての多体固有状態は熱力学的平衡へ到達することなく、強相関効果によって未知の非平衡トポロジカル定常状態へと捕捉された。"
        "この巨視的コヒーレンスの崩壊は、基底状態の縮退度とベリー位相の幾何学的性質が非自明な絡み合いを示していることを実験的に証明している。"
        "ゆえに超流動密度と相関長の特異な臨界指数が普遍性クラスの境界において発散し、量子カオスの領域へと完全に遷移した。"
    )
    assert len(cand_divergent) >= 400

    def mock_continuous_generator(x_0: str, x_mid: str, x_T: str) -> list[str]:
        assert len(x_0) > 0
        assert len(x_mid) > 0
        assert len(x_T) > 0
        return [cand_repetitive, cand_optimal, cand_divergent]

    res = solve_bidirectional_collision_bridge(
        anchors,
        forward_candidates=TEST_FWD_NOVEL,
        backward_candidates=TEST_BWD_NOVEL,
        continuous_rollout_generator=mock_continuous_generator,
        k_candidates=4,
    )

    # 1. Midpoint pivot must be extracted and non-empty
    assert res.midpoint_pivot is not None
    assert len(res.midpoint_pivot) > 0

    # 2. Minimum action must choose cand_optimal
    assert res.welded_text == cand_optimal
    assert len(res.welded_text) >= 400

    # 3. No string-stitching infill artifact
    assert res.infilling_step is None
    assert len(res.welded_trajectory) >= 4
    assert res.m_bridge > 0.0
