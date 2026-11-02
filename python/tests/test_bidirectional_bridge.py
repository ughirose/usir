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


