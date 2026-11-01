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
    pin_boundary_anchors,
    solve_bidirectional_collision_bridge,
    solve_tensor_collision_bridge,
    derive_endogenous_goal_hypotheses,
    execute_closed_loop_rewrite,
)


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
    res = solve_bidirectional_collision_bridge(anchors, k_candidates=4)

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
    assert len(res.welded_text) >= 400


def test_solve_bidirectional_collision_bridge_article():
    """Verify bidirectional meet-in-the-middle collision on an analytical article."""
    anchors = pin_boundary_anchors(theme="組織の非同期コミュニケーションと生産性", domain="article")
    res = solve_bidirectional_collision_bridge(anchors, k_candidates=4)

    assert res.domain == "article"
    assert res.optimal_pair[0] in range(4)
    assert res.optimal_pair[1] in range(4)
    assert res.m_bridge > 0.0
    assert "コミュニケーション" in res.welded_text or "組織" in res.welded_text
    assert len(res.welded_text) >= 400


def test_solve_bidirectional_collision_bridge_cot_reasoning():
    """Verify bidirectional meet-in-the-middle collision on a CoT reasoning problem."""
    anchors = pin_boundary_anchors(theme="コンパクト群上の不変測度の存在証明", domain="cot_reasoning")
    res = solve_bidirectional_collision_bridge(anchors, k_candidates=4)

    assert res.domain == "cot_reasoning"
    assert res.min_action < 10.0
    assert res.m_bridge > 0.0
    # Must preserve step structure and authentic proof paragraph scale
    assert any("ステップ" in step for step in res.welded_trajectory)
    assert len(res.welded_text) >= 400


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
    res = execute_closed_loop_rewrite(short_draft, domain="novel")

    assert isinstance(res, CollisionBridgeResult)
    assert res.anchors.x_0 == short_draft
    # x_T must not be identical to x_0
    assert res.anchors.x_T != short_draft
    assert len(res.anchors.x_T) > 10
    assert len(res.welded_text) >= 300


def test_derive_endogenous_goal_hypotheses_degenerate_input():
    """Verify derive_endogenous_goal_hypotheses handles empty or symbol-only inputs safely."""
    for bad_input in ["", "   \n\t  ", "!@#$%^&*()"]:
        optimal_x_T, scored = derive_endogenous_goal_hypotheses(bad_input, domain="article", k_candidates=3)
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
    optimal_x_T, scored = derive_endogenous_goal_hypotheses(x_0, domain="novel", k_candidates=3)

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
