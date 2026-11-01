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
    compute_collision_action,
    pin_boundary_anchors,
    solve_bidirectional_collision_bridge,
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
    """Verify bidirectional meet-in-the-middle collision on a novel theme."""
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

    # Welded trajectory contains both waves
    assert len(res.welded_trajectory) >= 4
    assert len(res.welded_text) > 50


def test_solve_bidirectional_collision_bridge_article():
    """Verify bidirectional meet-in-the-middle collision on an analytical article."""
    anchors = pin_boundary_anchors(theme="組織の非同期コミュニケーションと生産性", domain="article")
    res = solve_bidirectional_collision_bridge(anchors, k_candidates=4)

    assert res.domain == "article"
    assert res.optimal_pair[0] in range(4)
    assert res.optimal_pair[1] in range(4)
    assert res.m_bridge > 0.0
    assert "コミュニケーション" in res.welded_text or "組織" in res.welded_text


def test_solve_bidirectional_collision_bridge_cot_reasoning():
    """Verify bidirectional meet-in-the-middle collision on a CoT reasoning problem."""
    anchors = pin_boundary_anchors(theme="コンパクト群上の不変測度の存在証明", domain="cot_reasoning")
    res = solve_bidirectional_collision_bridge(anchors, k_candidates=4)

    assert res.domain == "cot_reasoning"
    assert res.min_action < 10.0
    assert res.m_bridge > 0.0
    # Must preserve step structure
    assert any("ステップ" in step for step in res.welded_trajectory)


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
