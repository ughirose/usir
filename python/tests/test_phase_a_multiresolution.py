"""Phase A Multiresolution Pipeline & Contrastive Fisher Metric Test Suite.

Verifies:
1. Stage 1: 4 basic structural signals s_m = (Flux, Lattice, Void, Shock) in R^4.
2. Stage 2: 3-scale orthogonal Haar wavelets, ISTA sparse solitons, RQA I_arch -> 24D state.
3. Contrastive Fisher Distillation: positive definiteness of G_style and Rayleigh quotient discriminant separability.
4. embed_sentence_unit with SyntaxTreeFrame priority and hidden_states pooling hook.
5. Boundary and edge case robustness.
"""

from __future__ import annotations

import numpy as np
import pytest

from usir_py.extractor_24d import (
    extract_stage1_signals,
    extract_stage2_multiresolution,
    extract_multiresolution_24d,
    extract_usir_24d,
)
from usir_py.manifold_geodesic import (
    fit_contrastive_fisher_metric,
    _embed_sentence_unit,
    embed_sentence_unit,
)
from usir_py.syntax_frame import SyntaxEdge, SyntaxNode, SyntaxTreeFrame


def test_stage1_signal_extraction_basic():
    """Verify Stage 1 extraction of 4 basic signals on Japanese literary text."""
    text = (
        "硝子の破片が冷たい朝の光を浴びて、鋭くきらきらと輝いていた。"
        "少年は静かにその場に立ち止まり、息を殺した。"
        "しかし、何事も起こらなかった。"
        "風だけが、音もなく吹き抜けていった。"
    )
    signals = extract_stage1_signals(text)
    assert signals.ndim == 2
    assert signals.shape[1] == 4
    assert signals.shape[0] >= 3

    # All signals must lie in [0.0, 1.0]
    assert np.all(signals >= 0.0)
    assert np.all(signals <= 1.0)

    # Check signal channels
    flux = signals[:, 0]
    lattice = signals[:, 1]
    void = signals[:, 2]
    shock = signals[:, 3]

    assert np.mean(flux) > 0.0
    assert np.mean(lattice) > 0.0
    assert np.mean(void) > 0.0
    assert np.mean(shock) > 0.0


def test_stage1_with_explicit_syntax_tree_frame():
    """Verify that SyntaxTreeFrame properties directly modulate Stage 1 signals."""
    # Frame 1: High radial entropy + physical sensory nodes
    node1 = SyntaxNode(node_id=0, surface="冷たい", lemma="冷たい", pos="形容詞", pos_detail="", hth_primary=0x51b7 << 34 | 0xF)
    node2 = SyntaxNode(node_id=1, surface="氷を", lemma="氷", pos="名詞", pos_detail="格助詞", hth_primary=0x6c37 << 34 | 0xE)
    node3 = SyntaxNode(node_id=2, surface="落とす", lemma="落とす", pos="動詞", pos_detail="", hth_primary=0x8efd << 34 | 0xD)
    edge1 = SyntaxEdge(source_id=0, target_id=1, relation="mod", jump_distance=0.1)
    edge2 = SyntaxEdge(source_id=1, target_id=2, relation="obj", jump_distance=0.75)

    frame_dense = SyntaxTreeFrame(
        sentence_id=0,
        text="冷たい氷を落とす。",
        nodes=[node1, node2, node3],
        edges=[edge1, edge2],
        root_id=2,
        max_depth=2,
        depth_distribution=[0.33, 0.33, 0.34],
        radial_entropy=1.09,
    )

    # Frame 2: Negation scope and shock contrast
    neg_node1 = SyntaxNode(node_id=0, surface="声も", lemma="声", pos="名詞", pos_detail="係助詞", hth_primary=0x58f0 << 34 | 0x8)
    neg_node2 = SyntaxNode(node_id=1, surface="なかった", lemma="ない", pos="助動詞", pos_detail="終助詞", is_negation_scope=True)
    neg_edge = SyntaxEdge(source_id=0, target_id=1, relation="subj", jump_distance=0.85)

    frame_neg = SyntaxTreeFrame(
        sentence_id=1,
        text="声もなかった。",
        nodes=[neg_node1, neg_node2],
        edges=[neg_edge],
        root_id=1,
        max_depth=1,
        depth_distribution=[0.5, 0.5],
        radial_entropy=0.69,
    )

    signals = extract_stage1_signals("", frames=[frame_dense, frame_neg])
    assert signals.shape == (2, 4)

    # Dense frame should have higher flux than bare negation
    assert signals[0, 0] > 0.1
    # Frame with negation scope and hyperbolic jump should trigger Shock (s4)
    assert signals[1, 3] > 0.2
    # Short apophatic sentence should have healthy Void (s3)
    assert signals[1, 2] > 0.3


def test_stage2_multiresolution_wavelet_and_ista():
    """Verify Stage 2 wavelet decomposition, ISTA solitons, and RQA I_arch."""
    # Synthetic 4-signal series across 10 sentences
    np.random.seed(42)
    s_signals = np.random.uniform(0.1, 0.8, size=(10, 4))
    # Inject a dramatic shock spike at sentence 5
    s_signals[5, 3] = 0.95

    res = extract_stage2_multiresolution(s_signals, text="テスト文書。波形展開を検証する。")
    assert "q" in res
    assert "p" in res
    assert "z_24d" in res
    assert len(res["q"]) == 12
    assert len(res["p"]) == 12
    assert len(res["z_24d"]) == 24

    # All q coordinates must be bounded in [0, 1]
    assert np.all(res["q"] >= 0.0)
    assert np.all(res["q"] <= 1.0)

    # Conjugate momenta must be non-negative
    assert np.all(res["p"] >= 0.0)

    # Verify presence of ISTA solitons and RQA metrics
    assert "ista_solitons" in res
    assert len(res["ista_solitons"]) == 4
    assert "rqa_i_arch" in res
    assert 0.0 <= res["rqa_i_arch"] <= 1.0

    # Verify end-to-end multiresolution extraction
    full_res = extract_multiresolution_24d("白樺の梢が揺れる。湖面には波紋が広がる。風が止む。沈黙が戻る。")
    assert len(full_res["z_24d"]) == 24
    assert "stage1_signals" in full_res
    assert full_res["stage1_signals"].shape[1] == 4


def test_extract_usir_24d_multiresolution_mode():
    """Verify extract_usir_24d supports use_multiresolution=True and preserves default mode."""
    text = "雪が静かに降り積もる。冷たい夜が更けていく。明かりが消える。"

    # Default mode
    res_default = extract_usir_24d(text, use_multiresolution=False)
    assert len(res_default["z_24d"]) == 24

    # Multiresolution mode
    res_multi = extract_usir_24d(text, use_multiresolution=True)
    assert len(res_multi["z_24d"]) == 24
    assert "stage1_signals" in res_multi
    assert "rqa_i_arch" in res_multi


def test_contrastive_fisher_metric_positive_definiteness_and_separability():
    """Verify Contrastive Fisher Distillation G_style is strictly SPD and maximizes Rayleigh quotient."""
    np.random.seed(123)
    # Class 1: Positive cluster (e.g. Critical phase centered at +0.7)
    pos_z = np.random.normal(loc=0.7, scale=0.15, size=(12, 24))
    # Class 2: Negative cluster (e.g. Deadzone cluster centered at 0.2)
    neg_z = np.random.normal(loc=0.2, scale=0.15, size=(12, 24))

    fisher = fit_contrastive_fisher_metric(pos_z, neg_z, lambda_reg=1e-3)

    G_style = fisher["G_style"]
    w_star = fisher["optimal_projection_axis"]
    lambda_max = fisher["lambda_max"]

    assert G_style.shape == (24, 24)
    assert len(w_star) == 24

    # 1. Symmetry
    assert np.allclose(G_style, G_style.T, atol=1e-10)

    # 2. Strict Positive Definiteness: all eigenvalues must be > 0
    eigvals = np.linalg.eigvalsh(G_style)
    assert np.all(eigvals > 0.0), f"Minimum eigenvalue: {np.min(eigvals)}"
    assert np.min(eigvals) > 0.01

    # 3. Unit norm optimal projection axis
    assert pytest.approx(np.linalg.norm(w_star), abs=1e-5) == 1.0

    # 4. Strictly positive Rayleigh quotient eigenvalue
    assert lambda_max > 0.0

    # 5. Fisher discriminant separability
    proj_pos = pos_z @ w_star
    proj_neg = neg_z @ w_star
    assert np.mean(proj_pos) > np.mean(proj_neg)
    # The separation between class means on w* must be significant
    assert (np.mean(proj_pos) - np.mean(proj_neg)) > 0.5


def test_embed_sentence_unit_syntax_frame_and_gpu_hook():
    """Verify _embed_sentence_unit respects SyntaxTreeFrame priority and hidden_states hook."""
    sentence = "冷たい氷の破片が鋭く光る。"

    # 1. Fallback embedding
    e_raw = _embed_sentence_unit(sentence, dim=24)
    assert len(e_raw) == 24
    assert pytest.approx(np.linalg.norm(e_raw), abs=1e-5) == 1.0

    # 2. SyntaxTreeFrame priority
    node = SyntaxNode(node_id=0, surface="氷", lemma="氷", pos="名詞", pos_detail="", hth_primary=0x6c37 << 34 | 0xE)
    frame = SyntaxTreeFrame(
        sentence_id=0,
        text=sentence,
        nodes=[node],
        edges=[],
        root_id=0,
        max_depth=1,
        radial_entropy=0.5,
    )
    e_frame = _embed_sentence_unit(sentence, dim=24, frame=frame)
    assert len(e_frame) == 24
    assert pytest.approx(np.linalg.norm(e_frame), abs=1e-5) == 1.0
    # Sensory dimensions (0..7) should reflect the node
    assert np.any(e_frame[:8] > 0.0)

    # 3. GPU hidden_states average pooling hook
    mock_hidden_states = np.random.uniform(0.1, 1.0, size=(16, 24))
    e_gpu = _embed_sentence_unit(sentence, dim=24, hidden_states=mock_hidden_states)
    assert len(e_gpu) == 24
    assert pytest.approx(np.linalg.norm(e_gpu), abs=1e-5) == 1.0


def test_multiresolution_edge_cases():
    """Verify robustness to empty and degenerate text inputs."""
    # Empty string
    empty_res = extract_multiresolution_24d("")
    assert np.all(empty_res["z_24d"] == 0.0)
    assert len(empty_res["q"]) == 12
    assert len(empty_res["p"]) == 12

    # Empty list and 0-shape arrays to Stage 2
    s2_empty_list = extract_stage2_multiresolution([])
    assert len(s2_empty_list["z_24d"]) == 24
    assert np.all(s2_empty_list["z_24d"] == 0.0)

    s2_zero_arr = extract_stage2_multiresolution(np.zeros((0, 4)))
    assert len(s2_zero_arr["z_24d"]) == 24
    assert np.all(s2_zero_arr["z_24d"] == 0.0)

    # Empty inputs to Contrastive Fisher
    fisher_empty = fit_contrastive_fisher_metric([], [], lambda_reg=1e-3)
    assert fisher_empty["G_style"].shape == (24, 24)
    assert len(fisher_empty["optimal_projection_axis"]) == 24
    assert np.all(fisher_empty["optimal_projection_axis"] == 0.0)
    assert fisher_empty["lambda_max"] == 0.0

    fisher_zeros = fit_contrastive_fisher_metric(np.zeros((0, 24)), np.zeros((0, 24)), lambda_reg=1e-3)
    assert fisher_zeros["G_style"].shape == (24, 24)
    assert len(fisher_zeros["optimal_projection_axis"]) == 24

    # Single word
    single_res = extract_multiresolution_24d("静寂。")
    assert len(single_res["z_24d"]) == 24
    assert np.all(single_res["q"] >= 0.0)

    # Mimetic frame embed_sentence_unit
    assert embed_sentence_unit is _embed_sentence_unit
    mim_node = SyntaxNode(node_id=0, surface="きらきら", lemma="きらきら", pos="副詞", pos_detail="")
    mim_frame = SyntaxTreeFrame(sentence_id=0, text="きらきら", nodes=[mim_node], edges=[])
    e_mim = embed_sentence_unit("きらきら", frame=mim_frame)
    assert np.any(e_mim[:8] > 0.0)
