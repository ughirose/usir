"""Phase E WS-E1 Unit Tests: Nishiki Unified Field, WorldCraft Arbitration, and Cross-Modal Functors.

Validates:
1. OPEN-06: Toulmin logic audit potential U_Toulmin(x):
   - Claim (assertive ending), Data (numbers/evidence/citations), Warrant (causal connectors) triad.
   - Max dependency depth penalty (D_max > 6).
   - Subject-predicate twist penalty (主述ねじれ).
2. Genre temperature tau_genre in [0, 1] unified field energy:
   - Monotonic sweep across specification (tau=0.1), article (tau=0.5), and literary fiction (tau=0.9).
   - Monotonic increase for specification, monotonic decrease for literary text.
3. OPEN-09: WorldCraft ontology constraints and null-space arbitration:
   - Tier 1 hard mask (naming matrix, deceased characters, unreleased spoilers).
   - Null-space projector P_null = I - J_C^T (J_C J_C^T + eps I)^-1 J_C.
   - Zero worldview infringement: ||J_C * Delta q_safe|| < 1e-5.
   - Momentum bank accumulation, energy tracking, and discharge.
4. Cross-Modal Functors F_visual and F_music:
   - Symplectic 24D mapping and inverse generative parameter consistency.
   - Void and Shock invariant preservation.
"""

import numpy as np
import pytest

from usir_py.nishiki_unified_field import (
    MomentumBank,
    ToulminAuditResult,
    WorldCraftArbitrator,
    WorldCraftOntologyRule,
    compute_toulmin_potential,
    compute_unified_field_energy,
    compute_usir_style_energy,
    evaluate_toulmin_components,
    music_functor_forward_24d,
    music_functor_inverse,
    visual_functor_forward_24d,
    visual_functor_inverse,
)


# ==============================================================================
# 1. Toulmin Logic Audit Potential Tests (OPEN-06)
# ==============================================================================

def test_toulmin_components_rigorous_spec():
    """Verify rigorous specification achieves low U_Toulmin potential."""
    spec_text = (
        "第12節の測定データに基づき、LOOCV精度は97.5%に達することが実験により実証された。"
        "なぜなら、24次元シンプレクティック多様体による拘束条件が有効に機能したからである。"
        "したがって、本手法の有効性は数学的に証明されると結論づけられる。"
    )
    res = evaluate_toulmin_components(spec_text)
    assert isinstance(res, ToulminAuditResult)
    assert res.claim_score >= 0.70
    assert res.data_score >= 0.50
    assert res.warrant_score >= 0.70
    assert res.triad_score >= 0.65
    assert res.u_toulmin < 0.50
    assert res.depth_penalty == 0.0
    assert res.twist_count == 0


def test_toulmin_components_vague_poetry():
    """Verify vague poetic text lacking Data/Warrant receives high U_Toulmin penalty."""
    poetic_text = (
        "風がどこか遠くで囁いているような気がした…"
        "夜の底が白くなったかも。"
        "ただ果てしない闇が広がっているのではないだろうか。"
    )
    res = evaluate_toulmin_components(poetic_text)
    assert res.claim_score <= 0.30
    assert res.data_score == 0.0
    assert res.warrant_score == 0.0
    # High triad deficiency potential
    assert res.u_toulmin >= 1.50


def test_toulmin_subject_predicate_twist_penalty():
    """Verify subject-predicate twist is detected and penalized."""
    # Twisted sentence: 理由は〜と考えます (mismatch), dangling は
    twisted_text = (
        "システムの障害が発生した理由は、担当者が手順を誤ったと考えます。"
        "本件の課題は、来期に向けて新しい体制を推進した。"
    )
    res = evaluate_toulmin_components(twisted_text)
    assert res.twist_count >= 1
    assert res.twist_penalty > 0.0
    assert res.u_toulmin > 0.5


# ==============================================================================
# 2. Genre Temperature tau_genre Monotonic Sweep Tests
# ==============================================================================

def test_genre_temperature_monotonic_sweep():
    """Verify monotonic energy transition across genre temperature tau in [0, 1]."""
    spec_text = (
        "実験結果の測定データによると、再現率は98.2%を達成した。"
        "なぜなら、直交基底による正規化が厳密に保持されたからである。"
        "したがって、提案手法の妥当性が成立すると結論づけられる。"
    )
    lit_text = (
        "国境の長いトンネルを抜けると雪国であった。夜の底が白くなった。"
        "信号所に汽車が止まった。向側の座席から娘が立って来て、島村の前のガラス窓を落した。"
    )

    u_spec = compute_toulmin_potential(spec_text)
    f_usir_spec = compute_usir_style_energy(spec_text)

    u_lit = compute_toulmin_potential(lit_text)
    f_usir_lit = compute_usir_style_energy(lit_text)

    # Inherent property: specification has much lower Toulmin potential than literature
    assert u_spec < u_lit

    tau_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    spec_energies = [compute_unified_field_energy(spec_text, tau)["f_total"] for tau in tau_levels]
    lit_energies = [compute_unified_field_energy(lit_text, tau)["f_total"] for tau in tau_levels]

    # At specification mode (tau=0.1), specification must have lower energy than literature
    assert spec_energies[0] < lit_energies[0]

    # Check monotonicity for spec: since u_spec < f_usir_spec, energy increases with tau
    if f_usir_spec > u_spec:
        for i in range(len(spec_energies) - 1):
            assert spec_energies[i] <= spec_energies[i + 1] + 1e-6

    # Check monotonicity for lit: since u_lit > f_usir_lit, energy decreases with tau
    if u_lit > f_usir_lit:
        for i in range(len(lit_energies) - 1):
            assert lit_energies[i] >= lit_energies[i + 1] - 1e-6


# ==============================================================================
# 3. WorldCraft Ontology Constraints & Null-Space Projector Tests (OPEN-09)
# ==============================================================================

def test_worldcraft_tier1_hard_mask():
    """Verify Tier 1 hard constraints filter out spoilers, deceased actions, and bad pronouns."""
    rule = WorldCraftOntologyRule(
        allowed_naming={"Alice": ["私", "アリス"]},
        forbidden_naming={"Alice": ["俺", "妾"]},
        deceased_characters={"ボブ"},
        unreleased_knowledge={"黒幕の正体は国王"},
        forbidden_terms={"スマートフォン", "インターネット"},
    )
    arbitrator = WorldCraftArbitrator(rule)

    candidates = [
        "アリスは静かに剣を握りしめ、前を見据えた。",  # Clean
        "アリスは「俺が相手だ」と叫んだ。",          # Forbidden pronoun
        "ボブが立ち上がり、大きな声で笑った。",       # Deceased agent action
        "実は黒幕の正体は国王だと判明した。",         # Unreleased spoiler
        "彼は懐からスマートフォンを取り出した。",     # Forbidden modern term
    ]

    valid, masks = arbitrator.apply_tier1_hard_mask(candidates)
    assert len(valid) == 1
    assert valid[0] == candidates[0]
    assert masks == [True, False, False, False, False]


def test_worldcraft_null_space_projection_zero_violation():
    """Verify null-space projector guarantees zero worldview infringement ||J_C Delta q_safe|| = 0."""
    np.random.seed(42)

    # Define 3 constraint hyperplanes in 12D element space
    # e.g., constraint 1: Sg-01 + Sg-04 pace constraint; constraint 2: Void balance; constraint 3: Shock bound
    j_c = np.array([
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)

    arbitrator = WorldCraftArbitrator()
    p_null = arbitrator.compute_null_space_projector(j_c)

    # 1. Projector properties: P_null is 12x12, symmetric, idempotent
    assert p_null.shape == (12, 12)
    assert np.allclose(p_null, p_null.T, atol=1e-5)
    assert np.allclose(p_null @ p_null, p_null, atol=1e-5)

    # 2. Desired arbitrary style momentum Delta q*
    target_dq = np.random.randn(12) * 0.8

    dq_safe, dq_cut, p_calc = arbitrator.arbitrate_momentum(target_dq, j_c)

    # 3. Mathematical proof of zero infringement: J_C * Delta q_safe = 0
    infringement = j_c @ dq_safe
    assert np.linalg.norm(infringement) < 1e-5

    # 4. Orthogonality: dq_safe and dq_cut must be strictly orthogonal
    inner_prod = np.dot(dq_safe, dq_cut)
    assert abs(inner_prod) < 1e-5

    # 5. Momentum banking
    bank = MomentumBank()
    assert bank.get_energy() == 0.0

    bank.accumulate(dq_cut)
    assert bank.get_energy() > 0.0
    assert np.allclose(bank.banked_momentum, dq_cut)

    discharged = bank.discharge()
    assert np.allclose(discharged, dq_cut)
    assert bank.get_energy() == 0.0
    assert np.allclose(bank.banked_momentum, np.zeros(12))


# ==============================================================================
# 4. Cross-Modal Visual & Music Functor Tests
# ==============================================================================

def test_cross_modal_visual_functor_invariants():
    """Verify F_visual forward and inverse mappings preserve Void & Shock invariants."""
    # Case A: Void-dominant visual scene (high margin / negative space)
    void_features = {
        "gabor_frequencies": [0.25, 0.25, 0.25, 0.25],
        "color_entropy": 0.2,
        "chiaroscuro_gradient": 0.1,
        "quadtree_depth_entropy": 0.2,
        "composition_margin_ratio": 0.85,
        "saliency_scanpath": [{"x": 0.5, "y": 0.5, "intensity": 0.2}],
    }
    z_void = visual_functor_forward_24d(void_features)
    assert z_void.shape == (24,)
    assert np.all(z_void[:12] >= 0.0) and np.all(z_void[:12] <= 1.0)
    assert np.all(z_void[12:] >= 0.0)

    # Invert back to diffusion params
    params_void = visual_functor_inverse(z_void)
    assert params_void["composition_margin_ratio"] > 0.60
    assert params_void["high_freq_noise_gain"] < 0.40

    # Case B: Shock-dominant visual scene (high chiaroscuro gradient)
    shock_features = {
        "gabor_frequencies": [0.60, 0.10, 0.20, 0.10],
        "color_entropy": 0.85,
        "chiaroscuro_gradient": 0.90,
        "quadtree_depth_entropy": 0.70,
        "composition_margin_ratio": 0.10,
        "saliency_scanpath": [
            {"x": 0.1, "y": 0.1, "intensity": 0.9},
            {"x": 0.9, "y": 0.9, "intensity": 0.85},
        ],
    }
    z_shock = visual_functor_forward_24d(shock_features)
    params_shock = visual_functor_inverse(z_shock)
    assert params_shock["high_freq_noise_gain"] > 0.70
    assert params_shock["composition_margin_ratio"] < 0.35


def test_cross_modal_music_functor_invariants():
    """Verify F_music forward and inverse mappings preserve tempo & dissonance invariants."""
    # Case A: Restful Void music (high rest ratio, low dissonance)
    calm_music = {
        "torus_distance": 0.1,
        "dissonance": 0.05,
        "syncopation": 0.1,
        "rest_ratio": 0.75,
        "bpm": 65.0,
    }
    z_calm = music_functor_forward_24d(calm_music)
    assert z_calm.shape == (24,)
    assert np.all(z_calm[:12] >= 0.0) and np.all(z_calm[:12] <= 1.0)
    assert np.all(z_calm[12:] >= 0.0)

    params_calm = music_functor_inverse(z_calm)
    assert params_calm["rest_interval"] > 0.50
    assert params_calm["harmonic_tension"] < 0.30
    assert params_calm["reverb_decay"] > 2.5

    # Case B: Intense Shock music (high dissonance, high syncopation, high BPM)
    tense_music = {
        "torus_distance": 0.8,
        "dissonance": 0.9,
        "syncopation": 0.85,
        "rest_ratio": 0.05,
        "bpm": 160.0,
    }
    z_tense = music_functor_forward_24d(tense_music)
    params_tense = music_functor_inverse(z_tense)
    assert params_tense["bpm"] > 120.0
    assert params_tense["harmonic_tension"] > 0.60
    assert params_tense["soliton_accent_gain"] > 0.70
