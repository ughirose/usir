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
    # Inherent property: specification has much higher style free energy than literature
    assert f_usir_spec > f_usir_lit
    # Specification: logical potential is lower than style free energy
    assert u_spec < f_usir_spec
    # Literature: logical potential is higher than style free energy
    assert u_lit > f_usir_lit

    tau_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    spec_energies = [compute_unified_field_energy(spec_text, tau)["f_total"] for tau in tau_levels]
    lit_energies = [compute_unified_field_energy(lit_text, tau)["f_total"] for tau in tau_levels]

    # At specification mode (tau=0.1), specification must have lower energy than literature
    assert spec_energies[0] < lit_energies[0]

    # At literary mode (tau=0.9), literature must have lower energy than specification
    assert lit_energies[-1] < spec_energies[-1]

    # Monotonicity for specification: energy increases strictly monotonically with tau
    for i in range(len(spec_energies) - 1):
        assert spec_energies[i] < spec_energies[i + 1]

    # Monotonicity for literature: energy decreases strictly monotonically with tau
    for i in range(len(lit_energies) - 1):
        assert lit_energies[i] > lit_energies[i + 1]


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


def test_toulmin_dependency_depth_penalty():
    """Verify D_max > 6 triggers dependency depth penalty."""
    deep_text = "私の兄の友人の会社の社長の秘書の父親の家の庭の桜の木が倒れた。"
    res = evaluate_toulmin_components(deep_text)
    assert res.max_depth_found > 6
    assert res.depth_penalty > 0.0
    assert res.u_toulmin > res.triad_deficiency


def test_worldcraft_empty_constraints():
    """Verify empty constraint Jacobian yields identity projector without error."""
    arbitrator = WorldCraftArbitrator()
    p_null_none = arbitrator.compute_null_space_projector(None)
    assert np.allclose(p_null_none, np.eye(12))

    p_null_empty = arbitrator.compute_null_space_projector(np.empty((0, 12)))
    assert np.allclose(p_null_empty, np.eye(12))

    target_dq = np.ones(12)
    dq_safe, dq_cut, _ = arbitrator.arbitrate_momentum(target_dq, np.empty((0, 12)))
    assert np.allclose(dq_safe, target_dq)
    assert np.allclose(dq_cut, np.zeros(12))


def test_worldcraft_deceased_passive_voice_not_flagged():
    """Verify passive voice about deceased character is not falsely flagged as live agent action."""
    rule = WorldCraftOntologyRule(deceased_characters={"ボブ"})
    arbitrator = WorldCraftArbitrator(rule)
    text = "ボブは静かに墓地に埋葬された。"
    violations = arbitrator.check_tier1_violations(text)
    assert len(violations) == 0


def test_momentum_bank_dimension_check():
    """Verify MomentumBank validates vector dimension."""
    bank = MomentumBank()
    with pytest.raises(ValueError):
        bank.accumulate(np.ones(5))


def test_music_functor_inverse_sequence():
    """Verify Python music_functor_inverse produces web_audio_sequence directives."""
    z = np.zeros(24)
    z[0] = 0.5
    res = music_functor_inverse(z)
    assert "web_audio_sequence" in res
    assert 1 <= len(res["web_audio_sequence"]) <= 16
    first = res["web_audio_sequence"][0]
    assert "note" in first and "time" in first and "duration" in first and "gain" in first


# ==============================================================================
# Tautology (non-progressing warrant) detection
# ==============================================================================

TAUTOLOGY_TEXT = (
    "データの数値は100%である。"
    "なぜならデータの数値は100%だからである。"
    "したがってデータの数値は100%であると結論づけられる。"
)
PROGRESSING_TEXT = (
    "実測では応答時間が120msであった。"
    "なぜなら通信経路が3段から1段に短縮されたからである。"
    "したがって遅延の主因は経路長であると結論づけられる。"
)


def test_toulmin_tautology_gets_no_warrant_credit():
    res = evaluate_toulmin_components(TAUTOLOGY_TEXT)
    assert res.tautology_count == 2
    assert res.warrant_score == 0.0
    assert res.tautology_penalty > 0.0


def test_toulmin_progressing_argument_is_not_tautology():
    res = evaluate_toulmin_components(PROGRESSING_TEXT)
    assert res.tautology_count == 0
    assert res.warrant_score > 0.0
    assert res.u_toulmin < evaluate_toulmin_components(TAUTOLOGY_TEXT).u_toulmin


# ==============================================================================
# Conjugate momentum from window series and 24D-driven music sequence
# ==============================================================================

def _patch(chiaro, color, quad, gabor):
    return {
        "gabor_frequencies": gabor,
        "color_entropy": color,
        "chiaroscuro_gradient": chiaro,
        "quadtree_depth_entropy": quad,
        "saliency_scanpath": [{"x": 0.1, "y": 0.1, "intensity": 0.5}, {"x": 0.1 + chiaro * 0.5, "y": 0.4, "intensity": 0.5}],
    }


def test_visual_momentum_all_twelve_dims_from_patches():
    patches = [
        _patch(0.1, 0.2, 0.3, [0.1, 0.2, 0.3, 0.4]),
        _patch(0.7, 0.6, 0.8, [0.5, 0.1, 0.2, 0.9]),
        _patch(0.3, 0.9, 0.1, [0.2, 0.8, 0.1, 0.3]),
    ]
    feats = {**patches[0], "patch_features": patches}
    z = visual_functor_forward_24d(feats)
    p = z[12:]
    assert p.shape == (12,)
    assert np.all(p >= 0.0)
    assert np.count_nonzero(p) == 12
    assert not np.allclose(p, z[:12])  # not a copy of static q


def test_music_momentum_all_twelve_dims_from_bars():
    bars = [
        {"torus_distance": 0.1, "dissonance": 0.1, "syncopation": 0.2, "rest_ratio": 0.1, "bpm": 80},
        {"torus_distance": 0.6, "dissonance": 0.5, "syncopation": 0.7, "rest_ratio": 0.4, "bpm": 120},
        {"torus_distance": 0.3, "dissonance": 0.9, "syncopation": 0.1, "rest_ratio": 0.8, "bpm": 100},
    ]
    z = music_functor_forward_24d({**bars[0], "bar_features": bars})
    assert np.count_nonzero(z[12:]) == 12


def test_momentum_is_zero_for_constant_window_series():
    bar = {"torus_distance": 0.3, "dissonance": 0.2, "syncopation": 0.2, "rest_ratio": 0.2, "bpm": 100}
    z = music_functor_forward_24d({**bar, "bar_features": [bar, bar, bar]})
    assert np.allclose(z[12:], 0.0)


def _z_music(rest, diss, synco=0.3, bpm=110.0):
    return music_functor_forward_24d(
        {"torus_distance": 0.4, "dissonance": diss, "syncopation": synco, "rest_ratio": rest, "bpm": bpm}
    )


def test_music_inverse_void_removes_notes():
    dense = music_functor_inverse(_z_music(rest=0.05, diss=0.3))["web_audio_sequence"]
    sparse = music_functor_inverse(_z_music(rest=0.9, diss=0.3))["web_audio_sequence"]
    assert len(sparse) < len(dense)
    assert len(sparse) >= 1


def test_music_inverse_is_not_a_fixed_pentatonic_loop():
    seq = music_functor_inverse(_z_music(rest=0.05, diss=0.5, synco=0.8, bpm=170.0))["web_audio_sequence"]
    notes = [n["note"] for n in seq]
    assert len(set(notes)) > 5
    assert notes != [[60, 62, 64, 67, 69][i % 5] for i in range(len(notes))]


def test_music_inverse_shock_increases_leaps_and_tritone():
    calm = music_functor_inverse(_z_music(rest=0.1, diss=0.05))["web_audio_sequence"]
    tense = music_functor_inverse(_z_music(rest=0.1, diss=0.95))["web_audio_sequence"]

    def mean_leap(seq):
        n = [x["note"] for x in seq]
        return float(np.mean([abs(b - a) for a, b in zip(n, n[1:])]))

    assert mean_leap(tense) > mean_leap(calm)


def test_music_inverse_is_deterministic_and_time_ordered_slots():
    z = _z_music(rest=0.2, diss=0.4)
    a = music_functor_inverse(z)["web_audio_sequence"]
    b = music_functor_inverse(z)["web_audio_sequence"]
    assert a == b
    assert all(n["duration"] > 0.0 and 36 <= n["note"] <= 96 for n in a)


# ==============================================================================
# WorldCraft constraint Jacobian compiler and HSMM integration
# ==============================================================================

def test_compile_constraint_jacobian_modes():
    arb = WorldCraftArbitrator()
    j = arb.compile_constraint_jacobian(active_constraints=["evidence"])
    assert j.shape == (2, 12)
    assert set(np.flatnonzero(j.sum(axis=0))) == {7, 11}

    j_all = arb.compile_constraint_jacobian(active_constraints=["evidence", "logic_voice", "stillness"])
    assert j_all.shape == (7, 12)
    assert np.allclose(j_all @ j_all.T, np.eye(7))

    assert arb.compile_constraint_jacobian("晴れた朝。", None).shape == (0, 12)
    assert arb.compile_constraint_jacobian("検証した記録を報告する。", None).shape == (2, 12)
    with pytest.raises(ValueError):
        arb.compile_constraint_jacobian(active_constraints=["no_such_mode"])


def test_compiled_jacobian_freezes_elements_in_null_space():
    arb = WorldCraftArbitrator()
    j = arb.compile_constraint_jacobian(active_constraints=["stillness"])
    dq = np.linspace(0.2, 1.4, 12)
    safe, cut, _ = arb.arbitrate_momentum(dq, j)
    assert np.allclose(safe[[2, 6, 10]], 0.0, atol=1e-4)
    assert np.allclose(safe[[0, 1, 3]], dq[[0, 1, 3]], atol=1e-4)
    assert np.allclose(safe + cut, dq)


def test_hsmm_director_banks_cut_momentum_and_releases_in_climax():
    from usir_py.hsmm_longform_director import HSMMLongformDirector

    # Paragraph 0 (Narrative) -> forced Climax at paragraph 1.
    forced = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

    base = HSMMLongformDirector(transition_matrix=forced, seed=1).generate_longform(target_paragraphs=6)
    arb = HSMMLongformDirector(
        transition_matrix=forced,
        seed=1,
        arbitrator=WorldCraftArbitrator(),
        scene_constraints={0: ["stillness"]},
    ).generate_longform(target_paragraphs=6)

    assert base["wells"][:2] == [0, 2] and arb["wells"][:2] == [0, 2]
    assert base["paragraphs"][0]["cut_momentum_norm"] == 0.0
    assert arb["paragraphs"][0]["cut_momentum_norm"] > 0.0
    assert arb["paragraphs"][0]["constraint_modes"] == ["stillness"]
    # The cut momentum is released at the Climax: more is discharged than without arbitration.
    assert arb["paragraphs"][1]["was_discharged"]
    assert arb["paragraphs"][1]["discharged_momentum_norm"] > base["paragraphs"][1]["discharged_momentum_norm"]
