"""Unit & Integration Tests for 3-Mode HSMM Multi-Well Director & Longform Generation.

Phase B WS-B2 Verifications:
1. 3-Mode HSMM transition matrix stochasticity, row normalization, and refractory period guard.
2. OPEN-09 Momentum Banking unreleased accumulation and Climax discharge burst.
3. OPEN-10 Wundt-Berlyne inverted-U utility and consecutive Climax prohibition.
4. 3,000-char longform generation simulation maintaining CV_L in [0.40, 0.70] without Deadzone (< 0.28).
"""

from __future__ import annotations

import numpy as np
import pytest

from usir_py.hsmm_longform_director import (
    HSMMLongformDirector,
    PhaseWell,
    create_default_wells,
    compute_cv_l,
    compute_wundt_berlyne_utility,
    split_sentences_japanese,
    DEFAULT_TRANSITION_MATRIX,
)
from usir_py.realtime_decoder import ClauseTwistedSMCDecoder, AdaptiveBAOABIntegrator


def test_hsmm_wells_and_transition_matrix_normalization():
    """Verify 3 wells specification and Markov transition matrix normalization."""
    wells = create_default_wells(dim=12)
    assert len(wells) == 3

    # Well 0: Narrative (Crystal-Void)
    assert wells[0].name == "Narrative"
    assert wells[0].target_cv_l == pytest.approx(0.48)
    assert wells[0].target_phi_void == pytest.approx(0.65)
    assert wells[0].target_kappa_kin == pytest.approx(0.20)

    # Well 1: Dialogue (Polyphony)
    assert wells[1].name == "Dialogue"
    assert wells[1].target_cv_l == pytest.approx(0.55)
    assert wells[1].delta_voice >= 0.25

    # Well 2: Climax (Critical-Shock)
    assert wells[2].name == "Climax"
    assert wells[2].target_cv_l == pytest.approx(0.65)
    assert wells[2].target_kappa_kin == pytest.approx(0.70)

    # Stochastic matrix: all rows sum to 1.0, non-negative
    director = HSMMLongformDirector(dim=12)
    T = director.transition_matrix
    assert T.shape == (3, 3)
    assert np.all(T >= 0.0)
    for i in range(3):
        assert np.sum(T[i]) == pytest.approx(1.0)

    # OPEN-10: Climax row strictly forbids staying in Climax
    assert T[2, 2] == 0.0
    assert T[2, 0] == 1.0  # Transition to Well 0 Narrative / Void (残心)


def test_refractory_period_guard_prevents_consecutive_climaxes():
    """Verify OPEN-10 Refractory Period Gate blocks back-to-back Climax states and tracks lifecycle."""
    director = HSMMLongformDirector(dim=12, seed=42)

    # 1. Immediate exit from Climax (2) always forces Well 0 and activates refractory
    assert not director.is_in_refractory()
    next_well = director.step_transition(2)
    assert next_well == 0, f"Expected Well 0 after Climax, got {next_well}"
    assert director.is_in_refractory()
    # Next transition decrements cooldown and clears refractory state
    after_refractory = director.step_transition(0)
    assert not director.is_in_refractory()

    # 2. Even with adversarial transition matrix with T[2, 2] = 1.0, gate forces Well 0
    adversarial_T = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],  # Attempts to stay in Climax forever
    ])
    adv_director = HSMMLongformDirector(dim=12, transition_matrix=adversarial_T, refractory_cooldown=2, seed=42)
    w1 = adv_director.step_transition(2)
    assert w1 == 0  # Forced Well 0
    assert adv_director.is_in_refractory()
    # Step 2: cooldown=2 still blocks Climax even though T[0, 2]=1.0
    w2 = adv_director.step_transition(w1)
    assert w2 != 2  # Climax strictly blocked
    assert adv_director.is_in_refractory()

    # 3. Across a 200-step trajectory, consecutive Climaxes (2 -> 2) must NEVER occur
    cur_well = 0
    trajectory = [cur_well]
    for _ in range(200):
        next_well = director.step_transition(cur_well)
        trajectory.append(next_well)
        cur_well = next_well

    for idx in range(len(trajectory) - 1):
        if trajectory[idx] == 2:
            assert trajectory[idx + 1] != 2, f"Consecutive Climax violation at step {idx}"


def test_momentum_banking_accumulation_and_discharge():
    """Verify OPEN-09 unreleased momentum banking in restrained wells and climax discharge."""
    director = HSMMLongformDirector(dim=12, seed=123)
    assert director.get_banked_energy() == 0.0

    target_p = np.full(12, 0.40)
    # Measured momentum is restrained / suppressed (0.15 < 0.40)
    measured_p = np.full(12, 0.15)

    delta1 = director.accumulate_unreleased_momentum(target_p, measured_p)
    assert np.allclose(delta1, 0.25)
    assert np.allclose(director.p_banked, 0.25)
    assert director.get_banked_energy() > 0.0

    # Accumulate second step
    delta2 = director.accumulate_unreleased_momentum(target_p, measured_p)
    assert np.allclose(delta2, 0.25)
    assert np.allclose(director.p_banked, 0.50)

    # When measured_p exceeds target_p, max(0, target - measured) does not subtract
    fast_measured_p = np.full(12, 0.80)
    delta3 = director.accumulate_unreleased_momentum(target_p, fast_measured_p)
    assert np.allclose(delta3, 0.0)
    assert np.allclose(director.p_banked, 0.50)

    # Climax discharge burst
    climax_base_p = np.full(12, 0.70)
    boosted_p, discharged_p = director.discharge_momentum(climax_base_p, apply_climax_base=False)   # previous behaviour

    assert np.allclose(discharged_p, 0.50)
    assert np.allclose(boosted_p, 1.20)  # 0.70 + 0.50 banked momentum!
    # Bank is completely discharged
    assert np.allclose(director.p_banked, 0.0)
    assert director.get_banked_energy() == 0.0


def test_wundt_berlyne_utility_curve():
    """Verify Wundt-Berlyne inverted-U utility dynamics."""
    a_star = 0.60
    u_peak = compute_wundt_berlyne_utility(a_star, a_star=a_star)
    assert u_peak == pytest.approx(1.0, rel=1e-3)

    # Under-arousal (deadzone) utility is low
    u_deadzone = compute_wundt_berlyne_utility(0.15, a_star=a_star)
    assert u_deadzone < 0.60

    # Over-arousal (purple prose / shock fatigue) utility decays
    u_purple = compute_wundt_berlyne_utility(1.80, a_star=a_star)
    assert u_purple < 0.60
    assert u_purple < u_peak


def test_cv_l_computation_and_edge_cases():
    """Verify sentence length coefficient of variation computation."""
    # Monotonous lengths -> CV_L = 0.0
    monotonous = [20, 20, 20, 20]
    assert compute_cv_l(monotonous) == pytest.approx(0.0)

    # Varied lengths
    varied = [10, 40, 15, 55]
    cv = compute_cv_l(varied)
    assert 0.50 <= cv <= 0.80

    # Edge cases
    assert compute_cv_l([]) == 0.0
    assert compute_cv_l([25]) == 0.0
    assert compute_cv_l("") == 0.0

    # Japanese text
    sample_text = "静寂が訪れた。風は止み、空には無数の星が瞬いていた。誰もいない。"
    cv_text = compute_cv_l(sample_text)
    assert cv_text > 0.0


def test_longform_generation_3000_chars_and_deadzone_defense():
    """Verify 3,000-char generation maintains CV_L in [0.40, 0.70] and avoids Deadzone (< 0.28)."""
    director = HSMMLongformDirector(dim=12, seed=999)

    res = director.generate_longform(
        prompt="冬枯れの干拓地と渡り鳥の影",
        target_paragraphs=7,
        target_total_chars=3000,
    )

    # 1. Structural guarantees
    assert res["paragraph_count"] == 7
    assert len(res["paragraphs"]) == 7
    assert res["total_chars"] >= 2800, f"Total characters {res['total_chars']} < 2800"

    # 2. Deadzone defense
    overall_cv_l = res["overall_cv_l"]
    assert not res["in_deadzone"], f"CV_L {overall_cv_l} sank into Deadzone (< 0.28)"
    assert res["is_healthy"], f"CV_L {overall_cv_l} outside healthy band [0.40, 0.70]"
    assert 0.40 <= overall_cv_l <= 0.70

    # 3. Refractory period invariant check
    wells = res["wells"]
    for i in range(len(wells) - 1):
        if wells[i] == 2:
            assert wells[i + 1] == 0, f"Climax at para {i} not followed by Well 0 Narrative"

    # 4. Banked history validation
    assert len(res["banked_history"]) == 7
    for energy in res["banked_history"]:
        assert energy >= 0.0


def test_longform_generation_with_clause_twisted_smc_integration():
    """Verify HSMMLongformDirector interoperates with ClauseTwistedSMCDecoder."""
    integrator = AdaptiveBAOABIntegrator(dim=12, seed=42)
    decoder = ClauseTwistedSMCDecoder(integrator=integrator, seed=42)
    director = HSMMLongformDirector(dim=12, seed=42)

    res = director.generate_longform(
        prompt="夜明け前の港町",
        target_paragraphs=6,
        target_total_chars=1200,
        decoder=decoder,
    )

    assert res["paragraph_count"] == 6
    assert len(res["paragraphs"]) == 6
    assert len(res["full_text"]) > 0
    assert res["total_chars"] >= 1000
    assert not res["in_deadzone"]
    assert res["is_healthy"], f"SMC CV_L {res['overall_cv_l']} outside healthy band [0.40, 0.70]"


def test_paragraph_text_diversity_across_same_well():
    """Verify that multiple paragraphs of the same well do not duplicate text verbatim."""
    director = HSMMLongformDirector(dim=12, seed=42)
    res = director.generate_longform(
        prompt="秋の夕暮れの枯野",
        target_paragraphs=8,
        target_total_chars=3200,
    )

    # Group texts by well
    texts_by_well: dict[int, list[str]] = {}
    for p in res["paragraphs"]:
        w = p["well_index"]
        texts_by_well.setdefault(w, []).append(p["text"])

    # For wells that appear multiple times, pairwise texts must differ
    for w, t_list in texts_by_well.items():
        if len(t_list) > 1:
            for i in range(len(t_list)):
                for j in range(i + 1, len(t_list)):
                    assert t_list[i] != t_list[j], f"Well {w} produced identical text at paragraphs {i} and {j}"


# ==============================================================================
# Production runner wiring (run_v4_bidirectional_experiment.run_longform_director)
# ==============================================================================

def test_run_v4_longform_director_wiring_with_worldcraft_arbitration():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "colab" / "run_v4_bidirectional_experiment.py"
    spec = importlib.util.spec_from_file_location("run_v4_for_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    res = mod.run_longform_director(
        "静謐な山寺の夜明け",
        seed=3,
        target_paragraphs=6,
        scene_constraints={0: ["stillness"], 1: ["evidence"]},
    )
    assert res["generator"] == "procedural_template"  # no model: must not be labelled as LLM text
    assert res["paragraph_count"] == 6
    assert res["constraint_modes"][0] == ["stillness"]
    assert res["constraint_modes"][1] == ["evidence"]
    assert res["total_cut_momentum_norm"] > 0.0
    assert res["qda"]["phase_regime"] in {"Phase_I_Deadzone", "Phase_II_Critical", "Phase_III_Chaos"}
    assert len(res["full_text"]) > 1000

    plain = mod.run_longform_director("静謐な山寺の夜明け", seed=3, target_paragraphs=6, use_arbitrator=False)
    assert plain["total_cut_momentum_norm"] == 0.0


def test_climax_only_steering_schedule_and_paragraph_hook():
    from usir_py.realtime_decoder import ClauseTwistedSMCDecoder, AdaptiveBAOABIntegrator

    forced = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    dec = ClauseTwistedSMCDecoder(integrator=AdaptiveBAOABIntegrator(dim=12, seed=0), seed=0, auto_gain=True)
    seen, hooked = [], []
    director = HSMMLongformDirector(transition_matrix=forced, seed=0)
    orig_gen = dec.generate_with_clause_twisted_smc

    def spy(*a, **k):
        seen.append(dec.steering_active)
        return orig_gen(*a, max_clauses=3, **{x: v for x, v in k.items() if x != "max_clauses"})

    dec.generate_with_clause_twisted_smc = spy
    res = director.generate_longform(
        target_paragraphs=6, decoder=dec, climax_only_steering=True,
        paragraph_hook=lambda k: hooked.append(k), min_clauses_per_paragraph=2,
    )
    wells = res["wells"]
    expected = [bool(w == 2 or (i > 0 and wells[i - 1] == 2)) for i, w in enumerate(wells)]
    assert seen == expected
    assert hooked == list(range(6))


# ---------------------------------------------------------------------------------------------------------------
# Climax base momentum + multi-Climax carry
def _climax_base_vec():
    v = np.zeros(12)
    v[[3, 7, 11]] = 0.25
    v[[0, 4, 8]] = -0.20
    return v


def test_climax_base_momentum_zero_bank():
    director = HSMMLongformDirector(seed=0)
    base_p = np.full(12, 0.5)
    boosted, discharged = director.discharge_momentum(base_p)
    assert np.allclose(discharged, _climax_base_vec())
    assert np.allclose(boosted, base_p + _climax_base_vec())
    assert np.allclose(director.p_banked, 0.0)


def test_climax_base_momentum_with_bank_and_carry():
    director = HSMMLongformDirector(seed=0)
    bank = np.linspace(0.1, 1.2, 12)
    director.p_banked = bank.copy()
    _, d_last = director.discharge_momentum(np.zeros(12), later_climax=False)
    assert np.allclose(d_last, bank + _climax_base_vec())
    assert np.allclose(director.p_banked, 0.0)
    director.p_banked = bank.copy()
    boosted, d_mid = director.discharge_momentum(np.zeros(12), later_climax=True)
    assert np.allclose(d_mid, 0.7 * bank + _climax_base_vec())
    assert np.allclose(boosted, d_mid)
    assert np.allclose(director.p_banked, 0.3 * bank)
    # constructor parameters
    d2 = HSMMLongformDirector(seed=0, climax_base_shock=0.5, climax_base_flux=0.1, climax_release_frac=0.5)
    d2.p_banked = bank.copy()
    _, dd = d2.discharge_momentum(np.zeros(12), later_climax=True)
    exp = 0.5 * bank
    exp[[3, 7, 11]] += 0.5
    exp[[0, 4, 8]] -= 0.1
    assert np.allclose(dd, exp)
    assert np.allclose(d2.p_banked, 0.5 * bank)


def test_apply_climax_base_false_restores_previous_behaviour():
    director = HSMMLongformDirector(seed=0)
    bank = np.linspace(0.1, 1.2, 12)
    director.p_banked = bank.copy()
    boosted, discharged = director.discharge_momentum(np.full(12, 0.5), later_climax=True, apply_climax_base=False)
    assert np.allclose(discharged, bank) and np.allclose(boosted, 0.5 + bank)
    assert np.allclose(director.p_banked, 0.0)


def test_plan_wells_has_no_side_effects_and_matches_generate_longform():
    tm = [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    director = HSMMLongformDirector(seed=7, transition_matrix=tm)
    state = director.rng.bit_generator.state
    planned = director.plan_wells(6)
    assert director.rng.bit_generator.state == state and director.refractory_counter == 0
    assert planned == [0, 2, 0, 0, 2, 0]                       # refractory cooldown after a Climax
    res = director.generate_longform(target_paragraphs=6, climax_only_steering=True)
    assert res["wells"] == planned
    dn = [p["discharged_momentum_norm"] for p in res["paragraphs"]]
    assert all(dn[i] > 0 for i in (1, 4))                    # Climax base is released at every Climax
    flux = [res["paragraphs"][i]["discharged_momentum"][0] for i in (1, 4)]
    assert all(f >= -0.20 - 1e-9 for f in flux)              # Flux base -0.20 plus released bank (>= 0)


def test_generate_longform_without_steering_keeps_old_discharge():
    tm = [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    director = HSMMLongformDirector(seed=7, transition_matrix=tm)
    res = director.generate_longform(target_paragraphs=6, climax_only_steering=False)
    for p in res["paragraphs"]:
        assert min(p["discharged_momentum"]) >= 0.0          # banked momentum only (no negative Flux base)
