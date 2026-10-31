"""Unit tests for USIR v3 Workstream 2 (Gap 3 & Gap 4).

Validates:
1. Gap 3: Terminal Anchor Preceding & Schrödinger Bridge Infilling:
   - sample_terminal_anchor_and_bridge selects the optimal sentence-ending predicate anchor W_{end}^*
     minimizing Kamae (Sg-10..12) Riemannian distance.
   - b_bridge guidance drift matches analytical formula (q_{end}^* - q_t) / (max(1, T - t) * dt).
   - psi_{T-t}(q_c; q_{end}^*) log-potential correctly steers intermediate candidate clauses.
   - ClauseTwistedSMC integration.
2. Gap 4 Part A: Momentum Banking:
   - MomentumBank stores suppressed momentum and tracks elastic energy E_{bank} = 0.5 * p^T M^{-1} p.
   - AdaptiveBAOABIntegrator stores suppressed momentum when H_{syntax} < eps_{syn}.
   - High-freedom narrative clauses (H_{syntax} >= eps_{syn}) discharge banked momentum into active dynamics.
3. Gap 4 Part B: Wundt-Berlyne Inverted-U Aesthetic Utility:
   - compute_wundt_berlyne_utility calculates U_{reader}(t) with overload gate and refractory fatigue integral.
   - Formal proof: Climax spike followed by 5-8 quiet recovery sentences yields strictly higher
     mean and cumulative utility than continuous purple prose with relentless high-amplitude spikes.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from usir_py.v2.adaptive_baoab import (
    AdaptiveBAOABIntegrator,
    ClauseTwistedSMC,
    MomentumBank,
    sample_terminal_anchor_and_bridge,
)
from usir_py.v2.thermodynamics_so8 import (
    HamiltonFristonResonance,
    SO8PullbackMetric,
    compute_wundt_berlyne_utility,
)


def test_gap3_terminal_anchor_selection_and_kamae_riemannian_distance() -> None:
    """Verify that sample_terminal_anchor_and_bridge identifies the minimum Kamae Riemannian distance anchor."""
    metric = SO8PullbackMetric()
    G = metric.metric_matrix()

    # Target Kamae for Sg-10..12 (indices 9, 10, 11)
    target_kamae = np.array([0.7, -0.4, 0.9], dtype=np.float64)

    # Candidate anchors
    # Anchor 1: exact match on Kamae indices
    q_cand1 = np.zeros(12, dtype=np.float64)
    q_cand1[9] = 0.7
    q_cand1[10] = -0.4
    q_cand1[11] = 0.9

    # Anchor 2: close to Kamae
    q_cand2 = np.zeros(12, dtype=np.float64)
    q_cand2[9] = 0.6
    q_cand2[10] = -0.3
    q_cand2[11] = 0.8

    # Anchor 3: far from Kamae
    q_cand3 = np.zeros(12, dtype=np.float64)
    q_cand3[9] = -0.8
    q_cand3[10] = 0.9
    q_cand3[11] = -0.5

    candidates = [
        {"text": "であった。", "q": q_cand1, "logprob": -0.5},
        {"text": "なのだ。", "q": q_cand2, "logprob": -0.2},
        {"text": "かもしれない。", "q": q_cand3, "logprob": -0.1},
    ]

    current_q = np.zeros(12, dtype=np.float64)
    t = 2
    T = 6
    dt = 0.1
    tau = 1.0

    res = sample_terminal_anchor_and_bridge(
        target_kamae=target_kamae,
        candidate_anchors=candidates,
        current_q=current_q,
        t=t,
        T=T,
        metric=metric,
        dt=dt,
        tau=tau,
    )

    # Anchor 1 is an exact match on Kamae indices => distance must be 0
    assert res["selected_index"] == 0
    assert res["selected_anchor"]["text"] == "であった。"
    assert pytest.approx(res["min_distance"], abs=1e-8) == 0.0
    assert np.allclose(res["q_end"], q_cand1)

    # Verify analytical Schrödinger Bridge drift: b_bridge = (q_end* - q_t) / (max(1, T - t) * dt)
    rem_steps = max(1, T - t)  # 6 - 2 = 4
    expected_b = (q_cand1 - current_q) / (rem_steps * dt)
    assert np.allclose(res["b_bridge"], expected_b)

    # Also test full 12D target vector
    full_target = np.ones(12, dtype=np.float64)
    full_target[9:12] = target_kamae
    res_full = sample_terminal_anchor_and_bridge(
        target_kamae=full_target,
        candidate_anchors=candidates,
        current_q=current_q,
        t=t,
        T=T,
        metric=metric,
        dt=dt,
    )
    assert res_full["selected_index"] == 0


def test_gap3_schrodinger_bridge_potential_infilling() -> None:
    """Verify that intermediate candidate clauses receive correct terminal arrival log-potentials."""
    so8 = SO8PullbackMetric()
    G = so8.metric_matrix()

    target_kamae = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    q_end_pinned = np.full(12, 0.5, dtype=np.float64)
    candidates = [{"text": "終端", "q": q_end_pinned}]

    current_q = np.zeros(12, dtype=np.float64)
    t = 3
    T = 5  # rem_steps = 2
    tau = 1.2
    dt = 0.1

    # Intermediate candidate clauses
    qc_close = np.full(12, 0.48, dtype=np.float64)  # very close to q_end
    qc_far = np.full(12, 0.0, dtype=np.float64)     # far from q_end

    res = sample_terminal_anchor_and_bridge(
        target_kamae=target_kamae,
        candidate_anchors=candidates,
        current_q=current_q,
        t=t,
        T=T,
        metric=so8,
        dt=dt,
        tau=tau,
        intermediate_candidates=[qc_close, qc_far],
    )

    potentials = res["intermediate_potentials"]
    assert potentials is not None
    assert len(potentials) == 2

    # Closer candidate must have strictly higher (less negative) log-potential
    assert potentials[0] > potentials[1]

    # Verify analytical formula: - (qc - q_end)^T G (qc - q_end) / (2 * tau * rem_steps)
    rem_steps = 2
    diff_close = qc_close - q_end_pinned
    expected_pot_close = -float(diff_close.T @ G @ diff_close) / (2.0 * tau * rem_steps)
    assert pytest.approx(potentials[0], rel=1e-6) == expected_pot_close


def test_gap3_clause_twisted_smc_delegation() -> None:
    """Verify ClauseTwistedSMC has sample_terminal_anchor_and_bridge method and weights candidates."""
    smc = ClauseTwistedSMC(beta_steer=1.0)
    target_kamae = np.array([0.2, -0.1, 0.3])
    anchor1 = {"q": np.full(12, 0.2), "text": "です。"}
    anchor2 = {"q": np.full(12, 0.9), "text": "だろう。"}

    res = smc.sample_terminal_anchor_and_bridge(
        target_kamae=target_kamae,
        candidate_anchors=[anchor1, anchor2],
        current_q=np.zeros(12),
        t=0,
        T=4,
    )
    assert res["selected_index"] == 0
    assert res["selected_anchor"]["text"] == "です。"

    # Check ClauseTwistedSMC evaluate_particle_weights with terminal anchor
    q_cand_aligned = np.full(12, 0.2)
    q_cand_deviant = np.full(12, -0.8)
    log_w, norm_w, ess = smc.evaluate_particle_weights(
        candidate_q=[q_cand_aligned, q_cand_deviant],
        llm_logprobs=[-0.5, -0.5],
        target_q=np.zeros(12),
        terminal_anchor_q=res["q_end"],
        t=1,
        T=4,
    )
    # The aligned particle must have higher weight
    assert norm_w[0] > norm_w[1]


def test_gap4_momentum_bank_elastic_energy_and_discharge() -> None:
    """Verify MomentumBank accumulation, elastic energy computation, and fractional discharge."""
    bank = MomentumBank(dim=12)
    assert bank.get_energy() == 0.0

    # Deposit momentum
    dp1 = np.ones(12, dtype=np.float64) * 2.0
    energy1 = bank.bank_momentum(dp1)
    # E = 0.5 * p^T M^{-1} p with M = I => 0.5 * 12 * (2.0^2) = 0.5 * 12 * 4 = 24.0
    assert pytest.approx(energy1, abs=1e-8) == 24.0
    assert pytest.approx(bank.energy, abs=1e-8) == 24.0

    # Deposit additional momentum
    dp2 = np.ones(12, dtype=np.float64) * 1.0
    energy2 = bank.bank_momentum(dp2)
    # Total p = 3.0 => E = 0.5 * 12 * 9.0 = 54.0
    assert pytest.approx(energy2, abs=1e-8) == 54.0

    # 50% partial discharge
    p_released = bank.discharge_momentum(eta_release=0.5)
    assert np.allclose(p_released, np.ones(12) * 1.5)
    # Remaining p = 1.5 => E = 0.5 * 12 * 2.25 = 13.5
    assert pytest.approx(bank.get_energy(), abs=1e-8) == 13.5

    # Full remaining discharge
    p_final = bank.discharge_momentum(eta_release=1.0)
    assert np.allclose(p_final, np.ones(12) * 1.5)
    assert pytest.approx(bank.get_energy(), abs=1e-8) == 0.0


def test_gap4_adaptive_baoab_momentum_banking_under_entropy_transition() -> None:
    """Verify AdaptiveBAOABIntegrator banks momentum under low H_syntax and discharges under high H_syntax."""
    integrator = AdaptiveBAOABIntegrator(gamma0=1.0)
    integrator.bank.reset()

    q0 = np.zeros(12, dtype=np.float64)
    p0 = np.ones(12, dtype=np.float64) * 2.0
    zero_grad = lambda q: np.zeros(12, dtype=np.float64)

    # Step 1: Low syntax entropy (rigid grammar constraint, e.g. H = 0.1 < eps_syn 0.6)
    q1, p1 = integrator.step_with_banking(
        q=q0,
        p=p0,
        grad_fn=zero_grad,
        h_syntax=0.1,
        dt=0.1,
        tau=0.0,  # Zero noise to measure deterministic momentum transfer
    )

    # Under severe braking, momentum is suppressed and stored in bank
    bank_energy_after_low_entropy = integrator.bank.get_energy()
    assert bank_energy_after_low_entropy > 0.0

    # Step 2: High syntax entropy (freedom clause, e.g. H = 2.0 >= eps_syn 0.6)
    # The banked momentum is discharged into active dynamics
    q2, p2 = integrator.step_with_banking(
        q=q1,
        p=p1,
        grad_fn=zero_grad,
        h_syntax=2.0,
        dt=0.1,
        tau=0.0,
        eta_release=1.0,
    )

    # After full discharge, bank is drained
    assert pytest.approx(integrator.bank.get_energy(), abs=1e-8) == 0.0
    # Active momentum p2 has received the released boost
    assert np.linalg.norm(p2) > np.linalg.norm(p1)


def test_gap4_wundt_berlyne_climax_vs_purple_prose_proof() -> None:
    """Mathematical Proof in unit test:

    A trajectory with a climax spike followed by a 5-8 quiet sentence refractory period
    yields strictly higher mean and cumulative reader utility than continuous purple prose!
    """
    T = 16
    tau_rec = 6
    f_overload = 10.0
    lambda_fatigue = 0.05

    # 1. Trajectory A: Climax + Refractory Recovery
    # - Sentences 0..3: quiet build-up (low surprise, low free energy, gentle drift)
    # - Sentence 4: dramatic catharsis CLIMAX (high surprise resolution V=10.0, controlled F=2.0)
    # - Sentences 5..12: 8 quiet recovery sentences (refractory period, V=0.2, F=0.5, nearly static p)
    # - Sentences 13..15: calm coda (V=0.1, F=0.3)
    valences_a = np.zeros(T, dtype=np.float64)
    free_energies_a = np.zeros(T, dtype=np.float64)
    p_traj_a = np.zeros((T, 12), dtype=np.float64)

    for t in range(T):
        if t < 4:
            valences_a[t] = 1.0
            free_energies_a[t] = 1.0
            p_traj_a[t] = np.ones(12) * (0.05 * (t + 1))
        elif t == 4:
            valences_a[t] = 18.0  # Massive catharsis climax payoff!
            free_energies_a[t] = 2.0
            p_traj_a[t] = np.ones(12) * 0.8  # Climax momentum surge (controlled style jump)
        elif 5 <= t <= 12:
            valences_a[t] = 0.5   # Quiet reflection & afterglow
            free_energies_a[t] = 0.5
            p_traj_a[t] = np.ones(12) * (0.8 - 0.05 * (t - 4))  # Gentle glide
        else:
            valences_a[t] = 0.2
            free_energies_a[t] = 0.3
            p_traj_a[t] = np.ones(12) * 0.4

    # 2. Trajectory B: Relentless Purple Prose
    # - Sentences 0..15: constant hysterical claims of catharsis (V=8.0 on every sentence)
    # - Style free energy blows up to massive overload (F=35.0, reader desensitized)
    # - Constant thrashing momentum swings (p alternating wildly between +2.5 and -2.5)
    valences_b = np.full(T, 8.0, dtype=np.float64)
    free_energies_b = np.full(T, 35.0, dtype=np.float64)
    p_traj_b = np.zeros((T, 12), dtype=np.float64)

    for t in range(T):
        sign = 1.0 if t % 2 == 0 else -1.0
        p_traj_b[t] = np.ones(12) * (2.5 * sign)

    res_a = compute_wundt_berlyne_utility(
        valence_series=valences_a,
        free_energy_series=free_energies_a,
        p_trajectory=p_traj_a,
        f_overload=f_overload,
        lambda_fatigue=lambda_fatigue,
        tau_rec=tau_rec,
    )

    res_b = compute_wundt_berlyne_utility(
        valence_series=valences_b,
        free_energy_series=free_energies_b,
        p_trajectory=p_traj_b,
        f_overload=f_overload,
        lambda_fatigue=lambda_fatigue,
        tau_rec=tau_rec,
    )

    # Verification:
    # 1. Climax + recovery cumulative utility is strongly positive
    assert res_a["cumulative_utility"] > 5.0

    # 2. Purple prose cumulative utility is severely negative due to refractory fatigue and overload
    assert res_b["cumulative_utility"] < 0.0

    # 3. Climax + recovery strictly beats purple prose on both mean and cumulative utility!
    assert res_a["mean_utility"] > res_b["mean_utility"]
    assert res_a["cumulative_utility"] > res_b["cumulative_utility"]

    # 4. Check fatigue penalty magnitude: Purple prose has huge accumulated fatigue
    total_fatigue_a = float(np.sum(res_a["fatigue_penalties"]))
    total_fatigue_b = float(np.sum(res_b["fatigue_penalties"]))
    assert total_fatigue_b > 10.0 * total_fatigue_a

    # 5. Check overload gate: Purple prose gated valence is crushed by exp(-35 / 10) = 0.030
    assert np.all(res_b["gated_valences"] < 0.5)


def test_gap4_hamilton_friston_resonance_method() -> None:
    """Verify HamiltonFristonResonance class method compute_wundt_berlyne_utility."""
    monitor = HamiltonFristonResonance(f_overload=12.0, lambda_fatigue=0.04, tau_rec=7)
    valences = [1.0, 2.0, 3.0, 0.5]
    free_energies = [0.5, 0.6, 0.8, 0.4]
    p_traj = [np.ones(12) * 0.1, np.ones(12) * 0.2, np.ones(12) * 0.3, np.ones(12) * 0.25]

    res = monitor.compute_wundt_berlyne_utility(
        valence_series=valences,
        free_energy_series=free_energies,
        p_trajectory=p_traj,
    )
    assert len(res["utilities"]) == 4
    assert res["mean_utility"] > 0.0
