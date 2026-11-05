"""Unit & Integration Tests for USIR v4.0 Realtime Decoder & LogitsProcessor.

Verifies:
1. LogitsStepper Top-K constraint, Refractory Period Gate, and Momentum updating.
2. USIRPyTorchLogitsProcessor tensor processing and batch safety.
3. run_v4_bidirectional_experiment pipeline execution across 3 domains.
4. usir_v4_bidirectional_bridge_runner.ipynb JSON syntax and cell integrity.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np
import pytest

from usir_py.realtime_decoder import (
    LogitsStepper,
    USIRPyTorchLogitsProcessor,
    TORCH_AVAILABLE,
    AdaptiveBAOABIntegrator,
    ClauseTwistedSMCDecoder,
    step_baoab_clause,
    generate_1_over_f_noise,
    BAOABStepResult,
)
from colab.run_v4_bidirectional_experiment import (
    detect_execution_tier,
    run_experiment_on_domain,
    EXPERIMENT_DOMAINS,
)

if TORCH_AVAILABLE:
    import torch


# ==============================================================================
# 1. LogitsStepper Tests
# ==============================================================================

def test_logits_stepper_refractory_blocks_punctuation_early():
    """Verify refractory period gate suppresses punctuation during early tokens."""
    stepper = LogitsStepper(top_k=10, refractory_min=5, refractory_max=30)
    stepper.register_punctuation_ids([42])

    logits = np.zeros(100, dtype=float)
    logits[42] = 10.0  # Highly likely punctuation token

    stepper.tokens_since_punctuation = 2  # < refractory_min (5)
    steered, telemetry = stepper.steer_logits(logits)

    assert telemetry["is_refractory_blocked"]
    assert steered[42] < -500.0  # Punished


def test_logits_stepper_refractory_forces_punctuation_late():
    """Verify refractory period gate boosts punctuation when run-on length exceeded."""
    stepper = LogitsStepper(top_k=10, refractory_min=5, refractory_max=30)
    stepper.register_punctuation_ids([42])

    logits = np.zeros(100, dtype=float)
    logits[42] = 2.0

    stepper.tokens_since_punctuation = 35  # > refractory_max (30)
    steered, telemetry = stepper.steer_logits(logits)

    assert telemetry["is_refractory_forced"]
    assert steered[42] == pytest.approx(5.5)  # Boosted by +3.5 (from 2.0)


def test_logits_stepper_clause_step_momentum_reset():
    """Verify that committing a punctuation token resets tokens_since_punctuation."""
    stepper = LogitsStepper(top_k=10, refractory_min=5, refractory_max=30)
    stepper.register_punctuation_ids([42])

    stepper.tokens_since_punctuation = 15
    stepper.step_token_selected(10)  # Regular word
    assert stepper.tokens_since_punctuation == 16

    stepper.step_token_selected(42)  # Punctuation
    assert stepper.tokens_since_punctuation == 0


def test_logits_stepper_backward_bridge_attraction():
    """Verify that backward bridge attraction steers candidate logits along Jacobian direction."""
    attract = np.zeros(12, dtype=float)
    attract[0] = 1.0  # Target direction in dimension 0

    stepper = LogitsStepper(
        top_k=5,
        eps_syn=0.1,
        beta_steer=1.0,
        refractory_min=2,
        backward_attraction=attract,
    )
    stepper.tokens_since_punctuation = 10  # within healthy window

    logits = np.array([2.0, 2.0, 2.0, 2.0, 2.0] + [0.0] * 95, dtype=float)
    # Token 0 aligns with attraction (+1.0), Token 1 opposes attraction (-1.0)
    token_jacobians = {
        0: np.array([1.0] + [0.0] * 11),
        1: np.array([-1.0] + [0.0] * 11),
    }

    steered, telemetry = stepper.steer_logits(logits, token_jacobians=token_jacobians)
    assert not telemetry["bypassed"]
    assert steered[0] > logits[0]
    assert steered[1] < logits[1]


# ==============================================================================
# 2. USIRPyTorchLogitsProcessor Tests
# ==============================================================================

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_pytorch_logits_processor_batch_processing():
    """Verify USIRPyTorchLogitsProcessor correctly processes 2D batch tensors."""
    processor = USIRPyTorchLogitsProcessor(
        dim=12,
        top_k=16,
        eps_syn=0.5,
        beta_steer=1.0,
        refractory_min=3,
        punctuation_token_ids=[99],
    )

    batch_size = 2
    vocab_size = 1000
    scores = torch.randn(batch_size, vocab_size)
    scores[:, 99] = 5.0

    input_ids = torch.tensor([[10, 20, 30], [5, 15, 99]])  # Batch 1 ended in punct

    output = processor(input_ids, scores.clone())
    assert output.shape == (batch_size, vocab_size)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_pytorch_logits_processor_register_bridge_embeddings():
    """Verify register_bridge_embeddings computes v_bridge and steers logits along bridge direction."""
    processor = USIRPyTorchLogitsProcessor(
        top_k=10,
        eps_syn=0.1,
        beta_steer=1.0,
        refractory_min=2,
    )

    class MockEmbedding(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = weight

    class MockModel:
        def __init__(self, weight):
            self.emb = MockEmbedding(weight)

        def get_input_embeddings(self):
            return self.emb

    class MockTokenizer:
        def encode(self, text, add_special_tokens=False):
            if "start" in text:
                return [1]
            return [2]

    vocab_size = 50
    hidden_dim = 16
    weight = torch.zeros(vocab_size, hidden_dim)
    weight[1, 0] = 1.0
    weight[2, 1] = 1.0
    weight[3, 1] = 1.0
    weight[3, 0] = -1.0
    weight[4, 0] = 1.0
    weight[4, 1] = -1.0

    model = MockModel(weight)
    tokenizer = MockTokenizer()
    processor.register_bridge_embeddings(model, tokenizer, x_0="start", x_T="end")

    assert processor.v_bridge is not None
    scores = torch.zeros(1, vocab_size)
    scores[0, 3] = 2.0
    scores[0, 4] = 2.0
    scores[0, 5] = 2.0
    input_ids = torch.tensor([[10, 20]])

    steered = processor(input_ids, scores.clone())
    assert steered[0, 3] > scores[0, 3]
    assert steered[0, 4] < scores[0, 4]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_pytorch_logits_processor_cjk_and_backtick_masking():
    """Verify register_tokenizer_punctuation identifies CJK/backtick tokens and masks them by -100.0."""
    processor = USIRPyTorchLogitsProcessor(top_k=10, eps_syn=0.1, refractory_min=2)

    class MockTokenizer:
        def encode(self, text, add_special_tokens=False):
            mapping = {"。": [10], "是": [20], "`": [21]}
            return mapping.get(text, [])

    processor.register_tokenizer_punctuation(MockTokenizer())
    assert 10 in processor.punctuation_token_ids
    assert 20 in processor.banned_cjk_token_ids
    assert 21 in processor.banned_cjk_token_ids

    scores = torch.zeros(1, 30)
    scores[0, 20] = 5.0
    scores[0, 21] = 4.0
    input_ids = torch.tensor([[1, 2, 3]])
    out = processor(input_ids, scores.clone())
    assert out[0, 20] <= -95.0
    assert out[0, 21] <= -95.0


def test_pytorch_logits_processor_preserves_japanese_joyo_kanji():
    """Verify Japanese Joyo kanji (希望, 著しい, 据える, 与える, 若い) are strictly NOT banned."""
    from usir_py.realtime_decoder import BANNED_CJK_AND_META_CHARS

    joyo_kanji = ["希", "著", "据", "与", "若", "訴", "説", "湿"]
    for j in joyo_kanji:
        assert j not in BANNED_CJK_AND_META_CHARS, f"Fatal regression: Joyo kanji '{j}' must not be banned!"

    # Verify Chinese simplified counterparts are properly targeted
    assert "诉" in BANNED_CJK_AND_META_CHARS
    assert "说" in BANNED_CJK_AND_META_CHARS
    assert "设" in BANNED_CJK_AND_META_CHARS
    assert "闭" in BANNED_CJK_AND_META_CHARS
    assert "为" in BANNED_CJK_AND_META_CHARS
    assert "则" in BANNED_CJK_AND_META_CHARS


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_pytorch_logits_processor_topk_exceeds_vocab_size():
    """Verify USIRPyTorchLogitsProcessor safely clamps top_k when top_k > vocab_size."""
    processor = USIRPyTorchLogitsProcessor(top_k=128, refractory_min=2)
    scores = torch.randn(2, 16)  # vocab_size = 16 < top_k = 128
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    out = processor(input_ids, scores.clone())
    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()


# ==============================================================================
# 3. Experiment Runner & Auto-Detection Tests

# ==============================================================================

def test_detect_execution_tier():
    """Verify that detect_execution_tier returns a valid environment tier dictionary."""
    tier_info = detect_execution_tier()
    assert "tier" in tier_info
    assert "device" in tier_info
    assert tier_info["tier"] in [
        "Tier_1_GPU_Transformers",
        "Tier_2_Local_Ollama",
        "Tier_3_Offline_Lattice_Solver",
    ]


def test_run_experiment_on_all_domains():
    """Verify that run_experiment_on_domain executes cleanly on novel, article, and CoT reasoning."""
    tier_info = detect_execution_tier()
    for domain_cfg in EXPERIMENT_DOMAINS:
        res = run_experiment_on_domain(domain_cfg, tier_info)
        assert res["domain"] == domain_cfg["domain"]
        assert "vanilla_unidirectional" in res["methods"]
        assert "usir_v3_forward_only" in res["methods"]
        assert "usir_v4_bidirectional_collision" in res["methods"]

        v4 = res["methods"]["usir_v4_bidirectional_collision"]
        assert v4["text_length"] > 20
        assert "r_grpo" in v4
        assert "d_3phase" in v4


# ==============================================================================
# 4. Colab Notebook Integrity Tests
# ==============================================================================

def test_colab_notebook_json_and_cell_integrity():
    """Verify that usir_v4_bidirectional_bridge_runner.ipynb is valid JSON and well-formed."""
    nb_path = Path(__file__).resolve().parent.parent / "colab" / "usir_v4_bidirectional_bridge_runner.ipynb"
    assert nb_path.exists(), f"Notebook not found at: {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert nb_data.get("nbformat") == 4
    assert "cells" in nb_data
    cells = nb_data["cells"]
    assert len(cells) >= 6

    # Verify all cells have valid types and non-empty sources
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    markdown_cells = [c for c in cells if c.get("cell_type") == "markdown"]

    assert len(code_cells) >= 5
    assert len(markdown_cells) >= 1

    for c in cells:
        assert len(c.get("source", [])) > 0


# ==============================================================================
# 5. Phase B WS-B1: Adaptive-BAOAB Integrator & Clause Twisted SMC Tests
# ==============================================================================

def test_adaptive_baoab_integrator_thermal_equilibrium():
    """Verify Theorem 5.1: 5-Stage Adaptive-BAOAB thermal equilibrium, ghost history, and syntax brake."""
    dim = 12
    mu_star = np.full(dim, 0.5, dtype=float)
    integrator = AdaptiveBAOABIntegrator(
        dim=dim,
        mu_star=mu_star,
        dt=0.25,
        tau=0.04,
        eps_syn=0.6,
        gamma_low=1.2,
        gamma_high=50.0,
        lambda_bridge=1.0,
        L_ghost=16,
        variance_ghost=0.02,
        seed=123,
    )

    # 1. Ghost history verification (OPEN-08)
    assert len(integrator.ghost_history) == 16
    ghost_arr = np.array(list(integrator.ghost_history))
    assert ghost_arr.shape == (16, 12)
    diff_ghost = ghost_arr - mu_star
    mean_ghost = np.mean(diff_ghost, axis=0)
    assert np.all(np.abs(mean_ghost) < 0.1)  # Zero mean within statistical bounds
    var_ghost = np.var(diff_ghost, axis=0)
    assert np.all(var_ghost > 0.005) and np.all(var_ghost < 0.05)  # Matches variance ~0.02

    # 2. Bayesian shrinkage gain verification (m / (m + 5))
    q_meas = np.ones(dim, dtype=float) * 0.9
    shrunk_m0 = integrator.compute_shrinkage(q_meas, m=0)
    assert np.allclose(shrunk_m0, mu_star)  # alpha_0 = 0.0 -> returns mu_star

    shrunk_m1 = integrator.compute_shrinkage(q_meas, m=1)
    alpha_1 = 1.0 / 6.0
    expected_m1 = alpha_1 * q_meas + (1.0 - alpha_1) * mu_star
    assert np.allclose(shrunk_m1, expected_m1)

    shrunk_m5 = integrator.compute_shrinkage(q_meas, m=5)
    alpha_5 = 5.0 / 10.0  # 0.5
    expected_m5 = 0.5 * q_meas + 0.5 * mu_star
    assert np.allclose(shrunk_m5, expected_m5)

    # 3. Variable friction brake verification (Stage O)
    # Case A: Low syntax entropy H_syntax < eps_syn (0.2 < 0.6) -> Strong brake gamma_t = 50.0
    integrator.p = np.full(dim, 2.0, dtype=float)  # High initial momentum
    step_low = integrator.step_clause(q_measured=q_meas, h_syntax=0.2)
    assert step_low.gamma_t == 50.0
    assert step_low.theta_t == pytest.approx(math.exp(-0.25 * 50.0))  # exp(-12.5) ~ 3.7e-6
    assert step_low.alpha_m == pytest.approx(1.0 / 6.0)
    assert isinstance(step_low, BAOABStepResult)
    assert len(step_low.delta_q) == dim

    # Case B: High syntax entropy H_syntax >= eps_syn (1.2 >= 0.6) -> Underdamped gamma_t = 1.2
    integrator.p = np.full(dim, 2.0, dtype=float)
    step_high = integrator.step_clause(q_measured=q_meas, h_syntax=1.2)
    assert step_high.gamma_t == 1.2
    assert step_high.theta_t == pytest.approx(math.exp(-0.25 * 1.2))  # exp(-0.3) ~ 0.7408

    # 4. Theorem 5.1: Thermal equilibrium invariant measure under variable friction
    eq_res = integrator.verify_invariant_momentum_measure(n_steps=10000, tau=0.04, seed=42)
    assert eq_res["passed"]
    assert eq_res["max_abs_mean"] < 0.05
    assert eq_res["cov_relative_error"] < 0.15


def test_clause_twisted_smc_decoder_particle_selection():
    """Verify Clause-Level Twisted SMC P=4 particle evaluation, resampling, and anchor tracking."""
    dim = 12
    mu_star = np.full(dim, 0.5, dtype=float)
    integrator = AdaptiveBAOABIntegrator(dim=dim, mu_star=mu_star, seed=42)

    decoder = ClauseTwistedSMCDecoder(
        integrator=integrator,
        n_particles=4,
        temperatures=(0.65, 0.72, 0.80, 0.88),
        gamma_reach=2.0,
        beta_steer=1.5,
        seed=42,
    )

    prefix = "深い霧が立ち込める森で、"
    target_end = "無事に出口へと到達した。"

    # Setup 4 distinct candidates with controlled style and likelihood
    q_target = mu_star.copy()
    q_good = q_target + 0.01  # Very close to target
    q_bad_style = q_target + 2.0  # Huge style discrepancy
    q_mediocre = q_target + 0.5

    candidates = [
        # Candidate 0: Good local fluency, but terrible style and terrible reachability
        {"text": "鳥のさえずりが遠く途絶え、", "llm_logprob": -0.1, "reach_logprob": -5.0, "q": q_bad_style},
        # Candidate 1: Moderate fluency, moderate style, poor reachability
        {"text": "足元を踏みしめて前へ進み、", "llm_logprob": -1.0, "reach_logprob": -3.0, "q": q_mediocre},
        # Candidate 2: Optimal particle (good fluency, minimal style error, excellent reachability to W_end^*)
        {"text": "微かな光の導きを見出し、", "llm_logprob": -0.4, "reach_logprob": -0.1, "q": q_good},
        # Candidate 3: Poor local fluency distractor
        {"text": "風雪が強く吹き荒れて、", "llm_logprob": -8.0, "reach_logprob": -2.0, "q": q_mediocre},
    ]

    # Execute single SMC clause step
    step_res = decoder.step_smc_clause(
        prefix=prefix,
        candidate_clauses=candidates,
        target_end=target_end,
        target_q=q_target,
    )

    # 1. Verification of particle weights and terms
    assert len(step_res["weights"]) == 4
    assert len(step_res["term1_llm"]) == 4
    assert len(step_res["term2_reach"]) == 4
    assert len(step_res["term3_style"]) == 4

    # Candidate 2 must have highest overall weight w^{(2)}
    assert step_res["selected_idx"] == 2
    assert step_res["selected_clause"] == "微かな光の導きを見出し、"
    assert step_res["next_prefix"] == prefix + "微かな光の導きを見出し、"

    # Normalized weights sum to 1.0
    norm_w = step_res["normalized_weights"]
    assert np.sum(norm_w) == pytest.approx(1.0)
    assert 1.0 <= step_res["ess"] <= 4.0

    # Verify BAOAB state updated
    assert integrator.m == 1
    assert len(integrator.history_q) == 1

    # 2. Multi-step generation rollout with mock/heuristic generator
    gen_res = decoder.generate_with_clause_twisted_smc(
        prefix="静かな夕暮れの街角で、",
        target_end="やがて街灯が灯った。",
        max_clauses=3,
    )
    assert "welded_text" in gen_res
    assert len(gen_res["clauses"]) >= 1
    assert len(gen_res["telemetry"]) >= 1
    assert gen_res["final_q"].shape == (12,)
    assert np.all(np.isfinite(gen_res["final_q"]))


def test_adaptive_baoab_integrator_bridge_attraction_and_full_invariant():
    """Verify bridge attraction force lambda_bridge * phi_bridge and full 5-stage BAOAB Theorem 5.1."""
    dim = 12
    mu_star = np.full(dim, 0.5, dtype=float)

    # 1. Bridge attraction steering test
    phi_pull = np.zeros(dim, dtype=float)
    phi_pull[0] = 2.0  # Strong positive bridge pull on dim 0

    integrator_nobridge = AdaptiveBAOABIntegrator(dim=dim, mu_star=mu_star, lambda_bridge=0.0, seed=10)
    integrator_bridge = AdaptiveBAOABIntegrator(dim=dim, mu_star=mu_star, lambda_bridge=2.0, phi_bridge=phi_pull, seed=10)

    # Freeze noise for deterministic comparison
    zero_rng = np.random.default_rng(999)
    step_nobridge = integrator_nobridge.step_clause(q_measured=mu_star, h_syntax=1.2, rng=zero_rng)

    zero_rng2 = np.random.default_rng(999)
    step_bridge = integrator_bridge.step_clause(q_measured=mu_star, h_syntax=1.2, rng=zero_rng2)

    # Dim 0 must be pulled positively in the direction of phi_pull
    assert step_bridge.q[0] > step_nobridge.q[0]
    assert step_bridge.p[0] > step_nobridge.p[0]

    # 2. Theorem 5.1: Full 5-stage coupled Hamiltonian Langevin equilibrium
    eq_full = integrator_bridge.verify_invariant_momentum_measure(
        n_steps=10000,
        tau=0.04,
        seed=42,
        full_baoab=True,
    )
    assert eq_full["passed"]
    assert eq_full["max_abs_mean"] < 0.05
    assert eq_full["cov_relative_error"] < 0.15


def test_adaptive_baoab_integrator_edge_cases():
    """Probe edge cases: zero/negative L, zero variance, dim mismatch, non-identity mass matrix."""
    dim = 12

    # 1. generate_1_over_f_noise edge cases
    noise_zero = generate_1_over_f_noise(L=0, dim=dim)
    assert noise_zero.shape == (0, dim)

    noise_neg = generate_1_over_f_noise(L=-3, dim=dim)
    assert noise_neg.shape == (0, dim)

    noise_zero_dim = generate_1_over_f_noise(L=16, dim=0)
    assert noise_zero_dim.shape == (16, 0)

    noise_zero_var = generate_1_over_f_noise(L=16, dim=dim, variance=0.0)
    assert np.all(noise_zero_var == 0.0)

    # 2. Dimension mismatch check in step_baoab_clause
    with pytest.raises(ValueError, match="Dimension mismatch"):
        step_baoab_clause(q=np.zeros(12), p=np.zeros(8))

    # 3. compute_shrinkage edge values
    integrator = AdaptiveBAOABIntegrator(dim=dim, seed=42)
    q_meas = np.full(dim, 1.0)
    shrunk_neg = integrator.compute_shrinkage(q_meas, m=-2)
    assert np.allclose(shrunk_neg, integrator.mu_star)  # m <= 0 gives alpha = 0

    shrunk_large = integrator.compute_shrinkage(q_meas, m=995)
    assert np.allclose(shrunk_large, q_meas, atol=0.01)  # alpha = 995/1000 = 0.995

    # 4. Non-identity diagonal mass matrix M = 2.0 * I
    M_custom = np.eye(dim) * 2.0
    integrator_m = AdaptiveBAOABIntegrator(dim=dim, mass_matrix=M_custom, seed=42)
    assert np.allclose(integrator_m.M_inv, np.eye(dim) * 0.5)
    res_m = integrator_m.step_clause(q_measured=integrator_m.mu_star)
    assert res_m.q.shape == (dim,)
    assert res_m.p.shape == (dim,)

    # 5. Integrator reset
    integrator.reset(seed=99)
    assert integrator.m == 0
    assert len(integrator.history_q) == 0
    assert len(integrator.ghost_history) == 16


def test_clause_twisted_smc_decoder_edge_cases_and_resampling():
    """Verify raw string inputs, empty candidate recovery, stochastic resampling, and steering limits."""
    dim = 12
    decoder = ClauseTwistedSMCDecoder(dim=dim, n_particles=4, seed=42)

    # 1. Raw string candidates (without dict wrapper)
    raw_candidates = [
        "静かな風が通り抜け、",
        "熱い思いを胸に秘め、",
        "冷ややかな視線を交わし、",
        "遠くの鐘の音が響き、",
    ]
    res_raw = decoder.step_smc_clause(prefix="旅人は歩き出し、", candidate_clauses=raw_candidates)
    assert res_raw["selected_clause"] in raw_candidates
    assert len(res_raw["weights"]) == 4
    assert np.sum(res_raw["normalized_weights"]) == pytest.approx(1.0)

    # 2. Empty candidate clauses list -> recovers via fallback palette without crashing
    res_empty = decoder.step_smc_clause(prefix="旅人は歩き出し、", candidate_clauses=[])
    assert len(res_empty["candidate_texts"]) == 4
    assert res_empty["selected_clause"] != ""

    # 3. Direct call to compute_particle_weights with empty list raises ValueError
    with pytest.raises(ValueError, match="candidates sequence must not be empty"):
        decoder.compute_particle_weights(prefix="旅人は歩き出し、", candidates=[])

    # 4. Stochastic resampling vs MAP
    weights_map = decoder.compute_particle_weights(
        prefix="テスト、",
        candidates=raw_candidates,
        resample_method="map",
    )
    assert weights_map["selected_idx"] == int(np.argmax(weights_map["weights"]))

    # Multinomial resampling must select a valid index
    weights_sample = decoder.compute_particle_weights(
        prefix="テスト、",
        candidates=raw_candidates,
        resample_method="multinomial",
    )
    assert 0 <= weights_sample["selected_idx"] < 4

    # 5. Extreme reachability dominance
    decoder_reach = ClauseTwistedSMCDecoder(dim=dim, gamma_reach=100.0, beta_steer=0.01, seed=42)
    cands_reach = [
        {"text": "節A、", "llm_logprob": -0.1, "reach_logprob": -10.0, "q": np.zeros(dim)},
        {"text": "節B、", "llm_logprob": -0.5, "reach_logprob": -0.01, "q": np.zeros(dim)},
    ]
    res_reach = decoder_reach.step_smc_clause(prefix="前文、", candidate_clauses=cands_reach)
    assert res_reach["selected_idx"] == 1  # Candidate with superior reachability dominates

    # 6. Extreme steering dominance
    q_target = np.full(dim, 0.8)
    decoder_steer = ClauseTwistedSMCDecoder(dim=dim, gamma_reach=0.01, beta_steer=100.0, seed=42)
    cands_steer = [
        {"text": "節A、", "llm_logprob": -0.01, "reach_logprob": -0.1, "q": np.full(dim, 0.0)},
        {"text": "節B、", "llm_logprob": -2.0, "reach_logprob": -0.5, "q": q_target.copy()},
    ]
    res_steer = decoder_steer.step_smc_clause(prefix="前文、", candidate_clauses=cands_steer, target_q=q_target)
    assert res_steer["selected_idx"] == 1  # Candidate matching target style dominates



# ==============================================================================
# Steering gain, trailing style window and momentum-coupled temperature
# ==============================================================================

def _cands(llm_a, llm_b, qa, qb):
    return [
        {"text": "A、", "llm_logprob": llm_a, "q": np.full(12, qa)},
        {"text": "B、", "llm_logprob": llm_b, "q": np.full(12, qb)},
    ]


def test_auto_gain_lets_style_term_overrule_a_small_llm_preference():
    target = np.full(12, 0.50)
    # LLM prefers A by 0.4 nats; the style of B is closer to the target, but only slightly.
    cands = _cands(-1.0, -1.4, 0.56, 0.52)

    legacy = ClauseTwistedSMCDecoder(dim=12, beta_steer=1.0, seed=0)
    r_legacy = legacy.compute_particle_weights("", cands, target_q=target)
    assert r_legacy["selected_clause"] == "A、"  # style term is ~0.02: cannot compete

    steered = ClauseTwistedSMCDecoder(dim=12, beta_steer=1.0, seed=0, auto_gain=True, steer_balance=2.0)
    r_steer = steered.compute_particle_weights("", cands, target_q=target)
    assert r_steer["selected_clause"] == "B、"
    assert r_steer["steer_gain"] > 10.0
    # std of the rescaled style term equals steer_balance * std of the LLM term
    assert np.std(r_steer["term3_style"]) == pytest.approx(2.0 * np.std(r_steer["term1_llm"]), rel=1e-6)


def test_auto_gain_off_by_default_keeps_weights():
    dec = ClauseTwistedSMCDecoder(dim=12, seed=0)
    r = dec.compute_particle_weights("", _cands(-1.0, -1.4, 0.56, 0.52), target_q=np.full(12, 0.5))
    assert r["steer_gain"] == 1.0


def test_style_window_uses_trailing_text_only():
    seen = []

    def fe(text):
        seen.append(text)
        return np.full(12, 0.5)

    dec = ClauseTwistedSMCDecoder(dim=12, seed=0, feature_extractor=fe, style_window_chars=10)
    dec.compute_particle_weights("あ" * 100, ["い、", "う、"], target_q=np.full(12, 0.5))
    assert all(len(t) == 10 for t in seen)


def test_temperature_scale_follows_momentum_norm():
    dec = ClauseTwistedSMCDecoder(dim=12, seed=0, temp_gain=0.15, p_ref=0.5)
    dec.integrator.p = np.full(12, 0.1)
    low = dec._temperature_scale()
    dec.integrator.p = np.full(12, 0.7)
    high = dec._temperature_scale()
    assert low < 1.0 < high <= 1.35
    assert ClauseTwistedSMCDecoder(dim=12, seed=0)._temperature_scale() == 1.0


# ==============================================================================
# Stability guards: gain clip, temperature bounds, repetition / latin guards, steering switch
# ==============================================================================

def test_gain_max_clips_auto_gain():
    cands = _cands(-1.0, -1.4, 0.5001, 0.5000)  # almost identical style -> huge raw gain
    free = ClauseTwistedSMCDecoder(dim=12, seed=0, auto_gain=True)
    clipped = ClauseTwistedSMCDecoder(dim=12, seed=0, auto_gain=True, gain_max=5.0)
    g_free = free.compute_particle_weights("", cands, target_q=np.full(12, 0.5))["steer_gain"]
    r = clipped.compute_particle_weights("", cands, target_q=np.full(12, 0.5))
    assert g_free > 5.0
    assert r["steer_gain"] == 5.0
    # the style term can never exceed steer_balance x the LLM spread
    assert np.std(r["term3_style"]) <= np.std(r["term1_llm"]) + 1e-9


def test_temperature_scale_respects_bounds_and_steering_switch():
    dec = ClauseTwistedSMCDecoder(dim=12, seed=0, temp_gain=0.15, p_ref=0.5, temp_min=0.85, temp_max=1.10)
    dec.integrator.p = np.full(12, 3.0)
    assert dec._temperature_scale() == pytest.approx(1.10)
    dec.integrator.p = np.full(12, 0.01)
    assert dec._temperature_scale() == pytest.approx(0.85)
    dec.steering_active = False
    assert dec._temperature_scale() == 1.0


def test_steering_inactive_disables_auto_gain():
    dec = ClauseTwistedSMCDecoder(dim=12, seed=0, auto_gain=True)
    dec.steering_active = False
    r = dec.compute_particle_weights("", _cands(-1.0, -1.4, 0.56, 0.52), target_q=np.full(12, 0.5))
    assert r["steer_gain"] == 1.0


def test_repeat_guard_blocks_third_occurrence():
    dec = ClauseTwistedSMCDecoder(dim=12, seed=0, repeat_guard=True, style_window_chars=120)
    phrase = "その光が寺内を照らし出す瞬間は、"
    prefix = phrase + "静かだった。" + phrase
    cands = [
        {"text": phrase, "llm_logprob": -0.5, "q": np.full(12, 0.5)},
        {"text": "霧が低く流れていた、", "llm_logprob": -2.0, "q": np.full(12, 0.5)},
    ]
    r = dec.compute_particle_weights(prefix, cands, target_q=np.full(12, 0.5))
    assert r["selected_clause"] == "霧が低く流れていた、"
    assert np.isneginf(r["weights"][0])


def test_repeat_guard_off_by_default_and_never_blocks_all():
    cands = [{"text": "同じ言葉が続く、", "llm_logprob": -1.0, "q": np.full(12, 0.5)}] * 2
    prefix = "同じ言葉が続く、同じ言葉が続く、"
    off = ClauseTwistedSMCDecoder(dim=12, seed=0).compute_particle_weights(prefix, cands, target_q=np.full(12, 0.5))
    assert np.all(np.isfinite(off["weights"]))
    on = ClauseTwistedSMCDecoder(dim=12, seed=0, repeat_guard=True).compute_particle_weights(prefix, cands, target_q=np.full(12, 0.5))
    assert np.all(np.isfinite(on["weights"]))  # all blocked -> guard is skipped, no NaN


def test_ascii_guard_blocks_latin_word_when_context_has_none():
    dec = ClauseTwistedSMCDecoder(dim=12, seed=0, ascii_guard=True)
    cands = [
        {"text": "寺の鐘 sounded した、", "llm_logprob": -0.5, "q": np.full(12, 0.5)},
        {"text": "寺の鐘が鳴った、", "llm_logprob": -1.5, "q": np.full(12, 0.5)},
    ]
    r = dec.compute_particle_weights("静かな朝、", cands, target_q=np.full(12, 0.5))
    assert r["selected_clause"] == "寺の鐘が鳴った、"


def test_min_clauses_toggles_eos_suppression_flag():
    flags = []
    dec = ClauseTwistedSMCDecoder(dim=12, seed=0)
    orig = dec.step_smc_clause

    def spy(*a, **k):
        flags.append(dec._suppress_eos)
        return orig(*a, **k)

    dec.step_smc_clause = spy
    dec.generate_with_clause_twisted_smc("", max_clauses=5, min_clauses=3, early_stop_on_punct=False)
    assert flags == [True, True, True, False, False]
    assert dec._suppress_eos is False


def test_script_guard_blocks_simplified_chinese_and_fullwidth_comma():
    dec = ClauseTwistedSMCDecoder(dim=12, seed=0, script_guard=True)
    cands = [
        {"text": "这座古老的寺庙，", "llm_logprob": -0.5, "q": np.full(12, 0.5)},
        {"text": "古い寺の朝、", "llm_logprob": -1.5, "q": np.full(12, 0.5)},
    ]
    r = dec.compute_particle_weights("静かな朝、", cands, target_q=np.full(12, 0.5))
    assert r["selected_clause"] == "古い寺の朝、"


def test_repeat_window_can_exceed_style_window():
    phrase = "その光が寺内を照らし出す瞬間は、"
    filler = "あ" * 60
    prefix = phrase + filler + phrase + filler  # two earlier occurrences ~150 chars back
    cands = [
        {"text": phrase, "llm_logprob": -0.5, "q": np.full(12, 0.5)},
        {"text": "霧が低く流れていた、", "llm_logprob": -2.0, "q": np.full(12, 0.5)},
    ]
    short = ClauseTwistedSMCDecoder(dim=12, seed=0, repeat_guard=True, style_window_chars=120)
    long_ = ClauseTwistedSMCDecoder(dim=12, seed=0, repeat_guard=True, style_window_chars=120, repeat_window_chars=400)
    assert short.compute_particle_weights(prefix, cands, target_q=np.full(12, 0.5))["selected_clause"] == phrase
    assert long_.compute_particle_weights(prefix, cands, target_q=np.full(12, 0.5))["selected_clause"] == "霧が低く流れていた、"


# ==============================================================================
# Proposal-side momentum injection, JIS boundary, sentence closing
# ==============================================================================

def test_jis_boundary_flags_simplified_forms_without_a_character_list():
    from usir_py.realtime_decoder import has_non_jis_kanji, count_non_jis_kanji

    assert has_non_jis_kanji("雰囲気が一变した")        # 变: simplified form, not in JIS X 0208
    assert has_non_jis_kanji("这座古老的寺庙")
    assert not has_non_jis_kanji("雰囲気が一変した")
    assert not has_non_jis_kanji("静謐な山寺の夜明けは、僧侶たちの祈りを包む。")
    assert count_non_jis_kanji("这们变") == 3


def test_spread_temperatures_follow_momentum_and_require_active_steering():
    dec = ClauseTwistedSMCDecoder(dim=12, seed=0, spread_temps=True, p_ref=0.5)
    dec.integrator.p = np.full(12, 0.05)
    cold = dec._candidate_temperatures(8)
    dec.integrator.p = np.full(12, 1.2)
    hot = dec._candidate_temperatures(8)
    assert cold[0] == pytest.approx(0.7) and hot[0] == pytest.approx(0.7)
    assert hot[-1] == pytest.approx(1.3) and cold[-1] < hot[-1]
    assert hot == sorted(hot)
    dec.steering_active = False
    assert dec._candidate_temperatures(4) == [dec.temperatures[i % len(dec.temperatures)] for i in range(4)]


def test_row_period_bias_orders_rows_short_to_long_and_scales_with_momentum():
    dec = ClauseTwistedSMCDecoder(dim=12, seed=0, short_bias=True, p_ref=0.5, period_bias_max=4.0)
    dec.integrator.p = np.full(12, 1.2)
    b = dec._row_period_bias(8)
    assert b[0] == 0.0 and b == sorted(b) and b[-1] == pytest.approx(4.0)
    dec.integrator.p = np.full(12, 0.05)
    assert dec._row_period_bias(8) == [0.0] * 8
    dec.steering_active = False
    assert dec._row_period_bias(8) == [0.0] * 8


def test_close_after_sets_force_close_flag_and_normalize_end_replaces_trailing_comma():
    flags = []
    dec = ClauseTwistedSMCDecoder(dim=12, seed=0, close_after=3, normalize_end=True)
    orig = dec.step_smc_clause

    def spy(*a, **k):
        flags.append(dec._force_close)
        return orig(*a, **k)

    dec.step_smc_clause = spy
    res = dec.generate_with_clause_twisted_smc(
        "", max_clauses=5, early_stop_on_punct=False,
        candidate_generator=lambda ctx, p, t: "静かな朝、",
    )
    assert flags == [False, False, False, True, True]
    assert res["welded_text"].endswith("。") and not res["welded_text"].endswith("、。")
    assert res["clauses"][-1].endswith("。")
    assert dec._force_close is False


def test_director_switches_pool_size_for_climax_only():
    from usir_py.realtime_decoder import AdaptiveBAOABIntegrator

    forced = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    dec = ClauseTwistedSMCDecoder(integrator=AdaptiveBAOABIntegrator(dim=12, seed=0), seed=0, climax_particles=8)
    sizes = []
    orig = dec.generate_with_clause_twisted_smc

    def spy(*a, **k):
        sizes.append(dec.n_particles)
        return orig(*a, **{x: v for x, v in k.items() if x != "max_clauses"}, max_clauses=2)

    dec.generate_with_clause_twisted_smc = spy
    from usir_py.hsmm_longform_director import HSMMLongformDirector

    res = HSMMLongformDirector(transition_matrix=forced, seed=0).generate_longform(
        target_paragraphs=6, decoder=dec, climax_only_steering=True
    )
    expected = [8 if (w == 2 or (i > 0 and res["wells"][i - 1] == 2)) else 4 for i, w in enumerate(res["wells"])]
    assert sizes == expected


def test_momentum_factor_uses_released_momentum_and_does_not_saturate():
    dec = ClauseTwistedSMCDecoder(dim=12, seed=0, spread_temps=True, short_bias=True)
    dec.integrator.p = np.full(12, 3.0)  # a large ||p|| would saturate the ||p||-based factor
    dec.discharge_norm = 0.265
    a = dec._momentum_factor()
    dec.discharge_norm = 0.007
    b = dec._momentum_factor()
    assert 0.0 < b < a < 0.6
    assert dec._candidate_temperatures(8)[-1] < dec._row_period_bias(8)[-1] + 1.3  # sanity of ranges
    # two different banked amounts give different proposal distributions
    dec.discharge_norm = 0.265
    t1, b1 = dec._candidate_temperatures(8), dec._row_period_bias(8)
    dec.discharge_norm = 0.007
    t2, b2 = dec._candidate_temperatures(8), dec._row_period_bias(8)
    assert t1 != t2 and b1 != b2
    dec.discharge_norm = 0.0
    assert dec._row_period_bias(8) == [0.0] * 8


def test_fullwidth_comma_is_outside_japanese_punctuation():
    from usir_py.realtime_decoder import has_non_jis_kanji

    assert has_non_jis_kanji("この夜明け，それが")
    assert not has_non_jis_kanji("この夜明け、それが。")


def test_close_after_stops_generation_once_the_sentence_is_closed():
    dec = ClauseTwistedSMCDecoder(dim=12, seed=0, close_after=2, normalize_end=True)
    seq = iter(["静かな朝、", "霧が晴れ、", "光が差した。", "余計な文。", "さらに続く。"])
    res = dec.generate_with_clause_twisted_smc(
        "", max_clauses=6, early_stop_on_punct=False,
        candidate_generator=lambda ctx, p, t: seq.__next__() if p == 0 else "静かな朝、",
    )
    assert res["welded_text"].endswith("光が差した。")
    assert "余計な文" not in res["welded_text"]


# ---- assistant-register guards / direction steering ----

def _qc(text, lp=-1.0):
    return {"text": text, "llm_logprob": lp, "q": np.full(12, 0.5)}


def test_meta_guard_blocks_omission_and_parenthetical_notes():
    from usir_py.realtime_decoder import ClauseTwistedSMCDecoder
    dec = ClauseTwistedSMCDecoder(dim=12, seed=0, meta_guard=True)
    cands = [_qc("（残りは省略）", -0.2), _qc("（続く）", -0.3), _qc("霧が低く流れていた、", -2.0)]
    r = dec.compute_particle_weights("静かな朝、", cands, target_q=np.full(12, 0.5))
    assert r["selected_clause"] == "霧が低く流れていた、"


def test_polite_guard_blocks_narration_but_allows_quoted_speech():
    from usir_py.realtime_decoder import ClauseTwistedSMCDecoder
    dec = ClauseTwistedSMCDecoder(dim=12, seed=0, polite_guard=True)
    cands = [_qc("大切ですね。", -0.2), _qc("「行きますよ」と彼は言った、", -1.0)]
    r = dec.compute_particle_weights("朝が来た。", cands, target_q=np.full(12, 0.5))
    assert r["selected_clause"] == "「行きますよ」と彼は言った、"
    # candidate that begins inside an already-open quote is exempt up to the closing bracket
    r2 = dec.compute_particle_weights("彼は「", [_qc("行きます」と言った、", -0.5), _qc("遠くを見た、", -3.0)], target_q=np.full(12, 0.5))
    assert r2["selected_clause"] == "行きます」と言った、"


def test_meta_polite_guards_off_by_default():
    from usir_py.realtime_decoder import ClauseTwistedSMCDecoder
    dec = ClauseTwistedSMCDecoder(dim=12, seed=0)
    r = dec.compute_particle_weights("", [_qc("（残りは省略）", -0.1), _qc("風が吹いた、", -2.0)], target_q=np.full(12, 0.5))
    assert r["selected_clause"] == "（残りは省略）"


def test_count_narration_meta():
    from usir_py.realtime_decoder import count_narration_meta
    c = count_narration_meta("「ありますよ」と言った。大切ですね。（残りは省略）")
    assert c["polite"] == 1 and c["meta"] == 1 and c["paren"] == 1


def test_regen_when_all_candidates_blocked():
    from usir_py.realtime_decoder import ClauseTwistedSMCDecoder
    dec = ClauseTwistedSMCDecoder(dim=12, seed=0, meta_guard=True, max_regen=2)
    batches = [["（省略）", "（続く）"], ["水面が揺れた、", "（略）"]]
    calls = []
    orig = dec.generate_candidate_clauses
    dec.generate_candidate_clauses = lambda prefix, target_end=None: (calls.append(1) or batches[min(len(calls) - 1, 1)])
    res = dec.step_smc_clause("朝、")
    assert res["selected_clause"] == "水面が揺れた、"
    assert dec.stats.get("regen", 0) >= 1


def _dir_dec(vec):
    from usir_py.realtime_decoder import ClauseTwistedSMCDecoder
    dec = ClauseTwistedSMCDecoder(dim=12, seed=0, direction_steer=True)
    dec.discharge_norm = float(np.linalg.norm(vec))
    dec.discharge_vec = np.asarray(vec, dtype=float)
    return dec


def _vec(shock=0.0, flux=0.0):
    v = np.zeros(12); v[[3, 7, 11]] = shock; v[[0, 4, 8]] = flux
    return v


def test_phase_bias_shock_forbids_commas_flux_suppresses_early_periods():
    sp, sc = _dir_dec(_vec(shock=1.0))._phase_bias("")
    assert sp > 4.0 and sc < -7.0                       # Shock: '。' +5d, '、' -8d  (single-clause sentences)
    fp, fc = _dir_dec(_vec(flux=1.0))._phase_bias("\u9759\u304b\u306a\u591c\u3001")
    assert fp < -4.0 and fc > 0                         # Flux, first clause of the sentence: '。' suppressed, '、' up
    fp2, fc2 = _dir_dec(_vec(flux=1.0))._phase_bias("\u9759\u304b\u306a\u3001\u591c\u306e\u3001")
    assert (fp2, fc2) == (0.0, 0.0)                     # after 2 commas the sentence may close
    assert _dir_dec(_vec(shock=1.0, flux=0.9))._phase_bias("") == (0.0, 0.0)  # inside |d| <= 0.25: no modulation


def test_direction_default_off_and_row_bias_not_graded():
    from usir_py.realtime_decoder import ClauseTwistedSMCDecoder
    off = ClauseTwistedSMCDecoder(dim=12, seed=0)
    off.discharge_vec = _vec(shock=1.0)
    assert off._phase_bias("") == (0.0, 0.0) and off._row_period_bias(4) == [0.0] * 4
    assert _dir_dec(_vec(shock=1.0))._row_period_bias(4) == [0.0] * 4


def test_shock_clause_targets_cycle_for_sentence_length_variety_and_flux_is_long():
    dec = _dir_dec(_vec(shock=1.0))
    caps = []
    for m in range(3):
        dec._clause_idx = m
        caps.append(dec._row_caps(4)[0])
    assert caps[1] < caps[0] < caps[2] and min(caps) >= 6      # short / very short / medium
    cf = _dir_dec(_vec(flux=1.0))._row_caps(4)
    assert max(cf) == 32 and min(cf) > 20
    ts, tf = _dir_dec(_vec(shock=1.0))._candidate_temperatures(4), _dir_dec(_vec(flux=1.0))._candidate_temperatures(4)
    assert ts[-1] > tf[-1]


def test_local_potential_prefers_candidate_whose_sentence_moves_along_p():
    from usir_py.realtime_decoder import ClauseTwistedSMCDecoder

    def fe(t):
        q = np.full(12, 0.5)
        q[3] = 0.9 if "B" in t else 0.1
        return q

    dec = ClauseTwistedSMCDecoder(dim=12, seed=0, direction_steer=True, feature_extractor=fe)
    dec.discharge_vec = _vec(shock=1.0)
    cands = [{"text": "A\u3001", "llm_logprob": -1.0}, {"text": "B\u3001", "llm_logprob": -1.0}]
    r = dec.compute_particle_weights("\u524d\u6587\u3002", cands, target_q=np.full(12, 0.5))
    assert r["align_active"] and r["align_cos"][1] > 0.5 and r["align_cos"][0] < -0.5
    assert r["term4_align"][1] > 0 > r["term4_align"][0] and r["selected_clause"] == "B\u3001"
    # gamma_dir >= 5 couples strongly: |term4| exceeds an LLM log-prob gap of 1.0
    assert abs(r["term4_align"][1] - r["term4_align"][0]) > 1.0


def test_register_penalty_has_no_all_blocked_passthrough():
    from usir_py.realtime_decoder import ClauseTwistedSMCDecoder
    dec = ClauseTwistedSMCDecoder(dim=12, seed=0, polite_guard=True)
    r = dec.compute_particle_weights("", [_qc("良いですね、", -0.1), _qc("風が吹いた、", -3.0)], target_q=np.full(12, 0.5))
    assert r["weights"][0] < -9000 and r["selected_clause"] == "風が吹いた、"
    r2 = dec.compute_particle_weights("", [_qc("良いですね、", -0.1), _qc("悪いですね、", -0.1)], target_q=np.full(12, 0.5))
    assert np.all(r2["weights"] < -9000)  # penalised even when every candidate violates


def test_forbidden_token_ids_from_surfaces():
    from usir_py.realtime_decoder import ClauseTwistedSMCDecoder

    class Tok:
        surf = ["静か", "です", "ますね", "（", "た", "風", "ょう", "ん", "※"]
        def __len__(self): return len(self.surf)
        def batch_decode(self, ids, skip_special_tokens=False): return [self.surf[i[0]] for i in ids]

    dec = ClauseTwistedSMCDecoder(dim=12, seed=0, tokenizer=Tok(), narration_mask=True)
    bad, post = dec._forbidden_token_ids()
    assert set(bad) == {1, 2, 3, 8} and set(post) == {4, 6, 7}


def test_repeat_guard_window_covers_carried_context_when_enabled():
    from usir_py.realtime_decoder import ClauseTwistedSMCDecoder
    phrase = "彼は自己認識を深めることで逃げ方を選ぶ"
    kw = dict(dim=12, seed=0, repeat_guard=True, repeat_window_chars=400, chat_prefix=phrase + phrase + "。")
    cands = [_qc(phrase + "、", -0.1), _qc("霧が低く流れていた、", -2.0)]
    off = ClauseTwistedSMCDecoder(**kw)
    assert off.compute_particle_weights("", cands, target_q=np.full(12, 0.5))["selected_clause"] == phrase + "、"
    on = ClauseTwistedSMCDecoder(**kw)
    on.repeat_use_context = True
    assert on.compute_particle_weights("", cands, target_q=np.full(12, 0.5))["selected_clause"] == "霧が低く流れていた、"


def test_history_ngram_map_and_first_token_ban_state():
    from usir_py.realtime_decoder import ClauseTwistedSMCDecoder

    class Tok:
        def encode(self, t, add_special_tokens=False): return [ord(c) % 997 for c in t]

    dec = ClauseTwistedSMCDecoder(dim=12, seed=0, tokenizer=Tok(), ngram_ban=4)
    dec.history_text = "ABCDEF"
    m = dec._history_ngram_map()
    assert (65 % 997, 66 % 997, 67 % 997) in m and (68 % 997) in m[(65 % 997, 66 % 997, 67 % 997)]
    dec.history_text = "XYZW"
    assert (65 % 997, 66 % 997, 67 % 997) not in dec._history_ngram_map()  # cache refreshed on new history


def test_all_repeat_blocked_is_penalised_not_passed_through():
    from usir_py.realtime_decoder import ClauseTwistedSMCDecoder
    phrase = "同じ言葉が続く、"
    dec = ClauseTwistedSMCDecoder(dim=12, seed=0, repeat_guard=True)
    r = dec.compute_particle_weights(phrase * 3, [_qc(phrase, -1.0), _qc(phrase, -1.2)], target_q=np.full(12, 0.5))
    assert np.all(r["weights"] < -9000) and dec.stats.get("repeat_unresolved") == 1


def test_chosen_clause_local_deviation_is_reported():
    from usir_py.realtime_decoder import ClauseTwistedSMCDecoder

    def fe(t):
        q = np.full(12, 0.5)
        q[3] = 0.9 if "A" in t else 0.5
        return q

    dec = ClauseTwistedSMCDecoder(dim=12, seed=0, direction_steer=True, feature_extractor=fe)
    dec.discharge_vec = _vec(shock=1.0)
    res = dec.step_smc_clause("\u524d\u6587\u3002", [{"text": "A\u3001", "llm_logprob": -1.0}, {"text": "B\u3001", "llm_logprob": -1.5}])
    assert res["align_active"] and abs(res["chosen_dq"][3] - 0.2) < 1e-9 and res["chosen_d_local"] is not None


def test_connective_and_terminal_endings_and_fragment_regex():
    from usir_py.realtime_decoder import _FRAGMENT_RE, _is_connective_end, _is_terminal_end
    for s in ("光が昇り始めるにつれ", "走って", "静かで", "雲を", "歩きながら", "降り出したから", "進まず", "音もなく"):
        assert _is_connective_end(s) and not _is_terminal_end(s), s
    assert not _is_connective_end("風が吹く")
    for s in ("扉が開いた", "夜だ", "風が鳴る", "冷たい", "山寺", "カメラ"):
        assert _is_terminal_end(s), s
    assert len(_FRAGMENT_RE.findall("光が昇り始めるにつれ。金色に染まった天辺。走って。歩きながら。")) == 3
    assert _FRAGMENT_RE.findall("扉が開いた。風が吹く。") == []


def test_fragment_candidates_are_blocked_and_char8_ceiling():
    from usir_py.realtime_decoder import ClauseTwistedSMCDecoder
    dec = ClauseTwistedSMCDecoder(dim=12, seed=0, syntactic_gate=True, char8_max=2)
    assert dec._blocked_candidates("", ["光が昇り始めるにつれ。", "扉が開いた。"]).tolist() == [True, False]
    off = ClauseTwistedSMCDecoder(dim=12, seed=0)
    assert not off._blocked_candidates("", ["光が昇り始めるにつれ。"]).any()
    # 8-char substring: allowed twice, rejected on the third occurrence
    dec.history_text = "朝の光が山を染める。\n朝の光が山を染める。"
    assert dec._char8_hits("", ["朝の光が山を染める。", "雨が静かに降る。"]).tolist() == [True, False]
    dec.history_text = "朝の光が山を染める。"
    assert dec._char8_hits("", ["朝の光が山を染める。"]).tolist() == [False]


def test_min_chars_floor_keeps_generating_and_max_chars_forces_close():
    from usir_py.realtime_decoder import ClauseTwistedSMCDecoder
    dec = ClauseTwistedSMCDecoder(dim=12, seed=0, close_after=2, normalize_end=True)
    flags = []

    def gen(ctx, p, t):
        flags.append((dec._suppress_eos, dec._force_close))
        return "あいうえお。"
    r = dec.generate_with_clause_twisted_smc("", max_clauses=10, early_stop_on_punct=False,
                                             candidate_generator=gen, min_chars=20)
    assert len(r["text"]) >= 20 and r["n_clauses"] >= 4
    assert flags[0][0] is True                       # below the char floor: EOS suppressed
    dec2 = ClauseTwistedSMCDecoder(dim=12, seed=0)
    r2 = dec2.generate_with_clause_twisted_smc("", max_clauses=10, early_stop_on_punct=False,
                                               candidate_generator=lambda c, p, t: "あいうえお。", max_chars=10)
    assert r2["n_clauses"] == 3


def test_soft_ngram_map_is_cached_per_n():
    from usir_py.realtime_decoder import ClauseTwistedSMCDecoder

    class Tk:
        def encode(self, t, add_special_tokens=False):
            return [ord(c) for c in t]
    dec = ClauseTwistedSMCDecoder(dim=12, seed=0, ngram_ban=5, ngram_soft=4)
    dec.tokenizer = Tk()
    dec.history_text = "abcdefg"
    m4, m5 = dec._history_ngram_map(4), dec._history_ngram_map(5)
    assert (ord("a"), ord("b"), ord("c")) in m4 and (ord("a"), ord("b"), ord("c"), ord("d")) in m5
    assert dec._history_ngram_map(4) is m4


def test_text_audit_detects_inflection_collapse():
    from usir_py.text_audit import audit_text
    assert audit_text("静けかな風が吹き荒れる。")["inflection_anomalies"] == ["静けかな"]
    assert audit_text("静かな風が吹く。静けさが満ちる。")["inflection_anomalies"] == []


def test_rhythm_cycles_short_medium_long_and_yields_to_regime():
    from usir_py.realtime_decoder import ClauseTwistedSMCDecoder
    dec = ClauseTwistedSMCDecoder(dim=12, seed=0, rhythm=True)
    short = dec._rhythm_params("")
    assert short[0] > 0 > short[1] and short[2] <= 18                 # sentence 1: short, period pushed
    med1 = dec._rhythm_params("光が差した。")
    med2 = dec._rhythm_params("光が差した。風が、")
    assert med1[0] < 0 and med2[0] > 0                                # sentence 2: 2 clauses
    lng = dec._rhythm_params("a。b。")
    assert lng[0] < 0 and dec._rhythm_params("a。b。" + "あ" * 70)[0] > 0   # sentence 3: long, closes after ~65 chars
    assert ClauseTwistedSMCDecoder(dim=12, seed=0)._rhythm_params("") is None   # default off
    dec.direction_steer = True
    dec.discharge_vec = np.array([0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0])
    assert dec._rhythm_params("") is None                              # Shock regime owns the bias


def test_token_classes_exempt_single_hiragana_and_grammar_tokens():
    from usir_py.realtime_decoder import ClauseTwistedSMCDecoder

    class Tk:
        vocab = ["さ", "か", "な", "ている", "静寂", "山", "カメラ", "、", "ー"]

        def __len__(self):
            return len(self.vocab)

        def batch_decode(self, ids, skip_special_tokens=False):
            return [self.vocab[i[0]] for i in ids]
    dec = ClauseTwistedSMCDecoder(dim=12, seed=0)
    dec.tokenizer = Tk()
    content, first_kanji, hira_only = dec._token_classes()
    assert content == [False, False, False, False, True, True, True, False, False]
    assert first_kanji == {"静": [4], "山": [5]}
    assert hira_only == [True, True, True, True, False, False, False, False, False]


def test_soft_landing_closes_at_first_sentence_end_after_floor():
    from usir_py.realtime_decoder import ClauseTwistedSMCDecoder
    dec = ClauseTwistedSMCDecoder(dim=12, seed=0)
    seq = iter(["あいうえお、", "かきくけこ、", "さしすせそ。", "たちつてと。"] * 3)
    r = dec.generate_with_clause_twisted_smc("", max_clauses=10, early_stop_on_punct=False,
                                             candidate_generator=lambda c, p, t: next(seq), min_chars=10)
    assert r["text"].endswith("さしすせそ。") and r["n_clauses"] == 3


def test_rhythm_long_seal_short_sentence_and_ngram_soft_all_tokens():
    from usir_py.realtime_decoder import ClauseTwistedSMCDecoder
    dec = ClauseTwistedSMCDecoder(dim=12, seed=0, rhythm=True)
    sp, sc, scap, need_noun = dec._rhythm_params("")
    assert sp > 0 and sc <= -12.0 and scap == 13 and need_noun                          # short: single clause, commas sealed
    long_p = dec._rhythm_params("a。b。" + "あ、" * 2 + "い" * 20)             # long sentence, 24 chars < 55
    assert long_p[0] <= -12.0 and long_p[1] > 0
    assert dec._rhythm_params("a。b。" + "あ" * 60)[0] > 0                  # >= 55: seal released
    assert dec._rhythm_params("a。b。" + "あ" * 90)[0] >= 10.0              # >= 85: pushed to close


def test_cv_potential_penalises_uniform_sentence_lengths():
    from usir_py.realtime_decoder import ClauseTwistedSMCDecoder
    dec = ClauseTwistedSMCDecoder(dim=12, seed=0, cv_gain=25.0, cv_floor=0.42)
    prefix = "あいうえおかきくけこ。さしすせそたちつてと。なにぬねのはひふへほ。"
    cands = [{"text": "まみむめもやゆよらり。", "llm_logprob": -1.0},       # same length -> CV stays ~0
             {"text": "まみむめもやゆよらりるれろわをんあいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろ。", "llm_logprob": -1.0}]
    r = dec.compute_particle_weights(prefix, cands, target_q=np.full(12, 0.5))
    assert r["weights"][0] < r["weights"][1] - 5.0
