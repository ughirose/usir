"""Unit and Stress Tests for Phase C: USIR-GRPO LoRA Weight Forge, Activation Steering, and Independent Blind Judge."""

from __future__ import annotations

import math
import numpy as np
import pytest

from usir_py.manifold_geodesic import compute_multiplicative_grpo_reward
from usir_py.lora_grpo_forge import (
    AttachedLoRALinear,
    attach_lora_to_model,
    set_lora_enabled,
    run_authentic_grpo_lora_forge,
)
from usir_py.activation_steerer import SymplecticActivationSteerer
from colab.run_v4_bidirectional_experiment import (
    evaluate_llm_judge,
    build_independent_judge_rubric_prompt,
    run_grpo_forge_training,
)

HEALTHY_PASSAGE = (
    "夜の静寂が広がる古い書斎で、青年は机上の羊皮紙に目を落とした。"
    "青白い月光が窓枠を斜めに切り裂き、微かなインクの匂いが漂う。"
    "指先でなぞった文字は冷たく、過ぎ去った日々の記憶を静かに呼び覚ましていた。"
)


# ===========================================================================
# 1. compute_multiplicative_grpo_reward Reference PPL Barrier Tests
# ===========================================================================

def test_multiplicative_grpo_reward_ppl_ref_barrier():
    """Verifies reference perplexity gate: M_semantic drops sharply when PPL exceeds threshold."""
    # 1. Baseline: ppl_ref is None -> m_semantic == 1.0
    res_none = compute_multiplicative_grpo_reward(HEALTHY_PASSAGE, ppl_ref=None)
    assert res_none["m_semantic"] == 1.0
    assert res_none["ppl_ref"] is None
    assert res_none["reward"] > 0.05
    assert not res_none["is_rejected"]

    # 2. Low PPL (e.g. 20.0 << threshold 45.0) -> m_semantic near 1.0
    res_low = compute_multiplicative_grpo_reward(
        HEALTHY_PASSAGE,
        ppl_ref=20.0,
        ppl_thresh=45.0,
        alpha_ppl=0.1,
    )
    # 1 / (1 + exp(0.1 * (20 - 45))) = 1 / (1 + exp(-2.5)) ~ 0.924
    assert 0.90 <= res_low["m_semantic"] <= 0.95
    assert res_low["reward"] > 0.04
    assert not res_low["is_rejected"]

    # 3. High PPL (e.g. 150.0 >> threshold 45.0) -> m_semantic drops to near 0
    res_high = compute_multiplicative_grpo_reward(
        HEALTHY_PASSAGE,
        ppl_ref=150.0,
        ppl_thresh=45.0,
        alpha_ppl=0.1,
    )
    # 1 / (1 + exp(0.1 * (150 - 45))) = 1 / (1 + exp(10.5)) ~ 2.75e-5
    assert res_high["m_semantic"] < 1e-4
    assert res_high["reward"] < 1e-4
    assert res_high["is_rejected"] is True

    # 4. Empty text edge case
    res_empty = compute_multiplicative_grpo_reward("", ppl_ref=30.0)
    assert res_empty["m_semantic"] == 0.0
    assert res_empty["ppl_ref"] == 30.0
    assert res_empty["reward"] == 0.0
    assert res_empty["is_rejected"] is True


# ===========================================================================
# 2. AttachedLoRALinear Layer & Toggle Tests
# ===========================================================================

class MockBaseLinear:
    """Mock linear layer compatible with both Torch-like and NumPy calls."""
    def __init__(self, in_features: int = 64, out_features: int = 64):
        self.in_features = in_features
        self.out_features = out_features
        # Orthogonal-ish weight
        rng = np.random.RandomState(42)
        self.weight = rng.randn(out_features, in_features).astype(np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32) @ self.weight.T

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def parameters(self):
        return [self.weight]


def test_attached_lora_linear_forward_and_toggle():
    """Verifies AttachedLoRALinear forward computation and instantaneous lora_enabled toggle."""
    base = MockBaseLinear(in_features=64, out_features=64)
    lora = AttachedLoRALinear(base_layer=base, r=8, alpha=16.0)

    assert lora.scaling == 2.0  # 16.0 / 8
    assert lora.lora_enabled is True

    # Synthetic input
    x = np.ones((2, 64), dtype=np.float32)

    # 1. Forward with lora_enabled = True (lora_B initially zeros -> output equals base)
    out_initial = lora(x)
    base_out = base(x)
    np.testing.assert_allclose(out_initial, base_out, rtol=1e-5)

    # 2. Mutate lora_B to non-zero -> output must differ from base
    if isinstance(lora.lora_B, np.ndarray):
        lora.lora_B.fill(0.1)
    else:
        # PyTorch Parameter
        lora.lora_B.data.fill_(0.1)


    out_steered = lora(x)
    assert not np.allclose(out_steered, base_out)

    # 3. Disable LoRA -> output must match base layer output exactly
    lora.lora_enabled = False
    out_disabled = lora(x)
    np.testing.assert_allclose(out_disabled, base_out, rtol=1e-5)

    # 4. Re-enable LoRA -> steered output restored
    lora.lora_enabled = True
    out_re_enabled = lora(x)
    np.testing.assert_allclose(out_re_enabled, out_steered, rtol=1e-5)


# ===========================================================================
# 3. Transformer In-Place Scanner & Module Attachment Tests
# ===========================================================================

class MockTransformerBlock:
    def __init__(self, in_f: int = 64, out_f: int = 64):
        self.q_proj = MockBaseLinear(in_f, out_f)
        self.k_proj = MockBaseLinear(in_f, out_f)
        self.v_proj = MockBaseLinear(in_f, out_f)
        self.out_proj = MockBaseLinear(in_f, out_f)


class MockTransformerModel:
    def __init__(self, num_layers: int = 4):
        self.layers = [MockTransformerBlock() for _ in range(num_layers)]


def test_attach_lora_to_model():
    """Verifies attach_lora_to_model scans transformer layers and performs in-place replacement."""
    model = MockTransformerModel(num_layers=4)

    # Attach LoRA to all 4 layers (q_proj and v_proj)
    res = attach_lora_to_model(
        model,
        target_modules=["q_proj", "v_proj"],
        r=8,
        alpha=16.0,
    )

    assert res["num_attached"] == 8  # 4 layers * 2 modules
    for layer in model.layers:
        assert isinstance(layer.q_proj, AttachedLoRALinear)
        assert isinstance(layer.v_proj, AttachedLoRALinear)
        # k_proj and out_proj must remain untransformed
        assert not isinstance(layer.k_proj, AttachedLoRALinear)
        assert not isinstance(layer.out_proj, AttachedLoRALinear)

    # Test toggling all LoRA modules at once
    count_off = set_lora_enabled(model, False)
    assert count_off == 8
    for layer in model.layers:
        assert layer.q_proj.lora_enabled is False
        assert layer.v_proj.lora_enabled is False

    count_on = set_lora_enabled(model, True)
    assert count_on == 8
    for layer in model.layers:
        assert layer.q_proj.lora_enabled is True
        assert layer.v_proj.lora_enabled is True


def test_attach_lora_target_layers_restriction():
    """Verifies target_layers parameter attaches only to specified layer subsets."""
    model = MockTransformerModel(num_layers=6)

    # Target only the last 2 layers (indices 4, 5)
    res = attach_lora_to_model(
        model,
        target_modules=["q_proj"],
        r=4,
        alpha=8.0,
        target_layers=2,
    )

    assert res["num_attached"] == 2
    assert not isinstance(model.layers[0].q_proj, AttachedLoRALinear)
    assert not isinstance(model.layers[3].q_proj, AttachedLoRALinear)
    assert isinstance(model.layers[4].q_proj, AttachedLoRALinear)
    assert isinstance(model.layers[5].q_proj, AttachedLoRALinear)


# ===========================================================================
# 4. Authentic GRPO Policy Gradient Forge Loop Tests
# ===========================================================================

def test_run_authentic_grpo_lora_forge_loop():
    """Verifies GRPO policy gradient training loop updates weights and tracks KL."""
    base = MockBaseLinear(64, 64)
    adapter = AttachedLoRALinear(base, r=8, alpha=16.0)

    # Candidate rollouts with varying stylistic quality
    candidates = [
        [
            "夜の静寂が広がる古い書斎で、青年は机上の羊皮紙に目を落とした。青白い月光が窓枠を切り裂く。",
            "雨の音が静かに屋根を打ち、かすかな湿気が古い本棚の隙間を満たしていく。",
            "激しい閃光が大気を引き裂き、天地を揺るがす轟音が地底から突き上がった。",
            "あああああああああああああああああああああああああああああ。",  # Low reward / degenerate
        ]
    ]

    res = run_authentic_grpo_lora_forge(
        model=adapter,
        candidate_completions=candidates,
        num_rollouts=4,
        num_steps=3,
        beta_kl=0.04,
        device="cpu",
    )

    assert len(res["steps"]) == 3
    assert res["final_mean_reward"] > 0.0
    for s in res["steps"]:
        assert "loss" in s
        assert "mean_reward" in s
        assert "mean_kl" in s
        assert s["mean_kl"] >= 0.0  # Schulman KL divergence is strictly non-negative


# ===========================================================================
# 5. SymplecticActivationSteerer Tests
# ===========================================================================

def test_symplectic_activation_steerer_orthonormality():
    """Verifies V_steer maintains strict column-orthonormality V^T V = I_12."""
    steerer = SymplecticActivationSteerer(d_model=128, num_layers=28)

    # Initial random Haar basis
    v_info = steerer.verify_orthonormality()
    assert v_info["is_orthonormal"] is True
    assert v_info["max_diag_error"] < 1e-4
    assert v_info["max_offdiag_error"] < 1e-4

    # Calibration on mini-corpus
    corpus = [
        "静かな雨が庭の青苔を濡らし、かすかな湿気が古い板壁を包み込む。",
        "激しい閃光が大気を切り裂き、爆音とともに建物の屋根が激しく揺れた。",
        "二人は黙って互いを見つめ合い、言葉にならぬ合意が交わされた。",
    ] * 4  # 12 samples
    # default (independent per-element ridge): unit-norm columns, not orthogonalised
    v_ind = steerer.calibrate_12_element_axes(corpus=corpus)
    assert v_ind.shape == (128, 12)
    norms = np.linalg.norm(v_ind, axis=0)
    # an axis whose q has zero variance on the corpus (3 distinct texts here) gets a zero column, never a made-up direction
    assert np.all((np.abs(norms - 1.0) < 1e-4) | (norms < 1e-6))
    assert (np.abs(norms - 1.0) < 1e-4).sum() >= 6

    # legacy QR mode: strict column-orthonormality
    v_calib = steerer.calibrate_12_element_axes(corpus=corpus, axis_mode="qr")
    assert v_calib.shape == (128, 12)

    v_calib_info = steerer.verify_orthonormality()
    assert v_calib_info["is_orthonormal"] is True
    assert v_calib_info["max_diag_error"] < 1e-4
    assert v_calib_info["max_offdiag_error"] < 1e-4


def test_symplectic_activation_steerer_norm_adaptive_injection():
    """Verifies potential injection h' = h + alpha * ||h||_2 * (V * dq) scales adaptively with norm."""
    steerer = SymplecticActivationSteerer(d_model=64, num_layers=20)

    # Desired shift along 3rd canonical axis (e.g. Void [間] +0.5)
    delta_q = np.zeros(12, dtype=np.float32)
    delta_q[2] = 0.5

    # 1. Base hidden state of unit norm
    h1 = np.ones((1, 4, 64), dtype=np.float32)  # shape [B, T, D]
    norm1 = np.linalg.norm(h1[0, 0])

    steered1 = steerer.steer_tensor(h1, delta_q_star=delta_q, alpha_act=0.15)
    diff1 = np.linalg.norm(steered1[0, 0] - h1[0, 0])

    # 2. Scaled hidden state (10x larger norm)
    h2 = h1 * 10.0
    steered2 = steerer.steer_tensor(h2, delta_q_star=delta_q, alpha_act=0.15)
    diff2 = np.linalg.norm(steered2[0, 0] - h2[0, 0])

    # Injection magnitude must scale proportionally to norm ||h||_2
    np.testing.assert_allclose(diff2 / diff1, 10.0, rtol=1e-4)


def test_symplectic_activation_steerer_context_manager():
    """Verifies steering_context safely manages state during execution."""
    steerer = SymplecticActivationSteerer(d_model=64)
    delta_q = np.zeros(12)
    delta_q[5] = 0.3

    assert steerer._current_delta_q is None

    with steerer.steering_context(delta_q_star=delta_q, alpha_act=0.2):
        assert steerer._current_delta_q is not None
        assert steerer._current_alpha == 0.2

    # Context exit resets active steering parameters
    assert steerer._current_delta_q is None


# ===========================================================================
# 6. Independent Blind LLM Judge Rubric Tests
# ===========================================================================

def test_independent_judge_rubric_scoring_and_prompt():
    """Verifies independent blind judge scores texts without circular self-scoring."""
    prompt = "雨上がりの古寺"
    target = "静寂が心を満たした"

    # 1. High quality literary text
    good_text = (
        "雨上がりの古寺に、静謐な風が木々の間を抜けて吹き渡る。"
        "濡れた青苔の上に落ちた雫が、夕暮れの淡い日差しを浴びて煌めいている。"
        "足音を忍ばせて石畳を進むと、遠くでかすかな鐘の音が響いた。"
        "日常の喧騒は消え去り、深い静寂が心を満たした。"
    )
    res_good = evaluate_llm_judge(good_text, domain="novel", prompt=prompt, target_conclusion=target)

    # 2. Cliché-ridden / AI slop contaminated text
    slop_text = (
        "雨上がりの古寺は、現代社会において非常に注目を集めている重要な課題である。"
        "様々な応用が考えられる初期条件において、社会や科学の観点からモデル化において重要である。"
        "解決策として期待結果が得られるといえるだろう。にほかならない。"
    )
    res_slop = evaluate_llm_judge(slop_text, domain="novel", prompt=prompt, target_conclusion=target)

    # Rubric keys check
    for key in ["coherence_flow", "logical_validity", "structural_sophistication", "absence_of_slop", "total_score", "normalized_quality", "rubric_prompt"]:
        assert key in res_good
        assert key in res_slop

    # High quality text must score substantially higher than slop text
    assert res_good["normalized_quality"] > res_slop["normalized_quality"]
    assert res_good["absence_of_slop"] > res_slop["absence_of_slop"]

    # Verify rubric prompt contains criteria and candidate text
    rubric_p = build_independent_judge_rubric_prompt(good_text, "novel", prompt, target)
    assert "INDEPENDENT BLIND EVALUATOR RUBRIC" in rubric_p
    assert prompt in rubric_p
    assert target in rubric_p
    assert "Coherence & Flow" in rubric_p


def test_colab_run_grpo_forge_training_adapter_integration():
    """Verifies run_grpo_forge_training in colab runner runs with AttachedLoRALinear."""
    mock_results = [
        {
            "methods": {
                "vanilla_unidirectional": {"text": "青白い月光が机上の古い書物を照らし出した。"},
                "usir_v3_forward_only": {"text": "静かな夜の空気が張り詰め、窓の外で木々がざわめく。"},
                "usir_v4_bidirectional_collision": {"text": "机の上のインク壺から、かすかな黒い香りが立ち上る。"},
            },
            "anchors": {
                "x_0": "青白い月光が机上の古い書物を照らし出した。",
                "x_T": "かすかな黒い香りが立ち上る。",
            },
            "domain": "novel",
        }
    ]

    res = run_grpo_forge_training(mock_results, device="cpu", num_steps=2)
    assert "mean_reward" in res
    assert "steps" in res
    assert len(res["steps"]) == 2


# ===========================================================================
# 7. Deep Edge Cases & Robustness Audit Tests
# ===========================================================================

def test_multiplicative_grpo_reward_nan_and_inf_safety():
    """Verifies compute_multiplicative_grpo_reward shuts down cleanly without propagating NaNs or Infs."""
    # 1. NaN perplexity
    res_nan = compute_multiplicative_grpo_reward(HEALTHY_PASSAGE, ppl_ref=float("nan"))
    assert res_nan["m_semantic"] == 0.0
    assert res_nan["reward"] == 0.0
    assert res_nan["is_rejected"] is True

    # 2. Infinite perplexity
    res_inf = compute_multiplicative_grpo_reward(HEALTHY_PASSAGE, ppl_ref=float("inf"))
    assert res_inf["m_semantic"] == 0.0
    assert res_inf["reward"] == 0.0
    assert res_inf["is_rejected"] is True

    # 3. Negative perplexity
    res_neg = compute_multiplicative_grpo_reward(HEALTHY_PASSAGE, ppl_ref=-10.0)
    assert res_neg["m_semantic"] == 0.0
    assert res_neg["reward"] == 0.0
    assert res_neg["is_rejected"] is True


def test_steerer_d_model_dimension_guard():
    """Verifies SymplecticActivationSteerer strictly validates d_model >= 12."""
    with pytest.raises(ValueError, match="d_model must be at least 12"):
        SymplecticActivationSteerer(d_model=8)

    steerer12 = SymplecticActivationSteerer(d_model=12)
    assert steerer12.v_steer.shape == (12, 12)
    assert steerer12.verify_orthonormality()["is_orthonormal"] is True


def test_attach_lora_named_modules_hierarchical_fallback():
    """Verifies attach_lora_to_model correctly traverses hierarchy even without get_submodule."""
    class MockAttention:
        def __init__(self):
            self.q_proj = MockBaseLinear(64, 64)
            self.v_proj = MockBaseLinear(64, 64)

    class MockLayerHierarchy:
        def __init__(self):
            self.self_attn = MockAttention()

    class MockModelHierarchy:
        def __init__(self):
            self.layer0 = MockLayerHierarchy()
            self.layer1 = MockLayerHierarchy()

        def named_modules(self):
            return [
                ("", self),
                ("layer0", self.layer0),
                ("layer0.self_attn", self.layer0.self_attn),
                ("layer0.self_attn.q_proj", self.layer0.self_attn.q_proj),
                ("layer0.self_attn.v_proj", self.layer0.self_attn.v_proj),
                ("layer1", self.layer1),
                ("layer1.self_attn", self.layer1.self_attn),
                ("layer1.self_attn.q_proj", self.layer1.self_attn.q_proj),
                ("layer1.self_attn.v_proj", self.layer1.self_attn.v_proj),
            ]

    model = MockModelHierarchy()
    res = attach_lora_to_model(model, target_modules=["q_proj", "v_proj"])
    assert res["num_attached"] == 4
    assert isinstance(model.layer0.self_attn.q_proj, AttachedLoRALinear)
    assert isinstance(model.layer1.self_attn.v_proj, AttachedLoRALinear)

    # Test target_layers <= 0 returns 0 attached modules
    res_zero = attach_lora_to_model(MockModelHierarchy(), target_modules=["q_proj"], target_layers=0)
    assert res_zero["num_attached"] == 0


def test_steerer_forward_hook_live_injection_and_removal():
    """Verifies SymplecticActivationSteerer registers, injects potential, and cleans up hook."""
    class HookHandle:
        def __init__(self, registry, hook):
            self.registry = registry
            self.hook = hook

        def remove(self):
            if self.hook in self.registry:
                self.registry.remove(self.hook)

    class MockHookableLayer:
        def __init__(self, d_model: int = 64):
            self.d_model = d_model
            self.hooks = []

        def register_forward_hook(self, fn):
            self.hooks.append(fn)
            return HookHandle(self.hooks, fn)

        def forward(self, x):
            out = np.asarray(x, dtype=np.float32)
            for h in self.hooks:
                out = h(self, (x,), out)
            return out

        def __call__(self, x):
            return self.forward(x)

    class CustomContainer:
        def __init__(self, items):
            self._items = items
        def __len__(self):
            return len(self._items)
        def __getitem__(self, idx):
            return self._items[idx]

    layer = MockHookableLayer(d_model=64)
    model = type("MockModel", (), {"layers": CustomContainer([layer])})()

    steerer = SymplecticActivationSteerer(d_model=64, layer_idx=0)
    x = np.ones((1, 4, 64), dtype=np.float32)

    # 1. Before steering context: unsteered
    out_before = layer(x)
    np.testing.assert_allclose(out_before, x)

    # 2. Within steering context: hook actively injects potential
    delta_q = np.zeros(12, dtype=np.float32)
    delta_q[0] = 0.5
    with steerer.steering_context(delta_q_star=delta_q, alpha_act=0.2, model=model):
        out_during = layer(x)
        assert not np.allclose(out_during, x)
        diff = np.linalg.norm(out_during - x)
        assert diff > 0.05

    # 3. After steering context: hook is cleanly removed
    out_after = layer(x)
    np.testing.assert_allclose(out_after, x)


def test_independent_judge_empty_string_and_technical_terms():
    """Verifies independent judge handles empty strings and katakana/alphanumeric keywords."""
    # 1. Empty string yields 0.0 quality
    res_empty = evaluate_llm_judge("", domain="novel", prompt="雨上がりの古寺")
    assert res_empty["normalized_quality"] == 0.0
    assert res_empty["total_score"] == 4.0

    # 2. Technical domain prompt with katakana and alphanumeric terms
    prompt_tech = "コンパクトHausdorff位相群上のHaar測度と3-SAT定理"
    text_tech = (
        "コンパクトHausdorff位相群において、Haar測度の構成は極めて自然な議論である。"
        "したがって、左不変性と正規化条件により、測度の一意性が確立される。"
        "一方で3-SAT定理との還元ステップにおける論理整合性も満たされる。"
    )
    res_tech = evaluate_llm_judge(text_tech, domain="cot_reasoning", prompt=prompt_tech)
    assert res_tech["normalized_quality"] > 60.0
    assert res_tech["logical_validity"] >= 4.0


def test_dual_ridge_regression_calibration():
    """Verifies calibrate_12_element_axes applies dual ridge regression and yields orthonormal basis (qr mode)."""
    steerer = SymplecticActivationSteerer(d_model=64, num_layers=10)
    v = steerer.calibrate_12_element_axes(ridge_lambda=1e-2, axis_mode="qr")
    assert v.shape == (64, 12)
    info = steerer.verify_orthonormality()
    assert info["is_orthonormal"] is True
    assert info["max_diag_error"] < 1e-4
    assert info["max_offdiag_error"] < 1e-4

