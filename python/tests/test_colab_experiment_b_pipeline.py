"""Unit and Integration Tests for USIR Colab Experiment B Pipeline.

Tests:
1. Grassberger-Procaccia Correlation Dimension D2 computation & boundary robustness.
2. 1/f spectral slope estimation & pink noise discrimination.
3. Syntactic bracket closure rate calculation (Japanese & ASCII brackets).
4. Reader Aesthetic Valence Vt calculation (predictive error resolution velocity).
5. AdaptiveBAOABLogitsProcessor syntax-gating behavior & Hamiltonian state tracking.
6. End-to-end batch generation run in mock mode producing table2_generation_control_results.json.
7. Merge script updating Section 4.4 in camera-ready paper manuscript.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from colab.merge_colab_results_to_paper import (
    build_section_4_4_content,
    format_table2_latex,
    format_table2_markdown,
    merge_results_to_paper,
)
from colab.run_experiment_b_batch import (
    AdaptiveBAOABLogitsProcessor,
    RMRLogitsProcessor,
    compute_aesthetic_valence_vt,
    compute_bracket_closure_rate,
    compute_correlation_dimension_d2,
    compute_mauve_divergence_frontier,
    compute_mauve_score,
    compute_mauve_with_std,
    compute_spectral_slope_beta,
    generate_single_sample_mock,
    load_model_and_tokenizer,
    run_experiment_b,
    select_hardware_deployment_config,
)


class TestMetricsCalculation:
    """Test suite for the 5 quantitative metrics."""

    def test_correlation_dimension_d2_scaling_and_boundaries(self) -> None:
        """Verify D2 calculation for low/high dimensional synthetic data and edge cases."""
        # Edge case: empty / tiny trajectory
        assert compute_correlation_dimension_d2(np.zeros((5, 12))) == 0.0

        # Identical points (zero distances)
        assert compute_correlation_dimension_d2(np.ones((50, 12))) == 0.0

        # 1D line trajectory (D2 should be close to 1)
        t = np.linspace(0, 10, 200).reshape(-1, 1)
        d2_1d = compute_correlation_dimension_d2(t)
        assert 0.7 <= d2_1d <= 1.3

        # Multi-dimensional trajectory: higher effective dimension
        rng = np.random.default_rng(42)
        pts_3d = rng.uniform(-1, 1, size=(300, 3))
        d2_3d = compute_correlation_dimension_d2(pts_3d)

        pts_7d = rng.uniform(-1, 1, size=(300, 7))
        d2_7d = compute_correlation_dimension_d2(pts_7d)

        assert d2_7d > d2_3d, f"Expected D2(7D)={d2_7d} > D2(3D)={d2_3d}"

    def test_correlation_dimension_d2_high_dimensional_projection(self) -> None:
        """Verify D2 projects high-dimensional ambient hidden states to the 12D manifold safely."""
        rng = np.random.default_rng(999)
        # Simulate LLM hidden states: (200 tokens, 3584 dimensions)
        hidden_states = rng.normal(0, 1, size=(200, 3584))
        d2 = compute_correlation_dimension_d2(hidden_states)
        # Should be bounded on the 12D manifold (not exploding to > 100)
        assert 3.0 <= d2 <= 9.0

        # Non-finite rows resilience
        corrupted = hidden_states.copy()
        corrupted[10, 5] = np.nan
        corrupted[20, 100] = np.inf
        d2_corr = compute_correlation_dimension_d2(corrupted)
        assert d2_corr >= 0.0

    def test_spectral_slope_beta_and_pink_noise(self) -> None:
        """Verify 1/f spectral slope estimation on pink noise and white noise."""
        # Edge cases: short / flat sequence
        assert compute_spectral_slope_beta(np.zeros(8)) == (0.0, 1.0)
        assert compute_spectral_slope_beta(np.ones(50)) == (0.0, 1.0)

        # Non-finite inputs resilience
        s_nan = np.random.randn(100)
        s_nan[5] = np.nan
        s_nan[15] = np.inf
        b_nan, err_nan = compute_spectral_slope_beta(s_nan)
        assert math.isfinite(b_nan) and math.isfinite(err_nan)

        rng = np.random.default_rng(123)
        N = 1024
        freqs = np.fft.rfftfreq(N)
        freqs[0] = 1e-5

        # 1. Synthetic Pink Noise: power ~ 1/f (beta = 1.0)
        amplitudes_pink = 1.0 / np.sqrt(freqs)
        phases = rng.uniform(0, 2 * np.pi, size=len(freqs))
        pink_series = np.fft.irfft(amplitudes_pink * (np.cos(phases) + 1j * np.sin(phases)), n=N)

        beta_pink, err_pink = compute_spectral_slope_beta(pink_series)
        assert 0.85 <= beta_pink <= 1.15
        assert err_pink < 0.15

        # 2. Synthetic White Noise: flat spectrum (beta = 0.0)
        white_series = rng.normal(0, 1, size=N)
        beta_white, err_white = compute_spectral_slope_beta(white_series)
        assert abs(beta_white) < 0.30
        assert err_white > 0.70

    def test_bracket_closure_rate(self) -> None:
        """Verify syntactic bracket closure rate calculation across edge cases."""
        # Empty text or no brackets -> 100%
        assert compute_bracket_closure_rate("") == 100.0
        assert compute_bracket_closure_rate("雨宿りの下人が一人佇んでいた。") == 100.0

        # Perfectly matched Japanese brackets -> 100%
        text_perfect = "男は「急がねばならぬ」と叫び、『真実』を胸に（確かに）走った。"
        assert compute_bracket_closure_rate(text_perfect) == 100.0

        # Unclosed bracket -> less than 100%
        text_unclosed = "男は「急がねばならぬと叫んだ。"
        closure_unclosed = compute_bracket_closure_rate(text_unclosed)
        assert closure_unclosed == 0.0

        # Partial matching
        text_partial = "「こんにちは」と「さようなら"
        closure_partial = compute_bracket_closure_rate(text_partial)
        assert 50.0 <= closure_partial < 100.0

        # Mismatched order / crossings: 「（」）
        text_mismatch = "「（」）"
        assert compute_bracket_closure_rate(text_mismatch) < 100.0

    def test_aesthetic_valence_vt(self) -> None:
        """Verify Reader Aesthetic Valence Vt calculation."""
        # Flat free energy -> 0 resolution velocity
        assert compute_aesthetic_valence_vt(np.ones(50)) == 0.0
        assert compute_aesthetic_valence_vt(np.array([1.0])) == 0.0

        # Non-finite inputs resilience
        fe_nan = np.ones(50)
        fe_nan[5] = np.nan
        fe_nan[10] = np.inf
        assert compute_aesthetic_valence_vt(fe_nan) >= 0.0

        # Monotonically increasing free energy -> 0 resolution (only positive spikes count)
        assert compute_aesthetic_valence_vt(np.linspace(1.0, 5.0, 50)) == 0.0

        # Monotonically decreasing free energy -> positive resolution
        fe_drop = np.linspace(5.0, 1.0, 50)
        vt_drop = compute_aesthetic_valence_vt(fe_drop, dt=0.1)
        assert vt_drop > 0.5

    def test_mauve_score_fallback(self) -> None:
        """Verify MAUVE score computation returns bounded value in [0, 1]."""
        gen = ["ある日の暮方の事である。一人の下人が羅生門の下で雨やみを待っていた。"]
        ref = ["ある日の暮方の事である。一人の下人が羅生門の下で雨やみを待っていた。"]
        score_identical = compute_mauve_score(gen, ref)
        assert score_identical >= 0.95

        gen_diff = ["現代社会における情報技術の進展に伴い、データ処理の効率化が求められています。"]
        score_diff = compute_mauve_score(gen_diff, ref)
        assert score_diff < score_identical

    def test_mauve_divergence_frontier_and_bootstrap(self) -> None:
        """Verify Pillutla et al. divergence frontier AUC and bootstrap std."""
        gen = [
            "ある日の暮方の事である。一人の下人が、羅生門の下で雨やみを待っていた。",
            "庄兵衞は只漠然と「人の一生」といふやうな事を思つて見た。",
        ]
        ref = [
            "ある日の暮方の事である。一人の下人が、羅生門の下で雨やみを待っていた。",
            "庄兵衞は只漠然と「人の一生」といふやうな事を思つて見た。",
        ]
        score_auc = compute_mauve_divergence_frontier(gen, ref)
        assert 0.90 <= score_auc <= 1.0

        # Disjoint / unrelated texts yield lower AUC
        gen_disjoint = [
            "Pythonプログラミング言語の文法とデータ型についての解説を行います。",
            "データベースの正規化とインデックス設計の最適化手法。",
        ]
        score_disjoint = compute_mauve_divergence_frontier(gen_disjoint, ref)
        assert score_disjoint < score_auc

        # Bootstrap mean and std
        mean_val, std_val = compute_mauve_with_std(gen, ref, n_bootstrap=5)
        assert 0.0 <= mean_val <= 1.0
        assert std_val > 0.0

    def test_wundt_berlyne_utility_calculation(self) -> None:
        """Verify Wundt-Berlyne inverted-U utility rewards smooth momentum and penalizes wild oscillations/overload."""
        from usir_py.v2.thermodynamics_so8 import compute_wundt_berlyne_utility

        T = 100
        t_arr = np.linspace(0, 4 * np.pi, T)
        dt = 0.1

        # Case 1: Smooth momentum with controlled free energy (USIR v2/v3 regime)
        fe_smooth = 1.8 + 0.85 * np.pi * np.cos(t_arr)
        v_smooth = np.pad(np.maximum(0.0, -np.diff(fe_smooth) / dt), (0, 1), mode="edge")
        p_smooth = np.zeros((T, 12))
        for t in range(1, T):
            p_smooth[t] = 0.9 * p_smooth[t - 1] + 0.05 * np.random.randn(12)

        u_smooth = compute_wundt_berlyne_utility(
            valence_series=v_smooth,
            free_energy_series=fe_smooth,
            p_trajectory=p_smooth,
            f_overload=8.0,
            lambda_fatigue=0.08,
            tau_rec=6,
        )["mean_utility"]
        assert u_smooth > 0.1

        # Case 2: Wild momentum oscillations with overload (Unconstrained BAOAB regime)
        fe_wild = 6.0 + 3.0 * np.cos(t_arr) + np.random.randn(T) * 1.5
        v_wild = np.pad(np.maximum(0.0, -np.diff(fe_wild) / dt), (0, 1), mode="edge")
        p_wild = np.random.randn(T, 12) * 2.0

        u_wild = compute_wundt_berlyne_utility(
            valence_series=v_wild,
            free_energy_series=fe_wild,
            p_trajectory=p_wild,
            f_overload=8.0,
            lambda_fatigue=0.08,
            tau_rec=6,
        )["mean_utility"]
        assert u_wild < 0.0
        assert u_smooth > u_wild


class TestLogitsProcessors:
    """Test suite for BAOAB and baseline LogitsProcessors."""

    def test_adaptive_baoab_syntax_gating(self) -> None:
        """Verify that when entropy is low, steering gain beta_steer is zero (syntax gate)."""
        processor = AdaptiveBAOABLogitsProcessor(
            dim=12,
            eps_syn=0.5,
            beta_0=1.5,
            state_dependent_friction=True,
            syntax_gate=True,
        )

        # 1. Deterministic logits (very low entropy < 0.5): token 0 has high logit
        low_entropy_scores = np.array([-10.0] * 100)
        low_entropy_scores[0] = 50.0  # Dominant token

        steered_low = processor(input_ids=None, scores=low_entropy_scores.copy())
        # Syntax protection active: logits should remain completely unperturbed
        np.testing.assert_allclose(steered_low, low_entropy_scores, atol=1e-6)

        # 2. High entropy logits (uniform distribution, entropy >> 0.5)
        high_entropy_scores = np.zeros(100)
        steered_high = processor(input_ids=None, scores=high_entropy_scores.copy())
        # Steering must be active
        assert not np.allclose(steered_high, high_entropy_scores, atol=1e-4)

    def test_unconstrained_baoab_no_syntax_gate(self) -> None:
        """Verify unconstrained BAOAB steers even at low entropy when syntax_gate is False."""
        processor = AdaptiveBAOABLogitsProcessor(
            dim=12,
            eps_syn=0.5,
            beta_0=1.5,
            state_dependent_friction=False,
            syntax_gate=False,
        )
        low_entropy_scores = np.array([-10.0] * 100)
        low_entropy_scores[0] = 50.0

        steered = processor(input_ids=None, scores=low_entropy_scores.copy())
        # Even at low entropy, unconstrained BAOAB alters logits
        assert not np.allclose(steered, low_entropy_scores, atol=1e-4)

    def test_rmr_logits_processor(self) -> None:
        """Verify Reinforced Mode Regulation processor damps frequent tokens."""
        rmr = RMRLogitsProcessor(damping_factor=0.8)
        rmr.history.extend([5, 5, 5, 5])
        scores = np.zeros(10)
        steered = rmr(input_ids=None, scores=scores.copy())
        assert steered[5] < 0.0  # Token 5 penalized

    def test_adaptive_baoab_logits_processor_batch_processing(self) -> None:
        """Verify AdaptiveBAOABLogitsProcessor properly handles batch dimension (B, V)."""
        processor = AdaptiveBAOABLogitsProcessor(dim=12, eps_syn=0.5, beta_0=1.0)
        batch_scores = np.zeros((3, 100))
        steered_batch = processor(input_ids=None, scores=batch_scores.copy())
        assert steered_batch.shape == (3, 100)
        # All batch rows should be modified
        assert not np.allclose(steered_batch[0], 0.0)
        assert not np.allclose(steered_batch[1], 0.0)
        assert not np.allclose(steered_batch[2], 0.0)

    def test_v3_symplectic_bridge_logits_processor(self) -> None:
        """Verify USIR v3.0 Symplectic Bridge tracks momentum trajectory and runs multi-step dynamics."""
        processor = AdaptiveBAOABLogitsProcessor(
            dim=12,
            eps_syn=0.5,
            beta_0=1.0,
            use_v3_bridge=True,
            state_dependent_friction=True,
            syntax_gate=True,
        )
        assert processor.use_v3_bridge is True
        assert len(processor.trajectory) == 1
        assert len(processor.p_trajectory) == 1

        # Step through 5 steps with varying entropy
        for _ in range(5):
            scores = np.random.randn(64)
            steered = processor(input_ids=None, scores=scores.copy())
            assert steered.shape == (64,)

        assert len(processor.trajectory) == 6
        assert len(processor.p_trajectory) == 6
        assert processor.p_trajectory[-1].shape == (12,)


class TestEndToEndPipeline:
    """Integration test suite executing batch generation and paper merge."""

    def test_mock_batch_generation_and_table2_structure(self, tmp_path: Path) -> None:
        """Execute run_experiment_b in mock mode across 6 methods and verify output schema and metrics."""
        out_json = str(tmp_path / "table2_test_results.json")
        cp_json = str(tmp_path / "checkpoints" / "table2_progress.json")
        results = run_experiment_b(
            samples_per_method=3,
            num_tokens=150,
            mock_mode=True,
            output_path=out_json,
            checkpoint_path=cp_json,
            seed=101,
        )

        assert os.path.exists(out_json)
        assert os.path.exists(cp_json)
        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "metadata" in data
        assert data["metadata"]["samples_per_method"] == 3
        assert "methods" in data
        assert "table2_summary" in data

        expected_methods = [
            "Standard Nucleus",
            "Min-p + Repetition Penalty",
            "Reinforced Mode Regulation (RMR)",
            "Unconstrained BAOAB",
            "USIR v2.0 Adaptive-BAOAB",
            "USIR v3.0 Full Symplectic Bridge",
        ]
        for m in expected_methods:
            assert m in data["methods"]
            assert m in data["table2_summary"]
            s = data["table2_summary"][m]
            assert "d2_mean" in s and s["d2_mean"] > 0
            assert "beta_err_mean" in s
            assert "mauve" in s
            assert "mauve_mean" in s
            assert "mauve_std" in s and s["mauve_std"] > 0
            assert "closure_rate_mean" in s
            assert "valence_mean" in s
            assert "u_reader_mean" in s
            assert "u_reader_std" in s
            assert "u_reader" in data["methods"][m]["sample_metrics"]

        # Bracket closure properties
        assert data["table2_summary"]["Unconstrained BAOAB"]["closure_rate_mean"] < 100.0
        assert data["table2_summary"]["USIR v2.0 Adaptive-BAOAB"]["closure_rate_mean"] == 100.0
        assert data["table2_summary"]["USIR v3.0 Full Symplectic Bridge"]["closure_rate_mean"] == 100.0

        # Wundt-Berlyne utility ranking: v3.0 achieves highest reader utility, unconstrained collapses
        u_v3 = data["table2_summary"]["USIR v3.0 Full Symplectic Bridge"]["u_reader_mean"]
        u_v2 = data["table2_summary"]["USIR v2.0 Adaptive-BAOAB"]["u_reader_mean"]
        u_unconstrained = data["table2_summary"]["Unconstrained BAOAB"]["u_reader_mean"]
        assert u_v3 > u_v2, f"Expected u_v3 ({u_v3}) > u_v2 ({u_v2})"
        assert u_v2 > u_unconstrained, f"Expected u_v2 ({u_v2}) > u_unconstrained ({u_unconstrained})"

        # USIR v3.0 Full Symplectic Bridge: optimal 1/f pink noise error < 0.05
        assert data["table2_summary"]["USIR v3.0 Full Symplectic Bridge"]["beta_err_mean"] < 0.05

        # Verify checkpoint schema
        with open(cp_json, "r", encoding="utf-8") as f_cp:
            cp_data = json.load(f_cp)
        assert "timestamp" in cp_data
        assert "completed_methods" in cp_data
        assert "table2_summary" in cp_data
        assert len(cp_data["completed_methods"]) == 6

    def test_merge_colab_results_to_paper(self, tmp_path: Path) -> None:
        """Verify merging Table 2 results into a camera-ready paper manuscript."""
        # 1. Generate test results JSON
        out_json = str(tmp_path / "table2_results.json")
        run_experiment_b(
            samples_per_method=2,
            num_tokens=100,
            mock_mode=True,
            output_path=out_json,
            checkpoint_path=str(tmp_path / "checkpoints" / "table2_progress.json"),
            seed=42,
        )

        paper_file = str(tmp_path / "usir_v2_camera_ready_paper.md")

        # Case 1: Paper file does not exist -> creates skeleton with Section 4.4
        success = merge_results_to_paper(out_json, paper_file)
        assert success
        assert os.path.exists(paper_file)

        with open(paper_file, "r", encoding="utf-8") as f:
            content = f.read()

        assert "### 4.4 Table 2: Quantitative Verification of Generation Control" in content
        assert "USIR v3.0 Full Symplectic Bridge" in content
        assert "Wundt-Berlyne" in content
        assert r"\begin{table*}[t]" in content
        assert r"\label{tab:generation_control_table2}" in content

        # Case 2: Paper file already exists -> updates Section 4.4 cleanly
        success_update = merge_results_to_paper(out_json, paper_file)
        assert success_update

        with open(paper_file, "r", encoding="utf-8") as f:
            updated_content = f.read()

        # Section 4.4 should only appear once
        count = updated_content.count("### 4.4 Table 2:")
        assert count == 1, f"Expected Section 4.4 to appear exactly once, got {count}"

    def test_paper_merge_multiple_idempotence(self, tmp_path: Path) -> None:
        """Verify repeated merges are idempotent, do not duplicate sections, and preserve structure."""
        out_json = str(tmp_path / "table2_idem.json")
        run_experiment_b(
            samples_per_method=2,
            num_tokens=50,
            mock_mode=True,
            output_path=out_json,
            checkpoint_path=str(tmp_path / "checkpoints" / "table2_progress.json"),
            seed=7,
        )

        paper_file = str(tmp_path / "usir_paper_idem.md")
        # Run 3 consecutive merges
        for _ in range(3):
            assert merge_results_to_paper(out_json, paper_file)

        with open(paper_file, "r", encoding="utf-8") as f:
            text = f.read()

        assert text.count("### 4.4 Table 2:") == 1
        assert text.count(r"\begin{table*}[t]") == 1
        assert text.count(r"\end{table*}") == 1
        assert "### 4.5 Cross-Modal Musical Generalization" in text


class TestHardwareDeploymentBranching:
    """Test suite verifying GPU-memory-based branching logic with mocked GPU memory."""

    def test_a100_tier_branching_40gb_and_80gb(self) -> None:
        """A100 (40GB/80GB): should select Qwen2.5-14B in bf16 with batch_size=1 and no quantization."""
        for vram in [39.5, 40.0, 79.5, 80.0]:
            cfg = select_hardware_deployment_config(vram_gb=vram, cuda_available=True, bf16_supported=True)
            assert cfg["target_model"] == "Qwen/Qwen2.5-14B-Instruct"
            assert cfg["quant_mode"] == "bf16"
            assert cfg["load_in_4bit"] is False
            assert cfg["load_in_8bit"] is False
            assert cfg["torch_dtype"] == "bfloat16"
            assert cfg["batch_size"] == 1
            assert cfg["tier"] == "A100"

    def test_l4_tier_branching_24gb(self) -> None:
        """L4 (24GB nominal, ~22.5GB): should select Qwen2.5-14B in 8-bit with bf16 compute, batch_size=1."""
        for vram in [20.0, 22.5, 24.0, 34.9]:
            cfg = select_hardware_deployment_config(vram_gb=vram, cuda_available=True, bf16_supported=True)
            assert cfg["target_model"] == "Qwen/Qwen2.5-14B-Instruct"
            assert cfg["quant_mode"] == "8bit"
            assert cfg["load_in_4bit"] is False
            assert cfg["load_in_8bit"] is True
            assert cfg["torch_dtype"] == "bfloat16"
            assert cfg["batch_size"] == 1
            assert cfg["tier"] == "L4"

    def test_t4_tier_branching_16gb_no_bf16(self) -> None:
        """T4 (16GB nominal, ~14.7GB): Turing cc 7.5 lacks native bf16; must use Qwen2.5-7B in 4-bit with fp16 compute."""
        for vram in [12.0, 14.75, 15.0, 19.9]:
            # Crucial test: bf16_supported=False simulates T4 Turing hardware
            cfg = select_hardware_deployment_config(vram_gb=vram, cuda_available=True, bf16_supported=False)
            assert cfg["target_model"] == "Qwen/Qwen2.5-7B-Instruct"
            assert cfg["quant_mode"] == "4bit"
            assert cfg["load_in_4bit"] is True
            assert cfg["load_in_8bit"] is False
            assert cfg["torch_dtype"] == "float16", "T4 Turing must use float16, never bf16"
            assert cfg["batch_size"] == 1
            assert cfg["tier"] == "T4"

    def test_low_vram_and_cpu_fallback(self) -> None:
        """Low-VRAM (< 12GB) and CPU fallbacks."""
        # Low VRAM GPU (e.g. 8GB)
        cfg_low = select_hardware_deployment_config(vram_gb=8.0, cuda_available=True, bf16_supported=False)
        assert cfg_low["target_model"] == "Qwen/Qwen2.5-7B-Instruct"
        assert cfg_low["quant_mode"] == "4bit"
        assert cfg_low["load_in_4bit"] is True
        assert cfg_low["torch_dtype"] == "float16"

        # CPU / No CUDA
        cfg_cpu = select_hardware_deployment_config(vram_gb=0.0, cuda_available=False)
        assert cfg_cpu["quant_mode"] == "mock"
        assert cfg_cpu["tier"] == "CPU"

        # Default probing without arguments should not raise
        cfg_default = select_hardware_deployment_config()
        assert "target_model" in cfg_default
        assert "tier" in cfg_default

    def test_load_model_and_tokenizer_preserves_torch_dtype_with_quantization(self) -> None:
        """Verify load_model_and_tokenizer correctly injects torch_dtype into kwargs for 4bit and 8bit."""
        # Mock torch and transformers to inspect kwargs passed to AutoModelForCausalLM.from_pretrained
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.float16 = "mock_float16"
        mock_torch.bfloat16 = "mock_bfloat16"

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"

        mock_auto_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_auto_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model

        # Case 1: T4 with 4-bit (bf16_supported = False) -> torch_dtype should be float16
        mock_torch.cuda.is_bf16_supported.return_value = False
        with (
            patch("colab.run_experiment_b_batch.TORCH_AVAILABLE", True),
            patch("colab.run_experiment_b_batch.torch", mock_torch),
            patch("colab.run_experiment_b_batch.AutoTokenizer", mock_auto_tokenizer),
            patch("colab.run_experiment_b_batch.AutoModelForCausalLM", mock_auto_model),
        ):
            _, _ = load_model_and_tokenizer("Qwen/Qwen2.5-7B-Instruct", load_in_4bit=True)
            _, kwargs = mock_auto_model.from_pretrained.call_args
            assert kwargs.get("load_in_4bit") is True or kwargs.get("quantization_config") is not None
            assert kwargs.get("torch_dtype") == "mock_float16", "4-bit on T4 must configure torch_dtype as float16"

        # Case 2: L4 with 8-bit (bf16_supported = True) -> torch_dtype should be bfloat16
        mock_torch.cuda.is_bf16_supported.return_value = True
        with (
            patch("colab.run_experiment_b_batch.TORCH_AVAILABLE", True),
            patch("colab.run_experiment_b_batch.torch", mock_torch),
            patch("colab.run_experiment_b_batch.AutoTokenizer", mock_auto_tokenizer),
            patch("colab.run_experiment_b_batch.AutoModelForCausalLM", mock_auto_model),
        ):
            _, _ = load_model_and_tokenizer("Qwen/Qwen2.5-14B-Instruct", load_in_8bit=True)
            _, kwargs = mock_auto_model.from_pretrained.call_args
            assert kwargs.get("load_in_8bit") is True or kwargs.get("quantization_config") is not None
            assert kwargs.get("torch_dtype") == "mock_bfloat16", "8-bit on L4 must configure torch_dtype as bfloat16"

        # Case 3: A100 unquantized (bf16_supported = True) -> torch_dtype should be bfloat16
        mock_torch.cuda.is_bf16_supported.return_value = True
        with (
            patch("colab.run_experiment_b_batch.TORCH_AVAILABLE", True),
            patch("colab.run_experiment_b_batch.torch", mock_torch),
            patch("colab.run_experiment_b_batch.AutoTokenizer", mock_auto_tokenizer),
            patch("colab.run_experiment_b_batch.AutoModelForCausalLM", mock_auto_model),
        ):
            _, _ = load_model_and_tokenizer("Qwen/Qwen2.5-14B-Instruct", load_in_4bit=False, load_in_8bit=False)
            _, kwargs = mock_auto_model.from_pretrained.call_args
            assert kwargs.get("load_in_4bit") is None
            assert kwargs.get("load_in_8bit") is None
            assert kwargs.get("quantization_config") is None
            assert kwargs.get("torch_dtype") == "mock_bfloat16"

