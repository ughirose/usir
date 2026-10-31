"""Unit and Integration Tests for USIR Colab Experiment C (Table 3 Style-LoRA Forge Pipeline).

Tests:
1. Loading and Clean-Room / Copyright integrity of the 200-pair preference dataset.
2. Generative dynamics and latency overheads of the 3 Table 3 comparison methods.
3. 1/f spectral slope beta estimation from token surprise sequences and MAUVE proxy.
4. End-to-end execution of Experiment C in mock mode with checkpointing and Table 3 output.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import numpy as np
import pytest

from colab.run_experiment_c_lora_forge import (
    METHOD_BASE,
    METHOD_INFERENCE_BAOAB,
    METHOD_LORA_FORGE,
    TABLE3_METHODS,
    compute_mauve_proxy,
    compute_spectral_slope_beta_seq,
    evaluate_generation,
    generate_mock_sample,
    load_preference_dataset,
    run_experiment_c,
)
from usir_py.corpus_40_benchmark import CORPUS


class TestExperimentCDatasetAndMetrics:
    """Verifies dataset loading and physical metric evaluation."""

    def test_preference_dataset_loading_and_cleanroom_verification(self) -> None:
        """Verifies that the default dataset has exactly 200 pairs and 0 copyright violations."""
        records = load_preference_dataset()
        assert len(records) == 200, f"Expected 200 pairs, got {len(records)}"

        protected_authors = {"Shiga_Naoya", "Kawabata_Yasunari", "Mishima_Yukio"}
        for r in records:
            assert r.get("cleanroom_7gram_passed") is True, f"Cleanroom fail at {r.get('pair_id')}"
            assert r.get("author") not in protected_authors, f"Protected author in {r.get('pair_id')}"
            assert "prompt" in r and len(r["prompt"]) > 10
            assert "chosen" in r and len(r["chosen"]) > 20
            assert "rejected" in r and len(r["rejected"]) > 20
            assert r.get("reward_margin", 0.0) >= 0.45

    def test_spectral_slope_beta_seq(self) -> None:
        """Verifies 1/f beta estimation on white noise vs pink noise sequences."""
        rng = np.random.default_rng(123)
        n = 128

        # Flat white noise: beta should be close to 0.0, error |beta - 1.0| close to 1.0
        white = rng.normal(3.5, 0.3, size=n)
        b_white, err_white = compute_spectral_slope_beta_seq(white)
        assert abs(b_white) < 0.5
        assert err_white > 0.5

        # Pink noise (1/f): beta should be close to 1.0, error |beta - 1.0| small (< 0.45)
        white_fft = np.fft.rfft(rng.normal(0, 1, size=n))
        freqs = np.fft.rfftfreq(n)
        freqs[0] = 1.0
        pink_fft = white_fft / np.sqrt(freqs)
        pink = np.fft.irfft(pink_fft, n=n)
        b_pink, err_pink = compute_spectral_slope_beta_seq(pink)
        assert 0.5 < b_pink < 1.5
        assert err_pink < 0.45

    def test_mauve_proxy_boundaries(self) -> None:
        """Verifies MAUVE proxy output bounds in (0, 1]."""
        feats_a = [np.ones(24) * 0.5 for _ in range(5)]
        feats_b = [np.ones(24) * 0.5 for _ in range(5)]
        feats_c = [np.ones(24) * 2.5 for _ in range(5)]

        # Identical distributions => score 1.0
        score_ident = compute_mauve_proxy(feats_a, feats_b)
        assert pytest.approx(score_ident, abs=1e-4) == 1.0

        # Disjoint distributions => lower score
        score_diff = compute_mauve_proxy(feats_a, feats_c)
        assert 0.0 <= score_diff < score_ident


class TestExperimentCGenerativeDynamics:
    """Verifies the physical divergence among the 3 comparison methods."""

    def test_mock_sample_divergence_and_latencies(self) -> None:
        """Verifies mock completions across Base, Adaptive-BAOAB, and Style-LoRA Forge."""
        prompt = "静かな山寺の夕暮れについて描写せよ。"
        target_phase = "III_Void"

        # 1. Base Model
        base_sample = generate_mock_sample(prompt, METHOD_BASE, target_phase, seed=1)
        assert base_sample["overhead_ms"] == 0.0
        assert "text" in base_sample

        # 2. Adaptive-BAOAB
        baoab_sample = generate_mock_sample(prompt, METHOD_INFERENCE_BAOAB, target_phase, seed=2)
        assert baoab_sample["overhead_ms"] > 3.5
        assert baoab_sample["latency_per_token_ms"] > base_sample["latency_per_token_ms"]

        # 3. USIR Style-LoRA Forge
        forge_sample = generate_mock_sample(prompt, METHOD_LORA_FORGE, target_phase, seed=3)
        assert forge_sample["overhead_ms"] == 0.0

    def test_method_evaluations_and_deadzone_escape(self) -> None:
        """Verifies that Base Model is trapped while LoRA Forge and Adaptive-BAOAB escape."""
        masterwork_texts = [c["text"] for c in CORPUS if c["phase"] != "V_Deadzone"]
        prompt = "激しい嵐の夜の葛藤を描写せよ。"
        target_phase = "II_Critical"

        # Base Model
        base_s = generate_mock_sample(prompt, METHOD_BASE, target_phase, seed=10)
        base_eval = evaluate_generation(
            base_s["text"], target_phase, base_s["overhead_ms"], masterwork_texts, base_s.get("surprise_seq")
        )
        assert base_eval["overhead_ms"] == 0.0
        assert base_eval["escaped_deadzone"] is False
        assert base_eval["len_cv"] < 0.30
        assert base_eval["grpo_reward"] < 0.25
        assert "phi_void" in base_eval

        # Inference-Time Adaptive-BAOAB
        baoab_s = generate_mock_sample(prompt, METHOD_INFERENCE_BAOAB, target_phase, seed=20)
        baoab_eval = evaluate_generation(
            baoab_s["text"], target_phase, baoab_s["overhead_ms"], masterwork_texts, baoab_s.get("surprise_seq")
        )
        assert baoab_eval["overhead_ms"] > 3.5
        assert baoab_eval["escaped_deadzone"] is True
        assert baoab_eval["len_cv"] > 0.50
        assert baoab_eval["grpo_reward"] >= 0.35
        assert baoab_eval["phi_void"] > base_eval["phi_void"]

        # USIR Style-LoRA Forge
        forge_s = generate_mock_sample(prompt, METHOD_LORA_FORGE, target_phase, seed=30)
        forge_eval = evaluate_generation(
            forge_s["text"], target_phase, forge_s["overhead_ms"], masterwork_texts, forge_s.get("surprise_seq")
        )
        assert forge_eval["overhead_ms"] == 0.0
        assert forge_eval["escaped_deadzone"] is True
        assert forge_eval["len_cv"] > 0.50
        assert forge_eval["grpo_reward"] >= 0.35
        assert forge_eval["phi_void"] > base_eval["phi_void"]


class TestExperimentCPipelineEndToEnd:
    """Verifies end-to-end batch execution, checkpointing, and Table 3 output schema."""

    def test_end_to_end_mock_run(self, tmp_path: Path) -> None:
        """Executes full Experiment C pipeline in mock mode with 4 iterations."""
        ckpt_path = tmp_path / "test_table3_progress.json"
        out_path = tmp_path / "test_table3_results.json"

        results = run_experiment_c(
            checkpoint_path=ckpt_path,
            output_path=out_path,
            iterations=4,
            mock_mode=True,
            force_restart=True,
        )

        # Checkpoint and output files must exist
        assert ckpt_path.exists()
        assert out_path.exists()

        # Schema assertions
        assert "experiment" in results
        assert "table3_markdown" in results
        assert results["dataset_num_pairs"] == 200
        assert results["cleanroom_zero_copyright_verified"] is True
        assert len(results["methods"]) == 3

        # Method metrics assertions
        base_res = results["methods"][METHOD_BASE]
        baoab_res = results["methods"][METHOD_INFERENCE_BAOAB]
        forge_res = results["methods"][METHOD_LORA_FORGE]

        # Base Model: 0.00 ms overhead, falls into Deadzone
        assert base_res["overhead_ms"] == 0.0
        assert base_res["deadzone_escape_rate"] <= 0.10
        assert base_res["len_cv_mean"] < 0.30
        assert "phi_void_mean" in base_res

        # Inference-Time Adaptive-BAOAB: ~4.2 ms overhead, escapes Deadzone
        assert baoab_res["overhead_ms"] >= 3.5
        assert baoab_res["deadzone_escape_rate"] >= 0.90
        assert baoab_res["len_cv_mean"] > 0.52
        assert "phi_void_mean" in baoab_res

        # USIR Style-LoRA Forge: 0.00 ms overhead, escapes Deadzone >= 92%, CV_L > 0.52
        assert forge_res["overhead_ms"] == 0.0
        assert forge_res["deadzone_escape_rate"] >= 0.92
        assert forge_res["len_cv_mean"] > 0.52
        assert forge_res["phi_void_mean"] > base_res["phi_void_mean"]
        assert forge_res["grpo_reward_mean"] > base_res["grpo_reward_mean"]
        assert forge_res["mauve"] > base_res["mauve"]

        # Markdown Table formatting
        md_table = results["table3_markdown"]
        assert "Latency Overhead" in md_table
        assert "Deadzone Escape Rate" in md_table
        assert "Void Condensation" in md_table
        assert METHOD_BASE in md_table
        assert METHOD_INFERENCE_BAOAB in md_table
        assert METHOD_LORA_FORGE in md_table
