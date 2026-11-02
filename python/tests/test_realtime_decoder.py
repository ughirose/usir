"""Unit & Integration Tests for USIR v4.0 Realtime Decoder & LogitsProcessor.

Verifies:
1. LogitsStepper Top-K constraint, Refractory Period Gate, and Momentum updating.
2. USIRPyTorchLogitsProcessor tensor processing and batch safety.
3. run_v4_bidirectional_experiment pipeline execution across 3 domains.
4. usir_v4_bidirectional_bridge_runner.ipynb JSON syntax and cell integrity.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from usir_py.realtime_decoder import (
    LogitsStepper,
    USIRPyTorchLogitsProcessor,
    TORCH_AVAILABLE,
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
    assert steered[42] > 10.0  # Boosted


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
