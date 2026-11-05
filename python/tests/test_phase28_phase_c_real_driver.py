"""Tests for python/colab/run_phase28_phase_c_real.py.

Torch-free tests cover the pure helpers. The end-to-end test builds a tiny randomly initialised Qwen2 model with a
character tokenizer on CPU and runs every stage of the driver; it is skipped when torch/transformers are missing.
It verifies code paths and integrity checks only; it says nothing about the effect size on a real model.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from colab import run_phase28_phase_c_real as drv
from usir_py.corpus_40_benchmark import CORPUS

SAMPLE = "雨が降っていた。窓の外は暗く、遠くで犬が鳴いた。男は黙って茶を飲んだ。やがて、夜が明けた。"


def test_prompts_disjoint_and_generic():
    assert len(drv.EVAL_PROMPTS) >= 12 and len(drv.TRAIN_PROMPTS) >= 8
    assert not set(drv.EVAL_PROMPTS) & set(drv.TRAIN_PROMPTS)
    banned = {str(c.get("author", "")) for c in CORPUS} | {str(c.get("title", "")) for c in CORPUS}
    banned = {b for b in banned if len(b) >= 2}
    for p in drv.EVAL_PROMPTS + drv.TRAIN_PROMPTS:
        assert not any(b in p for b in banned), p


def test_text_metrics_fields_and_deadzone_uses_repo_classifier():
    m = drv.text_metrics(SAMPLE, "雨が降っていた。")
    for k in ("len_cv", "sigma_norm", "phase_regime", "in_deadzone", "grpo_reward_no_ppl", "q", "p"):
        assert k in m
    assert len(m["q"]) == 12 and len(m["p"]) == 12
    assert m["in_deadzone"] == (m["phase_regime"] == "Phase_I_Deadzone")
    assert all(np.isfinite(m[k]) for k in ("len_cv", "sigma_norm", "d_3phase"))
    # Deadzone corpus chunks are classified by the repository classifier, not by a driver-local rule
    dead = [drv.text_metrics(c["text"], "")["in_deadzone"] for c in CORPUS if c["phase"] == "V_Deadzone"]
    assert all(dead)


def test_summarize_counts_errors_and_does_not_substitute():
    ok = {"prompt_idx": 0, "error": None, "wall_ms": 100.0, "n_new_tokens": 10, "ms_per_token": 10.0,
          "metrics": drv.text_metrics(SAMPLE, "x")}
    bad = {"prompt_idx": 1, "error": "RuntimeError('boom')"}
    s = drv.summarize([ok, bad])
    assert s["n_total"] == 2 and s["n_ok"] == 1 and s["n_errors"] == 1
    assert s["ms_per_token"]["n"] == 1
    assert s["deadzone"]["n"] == 1
    empty = drv.summarize([bad])
    assert empty["n_ok"] == 0 and empty["len_cv"]["mean"] is None and empty["deadzone"]["rate"] is None


def test_archive_line_roundtrip_and_nonfinite_sanitised():
    obj = {"a": np.float32(1.5), "b": float("nan"), "c": [np.int64(3)], "d": "日本語"}
    line = drv.build_archive_line(obj)
    assert re.fullmatch(r"ARCHIVE_B64 [0-9a-f]{8} \S+", line)
    back = drv.decode_archive_line(line)
    assert back == {"a": 1.5, "b": None, "c": [3], "d": "日本語"}


def test_source_has_no_hard_coded_expected_outcomes():
    src = open(drv.__file__, encoding="utf-8").read()
    for pat in (r"escape\s*>=", r"generate_mock_sample", r"rng\.normal", r"np\.random\.default_rng"):
        assert not re.search(pat, src), pat


# --------------------------------------------------------------------------------------------------
# End-to-end on a tiny random model (CPU)
# --------------------------------------------------------------------------------------------------

def _build_tiny():
    torch = pytest.importorskip("torch")
    tr = pytest.importorskip("transformers")
    chars = set("".join(drv.EVAL_PROMPTS + drv.TRAIN_PROMPTS) + "".join(c["text"] for c in CORPUS)) | set("。、！？「」")
    chars |= {chr(c) for c in range(0x3041, 0x30FF)}
    vocab = ["<pad>", "<eos>", "<unk>"] + sorted(chars)
    idx = {c: i for i, c in enumerate(vocab)}

    class CharTok:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, text, return_tensors="pt", **kw):
            ids = torch.tensor([[idx.get(c, 2) for c in text]], dtype=torch.long)
            return tr.BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

        def decode(self, ids, skip_special_tokens=True):
            ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
            return "".join(vocab[i] for i in ids if not (skip_special_tokens and i < 3))

    torch.manual_seed(0)
    cfg = tr.Qwen2Config(vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                         pad_token_id=0, eos_token_id=1, bos_token_id=1)
    model = tr.Qwen2ForCausalLM(cfg).eval()
    return model, CharTok()


def test_end_to_end_tiny_model_cpu():
    pytest.importorskip("torch")
    model, tok = _build_tiny()
    cfg = drv.Cfg(model_id="tiny-random-qwen2", dtype="fp32", device="cpu", n_prompts=2, max_new_tokens=40,
                  alphas=(0.1, 0.3), latency_reps=2, latency_tokens=8, grpo_steps=2, grpo_prompts_per_step=1,
                  grpo_rollouts=2, grpo_max_new_tokens=24, lora_r=4, lora_alpha=8.0, also_unmerged_gen=True)
    res = drv.run_all(model, tok, cfg, log=lambda m: None)
    assert res["errors"] == [], res["errors"]
    st = res["stages"]
    assert st["base"]["summary"]["n_ok"] == 2
    assert set(st["steering"]["conditions"]) >= {"steer_alpha_0.1", "steer_alpha_0.3", "steer_random_direction_alpha_0.3"}
    assert st["steering"]["calibration"]["orthonormality"]["is_orthonormal"]
    g = st["grpo_lora"]
    assert g["train"]["result"]["mode"] == "authentic_pytorch"
    assert len(g["train"]["result"]["steps"]) == 2
    assert g["trainable_params"] == g["lora"]["total_lora_params"]
    assert g["lora_merged"]["summary"]["n_ok"] == 2
    ck = res["checks"]
    assert ck["steer_hook_max_abs_logit_change"] > 0.0
    assert ck["steer_hook_removed_max_abs_logit_diff"] == 0.0
    assert ck["lora_original_vs_pretraining_max_abs_logit_diff"] == 0.0
    assert ck["lora_merged_vs_unmerged_max_abs_logit_diff"] < 1e-3
    for lat in (st["steering"]["latency"], g["latency"]):
        for entry in lat["states"].values():
            assert len(entry["series"]) == 2
            assert all(s["n_new_tokens"] == 8 for s in entry["series"])
    # archive line carries the same structure
    back = drv.decode_archive_line(drv.build_archive_line(res))
    assert back["model_id"] == "tiny-random-qwen2"
    # the model is restored to its original weights and unwrapped at the end of the GRPO stage
    assert not any(isinstance(m, drv.AttachedLoRALinear) for m in model.modules())


def test_stage_failure_is_recorded_not_substituted():
    pytest.importorskip("torch")
    model, tok = _build_tiny()
    cfg = drv.Cfg(model_id="tiny", dtype="fp32", device="cpu", n_prompts=1, max_new_tokens=8, do_grpo=False,
                  alphas=(0.1,), steer_control=False, latency_reps=1, latency_tokens=4)

    class Boom:
        def __init__(self, m):
            self.m = m

        def __getattr__(self, k):
            return getattr(self.m, k)

        def __call__(self, *a, **k):
            return self.m(*a, **k)

        def generate(self, *a, **k):
            raise RuntimeError("boom")

    res = drv.run_all(Boom(model), tok, cfg, log=lambda m: None)
    recs = res["stages"]["base"]["records"]
    assert recs[0]["error"] is not None and "metrics" not in recs[0]
    assert res["stages"]["base"]["summary"]["n_ok"] == 0
