"""Tests for Phase 33: independent-axis calibration, steering-vector cache, inject_positions, and the Phase 33 driver.

Torch-dependent tests use importorskip (the local uv env has no torch). The end-to-end test uses a tiny randomly
initialised Qwen2 with a character tokenizer on CPU; it verifies code paths only, not effect sizes.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from colab import run_phase33_phase_c as drv
from usir_py import activation_steerer as act
from usir_py.activation_steerer import SymplecticActivationSteerer
from usir_py.corpus_40_benchmark import CORPUS


# --------------------------------------------------------------------------------------------------
# axis extraction (numpy only)
# --------------------------------------------------------------------------------------------------

def _planted(n=60, d=100, seed=0, noise=0.02):
    rng = np.random.RandomState(seed)
    z = rng.randn(n, 12)
    common = rng.randn(d)
    a = rng.randn(12, d) + 1.0 * common  # correlated planted directions
    h = z @ a + noise * rng.randn(n, d)
    truth = np.linalg.pinv(a)                # (d, 12): H @ truth[:, k] == z_k exactly (noise-free)
    return z, h, truth


def _cos_cols(v, truth):
    return np.array([float(v[:, k] @ truth[:, k] / (np.linalg.norm(v[:, k]) * np.linalg.norm(truth[:, k]))) for k in range(12)])


def test_independent_axes_recover_planted_directions_better_than_qr(monkeypatch):
    z, h, truth = _planted()
    texts = [f"t{i}" for i in range(len(z))]
    table = {t: z[i] for i, t in enumerate(texts)}
    monkeypatch.setattr(act, "extract_usir_24d", lambda t: {"q": table[t]})
    st = SymplecticActivationSteerer(d_model=h.shape[1])
    v_ind, info = st.calibrate_12_element_axes(corpus=texts, hidden_states_list=list(h), ridge_lambda=1e-3,
                                               axis_mode="independent", return_info=True)
    v_qr = st.calibrate_12_element_axes(corpus=texts, hidden_states_list=list(h), ridge_lambda=1e-3, axis_mode="qr")
    c_ind, c_qr = _cos_cols(v_ind, truth), _cos_cols(v_qr, truth)
    assert c_ind.min() > 0.8, c_ind
    assert c_ind.mean() > c_qr.mean() + 0.1, (c_ind.mean(), c_qr.mean())
    # unit-norm, non-orthogonal columns
    np.testing.assert_allclose(np.linalg.norm(v_ind, axis=0), 1.0, atol=1e-5)
    assert np.max(np.abs(v_ind.T @ v_ind - np.eye(12))) > 0.05
    # calibration info: per-axis in-sample and 5-fold CV R^2
    assert info["axis_mode"] == "independent"
    assert len(info["r2_in_sample"]) == 12 and len(info["r2_cv"]) == 12 and info["cv_folds"] == 5
    assert min(info["r2_cv"]) > 0.9
    assert all(a >= c - 1e-9 for a, c in zip(info["r2_in_sample"], info["r2_cv"]))
    assert st.calibration_info["axis_mode"] == "qr"  # last call was qr


def test_axis_k_depends_only_on_q_k(monkeypatch):
    z, h, _ = _planted(seed=1)
    texts = [f"t{i}" for i in range(len(z))]
    table = {t: z[i].copy() for i, t in enumerate(texts)}
    monkeypatch.setattr(act, "extract_usir_24d", lambda t: {"q": table[t]})
    st = SymplecticActivationSteerer(d_model=h.shape[1])
    v1 = st.calibrate_12_element_axes(corpus=texts, hidden_states_list=list(h))
    for t in texts:
        table[t][5] = table[t][5] ** 3  # perturb element 5 only
    v2 = st.calibrate_12_element_axes(corpus=texts, hidden_states_list=list(h))
    for k in range(12):
        if k != 5:
            np.testing.assert_allclose(v1[:, k], v2[:, k], atol=1e-5)
    assert np.linalg.norm(v1[:, 5] - v2[:, 5]) > 1e-3


def test_invalid_axis_mode_rejected():
    st = SymplecticActivationSteerer(d_model=32)
    with pytest.raises(ValueError):
        st.calibrate_12_element_axes(axis_mode="nope")


def test_numpy_last_position_only():
    st = SymplecticActivationSteerer(d_model=16, inject_positions="last")
    h = np.random.RandomState(0).randn(2, 5, 16).astype(np.float32)
    dq = np.ones(12, dtype=np.float32)
    out = st.steer_tensor(h, dq, 0.3)
    np.testing.assert_array_equal(out[:, :-1], h[:, :-1])
    assert np.abs(out[:, -1] - h[:, -1]).max() > 1e-3
    full = st.steer_tensor(h, dq, 0.3, positions="all")
    np.testing.assert_allclose(out[:, -1], full[:, -1], atol=1e-6)


# --------------------------------------------------------------------------------------------------
# torch: cache and inject_positions
# --------------------------------------------------------------------------------------------------

def test_cached_device_tensor_reused_and_invalidated(monkeypatch):
    torch = pytest.importorskip("torch")
    st = SymplecticActivationSteerer(d_model=32)
    dq = np.linspace(-1, 1, 12).astype(np.float32)
    h = torch.randn(1, 4, 32)
    calls = {"n": 0}
    orig = torch.as_tensor

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(torch, "as_tensor", counting)
    st.steer_tensor(h, dq, 0.1)
    v1 = st._steer_cache["vec"]
    st.steer_tensor(h, dq, 0.1)
    st.steer_tensor(h * 2, dq, 0.1)
    assert st._steer_cache["vec"] is v1
    assert calls["n"] == 1  # only the first call built a tensor (one host->device transfer)
    assert v1.dtype == torch.float32 and v1.device == h.device
    st.steer_tensor(h, dq, 0.2)               # alpha change
    v2 = st._steer_cache["vec"]
    assert v2 is not v1 and calls["n"] == 2
    st.steer_tensor(h, dq * 2, 0.2)           # delta_q change
    v3 = st._steer_cache["vec"]
    assert v3 is not v2 and calls["n"] == 3
    st.v_steer = st._initialize_random_orthonormal(32, 12, seed=7)  # V change
    assert st._steer_cache is None
    st.steer_tensor(h, dq * 2, 0.2)
    assert st._steer_cache["vec"] is not v3 and calls["n"] == 4
    # values match the functional definition
    out = st.steer_tensor(h, dq, 0.1)
    expect = h + 0.1 * (h.norm(dim=-1, keepdim=True) + 1e-8) * torch.tensor(st.v_steer @ dq)
    torch.testing.assert_close(out, expect, atol=1e-5, rtol=1e-5)


def test_inject_positions_last_changes_only_last_position():
    torch = pytest.importorskip("torch")
    st = SymplecticActivationSteerer(d_model=32)
    dq = np.ones(12, dtype=np.float32)
    h = torch.randn(2, 6, 32)
    last = st.steer_tensor(h, dq, 0.3, positions="last")
    allp = st.steer_tensor(h, dq, 0.3, positions="all")
    assert torch.equal(last[:, :-1], h[:, :-1])
    assert (last[:, -1] - h[:, -1]).abs().max() > 1e-3
    torch.testing.assert_close(last[:, -1], allp[:, -1])
    assert (allp[:, :-1] - h[:, :-1]).abs().max() > 1e-3
    # single-token step (KV-cache decoding): last == all
    h1 = torch.randn(2, 1, 32)
    torch.testing.assert_close(st.steer_tensor(h1, dq, 0.3, positions="last"), st.steer_tensor(h1, dq, 0.3, positions="all"))

    # through the forward hook
    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

        def forward(self, x):
            for layer in self.model.layers:
                x = layer(x)
            return x

    m = M()
    st2 = SymplecticActivationSteerer(d_model=32, layer_idx=0)
    with st2.steering_context(dq, 0.3, model=m, inject_positions="last"):
        out = m(h)
    assert torch.equal(out[:, :-1], h[:, :-1]) and (out[:, -1] - h[:, -1]).abs().max() > 1e-3
    with st2.steering_context(dq, 0.3, model=m):  # default "all"
        out_all = m(h)
    assert (out_all[:, :-1] - h[:, :-1]).abs().max() > 1e-3
    assert torch.equal(m(h), h)  # hook removed


# --------------------------------------------------------------------------------------------------
# driver (pure helpers)
# --------------------------------------------------------------------------------------------------

def test_prompts_disjoint_generic_and_flat():
    assert len(drv.EVAL_PROMPTS) == 12 and len(drv.TRAIN_PROMPTS) == 12
    assert not set(drv.EVAL_PROMPTS) & set(drv.TRAIN_PROMPTS)
    banned = {str(c.get("author", "")) for c in CORPUS} | {str(c.get("title", "")) for c in CORPUS}
    banned = {b for b in banned if len(b) >= 2}
    for p in drv.EVAL_PROMPTS + drv.TRAIN_PROMPTS:
        assert not any(b in p for b in banned), p
    assert drv.Cfg().max_new_tokens == 400 and drv.Cfg().grpo_max_new_tokens == 256



def test_literary_prompt_set_generic_and_selectable():
    lit = drv.PROMPT_SETS["literary"]
    assert len(lit) == 12 and len(set(lit)) == 12
    assert not set(lit) & set(drv.EVAL_PROMPTS + drv.TRAIN_PROMPTS)
    banned = {str(c.get("author", "")) for c in CORPUS} | {str(c.get("title", "")) for c in CORPUS}
    banned = {b for b in banned if len(b) >= 2}
    for p in lit:
        assert not any(b in p for b in banned), p
    assert drv.Cfg().prompt_set == "expository" and drv.PROMPT_SETS["expository"] is drv.EVAL_PROMPTS


def test_store_rollouts_option_wired():
    import inspect
    from usir_py.lora_grpo_forge import run_authentic_grpo_lora_forge
    sig = inspect.signature(run_authentic_grpo_lora_forge)
    assert sig.parameters["store_rollouts"].default is False
    assert drv.Cfg().store_rollouts is False
    src = inspect.getsource(drv.run_grpo_stage33)
    assert "store_rollouts=cfg.store_rollouts" in src
    assert "strip_tail=cfg.reward_strip_tail" in src and drv.Cfg().reward_strip_tail is False
    body = inspect.getsource(run_authentic_grpo_lora_forge)
    assert 'step_entry["rollouts"] = step_rollouts' in body and "ppls.append(ppl_ref)" in body

def test_render_chat_fallback_and_enable_thinking():
    class NoTemplate:
        pass

    assert drv.render_chat(NoTemplate(), "abc") == "abc"

    class Tok:
        chat_template = "x"

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
            return f"<u>{msgs[0]['content']}</u>think={kw.get('enable_thinking')}"

    assert drv.render_chat(Tok(), "q") == "<u>q</u>think=False"

    class OldTok(Tok):
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return "old"

    assert drv.render_chat(OldTok(), "q") == "old"


def test_summary_has_both_deadzone_definitions():
    cfg = drv.Cfg()
    recs = []
    for i, text in enumerate(["雨が降っていた。窓の外は暗い。男は黙って茶を飲んだ。夜が明けた。", "手順は次のとおりです。まず電源を入れます。次に設定を確認します。最後に結果を保存して終わります。"]):
        m = drv.p28.text_metrics(text, "x")
        recs.append({"prompt_idx": i, "error": None, "wall_ms": 1.0, "n_new_tokens": 5, "ms_per_token": 0.2,
                     "metrics": m, "prompt": "p", "text": text})
    cond = {"records": recs}
    drv.finalize_condition(cond, ["u0", "u1"], cfg)
    s = cond["summary"]
    assert s["deadzone_classifier"]["n"] == 2 and s["deadzone_cv"]["n"] == 2
    assert s["deadzone_cv"]["wilson95"] is not None
    for r in recs:
        assert r["metrics"]["in_deadzone_cv"] == (r["metrics"]["len_cv"] < 0.28)


def test_source_has_no_mock_or_hard_coded_outcomes():
    src = open(drv.__file__, encoding="utf-8").read()
    for pat in (r"generate_mock_sample", r"rng\.normal", r"np\.random\.default_rng"):
        assert not re.search(pat, src), pat


# --------------------------------------------------------------------------------------------------
# driver end-to-end on a tiny random model (CPU)
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
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                         pad_token_id=0, eos_token_id=1, bos_token_id=1)
    return tr.Qwen2ForCausalLM(cfg).eval(), CharTok()


def test_phase33_driver_end_to_end_tiny_cpu():
    pytest.importorskip("torch")
    from usir_py.lora_grpo_forge import AttachedLoRALinear
    model, tok = _build_tiny()
    w_before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    cfg = drv.Cfg(model_id="tiny-random-qwen2", dtype="fp32", device="cpu", n_prompts=2, max_new_tokens=30,
                  alphas=(0.1, 0.3), inject_positions=("all", "last"), latency_reps=2, latency_tokens=8,
                  grpo_steps=2, grpo_prompts_per_step=1, grpo_rollouts=2, grpo_max_new_tokens=16, ppl_calib_samples=3,
                  lora_r=4, lora_alpha=8.0, also_unmerged_gen=True)
    res = drv.run_all(model, tok, cfg, log=lambda m: None)
    assert res["errors"] == [], res["errors"]
    assert res["provenance"]["mock_used"] is False and res["schema"] == drv.SCHEMA
    st = res["stages"]
    assert st["base"]["summary"]["n_ok"] == 2
    assert "deadzone_cv" in st["base"]["summary"] and "deadzone_classifier" in st["base"]["summary"]
    conds = st["steering"]["conditions"]
    expected = {f"steer_{k}_{m}_alpha_{a}" for k in ("cal", "rand") for m in ("all", "last") for a in (0.1, 0.3)}
    assert set(conds) == expected
    cal_info = st["steering"]["calibration"]["calibration"]
    assert cal_info["axis_mode"] == "independent" and len(cal_info["r2_cv"]) == 12 and len(cal_info["r2_in_sample"]) == 12
    for c in conds.values():
        assert c["summary"]["n_ok"] == 2
        assert "in_deadzone_cv" in c["paired_vs_base"] and c["paired_vs_base"]["n_pairs"] == 2
        assert all("ref_ppl_completion" in r for r in c["records"] if r["error"] is None)
    g = st["grpo_lora"]
    assert g["base_trainable_params_before_attach"] == 0
    assert g["train"]["result"]["mode"] == "authentic_pytorch" and len(g["train"]["result"]["steps"]) == 2
    assert g["trainable_params"] == g["lora"]["total_lora_params"]
    assert g["ppl_barrier"]["threshold"] > 0 and g["ppl_barrier"]["n_samples"] >= 1
    assert g["lora_merged"]["summary"]["n_ok"] == 2 and g["lora_unmerged"]["summary"]["n_ok"] == 2
    ck = res["checks"]
    assert ck["steer_hook_max_abs_logit_change"] > 0.0
    assert ck["steer_hook_removed_max_abs_logit_diff"] == 0.0
    assert ck["lora_original_vs_pretraining_max_abs_logit_diff"] == 0.0
    assert ck["lora_merged_vs_unmerged_max_abs_logit_diff"] < 1e-3
    for lat in (st["steering"]["latency"], g["latency"]):
        for entry in lat["states"].values():
            assert len(entry["series"]) == 2
    back = drv.p28.decode_archive_line(drv.p28.build_archive_line(res))
    assert back["model_id"] == "tiny-random-qwen2"
    # model restored to base weights and unwrapped
    assert not any(isinstance(m, AttachedLoRALinear) for m in model.modules())
    for k, v in model.state_dict().items():
        assert k in w_before and v.shape == w_before[k].shape
        assert not k.endswith("lora_A") and not k.endswith("lora_B")
        assert (v == w_before[k]).all(), k


def _batch_tok(tok):
    """Wraps the character tokenizer with list input and left/right padding (as a HF tokenizer would)."""
    torch = pytest.importorskip("torch")
    tr = pytest.importorskip("transformers")

    class BatchTok:
        pad_token_id, eos_token_id = tok.pad_token_id, tok.eos_token_id
        pad_token, eos_token = "<pad>", "<eos>"
        padding_side = "right"

        def __call__(self, text, return_tensors="pt", padding=False, **kw):
            if isinstance(text, str):
                return tok(text)
            rows = [tok(t)["input_ids"][0].tolist() for t in text]
            L = max(map(len, rows))
            pad = lambda r: ([0] * (L - len(r)) + r) if self.padding_side == "left" else (r + [0] * (L - len(r)))
            mask = lambda r: ([0] * (L - len(r)) + [1] * len(r)) if self.padding_side == "left" else ([1] * len(r) + [0] * (L - len(r)))
            return tr.BatchEncoding({"input_ids": torch.tensor([pad(r) for r in rows]),
                                     "attention_mask": torch.tensor([mask(r) for r in rows])})

        def decode(self, ids, skip_special_tokens=True):
            return tok.decode(ids, skip_special_tokens=skip_special_tokens)

    return BatchTok()


def test_batched_generation_left_padding_and_records():
    torch = pytest.importorskip("torch")
    model, tok = _build_tiny()
    btok = _batch_tok(tok)
    prompts = [drv.EVAL_PROMPTS[0], drv.EVAL_PROMPTS[1][:20], drv.EVAL_PROMPTS[2][:35]]
    # left padding keeps position -1 on each prompt's real last token: last-position logits match unpadded runs
    btok.padding_side = "left"
    enc = btok(prompts, padding=True)
    btok.padding_side = "right"
    with torch.no_grad():
        lb = model(**enc).logits[:, -1, :]
        for i, p in enumerate(prompts):
            l1 = model(**tok(p)).logits[:, -1, :]
            assert torch.allclose(lb[i], l1[0], atol=1e-4)
    cfg = drv.Cfg(model_id="tiny", dtype="fp32", device="cpu", n_prompts=3, max_new_tokens=12, gen_batch=2)
    cond = drv.run_condition_batched("base", model, btok, prompts, cfg, log=lambda m: None)
    recs = cond["records"]
    assert [r["prompt_idx"] for r in recs] == [0, 1, 2] and all(r["error"] is None for r in recs)
    assert all(0 <= r["n_new_tokens"] <= 12 and r["batch_size"] in (1, 2) for r in recs)
    assert recs[0]["seed"] == recs[1]["seed"] == cfg.seed and recs[2]["seed"] == cfg.seed + 2
    assert btok.padding_side == "right"  # restored


def test_gen_batch_dispatch(monkeypatch):
    called = {}
    monkeypatch.setattr(drv, "run_condition_batched", lambda *a, **k: called.setdefault("b", {"name": "x", "records": [], "summary": {}}))
    monkeypatch.setattr(drv.p28, "run_condition", lambda *a, **k: called.setdefault("s", {"name": "x", "records": [], "summary": {}}))
    monkeypatch.setattr(drv, "finalize_condition", lambda *a, **k: None)
    drv.run_condition33("x", None, None, [], [], drv.Cfg(gen_batch=1), print)
    assert "s" in called and "b" not in called
    drv.run_condition33("x", None, None, [], [], drv.Cfg(gen_batch=4), print)
    assert "b" in called
    assert drv.Cfg().gen_batch == 1


def test_per_row_steering_matches_steering_context():
    torch = pytest.importorskip("torch")
    model, tok = _build_tiny()
    st = SymplecticActivationSteerer(d_model=64, num_layers=4, layer_ratio=0.65)
    st.unit_direction = True
    dq = np.linspace(-1, 1, 12).astype(np.float32)
    enc = tok(drv.EVAL_PROMPTS[0])
    with torch.no_grad():
        l0 = model(**enc).logits
        for pos in ("last", "all"):
            with st.steering_context(dq, 0.3, model=model, inject_positions=pos):
                ref = model(**enc).logits
            st._steer_cache = None
            vec = st._cached_steer_vec("cpu", dq, 0.3)[None, :]
            with drv.per_row_steering(st, model, vec, pos):
                got = model(**enc).logits
            assert torch.allclose(got, ref, atol=1e-5) and not torch.allclose(got, l0)
        with drv.per_row_steering(st, model, torch.zeros(1, 64), "last"):
            assert torch.allclose(model(**enc).logits, l0, atol=1e-6)  # zero row = unsteered
        assert torch.allclose(model(**enc).logits, l0)  # hook removed


def test_mixed_steer_end_to_end_tiny_cpu():
    pytest.importorskip("torch")
    model, tok = _build_tiny()
    btok = _batch_tok(tok)
    cfg = drv.Cfg(model_id="tiny", dtype="fp32", device="cpu", n_prompts=3, max_new_tokens=10, alphas=(0.1, 0.3),
                  inject_positions=("last",), latency_reps=1, latency_tokens=4, do_grpo=False, steer_unit_dir=True,
                  gen_batch=5, mixed_steer=True)
    res = drv.run_all(model, btok, cfg, log=lambda m: None)
    assert res["errors"] == [], res["errors"]
    conds = res["stages"]["steering"]["conditions"]
    assert set(conds) == {f"steer_{k}_last_alpha_{a}" for k in ("cal", "rand") for a in (0.1, 0.3)}
    for name, c in conds.items():
        assert [r["prompt_idx"] for r in c["records"]] == [0, 1, 2] and c["summary"]["n_ok"] == 3
        assert c["direction"] == ("calibrated" if "_cal_" in name else "random_haar")
        assert all(r["mixed_batch"] for r in c["records"]) and c["paired_vs_base"]["n_pairs"] == 3
