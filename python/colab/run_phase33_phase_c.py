"""Phase 33 Phase C: real-model measurement on an evaluation set that induces Deadzone (Colab, single GPU).

Follow-up to python/colab/run_phase28_phase_c_real.py (Phase 28). Reuses its helpers by import. Differences:

  * Evaluation set: 12 chat-template instruct prompts asking for flat explanatory / summary prose (no author or
    character names), ~400 new tokens each, sampled (T=0.8, top_p=0.95). The Phase 28 raw literary openings gave a
    base Deadzone rate of 0/12 on Qwen3-8B, so no steering or training effect could be detected.
  * Deadzone is reported in two ways: (i) repository classifier
    compute_three_phase_qda_distance(domain='novel').phase_regime == 'Phase_I_Deadzone'; (ii) CV_L < 0.28.
  * Steering: axes from independent per-element ridge regressions (axis_mode='independent'), calibrated vs
    seeded-Haar random control at the same alphas, inject_positions as a config. Delta q* is expressed in
    standardised-q units and normalised to unit length, so alpha is the injected fraction of ||h|| along V dq*.
  * GRPO-LoRA: 256-token rollouts on instruct prompts disjoint from the evaluation prompts, physical reward
    times the reference-perplexity barrier M_semantic (reference = same model object, LoRA disabled, completion
    tokens only); threshold = 90th percentile of the base model's own completion PPL on the training prompts.
    Base weights are frozen before the adapters are attached.

No pass/fail thresholds are defined here. A stage that raises is recorded in result["errors"]; failed values are
never replaced by defaults.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

_current_dir = Path(__file__).resolve().parent
_python_dir = _current_dir.parent
for _p in (str(_python_dir), str(_current_dir)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_phase28_phase_c_real as p28  # noqa: E402  (sibling module; also bundled next to this file in the Colab zip)
from usir_py.activation_steerer import SymplecticActivationSteerer  # noqa: E402
from usir_py.corpus_40_benchmark import CORPUS  # noqa: E402
from usir_py.extractor_24d import extract_usir_24d  # noqa: E402
from usir_py.lora_grpo_forge import attach_lora_to_model, run_authentic_grpo_lora_forge, set_lora_enabled, strip_incomplete_tail  # noqa: E402

torch = p28.torch
TORCH_AVAILABLE = p28.TORCH_AVAILABLE

SCHEMA = "phase33_phase_c_v1"
DEFAULT_MODEL_ID = "Qwen/Qwen3-8B"
DEADZONE_REGIME = p28.DEADZONE_REGIME

PROSE_SUFFIX = "箇条書きや見出しは使わず、ですます調の文章で800字程度にまとめてください。"

# Generic task prompts (no author, work or character names). Evaluation and training sets are disjoint.
EVAL_TASKS: List[str] = [
    "電子レンジを安全に使うための手順と注意点を説明してください。",
    "図書館で本を借りるときの手続きの流れを説明してください。",
    "光合成のしくみについて、中学生にもわかるように説明してください。",
    "リサイクルの意義と家庭でできる取り組みについて要約してください。",
    "パスワードを安全に管理する方法を説明してください。",
    "日本の四季の特徴について要約してください。",
    "洗濯機で衣類を洗うときの手順を説明してください。",
    "地震が起きたときの避難の流れを順を追って説明してください。",
    "複利計算の考え方を例を挙げて説明してください。",
    "健康的な睡眠習慣を身につけるためのポイントを要約してください。",
    "インターネットで情報を調べるときの注意点を説明してください。",
    "天気予報で使われる降水確率の意味を説明してください。",
]
TRAIN_TASKS: List[str] = [
    "自転車のパンクを修理する手順を説明してください。",
    "コンビニエンスストアの仕組みについて要約してください。",
    "水の循環について、小学生にもわかるように説明してください。",
    "銀行口座を開設するときの流れを説明してください。",
    "電車の乗り換えを調べる方法を説明してください。",
    "日本の祝日の成り立ちについて要約してください。",
    "炊飯器でご飯を炊く手順を説明してください。",
    "火災報知器が鳴ったときの対応を順を追って説明してください。",
    "インフレーションとは何かを例を挙げて説明してください。",
    "運動不足を解消するための方法を要約してください。",
    "電子メールを書くときのマナーを説明してください。",
    "気温と湿度の関係について説明してください。",
]
EVAL_PROMPTS: List[str] = [t + PROSE_SUFFIX for t in EVAL_TASKS]
TRAIN_PROMPTS: List[str] = [t + PROSE_SUFFIX for t in TRAIN_TASKS]

# Phase 39: literary scene prompts (no author, work or character names), same length instruction as above
LITERARY_SUFFIX = "小説の地の文として、だ・である調で800字程度で書いてください。見出しや箇条書きは使わないでください。"
LITERARY_TASKS: List[str] = [
    "雨の降る夕方、駅のホームで電車を待つ人の様子を描いてください。",
    "古い家を片づけていて、子どものころの手紙を見つける場面を描いてください。",
    "夜の病院の廊下で、家族の手術が終わるのを待つ場面を描いてください。",
    "冬の朝、海沿いの小さな町を歩く場面を描いてください。",
    "長く会っていなかった友人と喫茶店で再会する場面を描いてください。",
    "山道で道に迷い、日が暮れていく場面を描いてください。",
    "祭りの夜、人混みの中で連れとはぐれる場面を描いてください。",
    "引っ越しの前日、空になった部屋で過ごす場面を描いてください。",
    "停電した夜、ろうそくの明かりで家族が過ごす場面を描いてください。",
    "夏の午後、川辺で釣りをする老人を描いてください。",
    "試験の結果を知らせる手紙を開ける直前の場面を描いてください。",
    "嵐が過ぎた翌朝、荒れた庭を見て回る場面を描いてください。",
]
LITERARY_PROMPTS: List[str] = [t + LITERARY_SUFFIX for t in LITERARY_TASKS]
PROMPT_SETS = {"expository": EVAL_PROMPTS, "literary": LITERARY_PROMPTS}


@dataclasses.dataclass
class Cfg(p28.Cfg):
    model_id: str = DEFAULT_MODEL_ID
    max_new_tokens: int = 400
    grpo_max_new_tokens: int = 256
    # steering
    inject_positions: Sequence[str] = ("all", "last")
    axis_mode: str = "independent"
    ridge_lambda: float = 0.1            # relative ridge (lam = ridge_lambda * mean diag of H H^T) in independent mode
    dq_normalise: bool = True            # unit-length Delta q* in standardised-q units
    match_control_norm: bool = True      # scale the random-control alpha so that ||V_rand dq|| == ||V_cal dq||
    # deadzone definition (ii)
    cv_deadzone_threshold: float = 0.28
    # GRPO reference-perplexity barrier
    ppl_percentile: float = 90.0
    ppl_calib_samples: int = 12
    alpha_ppl: float = 1.0
    score_ref_ppl: bool = True
    # Phase 36
    steer_unit_dir: bool = False         # inject alpha * ||h|| * (V dq)/||V dq|| (both calibrated and random)
    grpo_reward: str = "multiplicative"  # "multiplicative" (Phase 33) | "shaped" (see shaped_reward)
    reward_strip_tail: bool = False     # Phase 42: shaped reward on complete sentences only
    store_rollouts: bool = False        # Phase 40: keep GRPO rollout texts, rewards and ref PPL in the result
    prompt_set: str = "expository"      # evaluation prompts: "expository" (Phase 33-38) | "literary" (Phase 39)
    gen_batch: int = 1                  # >1: evaluation prompts generated together in one left-padded generate() call

    def as_dict(self) -> Dict[str, Any]:
        d = super().as_dict()
        d["inject_positions"] = list(self.inject_positions)
        return d


# ---------------------------------------------------------------------------
# prompts / chat template
# ---------------------------------------------------------------------------

def render_chat(tok: Any, user_text: str) -> str:
    """Chat-template string with the generation prompt appended (Qwen3: enable_thinking=False when supported).

    Tokenizers without a chat template (e.g. the character tokenizer used in the tests) get the plain text.
    """
    if getattr(tok, "chat_template", None) is None or not hasattr(tok, "apply_chat_template"):
        return user_text
    msgs = [{"role": "user", "content": user_text}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# metrics / summaries
# ---------------------------------------------------------------------------

def finalize_condition(cond: Dict[str, Any], user_prompts: Sequence[str], cfg: Cfg) -> Dict[str, Any]:
    """Adds the CV_L-based Deadzone flag and the Wilson interval for it; recomputes the summary."""
    for r in cond["records"]:
        if r.get("error") is None:
            r["user_prompt"] = user_prompts[r["prompt_idx"]]
            r["metrics"]["in_deadzone_cv"] = bool(r["metrics"]["len_cv"] < cfg.cv_deadzone_threshold)
    cond["summary"] = summarize33(cond["records"], cfg)
    return cond


def summarize33(records: List[Dict[str, Any]], cfg: Cfg) -> Dict[str, Any]:
    out = p28.summarize(records)
    ok = [r for r in records if r.get("error") is None and "in_deadzone_cv" in r.get("metrics", {})]
    k = sum(1 for r in ok if r["metrics"]["in_deadzone_cv"])
    out["deadzone_classifier"] = out["deadzone"]
    out["deadzone_cv"] = {"definition": f"CV_L < {cfg.cv_deadzone_threshold}", "count": k, "n": len(ok),
                          "rate": (k / len(ok)) if ok else None, "wilson95": p28.wilson_interval(k, len(ok))}
    out["n_hit_token_limit"] = sum(1 for r in ok if r.get("n_new_tokens", 0) >= cfg.max_new_tokens)
    return out


def paired33(cond: List[Dict[str, Any]], base: List[Dict[str, Any]]) -> Dict[str, Any]:
    res = p28.paired_vs_base(cond, base)
    bmap = {r["prompt_idx"]: r for r in base if r.get("error") is None}
    pairs = [(bmap[r["prompt_idx"]], r) for r in cond if r.get("error") is None and r["prompt_idx"] in bmap]

    def add(key: str, g: Callable[[Dict[str, Any]], Optional[float]]) -> None:
        d = np.asarray([g(c) - g(b) for b, c in pairs if g(b) is not None and g(c) is not None], dtype=float)
        res[key] = {"mean_diff": float(d.mean()) if d.size else None, "std_diff": float(d.std(ddof=1)) if d.size > 1 else None,
                    "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()), "n_zero": int((d == 0).sum())}

    add("in_deadzone_cv", lambda r: 1.0 if r["metrics"].get("in_deadzone_cv") else 0.0)
    if all("ref_ppl_completion" in r for _, r in pairs) and pairs:
        add("ref_ppl_completion", lambda r: r.get("ref_ppl_completion"))
    return res


def batched_generate(model: Any, tok: Any, prompts: Sequence[str], cfg: Cfg, seed: int) -> List[Dict[str, Any]]:
    """One sampled generate() call for several prompts (left padding, so position -1 is each prompt's real last token
    and the 'last' steering hook stays correct).  Batch-1 decoding of an 8B model is bound by per-step overhead and
    weight reads, so B rows cost about the same wall time per step as one row.

    Differences from the batch-1 path: one torch seed per batch (cfg.seed + first prompt index) instead of one per
    prompt, so samples are not identical to batch-1 runs with the same seed; wall time is shared by the rows."""
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    old_side = getattr(tok, "padding_side", "right")
    old_pad = tok.pad_token
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        enc = tok(list(prompts), return_tensors="pt", padding=True).to(cfg.device)
    finally:
        tok.padding_side = old_side
        tok.pad_token = old_pad
    plen = int(enc["input_ids"].shape[1])
    torch.manual_seed(int(seed))
    p28._sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=int(cfg.max_new_tokens), do_sample=True, temperature=cfg.temperature,
                             top_p=cfg.top_p, pad_token_id=pad_id)
    p28._sync()
    wall_ms = (time.perf_counter() - t0) * 1000.0
    eos = {int(x) for x in ([tok.eos_token_id] if isinstance(tok.eos_token_id, int) else (tok.eos_token_id or []))}
    eos |= {int(pad_id)}
    gen_cfg_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if gen_cfg_eos is not None:
        eos |= {int(x) for x in ([gen_cfg_eos] if isinstance(gen_cfg_eos, int) else gen_cfg_eos)}
    steps = int(out.shape[1]) - plen
    res = []
    for row in out[:, plen:].tolist():
        n = next((i for i, t in enumerate(row) if t in eos), len(row))  # tokens before the first EOS / pad
        res.append({"text": tok.decode(row[:n], skip_special_tokens=True), "n_new_tokens": n,
                    "wall_ms": wall_ms, "ms_per_token": (wall_ms / steps) if steps > 0 else None,
                    "batch_size": len(prompts), "batch_steps": steps})
    return res


def run_condition_batched(name: str, model: Any, tok: Any, prompts: Sequence[str], cfg: Cfg, log: Callable[[str], None],
                          ctx_factory: Optional[Callable[[], Any]] = None) -> Dict[str, Any]:
    """Same records as p28.run_condition, generated cfg.gen_batch prompts at a time.  ms_per_token is per decode step
    of the whole batch (wall of the batch / steps), not per row."""
    factory = ctx_factory or contextlib.nullcontext
    B = int(cfg.gen_batch)
    records: List[Dict[str, Any]] = []
    for s in range(0, len(prompts), B):
        idx = list(range(s, min(s + B, len(prompts))))
        try:
            with factory():
                gs = batched_generate(model, tok, [prompts[i] for i in idx], cfg, seed=cfg.seed + s)
            for i, g in zip(idx, gs):
                rec = {"prompt_idx": i, "prompt": prompts[i], "seed": cfg.seed + s, "error": None, **g}
                rec["metrics"] = p28.text_metrics(g["text"], prompts[i])
                records.append(rec)
            log(f"[{name}] batch {idx[0]}-{idx[-1]} steps={gs[0]['batch_steps']} wall={gs[0]['wall_ms'] / 1000:.1f}s")
        except Exception as exc:
            for i in idx:
                records.append({"prompt_idx": i, "prompt": prompts[i], "seed": cfg.seed + s, "error": repr(exc),
                                "traceback": traceback.format_exc()[-1500:]})
            log(f"[{name}] batch {idx[0]}-{idx[-1]} ERROR {exc!r}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return {"name": name, "records": records, "summary": p28.summarize(records)}


def run_condition33(name: str, model: Any, tok: Any, user_prompts: Sequence[str], rendered: Sequence[str], cfg: Cfg,
                    log: Callable[[str], None], base_records: Optional[List[Dict[str, Any]]] = None,
                    ctx_factory: Optional[Callable[[], Any]] = None) -> Dict[str, Any]:
    if int(getattr(cfg, "gen_batch", 1)) > 1:
        cond = run_condition_batched(name, model, tok, rendered, cfg, log, ctx_factory=ctx_factory)
    else:
        cond = p28.run_condition(name, model, tok, rendered, cfg, log, ctx_factory=ctx_factory)
    finalize_condition(cond, user_prompts, cfg)
    if base_records is not None:
        cond["paired_vs_base"] = paired33(cond["records"], base_records)
    return cond


# ---------------------------------------------------------------------------
# reference perplexity (completion tokens only)
# ---------------------------------------------------------------------------

def completion_ppl(model: Any, tok: Any, prompt: str, completion: str, device: str) -> Optional[float]:
    """exp(mean token NLL) of `completion` given `prompt` under the model as it currently is."""
    import torch.nn.functional as F
    enc = tok(prompt + completion, return_tensors="pt").to(device)
    plen = int(tok(prompt, return_tensors="pt")["input_ids"].shape[1])
    with torch.no_grad():
        logits = model(**enc).logits[:, :-1, :].float()
    labels = enc["input_ids"][:, 1:]
    s = max(0, plen - 1)
    if s >= labels.shape[1]:
        return None
    ce = F.cross_entropy(logits[:, s:, :].reshape(-1, logits.size(-1)), labels[:, s:].reshape(-1))
    return float(torch.exp(ce).item())


def add_ref_ppl(model: Any, tok: Any, cfg: Cfg, cond: Dict[str, Any]) -> None:
    for r in cond["records"]:
        if r.get("error") is None:
            try:
                r["ref_ppl_completion"] = completion_ppl(model, tok, r["prompt"], r["text"], cfg.device)
            except Exception as exc:
                r["ref_ppl_error"] = repr(exc)
    vals = [r["ref_ppl_completion"] for r in cond["records"] if r.get("ref_ppl_completion") is not None]
    cond["summary"]["ref_ppl_completion"] = p28._stats(vals)


def calibrate_ppl_threshold(model: Any, tok: Any, cfg: Cfg, train_rendered: Sequence[str],
                            log: Callable[[str], None]) -> Dict[str, Any]:
    """Threshold for the M_semantic barrier: percentile of the (LoRA-free) base model's own completion PPL."""
    ppls: List[float] = []
    torch.manual_seed(cfg.seed + 7)
    for i in range(int(cfg.ppl_calib_samples)):
        prompt = train_rendered[i % len(train_rendered)]
        enc = tok(prompt, return_tensors="pt").to(cfg.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=int(cfg.grpo_max_new_tokens), do_sample=True, temperature=0.85,
                                 pad_token_id=(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id))
        text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        v = completion_ppl(model, tok, prompt, text, cfg.device)
        if v is not None and np.isfinite(v):
            ppls.append(v)
    if not ppls:
        raise RuntimeError("PPL calibration produced no samples")
    thr = float(np.percentile(ppls, cfg.ppl_percentile))
    info = {"percentile": cfg.ppl_percentile, "threshold": thr, "alpha_ppl": cfg.alpha_ppl, "n_samples": len(ppls),
            "base_ppl_samples": ppls, "stats": p28._stats(ppls),
            "definition": "exp(mean NLL of completion tokens) under the base model (LoRA absent), 256-token samples at T=0.85 on the training prompts; barrier m = 1/(1+exp(alpha*(ppl-threshold)))"}
    log(f"[grpo] ppl barrier threshold={thr:.3f} (p{cfg.ppl_percentile:g} of {len(ppls)} base samples, median {info['stats']['median']:.3f})")
    return info


# ---------------------------------------------------------------------------
# steering
# ---------------------------------------------------------------------------

def calibrate_steerer33(model: Any, tok: Any, cfg: Cfg, log: Callable[[str], None]) -> Dict[str, Any]:
    hf_cfg = model.config
    d_model = int(hf_cfg.hidden_size)
    n_layers = int(hf_cfg.num_hidden_layers)
    st = SymplecticActivationSteerer(d_model=d_model, num_layers=n_layers, layer_ratio=cfg.steer_layer_ratio)
    texts = [c["text"] for c in CORPUS]
    hs: List[np.ndarray] = []
    for t in texts:
        enc = tok(t, return_tensors="pt").to(cfg.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        # output of the hooked decoder layer = hidden_states[layer_idx + 1]; position 0 (attention sink) excluded
        h = out.hidden_states[st.layer_idx + 1][0].float()
        h = h[1:] if h.shape[0] > 1 else h
        hs.append(h.mean(dim=0).cpu().numpy())
    _, cal = st.calibrate_12_element_axes(hidden_states_list=hs, corpus=texts, ridge_lambda=cfg.ridge_lambda,
                                          axis_mode=cfg.axis_mode, return_info=True)
    gram = st.v_steer.T @ st.v_steer
    info = {
        "d_model": d_model, "num_layers": n_layers, "layer_idx": st.layer_idx, "n_calibration_texts": len(texts),
        "calibration": cal, "column_cosine_max_abs_offdiag": float(np.max(np.abs(gram - np.diag(np.diag(gram))))),
        "column_norms": [float(x) for x in np.linalg.norm(st.v_steer, axis=0)],
    }
    if "r2_cv" in cal:
        info["r2_cv_mean"] = float(np.mean(cal["r2_cv"]))
        info["r2_in_sample_mean"] = float(np.mean(cal["r2_in_sample"]))
    log(f"[steer] calibrated mode={cfg.axis_mode} layer={st.layer_idx} "
        f"r2_cv_mean={info.get('r2_cv_mean')} r2_in_mean={info.get('r2_in_sample_mean')}")
    return {"steerer": st, "info": info}


def run_steering_stage33(model: Any, tok: Any, cfg: Cfg, user_prompts: Sequence[str], rendered: Sequence[str],
                         base_stage: Dict[str, Any], log: Callable[[str], None], checks: Dict[str, Any]) -> Dict[str, Any]:
    cal = calibrate_steerer33(model, tok, cfg, log)
    st: SymplecticActivationSteerer = cal["steerer"]
    base_q = base_stage["summary"].get("q_mean")
    if base_q is None:
        raise RuntimeError("no successful base generations: cannot define delta_q")
    target_q = np.mean([extract_usir_24d(c["text"])["q"] for c in CORPUS if c["phase"] in ("II_Critical", "III_Void")], axis=0)
    dq_raw = (np.asarray(target_q, dtype=float) - np.asarray(base_q, dtype=float))
    if st.q_std is not None:
        dq = np.where(st.q_std > 1e-12, dq_raw / np.where(st.q_std > 1e-12, st.q_std, 1.0), 0.0)  # standardised-q units
    else:
        dq = dq_raw
    dq_norm_before = float(np.linalg.norm(dq))
    if cfg.dq_normalise and dq_norm_before > 0:
        dq = dq / dq_norm_before
    dq = dq.astype(np.float32)

    rnd = SymplecticActivationSteerer(d_model=st.d_model, num_layers=st.num_layers, layer_ratio=cfg.steer_layer_ratio)
    rnd.v_steer = rnd._initialize_random_orthonormal(st.d_model, 12, seed=int(cfg.seed))  # seeded Haar
    st.unit_direction = bool(cfg.steer_unit_dir)
    rnd.unit_direction = bool(cfg.steer_unit_dir)
    n_cal = float(np.linalg.norm(st.steer_direction(dq)))
    n_rnd = float(np.linalg.norm(rnd.steer_direction(dq)))
    ctl_scale = (n_cal / n_rnd) if (cfg.match_control_norm and n_rnd > 0) else 1.0

    stage: Dict[str, Any] = {
        "calibration": cal["info"],
        "delta_q": {"value": [float(x) for x in dq], "norm_before_normalisation": dq_norm_before,
                    "normalised_to_unit": bool(cfg.dq_normalise),
                    "definition": "(mean q of CORPUS chunks in {II_Critical, III_Void} - mean q of stage (a) generations) / q_std of the calibration corpus",
                    "target_q": [float(x) for x in target_q], "base_q": [float(x) for x in base_q],
                    "norm_V_cal_dq": n_cal, "norm_V_random_dq": n_rnd, "control_alpha_scale": ctl_scale,
                    "norm_V_cal_dq_raw": float(np.linalg.norm(st.v_steer @ dq)), "unit_direction": bool(cfg.steer_unit_dir)},
        "conditions": {},
    }
    modes = [m for m in cfg.inject_positions]
    a_mid = float(cfg.alphas[min(1, len(cfg.alphas) - 1)])

    # integrity check: the hook changes logits and is fully removed afterwards
    probe = rendered[0]
    l0 = p28._probe_logits(model, tok, probe, cfg)
    with st.steering_context(dq, a_mid, model=model, inject_positions=modes[0]):
        l1 = p28._probe_logits(model, tok, probe, cfg)
    l2 = p28._probe_logits(model, tok, probe, cfg)
    checks["steer_hook_max_abs_logit_change"] = float((l1 - l0).abs().max())
    checks["steer_hook_removed_max_abs_logit_diff"] = float((l2 - l0).abs().max())

    for mode in modes:
        for a in cfg.alphas:
            name = f"steer_cal_{mode}_alpha_{a:g}"
            c = run_condition33(name, model, tok, user_prompts, rendered, cfg, log, base_stage["records"],
                                ctx_factory=lambda a=a, mode=mode: st.steering_context(dq, a, model=model, inject_positions=mode))
            c.update(alpha_act=float(a), inject_positions=mode, direction="calibrated")
            stage["conditions"][name] = c
            if cfg.steer_control:
                a_r = float(a) * ctl_scale
                name = f"steer_rand_{mode}_alpha_{a:g}"
                c = run_condition33(name, model, tok, user_prompts, rendered, cfg, log, base_stage["records"],
                                    ctx_factory=lambda a_r=a_r, mode=mode: rnd.steering_context(dq, a_r, model=model, inject_positions=mode))
                c.update(alpha_act=float(a), alpha_effective=a_r, inject_positions=mode, direction="random_haar")
                stage["conditions"][name] = c
    stage["latency"] = p28.latency_bench(
        model, tok, rendered[0],
        {"base": contextlib.nullcontext,
         f"steer_{modes[0]}_alpha_{a_mid:g}": lambda: st.steering_context(dq, a_mid, model=model, inject_positions=modes[0])}, cfg, log)
    return stage


# ---------------------------------------------------------------------------
# GRPO-LoRA
# ---------------------------------------------------------------------------

SHAPED_REWARD_DEF = ("Phase 36 shaped reward: clip(CV_L / 0.45, 0.05, 1.2) * cadence_alive * M_semantic, "
                     "CV_L and cadence_alive from extract_usir_24d, M_semantic = 1/(1+exp(alpha_ppl*(ppl_ref-threshold)))")


def make_shaped_reward(ppl_thresh: float, alpha_ppl: float,
                       strip_tail: bool = False) -> Callable[[str, float], float]:
    """strip_tail (Phase 37): score only the complete sentences (drop the fragment cut by the token limit)."""
    def fn(text: str, ppl_ref: float) -> float:
        if strip_tail and text:
            text = strip_incomplete_tail(text)
        if not text or not text.strip():
            return 0.0
        f = extract_usir_24d(text)
        arg = float(np.clip(alpha_ppl * (float(ppl_ref) - float(ppl_thresh)), -50.0, 50.0))
        m_sem = 1.0 / (1.0 + math.exp(arg))
        return float(np.clip(float(f["len_cv"]) / 0.45, 0.05, 1.2) * float(f["cadence_alive"]) * m_sem)
    return fn


def run_grpo_stage33(model: Any, tok: Any, cfg: Cfg, user_prompts: Sequence[str], rendered: Sequence[str],
                     train_user: Sequence[str], train_rendered: Sequence[str], base_stage: Dict[str, Any],
                     probe_logits_base: Any, log: Callable[[str], None], checks: Dict[str, Any]) -> Dict[str, Any]:
    for p in model.parameters():
        p.requires_grad_(False)  # base weights frozen BEFORE the adapters are attached
    n_base_trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    assert n_base_trainable == 0
    ppl_info = calibrate_ppl_threshold(model, tok, cfg, train_rendered, log)
    info = attach_lora_to_model(model, target_modules=("q_proj", "v_proj"), r=cfg.lora_r, alpha=cfg.lora_alpha)
    n_trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    if n_trainable != int(info["total_lora_params"]) or info["num_attached"] == 0:
        raise RuntimeError(f"trainable params {n_trainable} != LoRA params {info['total_lora_params']} (attached={info['num_attached']})")
    model.eval()
    if cfg.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(cfg.seed)
    items = [{"prompt": r, "x_0": u} for r, u in zip(train_rendered, train_user)]
    t0 = time.perf_counter()
    res = run_authentic_grpo_lora_forge(
        model, tokenizer=tok, prompts=items, num_rollouts=cfg.grpo_rollouts, beta_kl=cfg.grpo_beta_kl,
        learning_rate=cfg.grpo_lr, num_steps=cfg.grpo_steps, device=cfg.device, domain="novel",
        ppl_thresh=ppl_info["threshold"], alpha_ppl=cfg.alpha_ppl, ppl_completion_only=True,
        max_new_tokens=cfg.grpo_max_new_tokens, batched_rollouts=True, per_rollout_backward=True,
        prompts_per_step=cfg.grpo_prompts_per_step, store_rollouts=cfg.store_rollouts,
        reward_fn=(make_shaped_reward(ppl_info["threshold"], cfg.alpha_ppl, strip_tail=cfg.reward_strip_tail) if cfg.grpo_reward == "shaped" else None),
        progress_fn=lambda e: log(f"[grpo] step {e['step']}/{cfg.grpo_steps} reward={e['mean_reward']:.4f} kl={e['mean_kl']:.5f} loss={e['loss']:.5f} {e['sec']:.1f}s"))
    train_wall = time.perf_counter() - t0
    if res.get("mode") != "authentic_pytorch":
        raise RuntimeError(f"GRPO ran in mode {res.get('mode')!r}, not authentic_pytorch")
    stage: Dict[str, Any] = {
        "lora": {k: info[k] for k in ("num_attached", "total_lora_params", "r", "alpha")},
        "trainable_params": n_trainable, "base_trainable_params_before_attach": n_base_trainable,
        "train_prompts": list(TRAIN_PROMPTS), "ppl_barrier": ppl_info,
        "reward": ("compute_multiplicative_grpo_reward (physical terms) x M_semantic(ref PPL barrier), see ppl_barrier"
                   if cfg.grpo_reward != "shaped" else SHAPED_REWARD_DEF),
        "train": {"result": res, "wall_sec": train_wall,
                  "peak_cuda_mem_gb": (torch.cuda.max_memory_allocated() / 2 ** 30) if cfg.device.startswith("cuda") else None},
    }
    sw = p28.LoraStateSwitcher(model)
    probe = rendered[0]
    sw.set("unmerged"); l_un = p28._probe_logits(model, tok, probe, cfg)
    sw.set("merged"); l_me = p28._probe_logits(model, tok, probe, cfg)
    sw.set("original"); l_or = p28._probe_logits(model, tok, probe, cfg)
    checks["lora_merged_vs_unmerged_max_abs_logit_diff"] = float((l_me - l_un).abs().max())
    checks["lora_original_vs_pretraining_max_abs_logit_diff"] = float((l_or - probe_logits_base).abs().max())
    checks["lora_effect_merged_vs_original_max_abs_logit_diff"] = float((l_me - l_or).abs().max())

    if cfg.also_unmerged_gen:
        sw.set("unmerged")
        stage["lora_unmerged"] = run_condition33("lora_unmerged", model, tok, user_prompts, rendered, cfg, log, base_stage["records"])
    sw.set("merged")
    stage["lora_merged"] = run_condition33("lora_merged", model, tok, user_prompts, rendered, cfg, log, base_stage["records"])
    stage["latency"] = p28.latency_bench(
        model, tok, rendered[0],
        {"original": lambda: sw.ctx("original"), "lora_unmerged": lambda: sw.ctx("unmerged"), "lora_merged": lambda: sw.ctx("merged")},
        cfg, log)
    sw.set("original")  # model is back to base weights; used for reference-PPL scoring of every condition
    return stage


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def run_all(model: Any, tok: Any, cfg: Cfg, log: Callable[[str], None] = print) -> Dict[str, Any]:
    t_all = time.perf_counter()
    prompts = PROMPT_SETS[cfg.prompt_set]
    n = max(1, min(cfg.n_prompts, len(prompts)))
    user_prompts = prompts[:n]
    rendered = [render_chat(tok, u) for u in user_prompts]
    train_user = list(TRAIN_PROMPTS)
    train_rendered = [render_chat(tok, u) for u in train_user]
    result: Dict[str, Any] = {
        "schema": SCHEMA,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_id": cfg.model_id,
        "config": cfg.as_dict(),
        "provenance": {
            "mock_used": False,
            "thresholds_defined_by_driver": {"cv_deadzone_threshold": cfg.cv_deadzone_threshold,
                                             "ppl_barrier_percentile": cfg.ppl_percentile},
            "deadzone_definitions": {
                "classifier": "usir_py.manifold_geodesic.compute_three_phase_qda_distance(domain='novel').phase_regime == 'Phase_I_Deadzone'",
                "cv": f"len_cv (sentence-length CV, extractor definition) < {cfg.cv_deadzone_threshold}",
            },
            "generation_input": "Qwen chat template, enable_thinking=False when supported; sampled with per-prompt seeds shared across conditions",
            "eval_prompt_example_rendered": rendered[0],
        },
        "env": {
            "torch": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "param_count": int(sum(p.numel() for p in model.parameters())),
            "dtype": str(next(model.parameters()).dtype),
        },
        "eval_prompts": user_prompts,
        "stages": {},
        "checks": {},
        "errors": [],
    }
    try:
        import transformers
        result["env"]["transformers"] = transformers.__version__
    except Exception:
        pass
    checks = result["checks"]

    def record_error(stage: str, exc: BaseException) -> None:
        result["errors"].append({"stage": stage, "error": repr(exc), "traceback": traceback.format_exc()[-3000:]})
        log(f"[ERROR] stage {stage}: {exc!r}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    probe_base = None
    base_stage: Optional[Dict[str, Any]] = None
    try:
        model.eval()
        probe_base = p28._probe_logits(model, tok, rendered[0], cfg)
        base_stage = run_condition33("base", model, tok, user_prompts, rendered, cfg, log)
        result["stages"]["base"] = base_stage
    except Exception as exc:
        record_error("base", exc)

    if cfg.do_steer and base_stage is not None:
        try:
            result["stages"]["steering"] = run_steering_stage33(model, tok, cfg, user_prompts, rendered, base_stage, log, checks)
        except Exception as exc:
            record_error("steering", exc)

    if cfg.do_grpo and base_stage is not None and probe_base is not None:
        try:
            result["stages"]["grpo_lora"] = run_grpo_stage33(model, tok, cfg, user_prompts, rendered, train_user, train_rendered,
                                                              base_stage, probe_base, log, checks)
        except Exception as exc:
            record_error("grpo_lora", exc)

    # reference (base-weight) completion PPL of every generated text, then paired diffs with it
    if cfg.score_ref_ppl and base_stage is not None:
        try:
            conds: List[Dict[str, Any]] = [base_stage]
            conds += list(result["stages"].get("steering", {}).get("conditions", {}).values())
            g = result["stages"].get("grpo_lora", {})
            conds += [g[k] for k in ("lora_merged", "lora_unmerged") if k in g]
            set_lora_enabled(model, False)  # no-op unless a failed GRPO stage left adapters attached
            for c in conds:
                add_ref_ppl(model, tok, cfg, c)
            for c in conds:
                if c is not base_stage:
                    c["paired_vs_base"] = paired33(c["records"], base_stage["records"])
        except Exception as exc:
            record_error("ref_ppl", exc)

    result["total_wall_sec"] = time.perf_counter() - t_all
    return result


def main(argv: Optional[Sequence[str]] = None, model_id: Optional[str] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description="Phase 33 Phase C real-model measurement")
    ap.add_argument("--model-id", default=model_id or DEFAULT_MODEL_ID)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--device", default="cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu")
    ap.add_argument("--n-prompts", type=int, default=Cfg.n_prompts)
    ap.add_argument("--max-new-tokens", type=int, default=Cfg.max_new_tokens)
    ap.add_argument("--seed", type=int, default=Cfg.seed)
    ap.add_argument("--alphas", type=float, nargs="+", default=list(Cfg.alphas))
    ap.add_argument("--inject-positions", nargs="+", default=list(Cfg.inject_positions), choices=["all", "last"])
    ap.add_argument("--axis-mode", default=Cfg.axis_mode, choices=["independent", "qr"])
    ap.add_argument("--ridge-lambda", type=float, default=Cfg.ridge_lambda)
    ap.add_argument("--no-steer", action="store_true")
    ap.add_argument("--no-steer-control", action="store_true")
    ap.add_argument("--no-grpo", action="store_true")
    ap.add_argument("--grpo-steps", type=int, default=Cfg.grpo_steps)
    ap.add_argument("--grpo-prompts-per-step", type=int, default=Cfg.grpo_prompts_per_step)
    ap.add_argument("--grpo-rollouts", type=int, default=Cfg.grpo_rollouts)
    ap.add_argument("--grpo-max-new-tokens", type=int, default=Cfg.grpo_max_new_tokens)
    ap.add_argument("--grpo-lr", type=float, default=Cfg.grpo_lr)
    ap.add_argument("--alpha-ppl", type=float, default=Cfg.alpha_ppl)
    ap.add_argument("--ppl-percentile", type=float, default=Cfg.ppl_percentile)
    ap.add_argument("--lora-r", type=int, default=Cfg.lora_r)
    ap.add_argument("--lora-alpha", type=float, default=Cfg.lora_alpha)
    ap.add_argument("--latency-reps", type=int, default=Cfg.latency_reps)
    ap.add_argument("--latency-tokens", type=int, default=Cfg.latency_tokens)
    ap.add_argument("--also-unmerged-gen", action="store_true")
    ap.add_argument("--steer-unit-dir", action="store_true")
    ap.add_argument("--grpo-reward", default=Cfg.grpo_reward, choices=["multiplicative", "shaped"])
    ap.add_argument("--store-rollouts", action="store_true")
    ap.add_argument("--reward-strip-tail", action="store_true")
    ap.add_argument("--prompt-set", default=Cfg.prompt_set, choices=sorted(PROMPT_SETS))
    ap.add_argument("--gen-batch", type=int, default=Cfg.gen_batch)
    ap.add_argument("--out", default=None, help="write the result JSON here (in addition to the ARCHIVE_B64 line)")
    ap.add_argument("--no-archive-line", action="store_true")
    args = ap.parse_args(argv)
    if not TORCH_AVAILABLE:
        raise SystemExit("torch is required for the real-model driver")

    cfg = Cfg(
        model_id=args.model_id, dtype=args.dtype, device=args.device, n_prompts=args.n_prompts,
        max_new_tokens=args.max_new_tokens, seed=args.seed, do_steer=not args.no_steer, alphas=tuple(args.alphas),
        inject_positions=tuple(args.inject_positions), axis_mode=args.axis_mode, ridge_lambda=args.ridge_lambda,
        steer_control=not args.no_steer_control, latency_reps=args.latency_reps, latency_tokens=args.latency_tokens,
        do_grpo=not args.no_grpo, grpo_steps=args.grpo_steps, grpo_prompts_per_step=args.grpo_prompts_per_step,
        grpo_rollouts=args.grpo_rollouts, grpo_max_new_tokens=args.grpo_max_new_tokens, grpo_lr=args.grpo_lr,
        alpha_ppl=args.alpha_ppl, ppl_percentile=args.ppl_percentile,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, also_unmerged_gen=args.also_unmerged_gen,
        steer_unit_dir=args.steer_unit_dir, grpo_reward=args.grpo_reward,
        prompt_set=args.prompt_set, store_rollouts=args.store_rollouts,
        reward_strip_tail=args.reward_strip_tail, gen_batch=args.gen_batch,
    )
    t0 = time.time()

    def log(msg: str) -> None:
        sys.stdout.write(f"[{time.time() - t0:7.0f}s] {msg}\n")
        sys.stdout.flush()

    log(f"loading {cfg.model_id} dtype={cfg.dtype} device={cfg.device}")
    t_load = time.perf_counter()
    model, tok = p28.load_model(cfg.model_id, cfg.dtype, cfg.device)
    load_sec = time.perf_counter() - t_load
    log(f"loaded in {load_sec:.0f}s")
    result = run_all(model, tok, cfg, log)
    result["model_load_sec"] = load_sec
    if args.out:
        Path(args.out).write_text(json.dumps(p28._sanitize(result), ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"wrote {args.out}")
    if not args.no_archive_line:
        print(p28.build_archive_line(result))
        sys.stdout.flush()
    return result


if __name__ == "__main__":
    main()
