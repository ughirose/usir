"""USIR v4.0 Authentic Transformer-Attached GRPO LoRA Weight Forge.

Replaces mock adapters with:
1. AttachedLoRALinear: In-place wrapper for transformer linear layers (q_proj, v_proj),
   supporting instantaneous zero-overhead switching between reference model pi_ref (lora_enabled=False)
   and policy model pi_theta (lora_enabled=True).
2. attach_lora_to_model: In-place transformer scanner and LoRA attachment engine.
3. run_authentic_grpo_lora_forge: Genuine GRPO policy gradient training loop over G=4 rollouts,
   computing multiplicative GRPO rewards, relative advantages A_i = (R_i - mu) / (sigma + eps),
   token log-likelihoods, and token-level KL divergence regularization.
   Includes authentic execution on PyTorch models as well as safe simulation fallback for CPU/mock environments.
"""

from __future__ import annotations

import math
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

# Conditional PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

from .manifold_geodesic import compute_multiplicative_grpo_reward


# ---------------------------------------------------------------------------
# 1. AttachedLoRALinear Layer
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:
    class AttachedLoRALinear(nn.Module):
        """Authentic PyTorch Transformer In-Place LoRA Wrapper.

        Wraps an existing base_layer (e.g., nn.Linear in q_proj, v_proj),
        freezes base parameters, and learns low-rank adapters lora_A and lora_B.
        Enables instantaneous switching between reference distribution pi_ref (lora_enabled=False)
        and policy distribution pi_theta (lora_enabled=True).
        """

        def __init__(
            self,
            base_layer: Any,
            r: int = 8,
            alpha: float = 16.0,
            dropout: float = 0.0,
        ):
            super().__init__()
            self.base_layer = base_layer
            _params_fn = getattr(base_layer, "parameters", None)
            if callable(_params_fn):
                for p in _params_fn():
                    if isinstance(p, torch.Tensor):  # duck-typed mocks may yield numpy arrays
                        p.requires_grad = False

            self.in_features = getattr(base_layer, "in_features", None)
            self.out_features = getattr(base_layer, "out_features", None)
            if self.in_features is None and hasattr(base_layer, "weight"):
                self.in_features = base_layer.weight.shape[1]
                self.out_features = base_layer.weight.shape[0]
            if self.in_features is None:
                self.in_features = 64
                self.out_features = 64

            weight = getattr(base_layer, "weight", None)
            is_t = isinstance(weight, torch.Tensor)
            dtype = weight.dtype if is_t else torch.float32
            device = weight.device if is_t else torch.device("cpu")

            self.r = max(1, int(r))
            self.alpha = float(alpha)
            self.scaling = float(alpha) / float(self.r)
            self.lora_enabled = True

            # Gaussian init for lora_A, zero init for lora_B
            self.lora_A = nn.Parameter(
                torch.randn(self.r, self.in_features, device=device, dtype=torch.float32) * 0.01
            )
            self.lora_B = nn.Parameter(
                torch.zeros(self.out_features, self.r, device=device, dtype=torch.float32)
            )
            self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if not isinstance(x, torch.Tensor):
                # NumPy input (duck-typed base layers used in unit tests): compute with NumPy
                x_arr = np.asarray(x, dtype=np.float32)
                base_np = np.asarray(self.base_layer(x_arr))
                if not self.lora_enabled:
                    return base_np
                a_np = self.lora_A.detach().cpu().numpy()
                b_np = self.lora_B.detach().cpu().numpy()
                return base_np + (x_arr @ a_np.T @ b_np.T) * self.scaling
            base_out = self.base_layer(x)
            if not self.lora_enabled:
                return base_out
            x_fp32 = x.to(torch.float32)
            dropped = self.dropout(x_fp32)
            lora_A_fp32 = self.lora_A.to(torch.float32)
            lora_B_fp32 = self.lora_B.to(torch.float32)
            lora_out = (dropped @ lora_A_fp32.T @ lora_B_fp32.T) * self.scaling
            return base_out + lora_out.to(base_out.dtype)

        def extra_repr(self) -> str:
            return (
                f"in_features={self.in_features}, out_features={self.out_features}, "
                f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.4f}, "
                f"lora_enabled={self.lora_enabled}"
            )

else:
    class AttachedLoRALinear:  # type: ignore[no-redef]
        """NumPy / Pure-Python In-Place LoRA Wrapper for CPU / non-torch environments."""

        def __init__(
            self,
            base_layer: Any,
            r: int = 8,
            alpha: float = 16.0,
            dropout: float = 0.0,
        ):
            self.base_layer = base_layer
            self.in_features = getattr(base_layer, "in_features", None)
            self.out_features = getattr(base_layer, "out_features", None)
            if self.in_features is None and hasattr(base_layer, "weight"):
                w = np.asarray(base_layer.weight)
                self.in_features = w.shape[1] if w.ndim > 1 else w.shape[0]
                self.out_features = w.shape[0]
            if self.in_features is None:
                self.in_features = 64
                self.out_features = 64

            self.r = max(1, int(r))
            self.alpha = float(alpha)
            self.scaling = float(alpha) / float(self.r)
            self.lora_enabled = True

            rng = np.random.RandomState(42)
            self.lora_A = (rng.randn(self.r, self.in_features) * 0.01).astype(np.float32)
            self.lora_B = np.zeros((self.out_features, self.r), dtype=np.float32)
            self.dropout_p = float(dropout)

        def forward(self, x: Any) -> Any:
            x_arr = np.asarray(x, dtype=np.float32)
            if callable(self.base_layer):
                base_out = self.base_layer(x_arr)
            elif hasattr(self.base_layer, "forward"):
                base_out = self.base_layer.forward(x_arr)
            elif hasattr(self.base_layer, "weight"):
                w = np.asarray(self.base_layer.weight, dtype=np.float32)
                base_out = x_arr @ w.T
            else:
                base_out = x_arr

            if not self.lora_enabled:
                return base_out

            lora_out = (x_arr @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return base_out + lora_out

        def __call__(self, x: Any) -> Any:
            return self.forward(x)

        def parameters(self) -> List[Any]:
            return [self.lora_A, self.lora_B]


# ---------------------------------------------------------------------------
# 2. In-Place Transformer Scanner & Attachment Engine
# ---------------------------------------------------------------------------

def _extract_layer_index(name: str) -> Optional[int]:
    """Extracts integer layer index from module path (e.g. model.layers.27.self_attn.q_proj or layer0)."""
    m = re.search(r'\b(?:layers?|h|blocks|decoder_layers)[._\[]?(\d+)\b', name, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def attach_lora_to_model(
    model: Any,
    target_modules: Sequence[str] = ("q_proj", "v_proj"),
    r: int = 8,
    alpha: float = 16.0,
    target_layers: Optional[Union[int, Sequence[int], Set[int]]] = None,
) -> Dict[str, Any]:
    """Scans transformer architecture and wraps target attention layers with AttachedLoRALinear.

    Args:
        model: HuggingFace AutoModelForCausalLM, PyTorch nn.Module, or duck-typed container.
        target_modules: List of module sub-names to adapt (default: ["q_proj", "v_proj"]).
        r: LoRA low-rank dimension (default: 8).
        alpha: LoRA scaling coefficient (default: 16.0).
        target_layers: Optional restriction. If int K, attaches to the last K layers.
                       If sequence/set, attaches to those exact layer indices.
                       If None, attaches to all layers containing target_modules.

    Returns:
        Dict containing attached_modules list, total_lora_params count, and num_attached.
    """
    attached_names: List[str] = []
    total_params = 0

    # Determine target layer set if integer K is provided
    layer_indices_set: Optional[Set[int]] = None
    if isinstance(target_layers, int):
        if target_layers <= 0:
            layer_indices_set = set()
        else:
            # Discover maximum layer index
            max_idx = -1
            if hasattr(model, "named_modules"):
                for name, _ in model.named_modules():
                    idx = _extract_layer_index(name)
                    if idx is not None and idx > max_idx:
                        max_idx = idx
            elif hasattr(model, "layers") and isinstance(model.layers, (list, tuple)):
                max_idx = len(model.layers) - 1

            if max_idx >= 0:
                start_idx = max(0, max_idx - target_layers + 1)
                layer_indices_set = set(range(start_idx, max_idx + 1))
    elif target_layers is not None:
        layer_indices_set = set(target_layers)

    # 1. PyTorch named_modules traversal
    if hasattr(model, "named_modules") and callable(model.named_modules):
        # Find all matching children to replace
        modules_to_replace: List[Tuple[str, str, Any, Any]] = []  # (full_name, child_attr, parent, child)
        for name, module in model.named_modules():
            # Check if module matches target_modules
            leaf_name = name.split(".")[-1]
            if not any(leaf_name == tgt or name.endswith("." + tgt) for tgt in target_modules):
                continue

            # Check layer filter
            if layer_indices_set is not None:
                if len(layer_indices_set) == 0:
                    continue
                l_idx = _extract_layer_index(name)
                if l_idx is not None and l_idx not in layer_indices_set:
                    continue

            # Find parent module
            parts = name.split(".")
            parent_name = ".".join(parts[:-1])
            child_attr = parts[-1]
            if parent_name:
                if hasattr(model, "get_submodule") and callable(model.get_submodule):
                    parent = model.get_submodule(parent_name)
                else:
                    curr = model
                    for part in parts[:-1]:
                        curr = getattr(curr, part, None)
                    parent = curr
            else:
                parent = model
            if parent is not None:
                modules_to_replace.append((name, child_attr, parent, module))

        for full_name, child_attr, parent, old_module in modules_to_replace:
            if isinstance(old_module, AttachedLoRALinear):
                continue
            wrapped = AttachedLoRALinear(old_module, r=r, alpha=alpha)
            setattr(parent, child_attr, wrapped)
            attached_names.append(full_name)
            p_count = (wrapped.lora_A.numel() + wrapped.lora_B.numel()) if hasattr(wrapped.lora_A, "numel") else (wrapped.lora_A.size + wrapped.lora_B.size)
            total_params += p_count

    # 2. Duck-typed object / Mock traversal fallback
    else:
        for tgt in target_modules:
            if hasattr(model, tgt):
                old = getattr(model, tgt)
                if not isinstance(old, AttachedLoRALinear):
                    wrapped = AttachedLoRALinear(old, r=r, alpha=alpha)
                    setattr(model, tgt, wrapped)
                    attached_names.append(tgt)
                    p_count = (wrapped.lora_A.numel() + wrapped.lora_B.numel()) if hasattr(wrapped.lora_A, "numel") else (wrapped.lora_A.size + wrapped.lora_B.size)
                    total_params += p_count

        # Check sub-attributes (e.g. model.layers)
        if hasattr(model, "layers") and isinstance(model.layers, (list, tuple)):
            for idx, layer in enumerate(model.layers):
                if layer_indices_set is not None and idx not in layer_indices_set:
                    continue
                for tgt in target_modules:
                    if hasattr(layer, tgt):
                        old = getattr(layer, tgt)
                        if not isinstance(old, AttachedLoRALinear):
                            wrapped = AttachedLoRALinear(old, r=r, alpha=alpha)
                            setattr(layer, tgt, wrapped)
                            full_name = f"layers.{idx}.{tgt}"
                            attached_names.append(full_name)
                            p_count = (wrapped.lora_A.numel() + wrapped.lora_B.numel()) if hasattr(wrapped.lora_A, "numel") else (wrapped.lora_A.size + wrapped.lora_B.size)
                            total_params += p_count

    return {
        "attached_modules": attached_names,
        "total_lora_params": total_params,
        "num_attached": len(attached_names),
        "r": r,
        "alpha": alpha,
    }


def set_lora_enabled(model: Any, enabled: bool) -> int:
    """Toggles lora_enabled flag across all AttachedLoRALinear modules in model.

    Returns the count of toggled LoRA modules.
    """
    count = 0
    if hasattr(model, "modules") and callable(model.modules):
        for m in model.modules():
            if isinstance(m, AttachedLoRALinear):
                m.lora_enabled = bool(enabled)
                count += 1
    elif hasattr(model, "named_modules") and callable(model.named_modules):
        for _, m in model.named_modules():
            if isinstance(m, AttachedLoRALinear):
                m.lora_enabled = bool(enabled)
                count += 1
    else:
        # Duck-typed walk
        for attr in dir(model):
            try:
                val = getattr(model, attr)
                if isinstance(val, AttachedLoRALinear):
                    val.lora_enabled = bool(enabled)
                    count += 1
            except Exception:
                pass
        if hasattr(model, "layers") and isinstance(model.layers, (list, tuple)):
            for layer in model.layers:
                for attr in dir(layer):
                    try:
                        val = getattr(layer, attr)
                        if isinstance(val, AttachedLoRALinear):
                            val.lora_enabled = bool(enabled)
                            count += 1
                    except Exception:
                        pass
    return count


# ---------------------------------------------------------------------------
# 3. Authentic GRPO Policy Gradient Training Loop
# ---------------------------------------------------------------------------

_SENT_END = "。！？!?"


def strip_incomplete_tail(text: str) -> str:
    """Phase 37: drop the trailing fragment after the last sentence end (a rollout cut by the token limit).

    Returns the text unchanged when it already ends with a sentence end or contains none."""
    s = text.rstrip()
    if not s or s[-1] in _SENT_END:
        return s
    cut = max(s.rfind(c) for c in _SENT_END)
    return s[:cut + 1] if cut >= 0 else s


def run_authentic_grpo_lora_forge(
    model: Any,
    tokenizer: Optional[Any] = None,
    prompts: Optional[Sequence[Union[str, Dict[str, Any]]]] = None,
    candidate_completions: Optional[Sequence[Sequence[str]]] = None,
    num_rollouts: int = 4,
    beta_kl: float = 0.04,
    learning_rate: float = 2e-4,
    num_steps: int = 5,
    device: str = "cpu",
    domain: str = "novel",
    ppl_thresh: float = 45.0,
    alpha_ppl: float = 0.1,
    max_new_tokens: int = 48,
    batched_rollouts: bool = False,
    per_rollout_backward: bool = False,
    prompts_per_step: Optional[int] = None,
    progress_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
    ppl_completion_only: bool = False,
    reward_fn: Optional[Callable[[str, float], float]] = None,
) -> Dict[str, Any]:
    r"""Executes authentic USIR-GRPO fine-tuning loop for AttachedLoRALinear weights.

    Formula:
        A_i = (R_i - \mu_R) / (\sigma_R + 10^{-6})
        D_KL(\pi_\theta || \pi_ref) = \exp(\ln \pi_ref - \ln \pi_\theta) - (\ln \pi_ref - \ln \pi_\theta) - 1
        L_GRPO = - 1/G \sum_{i=1}^G [ A_i * 1/|y_i| \sum_t \ln \pi_\theta(y_{i,t}) - \beta_KL * D_KL ]

    Supports both genuine PyTorch model execution and CPU simulation mode.

    Phase 28 additions (defaults reproduce the previous behaviour; they only affect the authentic PyTorch path):
        max_new_tokens: completion length of model-generated rollouts (previously fixed at 48).
        batched_rollouts: draw the G rollouts with one generate(num_return_sequences=G) call instead of G calls.
        per_rollout_backward: call backward() on (loss_i / (n_prompts * G)) right after each rollout instead of
            stacking all losses; the accumulated gradient is the same, peak memory is one graph instead of all.
        prompts_per_step: if set, each optimisation step uses a rotating window of this many prompts
            (step s uses prompts[(s*k + j) % len(prompts)]); otherwise every step uses all prompts.
        progress_fn: called with each step_logs entry (e.g. for heartbeat logging on long Colab jobs).
    Phase 33 addition:
        ppl_completion_only: compute the reference perplexity (LoRA disabled) used by the M_semantic barrier over the
            completion tokens only, instead of prompt + completion. Matters for long chat-template prompts.
    Phase 36 addition:
        reward_fn: if given, reward = reward_fn(completion_text, ppl_ref) replaces compute_multiplicative_grpo_reward
            in the authentic PyTorch path (e.g. a continuous shaping reward without the steep exp(-0.5 d_3phase)).
    Caller responsibility: the authentic path optimises every parameter with requires_grad=True. Freeze the base
    model (p.requires_grad_(False) for all parameters) BEFORE attach_lora_to_model, otherwise the optimizer is
    built over the entire model.
    """
    if prompts is None:
        prompts = [
            "夜の静寂が広がる古い書斎で、青年は机上の羊皮紙に目を落とした。",
            "青白い月光が窓枠を斜めに切り裂き、微かなインクの匂いが漂う。",
        ]

    # Verify PyTorch capability
    is_real_torch_model = (
        TORCH_AVAILABLE
        and isinstance(model, nn.Module)
        and any(p.requires_grad for p in model.parameters())
    )

    if is_real_torch_model and tokenizer is not None:
        # --- Authentic PyTorch Path ---
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

        step_logs: List[Dict[str, Any]] = []
        all_rewards: List[float] = []
        all_kls: List[float] = []

        for step in range(num_steps):
            t_step = time.perf_counter()
            optimizer.zero_grad()
            step_rewards: List[float] = []
            group_stds: List[float] = []   # Phase 36: within-group reward std (before the +1e-6) per prompt
            step_losses: List[torch.Tensor] = []
            step_kls: List[float] = []
            step_loss_sum = 0.0

            if prompts_per_step is not None and 0 < int(prompts_per_step) < len(prompts):
                k_pps = int(prompts_per_step)
                step_items = [((step * k_pps + j) % len(prompts), prompts[(step * k_pps + j) % len(prompts)]) for j in range(k_pps)]
            else:
                step_items = list(enumerate(prompts))
            n_loss_expected = max(1, len(step_items) * int(num_rollouts))

            for p_idx, p_item in step_items:
                p_text = p_item["prompt"] if isinstance(p_item, dict) else str(p_item)
                x0 = p_item.get("x_0", p_text) if isinstance(p_item, dict) else p_text
                xT = p_item.get("x_T", None) if isinstance(p_item, dict) else None
                dom = p_item.get("domain", domain) if isinstance(p_item, dict) else domain

                # 1. Rollout G=4 completions
                if candidate_completions and p_idx < len(candidate_completions):
                    rollout_texts = list(candidate_completions[p_idx][:num_rollouts])
                else:
                    # Model generation
                    set_lora_enabled(model, True)
                    in_enc = tokenizer(p_text, return_tensors="pt").to(device)
                    prompt_len = in_enc["input_ids"].shape[1]
                    rollout_texts = []
                    with torch.no_grad():
                        if batched_rollouts:
                            out_ids = model.generate(
                                **in_enc,
                                max_new_tokens=int(max_new_tokens),
                                do_sample=True,
                                temperature=0.85,
                                num_return_sequences=int(num_rollouts),
                                pad_token_id=getattr(tokenizer, "pad_token_id", 0),
                            )
                            for row in range(out_ids.shape[0]):
                                rollout_texts.append(tokenizer.decode(out_ids[row][prompt_len:], skip_special_tokens=True))
                        else:
                            for _ in range(num_rollouts):
                                out_ids = model.generate(
                                    **in_enc,
                                    max_new_tokens=int(max_new_tokens),
                                    do_sample=True,
                                    temperature=0.85,
                                    pad_token_id=getattr(tokenizer, "pad_token_id", 0),
                                )
                                c_text = tokenizer.decode(out_ids[0][prompt_len:], skip_special_tokens=True)
                                rollout_texts.append(c_text)

                # 2. Compute physical rewards for all completions
                rewards: List[float] = []
                for c_text in rollout_texts:
                    # Reference PPL estimation using lora_enabled=False
                    set_lora_enabled(model, False)
                    enc = tokenizer(p_text + c_text, return_tensors="pt").to(device)
                    with torch.no_grad():
                        ref_out = model(**enc)
                        shift_logits = ref_out.logits[:, :-1, :].float().contiguous()
                        shift_labels = enc["input_ids"][:, 1:].contiguous()
                        if ppl_completion_only:
                            _pl = int(tokenizer(p_text, return_tensors="pt")["input_ids"].shape[1])
                            _gs = max(0, _pl - 1)
                            if _gs < shift_labels.shape[1]:
                                shift_logits = shift_logits[:, _gs:, :].contiguous()
                                shift_labels = shift_labels[:, _gs:].contiguous()
                        ce_loss = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                        )
                        ppl_ref = float(torch.exp(ce_loss).item())

                    if reward_fn is not None:
                        rewards.append(float(reward_fn(c_text, ppl_ref)))
                        continue
                    r_dict = compute_multiplicative_grpo_reward(
                        c_text,
                        x_0=x0,
                        x_T=xT,
                        domain=dom,
                        ppl_ref=ppl_ref,
                        ppl_thresh=ppl_thresh,
                        alpha_ppl=alpha_ppl,
                    )
                    rewards.append(r_dict["reward"])

                step_rewards.extend(rewards)

                # 3. Advantage calculation
                r_arr = np.array(rewards, dtype=np.float32)
                r_mean = float(np.mean(r_arr))
                group_stds.append(float(np.std(r_arr)))
                r_std = float(np.std(r_arr)) + 1e-6
                advantages = (r_arr - r_mean) / r_std

                # 4. Policy gradient and KL divergence per rollout
                for i, c_text in enumerate(rollout_texts):
                    full_text = p_text + c_text
                    enc = tokenizer(full_text, return_tensors="pt").to(device)
                    in_ids = enc["input_ids"]
                    if in_ids.shape[1] < 2:
                        continue

                    # Reference logits (no grad, lora off)
                    set_lora_enabled(model, False)
                    with torch.no_grad():
                        logits_ref = model(in_ids).logits[:, :-1, :].float()
                        log_probs_ref = F.log_softmax(logits_ref, dim=-1)

                    # Policy logits (grad, lora on)
                    set_lora_enabled(model, True)
                    logits_theta = model(in_ids).logits[:, :-1, :].float()
                    log_probs_theta = F.log_softmax(logits_theta, dim=-1)

                    # Mask prompt tokens out of GRPO loss: target tokens are in_ids[:, 1:]
                    p_enc = tokenizer(p_text, return_tensors="pt")
                    prompt_len = p_enc["input_ids"].shape[1]
                    gen_start_idx = max(0, prompt_len - 1)

                    # Gather token log probabilities
                    target_ids = in_ids[:, 1:].unsqueeze(-1)
                    tok_lp_ref = torch.gather(log_probs_ref, -1, target_ids).squeeze(-1)
                    tok_lp_theta = torch.gather(log_probs_theta, -1, target_ids).squeeze(-1)

                    # Slice to only generated completion tokens
                    if gen_start_idx < tok_lp_theta.shape[1]:
                        gen_lp_ref = tok_lp_ref[:, gen_start_idx:]
                        gen_lp_theta = tok_lp_theta[:, gen_start_idx:]
                    else:
                        gen_lp_ref = tok_lp_ref
                        gen_lp_theta = tok_lp_theta

                    # Token-level Schulman KL estimator: exp(r) - r - 1
                    log_ratio = gen_lp_ref - gen_lp_theta
                    kl_div = torch.exp(log_ratio) - log_ratio - 1.0
                    mean_kl = kl_div.mean()
                    step_kls.append(float(mean_kl.item()))

                    # GRPO Objective: Adv * (1/|y_i| \sum_t ln pi_theta) - beta * KL
                    adv_t = float(advantages[i])
                    loss_i = -(adv_t * gen_lp_theta.mean() - beta_kl * mean_kl)
                    if per_rollout_backward:
                        (loss_i / float(n_loss_expected)).backward()
                        step_loss_sum += float(loss_i.item()) / float(n_loss_expected)
                    else:
                        step_losses.append(loss_i)

            if per_rollout_backward:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                loss_val = step_loss_sum
            elif step_losses:
                total_loss = torch.stack(step_losses).mean()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                loss_val = float(total_loss.item())
            else:
                loss_val = 0.0

            m_rew = float(np.mean(step_rewards)) if step_rewards else 0.0
            s_rew = float(np.std(step_rewards)) if step_rewards else 0.0
            m_kl = float(np.mean(step_kls)) if step_kls else 0.0
            all_rewards.append(m_rew)
            all_kls.append(m_kl)

            step_entry = {
                "step": step + 1,
                "loss": round(loss_val, 6),
                "mean_reward": round(m_rew, 4),
                "std_reward": round(s_rew, 4),
                "mean_kl": round(m_kl, 6),
                "sec": round(time.perf_counter() - t_step, 3),
                "group_reward_std": [float(f"{x:.3g}") for x in group_stds],
            }
            step_logs.append(step_entry)
            if progress_fn is not None:
                progress_fn(step_entry)

        return {
            "mode": "authentic_pytorch",
            "final_mean_reward": round(float(all_rewards[-1]), 4) if all_rewards else 0.0,
            "final_std_reward": round(float(step_logs[-1]["std_reward"]), 4) if step_logs else 0.0,
            "mean_kl": round(float(all_kls[-1]), 6) if all_kls else 0.0,
            "steps": step_logs,
            "num_rollouts": num_rollouts,
            "device": device,
        }

    else:
        # --- Safe Simulation & Fallback Path (CPU, Mock Models, Non-Torch) ---
        step_logs = []
        all_rewards = []
        all_kls = []

        # Find attached LoRA modules to update simulation weights
        lora_modules: List[Any] = []
        if hasattr(model, "modules") and callable(model.modules):
            lora_modules = [m for m in model.modules() if isinstance(m, AttachedLoRALinear)]
        elif hasattr(model, "parameters"):
            lora_modules = [model]
        elif isinstance(model, AttachedLoRALinear):
            lora_modules = [model]

        default_completions = [
            "冷たい雨が静かに庭の石畳を濡らし、かすかな墨の香りが廊下に漂った。",
            "微風が古寺の釣鐘をかすかに揺らし、遠くで鳥の羽ばたく音が響いた。",
            "闇の中に浮かぶ障子の白さが、時の経過を静かに告げていた。",
            "足音を忍ばせて歩むと、かすかに軋む床板が夜の沈黙を乱した。",
        ]

        for step in range(num_steps):
            step_rewards = []
            step_kls = []
            accumulated_grad_norm = 0.0

            for p_idx, p_item in enumerate(prompts):
                p_text = p_item["prompt"] if isinstance(p_item, dict) else str(p_item)
                x0 = p_item.get("x_0", p_text) if isinstance(p_item, dict) else p_text
                xT = p_item.get("x_T", None) if isinstance(p_item, dict) else None
                dom = p_item.get("domain", domain) if isinstance(p_item, dict) else domain

                if candidate_completions and p_idx < len(candidate_completions):
                    rollout_texts = list(candidate_completions[p_idx][:num_rollouts])
                else:
                    rollout_texts = default_completions[:num_rollouts]

                # 1. Compute physical rewards
                rewards = []
                for i_r, c_text in enumerate(rollout_texts):
                    sim_ppl = 25.0 + float(i_r * 4.0)
                    r_dict = compute_multiplicative_grpo_reward(
                        c_text,
                        x_0=x0,
                        x_T=xT,
                        domain=dom,
                        ppl_ref=sim_ppl,
                        ppl_thresh=ppl_thresh,
                        alpha_ppl=alpha_ppl,
                    )
                    rewards.append(r_dict["reward"])
                step_rewards.extend(rewards)

                # 2. Relative advantage
                r_arr = np.array(rewards, dtype=np.float32)
                r_mean = float(np.mean(r_arr))
                r_std = float(np.std(r_arr)) + 1e-6
                adv = (r_arr - r_mean) / r_std

                # 3. Simulate log probabilities and KL divergence
                for i_r in range(len(rollout_texts)):
                    lp_theta = -1.2 + 0.1 * adv[i_r] - 0.02 * step
                    lp_ref = -1.2
                    ratio = lp_ref - lp_theta
                    kl_i = math.exp(ratio) - ratio - 1.0
                    step_kls.append(kl_i)
                    accumulated_grad_norm += abs(adv[i_r])

            # Simulate parameter descent on AttachedLoRALinear
            mean_adv = float(np.mean(step_rewards)) if step_rewards else 0.0
            for mod in lora_modules:
                if hasattr(mod, "lora_B"):
                    if TORCH_AVAILABLE and isinstance(mod.lora_B, torch.Tensor):
                        with torch.no_grad():
                            grad_sim = torch.randn_like(mod.lora_B) * 0.005 + (mean_adv * 0.01)
                            mod.lora_B.add_(grad_sim * learning_rate)
                    elif isinstance(mod.lora_B, np.ndarray):
                        grad_sim = (np.random.randn(*mod.lora_B.shape).astype(np.float32) * 0.005) + (mean_adv * 0.01)
                        mod.lora_B += grad_sim * np.float32(learning_rate)

            m_rew = float(np.mean(step_rewards)) if step_rewards else 0.0
            s_rew = float(np.std(step_rewards)) if step_rewards else 0.0
            m_kl = float(np.mean(step_kls)) if step_kls else 0.0
            loss_sim = float(-0.5 * accumulated_grad_norm + beta_kl * m_kl)
            all_rewards.append(m_rew)
            all_kls.append(m_kl)

            step_logs.append({
                "step": step + 1,
                "loss": round(loss_sim, 6),
                "mean_reward": round(m_rew, 4),
                "std_reward": round(s_rew, 4),
                "mean_kl": round(m_kl, 6),
            })

        return {
            "mode": "simulation_cpu",
            "final_mean_reward": round(float(all_rewards[-1]), 4) if all_rewards else 0.0,
            "final_std_reward": round(float(step_logs[-1]["std_reward"]), 4) if step_logs else 0.0,
            "mean_kl": round(float(all_kls[-1]), 6) if all_kls else 0.0,
            "steps": step_logs,
            "num_rollouts": num_rollouts,
            "device": device,
        }
