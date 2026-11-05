"""USIR v4.0 Realtime Decoder & USIRPyTorchLogitsProcessor.

Solves:
1. BPE Subword Boundary Problem (Bomb 3): Evaluates clause-level step boundaries and
   maintains continuous momentum state across subwords.
2. Candidate Word Explosion (Bomb 4): Operates strictly within the Top-K (K=64) candidate tokens
   selected by the base LLM context distribution, avoiding naive full-vocab inverse lookups.
3. Refractory Period Gate: Prevents punctuation spam / stutter by enforcing a minimum refractory
   window (min 5 tokens) and preventing runaway run-on sentences (max 45 tokens).
4. Backward Wave Attraction phi: Pulls forward generation toward the meet-in-the-middle target anchor.
"""

from __future__ import annotations

import collections
import math
import sys
import re
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple, Union

import numpy as np

# Conditional PyTorch / Transformers imports
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    F = None
    TORCH_AVAILABLE = False

try:
    from transformers import LogitsProcessor, StoppingCriteria, StoppingCriteriaList
except ImportError:
    class StoppingCriteria:  # type: ignore[no-redef]
        pass

    class StoppingCriteriaList(list):  # type: ignore[no-redef]
        pass

    class LogitsProcessor:  # type: ignore[no-redef]
        """Fallback base class when transformers is not installed."""
        def __call__(self, input_ids: Any, scores: Any) -> Any:
            return scores


# Clause boundary punctuation: ASCII and full-width variants (models emit U+FF0C / U+FF0E freely).
CLAUSE_PUNCT = ("\u3001", "\u3002", "\uff0c", "\uff0e", "\uff01", "\uff1f", "!", "?")
CLAUSE_PUNCT_RE = re.compile("[\u3001\u3002\uff0c\uff0e\uff01\uff1f!?]")


# Canonical Japanese and punctuation characters for refractory period gating
PUNCTUATION_CHARS: Set[str] = {"、", "。", "！", "？", "!", "?", ",", ".", "…", "—", "」", "』"}

# Chinese simplified function words, non-Japanese CJK variants, and markdown backquote tokens to suppress at logits level
# Note: Standard Japanese Joyo kanji (e.g. 希, 著, 据, 与, 若, 訴, 説) are strictly preserved.
BANNED_CJK_AND_META_CHARS: Set[str] = {
    "是", "恒", "且", "则", "为", "这", "个", "们",
    "设", "闭", "紧", "测", "军", "团", "态", "识", "证", "义",
    "单", "达", "处", "满", "须", "尔", "并", "濕",
    "诉", "说", "，", "`", "```",
    "假设", "是一个", "根据", "必须", "满足", "其中", "处于", "无法", "达成", "晨光",
    "首先", "注意到", "由于", "从而", "基于", "使得", "假定", "由前期提", "由此",
    "因此可以", "可以写作", "不动点", "我们", "至此", "单调性", "初照下",
}


def is_non_japanese_cjk(text: Union[str, bytes]) -> bool:
    """Detects whether text contains any non-Japanese CJK character.

    Mechanical JIS EUC-JP whitelist check:
    Returns True if any character in the CJK range [0x4E00, 0x9FFF] fails to encode in 'euc_jp'
    (e.g. 隐, 测, 这, 们, 为, 设, 证, 满, 导, 边) or is a simplified variant (e.g. 步 U+6B65 vs
    Japanese 歩 U+6B69, or 个 U+4E2A).
    Standard Japanese Joyo and Jinmeiyo kanji (希, 著, 据, 与, 若, etc.) encode cleanly in euc_jp
    and return False with 0 false positives.
    """
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")
    for ch in text:
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF:
            if code == 0x6B65 or code == 0x4E2A:  # 步 (U+6B65) and 个 (U+4E2A)
                return True
            try:
                ch.encode("euc_jp")
            except UnicodeEncodeError:
                return True
    return False



class LogitsStepper:
    """General-purpose standalone Logits Stepper for Top-K candidate reweighting,

    refractory period gating, and backward bridge attraction. Works on NumPy or lists.
    """

    def __init__(
        self,
        dim: int = 12,
        top_k: int = 64,
        eps_syn: float = 0.5,
        beta_steer: float = 1.0,
        refractory_min: int = 5,
        refractory_max: int = 45,
        backward_attraction: Optional[np.ndarray] = None,
    ) -> None:
        self.dim = dim
        self.top_k = top_k
        self.eps_syn = eps_syn
        self.beta_steer = beta_steer
        self.refractory_min = refractory_min
        self.refractory_max = refractory_max
        self.backward_attraction = backward_attraction

        # Internal state
        self.tokens_since_punctuation: int = 0
        self.q = np.full(dim, 0.5, dtype=float)
        self.p = np.zeros(dim, dtype=float)
        self.punctuation_token_ids: Set[int] = set()

    def register_punctuation_ids(self, token_ids: Union[List[int], Set[int]]) -> None:
        self.punctuation_token_ids.update(token_ids)

    def step_token_selected(self, token_id: int, is_punctuation: bool = False) -> None:
        """Called when a token is sampled and committed to the sequence."""
        if is_punctuation or token_id in self.punctuation_token_ids:
            self.tokens_since_punctuation = 0
            # Clause step momentum update: dissipate excess shock, update momentum
            self.p = self.p * 0.85
        else:
            self.tokens_since_punctuation += 1
            # Slight momentum decay per subword token
            self.p = self.p * 0.98

    def steer_logits(
        self,
        logits: np.ndarray,
        token_jacobians: Optional[Dict[int, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Applies Top-K extraction, refractory gating, and backward attraction.

        Args:
            logits: 1D array of vocab logits.
            token_jacobians: optional dict mapping token_id -> 12D derivative.

        Returns:
            (steered_logits, telemetry_dict)
        """
        steered = logits.copy()
        vocab_size = len(logits)
        K = min(self.top_k, vocab_size)

        # 1. Select top-K token indices
        top_indices = np.argpartition(logits, -K)[-K:]
        top_indices = top_indices[np.argsort(-logits[top_indices])]

        # 2. Shannon entropy of top-K base logits
        top_logits = logits[top_indices]
        max_l = np.max(top_logits)
        exp_l = np.exp(top_logits - max_l)
        probs = exp_l / np.sum(exp_l)

        h_llm = float(-np.sum([p * np.log(p + 1e-12) for p in probs]))

        # 3. Syntax Protection Gate
        if h_llm < self.eps_syn:
            # Low entropy => grammatically deterministic token (e.g. closing bracket)
            effective_beta = 0.0
            bypassed = True
        else:
            effective_beta = self.beta_steer * min(1.0, (h_llm - self.eps_syn) / self.eps_syn)
            bypassed = False

        # 4. Refractory Period Gating
        is_refractory_blocked = self.tokens_since_punctuation < self.refractory_min
        is_refractory_forced = self.tokens_since_punctuation > self.refractory_max

        for idx in top_indices:
            idx = int(idx)
            is_punct = idx in self.punctuation_token_ids

            if is_punct and is_refractory_blocked:
                # Strictly penalize punctuation spam (prevent stuttering)
                steered[idx] -= 1000.0
            elif is_punct and is_refractory_forced:
                # Strongly boost punctuation (prevent runaway run-on sentence)
                steered[idx] += 3.5
            elif not bypassed:
                # Steer towards backward bridge attraction phi and Hamiltonian direction
                steer_val = 0.0
                if self.backward_attraction is not None and token_jacobians and idx in token_jacobians:
                    j_vec = token_jacobians[idx][:len(self.backward_attraction)]
                    steer_val += float(np.dot(self.backward_attraction, j_vec))
                steered[idx] += effective_beta * steer_val

        telemetry = {
            "h_llm": h_llm,
            "bypassed": bypassed,
            "tokens_since_punctuation": self.tokens_since_punctuation,
            "is_refractory_blocked": is_refractory_blocked,
            "is_refractory_forced": is_refractory_forced,
            "effective_beta": effective_beta,
        }
        return steered, telemetry


class USIRPyTorchLogitsProcessor(LogitsProcessor):
    """PyTorch / HuggingFace LogitsProcessor direct class for USIR v4.0.

    Hooks directly into `model.generate(logits_processor=[...])`.
    Supports 2D (batch_size, vocab_size) and 1D tensors.
    """

    def __init__(
        self,
        dim: int = 12,
        top_k: int = 64,
        eps_syn: float = 0.5,
        beta_steer: float = 1.0,
        refractory_min: int = 5,
        refractory_max: int = 45,
        backward_attraction: Optional[Union[np.ndarray, List[float]]] = None,
        punctuation_token_ids: Optional[Union[List[int], Set[int]]] = None,
        token_jacobians: Optional[Dict[int, np.ndarray]] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.top_k = top_k
        self.eps_syn = eps_syn
        self.beta_steer = beta_steer
        self.refractory_min = refractory_min
        self.refractory_max = refractory_max

        self.backward_attraction = (
            np.asarray(backward_attraction, dtype=float) if backward_attraction is not None else None
        )
        self.punctuation_token_ids = set(punctuation_token_ids) if punctuation_token_ids else set()
        self.banned_cjk_token_ids: Set[int] = set()
        self._banned_token_ids: Set[int] = self.banned_cjk_token_ids
        self.token_jacobians = token_jacobians
        self.embedding_weight: Optional[Any] = None
        self._embed_weight: Optional[Any] = None
        self.v_bridge: Optional[Any] = None
        self._bridge_dir_tensor: Optional[Any] = None
        self._punct_tensor: Optional[Any] = None
        self._banned_tensor: Optional[Any] = None

        # Track per-batch sequence state
        self.batch_tokens_since_punctuation: Dict[int, int] = {}
        self.call_count: int = 0

    def reset(self) -> None:
        """Resets dynamic tracking state (call_count and batch token counters) for a new generation pass."""
        self.call_count = 0
        self.batch_tokens_since_punctuation.clear()

    def register_bridge_embeddings(self, model: Any, tokenizer: Any, x_0: str, x_T: str) -> None:
        """Computes and registers embedding-space bridge vector v_bridge = normalize(e_xT - e_x0)."""
        if not TORCH_AVAILABLE:
            return
        try:
            emb_layer = model.get_input_embeddings()
            if emb_layer is None or not hasattr(emb_layer, "weight"):
                return
            self.embedding_weight = emb_layer.weight
            self._embed_weight = emb_layer.weight

            device = self.embedding_weight.device
            ids_0 = tokenizer.encode(x_0, add_special_tokens=False)
            ids_T = tokenizer.encode(x_T, add_special_tokens=False)

            if not ids_0 or not ids_T:
                return

            t_ids_0 = torch.tensor(ids_0, dtype=torch.long, device=device)
            t_ids_T = torch.tensor(ids_T, dtype=torch.long, device=device)

            e_0 = self.embedding_weight[t_ids_0].mean(dim=0).to(dtype=torch.float32)
            e_T = self.embedding_weight[t_ids_T].mean(dim=0).to(dtype=torch.float32)

            diff = e_T - e_0
            norm = torch.norm(diff, p=2)
            if norm > 1e-7:
                self.v_bridge = diff / norm
                self._bridge_dir_tensor = self.v_bridge
        except Exception:
            pass

    def register_tokenizer_punctuation(self, tokenizer: Any) -> None:
        """Auto-detects and registers punctuation token IDs and banned CJK/meta token IDs from HuggingFace tokenizer."""
        for punct in PUNCTUATION_CHARS:
            try:
                ids = tokenizer.encode(punct, add_special_tokens=False)
                for tid in ids:
                    self.punctuation_token_ids.add(tid)
            except Exception:
                pass
        for bad_tok in BANNED_CJK_AND_META_CHARS:
            try:
                ids = tokenizer.encode(bad_tok, add_special_tokens=False)
                if len(bad_tok) == 1 or bad_tok == "```":
                    for tid in ids:
                        self.banned_cjk_token_ids.add(tid)
                elif len(ids) == 1:
                    self.banned_cjk_token_ids.add(ids[0])
            except Exception:
                pass

        # Full-vocabulary scan for non-Japanese CJK characters via EUC-JP whitelist
        vocab = None
        if hasattr(tokenizer, "get_vocab") and callable(tokenizer.get_vocab):
            try:
                vocab = tokenizer.get_vocab()
            except Exception:
                vocab = None
        elif hasattr(tokenizer, "vocab") and isinstance(tokenizer.vocab, dict):
            vocab = tokenizer.vocab

        if isinstance(vocab, dict):
            for token_str, token_id in vocab.items():
                if is_non_japanese_cjk(token_str):
                    self.banned_cjk_token_ids.add(token_id)
        elif hasattr(tokenizer, "convert_ids_to_tokens") and hasattr(tokenizer, "__len__"):
            try:
                for tid in range(len(tokenizer)):
                    tok_text = tokenizer.convert_ids_to_tokens(tid)
                    if tok_text and is_non_japanese_cjk(tok_text):
                        self.banned_cjk_token_ids.add(tid)
            except Exception:
                pass
        self._banned_token_ids = self.banned_cjk_token_ids

        self._punct_tensor = None
        self._banned_tensor = None

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        """Called at each generation step by HuggingFace generate()."""
        self.call_count += 1

        if not TORCH_AVAILABLE or not isinstance(scores, torch.Tensor):
            return scores

        device = scores.device
        is_2d = scores.ndim == 2
        batch_size = scores.shape[0] if is_2d else 1

        # Cache punctuation and banned CJK/meta 1D tensors on target device
        if self.punctuation_token_ids and (
            self._punct_tensor is None or self._punct_tensor.device != device
        ):
            self._punct_tensor = torch.tensor(
                sorted(self.punctuation_token_ids), dtype=torch.long, device=device
            )
        if self.banned_cjk_token_ids and (
            self._banned_tensor is None or self._banned_tensor.device != device
        ):
            self._banned_tensor = torch.tensor(
                sorted(self.banned_cjk_token_ids), dtype=torch.long, device=device
            )

        embed_w = self._embed_weight if self._embed_weight is not None else self.embedding_weight
        if embed_w is not None and embed_w.device != device:
            embed_w = embed_w.to(device=device)
            self._embed_weight = embed_w

        bridge_dir = self._bridge_dir_tensor if self._bridge_dir_tensor is not None else self.v_bridge
        if bridge_dir is not None and bridge_dir.device != device:
            bridge_dir = F.normalize(bridge_dir.to(device=device, dtype=torch.float32), dim=-1)
            self._bridge_dir_tensor = bridge_dir

        # Suppress banned Chinese function words & markdown backticks across vocabulary (vectorized across batch)
        if self._banned_tensor is not None and self._banned_tensor.numel() > 0:
            valid_banned = self._banned_tensor[self._banned_tensor < scores.shape[-1]]
            if valid_banned.numel() > 0:
                scores[..., valid_banned] -= 100.0

        # Process each sequence in the batch with 100% vectorized GPU tensor ops
        for b in range(batch_size):
            curr_scores = scores[b] if is_2d else scores
            curr_input = input_ids[b] if input_ids is not None and input_ids.ndim == 2 else input_ids

            # Update tokens_since_punctuation based on last generated token
            if curr_input is not None and len(curr_input) > 0:
                last_token_id = int(curr_input[-1].item())
                if last_token_id in self.punctuation_token_ids:
                    self.batch_tokens_since_punctuation[b] = 0
                else:
                    self.batch_tokens_since_punctuation[b] = (
                        self.batch_tokens_since_punctuation.get(b, 0) + 1
                    )
            tokens_since_punct = self.batch_tokens_since_punctuation.get(b, 0)

            # 1. Top-K extraction (K=64)
            vocab_size = curr_scores.shape[-1]
            K = min(self.top_k, vocab_size)
            top_scores, top_indices = torch.topk(curr_scores, K)

            # 2. Shannon entropy over Top-K
            top_probs = F.softmax(top_scores, dim=-1)
            h_llm = float(-torch.sum(top_probs * torch.log(top_probs + 1e-12)).item())

            # 3. Syntax Gate: if H_LLM < eps_syn, bypass modification
            if h_llm < self.eps_syn:
                effective_beta = 0.0
            else:
                effective_beta = self.beta_steer * min(1.0, (h_llm - self.eps_syn) / self.eps_syn)

            # 4. Vectorized Refractory Period Gating on Top-K
            is_refractory_blocked = tokens_since_punct < self.refractory_min
            is_refractory_forced = tokens_since_punct > self.refractory_max

            if (is_refractory_blocked or is_refractory_forced) and self._punct_tensor is not None:
                is_punct_mask = torch.isin(top_indices, self._punct_tensor)
                if is_refractory_blocked:
                    top_scores = torch.where(is_punct_mask, top_scores - 1000.0, top_scores)
                elif is_refractory_forced:
                    top_scores = torch.where(is_punct_mask, top_scores + 3.5, top_scores)

            # 5. Vectorized Backward Attraction Steering on Top-K via single torch.mv
            if effective_beta > 0.0 and bridge_dir is not None and embed_w is not None:
                top_embs = F.normalize(embed_w[top_indices].float(), dim=-1)
                cos_sims = torch.mv(top_embs, bridge_dir)
                top_scores = top_scores + (effective_beta * cos_sims * 0.5).to(dtype=top_scores.dtype)

            curr_scores[top_indices] = top_scores

        return scores


# ==============================================================================
# Phase B WS-B1: Hierarchical Dual-Rate 5-Stage Adaptive-BAOAB & Clause-Level Twisted SMC
# ==============================================================================

def generate_1_over_f_noise(
    L: int = 16,
    dim: int = 12,
    variance: float = 0.02,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generates L steps of 1/f (pink) noise across dim channels with exact variance.

    OPEN-08: Used for preloading ghost history ring buffers with 1/f correlated
    Gaussian fluctuations around target attractor mu_star.
    """
    if L <= 0 or dim <= 0:
        return np.zeros((max(0, L), max(0, dim)), dtype=np.float64)
    if variance <= 0.0:
        return np.zeros((L, dim), dtype=np.float64)

    if rng is None:
        rng = np.random.default_rng()
    freqs = np.fft.rfftfreq(L)
    with np.errstate(divide="ignore"):
        scale = 1.0 / np.where(freqs == 0, float("inf"), np.sqrt(freqs))
    scale[0] = 0.0  # Zero DC component so mean is zero
    noise = rng.normal(0.0, 1.0, size=(dim, L))
    spec = np.fft.rfft(noise, axis=-1) * scale
    pink = np.fft.irfft(spec, n=L, axis=-1)
    current_std = np.std(pink, axis=-1, keepdims=True)
    current_std = np.where(current_std < 1e-8, 1.0, current_std)
    pink = pink / current_std * math.sqrt(variance)
    return pink.T  # Shape: (L, dim)


class BAOABStepResult(NamedTuple):
    """Result of a 5-stage Adaptive-BAOAB clause-level integration step."""

    q: np.ndarray          # Next target position q_{m+1}^* in R^12
    p: np.ndarray          # Next conjugate momentum p_{m+1}^* in R^12
    delta_q: np.ndarray    # Next target displacement Delta q_{m+1}^* = q_{m+1}^* - q_m^{shrunk}
    q_shrunk: np.ndarray   # Bayesian shrunk position hat{q}_m^{shrunk}
    gamma_t: float         # Effective friction coefficient (50.0 or 1.2)
    theta_t: float         # Friction dissipation factor exp(-Delta t * gamma_t)
    alpha_m: float         # Bayesian shrinkage gain m / (m + 5)
    m: int                 # Clause step counter


def step_baoab_clause(
    q: np.ndarray,
    p: np.ndarray,
    mu_star: Optional[np.ndarray] = None,
    G_style: Optional[np.ndarray] = None,
    h_syntax: Optional[float] = None,
    phi_bridge: Optional[np.ndarray] = None,
    q_measured: Optional[np.ndarray] = None,
    m: int = 1,
    dt: float = 0.25,
    tau: float = 0.04,
    eps_syn: float = 0.6,
    gamma_low: float = 1.2,
    gamma_high: float = 50.0,
    lambda_bridge: float = 1.0,
    mass_matrix: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> BAOABStepResult:
    """Executes a single 5-stage Adaptive-BAOAB symplectic step at a clause boundary.

    Outer Loop (Clause Boundary) equations:
        1. Bayesian Shrinkage:
           alpha_m = m / (m + 5)
           hat{q}_m^{shrunk} = alpha_m * hat{q}_m + (1 - alpha_m) * mu^*
        2. Potential Gradient:
           nabla_q V(q) = G_style (q - mu^*) - lambda_bridge * phi_bridge
        3. Stage B:
           p^{(1)} = p_m - (dt / 2) * nabla_q V(q_m)
        4. Stage A:
           q^{(1)} = q_m + (dt / 2) * M^{-1} p^{(1)}
        5. Stage O (Syntax-Entropy Dependent Friction Brake):
           gamma_t = 50.0 if H_syntax < eps_syn else 1.2
           theta_t = exp(-dt * gamma_t)
           p^{(2)} = theta_t * p^{(1)} + sqrt(tau * (1 - theta_t^2)) * M^{1/2} xi
        6. Stage A:
           q_{m+1}^* = q^{(1)} + (dt / 2) * M^{-1} p^{(2)}
        7. Stage B:
           p_{m+1}^* = p^{(2)} - (dt / 2) * nabla_q V(q_{m+1}^*)
        8. Target Displacement:
           Delta q_{m+1}^* = q_{m+1}^* - hat{q}_m^{shrunk}
    """
    if rng is None:
        rng = np.random.default_rng()

    q_curr = np.asarray(q, dtype=np.float64).ravel()
    p_curr = np.asarray(p, dtype=np.float64).ravel()
    dim = len(q_curr)
    if len(p_curr) != dim:
        raise ValueError(f"Dimension mismatch between q ({dim}) and p ({len(p_curr)})")

    if mu_star is None:
        mu = np.full(dim, 0.5, dtype=np.float64)
    else:
        mu = np.asarray(mu_star, dtype=np.float64).ravel()

    if G_style is None:
        G = np.eye(dim, dtype=np.float64)
    else:
        G = np.asarray(G_style, dtype=np.float64)

    if mass_matrix is None:
        M_inv = np.eye(dim, dtype=np.float64)
        M_sqrt = np.eye(dim, dtype=np.float64)
    else:
        M = np.asarray(mass_matrix, dtype=np.float64)
        M_inv = np.linalg.inv(M)
        eigvals, eigvecs = np.linalg.eigh(M)
        M_sqrt = eigvecs @ np.diag(np.sqrt(np.maximum(1e-12, eigvals))) @ eigvecs.T

    # 1. Bayesian shrinkage
    alpha_m = float(m) / float(m + 5) if m > 0 else 0.0
    if q_measured is not None:
        q_meas = np.asarray(q_measured, dtype=np.float64).ravel()
    else:
        q_meas = q_curr
    q_shrunk = alpha_m * q_meas + (1.0 - alpha_m) * mu

    # Gradient closure
    phi = np.zeros(dim, dtype=np.float64) if phi_bridge is None else np.asarray(phi_bridge, dtype=np.float64).ravel()

    def grad_V(x: np.ndarray) -> np.ndarray:
        return (G @ (x - mu)) - (lambda_bridge * phi)

    dt_half = 0.5 * dt

    # 2. Stage B
    p_1 = p_curr - dt_half * grad_V(q_curr)

    # 3. Stage A
    q_1 = q_curr + dt_half * (M_inv @ p_1)

    # 4. Stage O (Syntax Protection Brake)
    if h_syntax is not None and float(h_syntax) < eps_syn:
        gamma_t = float(gamma_high)  # 50.0 (Strong brake)
    else:
        gamma_t = float(gamma_low)   # 1.2 (Underdamped)

    theta_t = math.exp(-dt * gamma_t)
    sigma_noise = math.sqrt(max(0.0, tau * (1.0 - theta_t**2)))
    xi = rng.normal(0.0, 1.0, size=dim)
    p_2 = theta_t * p_1 + sigma_noise * (M_sqrt @ xi)

    # 5. Stage A
    q_next = q_1 + dt_half * (M_inv @ p_2)

    # 6. Stage B
    p_next = p_2 - dt_half * grad_V(q_next)

    # 7. Next clause target displacement
    delta_q = q_next - q_shrunk

    return BAOABStepResult(
        q=q_next,
        p=p_next,
        delta_q=delta_q,
        q_shrunk=q_shrunk,
        gamma_t=gamma_t,
        theta_t=theta_t,
        alpha_m=alpha_m,
        m=m,
    )


class AdaptiveBAOABIntegrator:
    """Hierarchical Dual-Rate 5-Stage Adaptive-BAOAB Symplectic Integrator (Phase B WS-B1).

    Maintains phase-space state (q, p) in R^12 x R^12, ghost history ring buffer of length L=16,
    Bayesian shrinkage against target attractor mu^*, and syntax-entropy-dependent variable friction.
    """

    def __init__(
        self,
        dim: int = 12,
        mu_star: Optional[Union[np.ndarray, Sequence[float]]] = None,
        G_style: Optional[np.ndarray] = None,
        mass_matrix: Optional[np.ndarray] = None,
        dt: float = 0.25,
        tau: float = 0.04,
        eps_syn: float = 0.6,
        gamma_low: float = 1.2,
        gamma_high: float = 50.0,
        lambda_bridge: float = 1.0,
        phi_bridge: Optional[np.ndarray] = None,
        L_ghost: int = 16,
        variance_ghost: float = 0.02,
        seed: Optional[int] = None,
    ) -> None:
        self.dim = int(dim)
        self.rng = np.random.default_rng(seed)
        self.dt = float(dt)
        self.tau = float(tau)
        self.eps_syn = float(eps_syn)
        self.gamma_low = float(gamma_low)
        self.gamma_high = float(gamma_high)
        self.lambda_bridge = float(lambda_bridge)
        self.phi_bridge = None if phi_bridge is None else np.asarray(phi_bridge, dtype=np.float64).ravel()
        self.L_ghost = int(L_ghost)
        self.variance_ghost = float(variance_ghost)

        if mu_star is not None:
            self.mu_star = np.asarray(mu_star, dtype=np.float64).ravel()
        else:
            self.mu_star = np.full(self.dim, 0.5, dtype=np.float64)

        if G_style is not None:
            self.G_style = np.asarray(G_style, dtype=np.float64)
        else:
            self.G_style = np.eye(self.dim, dtype=np.float64)

        if mass_matrix is not None:
            self.mass_matrix = np.asarray(mass_matrix, dtype=np.float64)
            self.M_inv = np.linalg.inv(self.mass_matrix)
            eigvals, eigvecs = np.linalg.eigh(self.mass_matrix)
            self.M_sqrt = eigvecs @ np.diag(np.sqrt(np.maximum(1e-12, eigvals))) @ eigvecs.T
        else:
            self.mass_matrix = np.eye(self.dim, dtype=np.float64)
            self.M_inv = np.eye(self.dim, dtype=np.float64)
            self.M_sqrt = np.eye(self.dim, dtype=np.float64)

        # Ring buffer for Ghost History (OPEN-08)
        self.ghost_history: collections.deque[np.ndarray] = collections.deque(maxlen=self.L_ghost)
        self.m: int = 0
        self.q = self.mu_star.copy()
        self.p = np.zeros(self.dim, dtype=np.float64)
        self.history_q: List[np.ndarray] = []

        self.preload_ghost_history()

    def preload_ghost_history(self, L: Optional[int] = None, variance: Optional[float] = None) -> None:
        """Preloads ring buffer with mu^* + 1/f noise (OPEN-08)."""
        l_len = self.L_ghost if L is None else int(L)
        var = self.variance_ghost if variance is None else float(variance)
        noise = generate_1_over_f_noise(L=l_len, dim=self.dim, variance=var, rng=self.rng)
        self.ghost_history.clear()
        for k in range(l_len):
            self.ghost_history.append(self.mu_star + noise[k])

    def compute_shrinkage(self, q_measured: np.ndarray, m: Optional[int] = None) -> np.ndarray:
        """Computes Bayesian shrinkage hat{q}_m^{shrunk} = alpha_m * hat{q}_m + (1 - alpha_m) * mu^*."""
        step_idx = self.m if m is None else int(m)
        alpha = float(step_idx) / float(step_idx + 5) if step_idx > 0 else 0.0
        meas = np.asarray(q_measured, dtype=np.float64).ravel()
        return alpha * meas + (1.0 - alpha) * self.mu_star

    def grad_V(self, q: np.ndarray, phi_bridge: Optional[np.ndarray] = None) -> np.ndarray:
        """Computes potential gradient nabla_q V(q) = G_style (q - mu^*) - lambda_bridge * phi_bridge."""
        x = np.asarray(q, dtype=np.float64).ravel()
        phi_eff = self.phi_bridge if phi_bridge is None else phi_bridge
        phi = np.zeros(self.dim, dtype=np.float64) if phi_eff is None else np.asarray(phi_eff, dtype=np.float64).ravel()
        return (self.G_style @ (x - self.mu_star)) - (self.lambda_bridge * phi)

    def step_clause(
        self,
        q_measured: Optional[np.ndarray] = None,
        h_syntax: Optional[float] = None,
        phi_bridge: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
        tau: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> BAOABStepResult:
        """Performs a 5-stage B-A-O-A-B update at clause boundary."""
        self.m += 1
        phi_eff = self.phi_bridge if phi_bridge is None else phi_bridge
        res = step_baoab_clause(
            q=self.q,
            p=self.p,
            mu_star=self.mu_star,
            G_style=self.G_style,
            h_syntax=h_syntax,
            phi_bridge=phi_eff,
            q_measured=q_measured,
            m=self.m,
            dt=self.dt if dt is None else float(dt),
            tau=self.tau if tau is None else float(tau),
            eps_syn=self.eps_syn,
            gamma_low=self.gamma_low,
            gamma_high=self.gamma_high,
            lambda_bridge=self.lambda_bridge,
            mass_matrix=self.mass_matrix,
            rng=self.rng if rng is None else rng,
        )
        self.q = res.q.copy()
        self.p = res.p.copy()
        self.ghost_history.append(res.q_shrunk.copy())
        self.history_q.append(self.q.copy())
        return res

    def reset(
        self,
        q_init: Optional[np.ndarray] = None,
        p_init: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Resets the integrator state for a new generation run."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.m = 0
        self.q = self.mu_star.copy() if q_init is None else np.asarray(q_init, dtype=np.float64).ravel().copy()
        self.p = np.zeros(self.dim, dtype=np.float64) if p_init is None else np.asarray(p_init, dtype=np.float64).ravel().copy()
        self.history_q.clear()
        self.preload_ghost_history()

    def verify_invariant_momentum_measure(
        self,
        n_steps: int = 10000,
        tau: Optional[float] = None,
        seed: int = 42,
        tol_cov: float = 0.15,
        tol_mean: float = 0.06,
        full_baoab: bool = False,
    ) -> Dict[str, Any]:
        """Verifies Theorem 5.1: Stationary momentum distribution invariance N(0, tau M) under variable friction.

        When full_baoab is True, executes the full coupled 5-stage B-A-O-A-B Hamiltonian Langevin system.
        When full_baoab is False, verifies the exact Stage O momentum thermalization substep.
        """
        eff_tau = self.tau if tau is None else float(tau)
        rng = np.random.default_rng(seed)
        p_curr = math.sqrt(eff_tau) * (self.M_sqrt @ rng.normal(0.0, 1.0, size=self.dim))
        dt = self.dt
        samples = np.zeros((n_steps, self.dim), dtype=np.float64)

        if full_baoab:
            q_curr = self.mu_star.copy()
            for i in range(n_steps):
                h_syn = 0.2 if rng.random() < 0.5 else 1.2
                res = step_baoab_clause(
                    q=q_curr,
                    p=p_curr,
                    mu_star=self.mu_star,
                    G_style=self.G_style,
                    h_syntax=h_syn,
                    phi_bridge=self.phi_bridge,
                    m=i + 1,
                    dt=dt,
                    tau=eff_tau,
                    eps_syn=self.eps_syn,
                    gamma_low=self.gamma_low,
                    gamma_high=self.gamma_high,
                    lambda_bridge=self.lambda_bridge,
                    mass_matrix=self.mass_matrix,
                    rng=rng,
                )
                q_curr = res.q
                p_curr = res.p
                samples[i] = p_curr
        else:
            for i in range(n_steps):
                # Toggle between strong brake (50.0) and underdamped (1.2) randomly
                gamma_k = self.gamma_high if rng.random() < 0.5 else self.gamma_low
                theta = math.exp(-gamma_k * dt)
                sigma_noise = math.sqrt(max(0.0, eff_tau * (1.0 - theta**2)))
                xi = rng.normal(0.0, 1.0, size=self.dim)
                p_curr = theta * p_curr + sigma_noise * (self.M_sqrt @ xi)
                samples[i] = p_curr

        mean_p = np.mean(samples, axis=0)
        target_cov = eff_tau * self.mass_matrix
        emp_cov = np.cov(samples, rowvar=False)

        cov_frob_err = float(np.linalg.norm(emp_cov - target_cov) / np.linalg.norm(target_cov))
        diag_err = float(np.mean(np.abs(np.diag(emp_cov) - np.diag(target_cov))) / eff_tau)
        max_abs_mean = float(np.max(np.abs(mean_p)))
        passed = bool(cov_frob_err < tol_cov and max_abs_mean < tol_mean)

        return {
            "n_steps": n_steps,
            "max_abs_mean": max_abs_mean,
            "cov_relative_error": cov_frob_err,
            "diag_relative_error": diag_err,
            "passed": passed,
            "empirical_mean": mean_p,
            "empirical_cov": emp_cov,
            "target_cov": target_cov,
        }


def has_non_jis_kanji(text: str) -> bool:
    """True if text contains an ideograph outside JIS X 0208 (strict Shift_JIS).

    Simplified-Chinese-only forms (这, 们, 变, ...) are not encodable in JIS X 0208, so no character list
    has to be maintained: the boundary is the Japanese kanji standard itself.
    """
    for ch in text:
        o = ord(ch)
        if ch in "\uff0c\uff0e":  # full-width comma / full stop: Chinese-style punctuation, not 、。
            return True
        if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF:
            try:
                ch.encode("shift_jis")
            except UnicodeEncodeError:
                return True
    return False


_QUOTE_RE = re.compile(r"[「『][^」』]*[」』]")
# Narration (outside quotes) must not use assistant-style registers. Syntactic classes only (no work-specific words):
#   polite   : desu/masu-family clause endings, volitional/imperative polite forms
#   meta     : omission / continuation / summary / second-person-address markers
#   paren    : parenthetical annotation (...) / （...） inside prose
_POLITE_RE = re.compile(r"(?:です|ます|でした|ました|でしょう|ましょう|ください|ません|ませんでした)(?:[ねよわ])?[、。！？!?]")
_META_RE = re.compile(r"省略|以下の|続きは|ご覧|いかがでしょう|参考にし|※|お楽しみ|ご紹介|解説|[【】]")
# Surfaces suppressed at sampling time outside quotes (token ids whose decoded text contains one of these)
_FORBIDDEN_SURFACES = ("です", "ます", "でした", "ました", "ましょう", "ですね", "ますね", "ください",
                       "（", "）", "(", ")", "【", "】", "※", "省略",
                       # review / commentary vocabulary (logit suppression only, never string censorship of output)
                       "小説", "作品", "読者", "筆者", "著者", "物語", "描写", "表現")
# tokens that complete a polite form after a stem token ("まし"+"た"); suppressed right after a stem
_POST_POLITE_TOKENS = ("た", "ょう", "ん")
_POLITE_STEMS = ("まし", "でし", "ませ")
# Syntactic Finality Gate: endings after which a sentence cannot close ('。' suppressed) and terminal forms after which it can.
_CONNECTIVE_SUFFIXES = ("つれ", "ながら", "から", "ので", "たら", "なら", "つつ")
_CONNECTIVE_CHARS = "てでにをがはもとばしりず"  # 'く' is handled separately (hiragana + く: なく よく; kanji + く: 吹く is terminal)
_TERMINAL_CHARS = "ただるいう"
_FRAGMENT_RE = re.compile("(?:" + "|".join(_CONNECTIVE_SUFFIXES) + "|[" + _CONNECTIVE_CHARS + "]|(?<=[ぁ-ゟ])く)。")


def _is_connective_end(s: str) -> bool:
    """True when `s` ends with a connective / case-particle / continuative form (a sentence cannot end here)."""
    s = s.rstrip()
    if not s:
        return False
    if s[-1] == "く":
        return len(s) > 1 and "ぁ" <= s[-2] <= "ゟ"
    return s.endswith(_CONNECTIVE_SUFFIXES) or s[-1] in _CONNECTIVE_CHARS


def _is_terminal_end(s: str) -> bool:
    """True when `s` ends with a terminal / past form (た だ る い う) or a noun (kanji / katakana)."""
    s = s.rstrip()
    if not s:
        return False
    ch = s[-1]
    return ch in _TERMINAL_CHARS or ("一" <= ch <= "鿿") or ("ァ" <= ch <= "ー")
_PAREN_RE = re.compile(r"[（(][^）)]*[）)]|[（(][^）)]*$")


def narration_text(text: str, in_quote: bool = False) -> str:
    """Text with quoted speech removed. If in_quote, the text starts inside an already-open 「."""
    if in_quote:
        m = re.search(r"[」』]", text)
        text = text[m.end():] if m else ""
    text = _QUOTE_RE.sub("", text)
    # a quote opened and not yet closed at the end: drop the open tail
    m = re.search(r"[「『][^」』]*$", text)
    return text[: m.start()] if m else text


def count_narration_meta(text: str) -> Dict[str, int]:
    """Counts assistant-style marks in narration (outside quotes): polite endings, meta words, parentheses."""
    nar = narration_text(text)
    return {
        "polite": len(_POLITE_RE.findall(nar)),
        "meta": len(_META_RE.findall(nar)),
        "paren": len(_PAREN_RE.findall(nar)),
    }


def count_non_jis_kanji(text: str) -> int:
    n = 0
    for ch in text:
        o = ord(ch)
        if ch in "\uff0c\uff0e":
            n += 1
            continue
        if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF:
            try:
                ch.encode("shift_jis")
            except UnicodeEncodeError:
                n += 1
    return n


class ClauseTwistedSMCDecoder:
    """Clause-Level Twisted Sequential Monte Carlo (SMC) Decoder (P=4 Particles, Phase B WS-B1).

    Evaluates P=4 candidate clauses at each clause boundary:
        w^{(p)} = ln P_LLM(c^{(p)} | w_{<=m})
                + gamma_reach * ln P_LLM(W_{end}^* | w_{<=m} o c^{(p)})
                - (beta_steer / 2) * ||hat{q}(w_{<=m} o c^{(p)}) - q_{m+1}^*||_{G_style}^2
    Selects the optimal particle via resampling and updates Adaptive-BAOAB state.
    Provides mock/heuristic generator and scoring fallbacks for CPU / offline unit tests.
    """

    def __init__(
        self,
        integrator: Optional[AdaptiveBAOABIntegrator] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        n_particles: int = 4,
        temperatures: Sequence[float] = (0.65, 0.72, 0.80, 0.88),
        gamma_reach: float = 1.0,
        beta_steer: float = 1.0,
        G_style: Optional[np.ndarray] = None,
        max_clause_tokens: int = 20,
        feature_extractor: Optional[Callable[[str], np.ndarray]] = None,
        resample_method: str = "map",
        dim: int = 12,
        seed: Optional[int] = None,
        chat_prefix: str = "",
        auto_gain: bool = False,
        steer_balance: float = 1.0,
        style_window_chars: Optional[int] = None,
        temp_gain: float = 0.0,
        p_ref: float = 0.5,
        gain_max: Optional[float] = None,
        temp_min: float = 0.8,
        temp_max: float = 1.35,
        repeat_guard: bool = False,
        repeat_ngram: int = 6,
        repeat_max: int = 2,
        ascii_guard: bool = False,
        eos_penalty: float = 10.0,
        repeat_window_chars: Optional[int] = None,
        script_guard: bool = False,
        spread_temps: bool = False,
        temp_spread_lo: float = 0.7,
        temp_spread_hi_max: float = 1.3,
        short_bias: bool = False,
        period_bias_max: float = 4.0,
        climax_particles: Optional[int] = None,
        close_after: Optional[int] = None,
        close_period_boost: float = 5.0,
        normalize_end: bool = False,
        meta_guard: bool = False,
        polite_guard: bool = False,
        max_regen: int = 0,
        direction_steer: bool = False,
        narration_mask: bool = False,
        narration_bias: float = -15.0,
        register_penalty: float = 1e4,
        direction_kappa: float = 2.5,
        len_gain: float = 0.9,
        len_ramp: float = 6.0,
        temp_dir_gain: float = 0.25,
        align_balance: float = 1.0,
        ngram_ban: int = 0,
        rep_penalty: float = 1.0,
        rep_regen: int = 3,
        min_cap: int = 8,
        gamma_dir: float = 5.0,
        min_p: float = 0.0,
        phase_thr: float = 0.25,
        syntactic_gate: bool = False,
        gate_bias: float = -12.0,
        ngram_soft: int = 0,
        ngram_soft_penalty: float = 4.0,
        char8_max: int = 0,
        content_rep_penalty: float = 1.0,
        lex_decay: float = 0.0,
        rhythm: bool = False,
        cv_gain: float = 0.0,
        cv_floor: float = 0.42,
    ) -> None:
        # cv_gain / cv_floor : SMC log-weight term -cv_gain * max(0, cv_floor - CV_L^(k)), where CV_L^(k) is the coefficient of
        #                      variation of the sentence lengths of the paragraph (confirmed + the sentence completed by candidate k)
        self.cv_gain = float(cv_gain)
        self.cv_floor = float(cv_floor)
        # content_rep_penalty : repetition penalty applied ONLY to content-word tokens (contain a kanji, or are katakana
        #                       words of >= 2 chars); single hiragana / punctuation / multi-hiragana grammar tokens are exempt
        #                       (okurigana, conjugation endings and particles are never penalised)
        # lex_decay           : logit -lex_decay for tokens with >= 2 kanji that already occurred >= 2 times in the current paragraph
        # rhythm              : base 3-beat sentence rhythm (short 1 clause / medium 2 clauses / long 3-4 clauses) outside the
        #                       Shock/Flux regimes
        self.content_rep_penalty = float(content_rep_penalty)
        self.lex_decay = float(lex_decay)
        self.rhythm = bool(rhythm)
        self._tok_class = None
        self._cur_prefix = ""
        # syntactic_gate : Syntactic Finality Gate. Logit stage: '。' gets `gate_bias` (-12) after a connective /
        #                  continuative ending (て で に を が は も と から ので ば たら なら つつ つれ ながら し り く ず ...);
        #                  the positive '。' bias fires only after a terminal form (た だ る い う) or a noun (kanji/katakana).
        #                  SMC stage: candidates ending in "<continuative>。" get register_penalty.
        # ngram_soft     : n of the soft token n-gram decay (logit -ngram_soft_penalty); ngram_ban (hard, -1e4) is separate
        # char8_max      : a candidate that would make any 8-char substring occur more than char8_max times in the
        #                  text so far (history + current paragraph) is rejected (SMC -register_penalty + resample)
        self.syntactic_gate = bool(syntactic_gate)
        self.gate_bias = float(gate_bias)
        self.ngram_soft = int(ngram_soft)
        self.ngram_soft_penalty = float(ngram_soft_penalty)
        self.char8_max = int(char8_max)
        # gamma_dir : coupling of the local Shock/Flux potential gamma_dir * <p_hat, q_local(y_k) - mean_k q_local>
        # min_p     : Min-p filter on the tempered proposal distribution (p < min_p * p_max -> -inf)
        # phase_thr : |d| above which the punctuation-phase modulation is active
        #   Shock (d > thr): '、' -8d, '。' +5d, clause token targets cycle short / very short / medium (one clause = one sentence)
        #   Flux  (d < -thr): '。' -5|d| and '、' +3|d| while the current sentence has < 2 clauses (long compound sentences)
        self.gamma_dir = float(gamma_dir)
        self.min_p = float(min_p)
        self.phase_thr = float(phase_thr)
        self._clause_idx = 0
        # ngram_ban   : context-wide token n-gram exclusion at proposal time (logit -1e4 for any token that would
        #               re-create an n-gram already present in history_text / the prompt / the row so far)
        # rep_penalty : HF repetition_penalty on tokens already in the row context
        # rep_regen   : when every candidate repeats recent text -> temperature +0.3, first tokens of the repeated
        #               candidates masked (-1e4), resample (up to rep_regen times)
        self.ngram_ban = int(ngram_ban)
        self.rep_penalty = float(rep_penalty)
        self.rep_regen = int(rep_regen)
        self.min_cap = int(min_cap)
        self.history_text = ""
        self._hist_cache = (None, None)
        self._temp_boost = 0.0
        self._first_ban_ids = []
        # narration_mask : NarrationStyleLogitsMask - outside 「」 the logits of polite-auxiliary / meta-mark tokens
        #                  get `narration_bias` (pre-generation control, not post-hoc string matching)
        # direction_steer (continuous, no dead band): d = tanh(kappa (S-F)/(S+F)) modulates the '。'/'、' bias,
        #   the per-candidate clause length cap (exp(-len_gain d ...), soft punctuation ramp `len_ramp`), the temperature
        #   spread (temp_dir_gain), and an SMC term align_balance * <p_discharge_hat, dq_hat_k> (cosine alignment)
        self.repeat_use_context = False
        self.narration_mask = bool(narration_mask)
        self.narration_bias = float(narration_bias)
        self.register_penalty = float(register_penalty)
        self.direction_kappa = float(direction_kappa)
        self.len_gain = float(len_gain)
        self.len_ramp = float(len_ramp)
        self.temp_dir_gain = float(temp_dir_gain)
        self.align_balance = float(align_balance)
        self._forbidden_cache = None
        # Assistant-register guards (narration only; quoted speech is exempt):
        #   meta_guard   : block candidates with meta words (omission/summary/address) or parenthetical notes
        #   polite_guard : block candidates with desu/masu-family endings
        #   max_regen    : when every candidate is blocked, regenerate up to this many times before giving up
        # Direction steering (needs steering_active): d = (S - F) / (S + F) from the discharged momentum vector,
        #   S = mean|p[Shock 3,7,11]|, F = mean|p[Flux 0,4,8]|. d > 0 -> '。' bias (short assertions),
        #   d < 0 -> '、' bias (long flowing sentences); |d| < direction_min -> no bias.
        self.meta_guard = bool(meta_guard)
        self.polite_guard = bool(polite_guard)
        self.max_regen = int(max_regen)
        self.direction_steer = bool(direction_steer)
        self.discharge_vec = None
        # Proposal-side momentum injection (Climax / steering_active only):
        #   spread_temps    : per-candidate temperatures linspace(temp_spread_lo, t_hi), t_hi grows with ln(||p||/p_ref)
        #   short_bias      : per-row bias that raises the "。" logit and lowers "、" (rows ordered short -> long),
        #                     scaled by the momentum factor, so short assertive clauses are always in the pool
        #   climax_particles: candidate pool size used while steering is active (director toggles n_particles)
        #   close_after     : from this clause index on, "、" is suppressed and "。" boosted (finish the sentence)
        #   normalize_end   : a text that still ends with "、" gets its last mark replaced by "。"
        self.spread_temps = bool(spread_temps)
        self.temp_spread_lo = float(temp_spread_lo)
        self.temp_spread_hi_max = float(temp_spread_hi_max)
        self.short_bias = bool(short_bias)
        self.period_bias_max = float(period_bias_max)
        self.climax_particles = None if climax_particles is None else int(climax_particles)
        self.close_after = None if close_after is None else int(close_after)
        self.close_period_boost = float(close_period_boost)
        self.normalize_end = bool(normalize_end)
        self._force_close = False
        self._punct_id_cache = None
        self.discharge_norm = None
        self.discharge_ref = 0.5
        # repeat_window_chars: window used by repeat_guard (default: the style window / 120 chars)
        # script_guard       : also block candidates containing simplified-Chinese-only characters or the
        #                      full-width comma (script collapse into Chinese), in addition to latin words
        self.repeat_window_chars = None if repeat_window_chars is None else int(repeat_window_chars)
        self.script_guard = bool(script_guard)
        # Stability guards (all off by default = legacy behaviour):
        #   gain_max      : upper clip of the auto_gain factor (no unbounded amplification of tiny style differences)
        #   temp_min/max  : bounds of the momentum-coupled temperature multiplier
        #   repeat_guard  : candidate whose repeat_ngram-char substring already occurs >= repeat_max times in the
        #                   trailing style window gets weight -inf (3rd occurrence of the same phrase is blocked)
        #   ascii_guard   : candidate containing a 3+ letter latin word is blocked when the context has none
        #   eos_penalty   : subtracted from EOS logits while fewer than min_clauses clauses were generated
        self.gain_max = None if gain_max is None else float(gain_max)
        self.temp_min = float(temp_min)
        self.temp_max = float(temp_max)
        self.repeat_guard = bool(repeat_guard)
        self.repeat_ngram = int(repeat_ngram)
        self.repeat_max = int(repeat_max)
        self.ascii_guard = bool(ascii_guard)
        self.eos_penalty = float(eos_penalty)
        self.steering_active = True
        self._suppress_eos = False
        # Steering controls (all off by default = legacy behaviour):
        #   auto_gain         : rescale the style term so that std_p(style) = steer_balance * std_p(ln P_LLM)
        #   style_window_chars: measure candidate style on the trailing window only (a cumulative prefix
        #                       dilutes the difference between candidates as the paragraph grows)
        #   temp_gain         : candidate temperatures scaled by clip(1 + temp_gain * ln(||p|| / p_ref), 0.8, 1.35)
        self.auto_gain = bool(auto_gain)
        self.steer_balance = float(steer_balance)
        self.style_window_chars = None if style_window_chars is None else int(style_window_chars)
        self.temp_gain = float(temp_gain)
        self.p_ref = float(p_ref)
        self.chat_prefix = str(chat_prefix)
        self.verbose = False
        self.eos_reached = False
        self.stats: Dict[str, int] = {"candidates": 0, "incomplete": 0, "logprob_errors": 0}
        self.dim = int(dim)
        self.n_particles = int(n_particles)
        self.base_n_particles = self.n_particles
        self.temperatures = list(temperatures)[:self.n_particles]
        while len(self.temperatures) < self.n_particles:
            self.temperatures.append(0.72)
        self.gamma_reach = float(gamma_reach)
        self.beta_steer = float(beta_steer)
        self.max_clause_tokens = int(max_clause_tokens)
        self.resample_method = str(resample_method).lower()
        self.rng = np.random.default_rng(seed)

        if G_style is not None:
            self.G_style = np.asarray(G_style, dtype=np.float64)
        else:
            self.G_style = np.eye(self.dim, dtype=np.float64)

        if integrator is not None:
            self.integrator = integrator
        else:
            self.integrator = AdaptiveBAOABIntegrator(dim=self.dim, G_style=self.G_style, seed=seed)

        self.model = model
        self.tokenizer = tokenizer

        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
        else:
            def _extract_q(text: str) -> np.ndarray:
                try:
                    from usir_py.extractor_24d import extract_usir_24d
                    res = extract_usir_24d(text)
                    return np.asarray(res["q"], dtype=np.float64)
                except Exception:
                    # Deterministic fallback mapping characters to 12D coordinates
                    q = np.full(self.dim, 0.5, dtype=np.float64)
                    if text:
                        chars = list(text)
                        for i, ch in enumerate(chars[:12]):
                            q[i % self.dim] = ((ord(ch) * 31) % 100) / 100.0
                    return q
            self.feature_extractor = _extract_q

    def _compute_hf_logprob(self, context: str, target: str) -> float:
        """Computes average token log-likelihood of target conditioned on context using HuggingFace model."""
        if not TORCH_AVAILABLE or self.model is None or self.tokenizer is None:
            return 0.0
        try:
            device = next(self.model.parameters()).device
            ctx_ids = self.tokenizer.encode(self.chat_prefix + context, add_special_tokens=False) if (self.chat_prefix or context) else []
            tgt_ids = self.tokenizer.encode(target, add_special_tokens=False) if target else []
            if not tgt_ids:
                return 0.0

            # If context is empty, use BOS token if available so first target token can be predicted
            if len(ctx_ids) == 0:
                bos_id = getattr(self.tokenizer, "bos_token_id", None)
                if bos_id is not None:
                    ctx_ids = [bos_id]

            full_ids = ctx_ids + tgt_ids
            input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = self.model(input_tensor).logits[0]

            if len(ctx_ids) > 0:
                start_pos = len(ctx_ids) - 1
                end_pos = len(full_ids) - 1
                target_logits = logits[start_pos:end_pos]
                target_labels = torch.tensor(tgt_ids, dtype=torch.long, device=device)
            else:
                # No context and no BOS token: logits[0] predicts tgt_ids[1]
                target_logits = logits[:len(tgt_ids) - 1]
                target_labels = torch.tensor(tgt_ids[1:], dtype=torch.long, device=device)

            if target_logits.shape[0] == 0 or target_labels.shape[0] == 0:
                return 0.0

            log_probs = F.log_softmax(target_logits, dim=-1)
            token_logprobs = log_probs.gather(1, target_labels.unsqueeze(-1)).squeeze(-1)
            return float(token_logprobs.mean().item())
        except Exception:
            # Never report an error as log-prob 0.0 (= certainty); use a strongly negative sentinel.
            self.stats["logprob_errors"] += 1
            return -20.0

    def _blocked_candidates(self, prefix: str, cand_texts: Sequence[str]) -> np.ndarray:
        """Boolean mask of candidates rejected by the repetition / latin-word guards."""
        blocked = np.zeros(len(cand_texts), dtype=bool)
        # exact 8-char substring ceiling (count would exceed char8_max) and continuative-ending fragments
        blocked |= self._char8_hits(prefix, cand_texts)
        blocked |= self._fragment_hits(cand_texts) > 0
        if not (self.repeat_guard or self.ascii_guard or self.script_guard):
            return blocked
        win_len = self.repeat_window_chars or self.style_window_chars or 120
        # With carried story context the repetition window must also cover the carried paragraphs
        window = ((self.chat_prefix if self.repeat_use_context else "") + prefix)[-win_len:]
        ctx_has_latin = re.search(r"[A-Za-z]{3,}", prefix) is not None
        n = self.repeat_ngram
        for i, c in enumerate(cand_texts):
            body = re.sub(r"[、。！？!?\s]+", "", c)
            if self.ascii_guard and not ctx_has_latin and re.search(r"[A-Za-z]{3,}", c):
                blocked[i] = True
                continue
            if self.script_guard and has_non_jis_kanji(c):
                blocked[i] = True
                continue
            if self.repeat_guard and len(body) >= n:
                win = re.sub(r"[、。！？!?\s]+", "", window)
                for j in range(0, len(body) - n + 1):
                    if win.count(body[j:j + n]) >= self.repeat_max:
                        blocked[i] = True
                        break
        return blocked

    def _register_hits(self, context: str, cand_texts: Sequence[str]) -> np.ndarray:
        """Number of assistant-register violations per candidate (narration only; quoted speech exempt).

        Last line of defence behind NarrationStyleLogitsMask: every hit costs `register_penalty` in the SMC weight.
        There is no all-blocked pass-through: the penalty is always applied.
        """
        hits = np.zeros(len(cand_texts), dtype=np.float64)
        if not (self.meta_guard or self.polite_guard):
            return hits
        full = self.chat_prefix + context
        in_quote = (full.count("「") + full.count("『")) > (full.count("」") + full.count("』"))
        for i, c in enumerate(cand_texts):
            nar = narration_text(c, in_quote=in_quote)
            if self.meta_guard:
                hits[i] += len(_META_RE.findall(nar)) + len(_PAREN_RE.findall(nar))
            if self.polite_guard:
                hits[i] += len(_POLITE_RE.findall(nar))
        return hits

    def _forbidden_token_ids(self):
        """Token ids whose surface contains a polite auxiliary / meta mark (cached per decoder)."""
        if self._forbidden_cache is None:
            tk = self.tokenizer
            n = len(tk)
            surfaces = tk.batch_decode([[i] for i in range(n)], skip_special_tokens=False)
            bad = [i for i, s in enumerate(surfaces) if any(f in s for f in _FORBIDDEN_SURFACES)]
            tail_next = [i for i, s in enumerate(surfaces) if s in _POST_POLITE_TOKENS]
            self._forbidden_cache = (bad, tail_next)
        return self._forbidden_cache

    def _row_caps(self, P: int) -> List[int]:
        """Per-candidate max clause length in tokens.

        Shock (d > phase_thr): targets cycle over the clause index (short / very short / medium) so that sentence
        lengths vary (CV_L) while every clause is a short single sentence. Flux: long ceiling. Otherwise graded by d.
        """
        base = self.max_clause_tokens
        rp = self._rhythm_params(self._cur_prefix)
        if rp is not None:
            return [int(rp[2])] * P
        if not (self.direction_steer and self.steering_active) or P < 2:
            return [base] * P
        d = self._direction()
        if d > self.phase_thr:
            target = base * (0.7, 0.4, 1.0)[self._clause_idx % 3]
            return [int(np.clip(round(target * (1.0 - 0.25 * i / (P - 1))), self.min_cap, int(1.6 * base))) for i in range(P)]
        mf = self._momentum_factor()
        caps = []
        for i in range(P):
            c = base * math.exp(-self.len_gain * d * mf * (1.0 + i / (P - 1)))
            caps.append(int(np.clip(round(c), self.min_cap, int(1.6 * base))))
        return caps

    def _phase_bias(self, prefix: str):
        """(period_bias, comma_bias) of the punctuation-phase modulation; (0, 0) when inactive."""
        rp = self._rhythm_params(prefix)
        if rp is not None:
            return rp[0], rp[1]
        if not (self.direction_steer and self.steering_active):
            return 0.0, 0.0
        d = self._direction()
        if d > self.phase_thr:
            return 5.0 * d, -8.0 * d
        if d < -self.phase_thr:
            tail = prefix.rsplit("\u3002", 1)[-1]
            if tail.count("\u3001") < 2:
                return -5.0 * abs(d), 3.0 * abs(d)
        return 0.0, 0.0

    def _momentum_factor(self) -> float:
        """Strength of the proposal-side injection in [0, 1).

        With a released momentum (discharge_norm set by the director) the factor is 1 - exp(-||p_discharge|| / d0):
        smooth and non-saturating, so different banked amounts give different proposal distributions.
        Without it, a log-scale function of ||p|| (0 at p_ref, 1 at 5 p_ref) is used.
        """
        if self.discharge_norm is not None:
            return float(1.0 - math.exp(-max(0.0, self.discharge_norm) / self.discharge_ref))
        p_norm = float(np.linalg.norm(np.asarray(self.integrator.p, dtype=np.float64)))
        if p_norm <= self.p_ref:
            return 0.0
        return float(np.clip(math.log(p_norm / self.p_ref) / math.log(5.0), 0.0, 1.0))

    def _candidate_temperatures(self, P: int) -> List[float]:
        """Temperature of each of the P candidates.

        Spread mode (steering active): linspace(lo, t_hi) with t_hi = 1.0 + 0.3 * momentum factor ... capped at
        temp_spread_hi_max, so a large ||p|| puts bold (high-temperature) candidates into the pool.
        """
        if self.spread_temps and self.steering_active:
            t_hi = min(self.temp_spread_hi_max, 1.0 + 0.3 * self._momentum_factor() * (self.temp_spread_hi_max - 1.0) / 0.3)
            t_hi = max(t_hi, self.temp_spread_lo)
            if P == 1:
                temps = [self.temp_spread_lo]
            else:
                temps = [float(self.temp_spread_lo + (t_hi - self.temp_spread_lo) * i / (P - 1)) for i in range(P)]
        else:
            base = [self.temperatures[i % len(self.temperatures)] for i in range(P)]
            sc = self._temperature_scale()
            temps = [float(t * sc) for t in base]
        if self.direction_steer and self.steering_active and P >= 2:
            # Shock-dominant -> bolder (hotter) later rows, Flux-dominant -> calmer: continuous in d and ||p||
            d = self._direction()
            mf = self._momentum_factor()
            temps = [float(np.clip(t * (1.0 + self.temp_dir_gain * d * mf * i / (P - 1)), 0.5, 1.5)) for i, t in enumerate(temps)]
        if self._temp_boost:
            temps = [float(t + self._temp_boost) for t in temps]
        return temps

    def _token_classes(self):
        """(content_mask, kanji2_mask) lists over the vocabulary, cached per decoder."""
        if self._tok_class is None:
            tk = self.tokenizer
            surfaces = tk.batch_decode([[i] for i in range(len(tk))], skip_special_tokens=False)
            content, kanji2 = [], []
            for sf in surfaces:
                st = sf.strip()
                nk = sum(1 for ch in st if "\u4e00" <= ch <= "\u9fff")
                kata = sum(1 for ch in st if "\u30a1" <= ch <= "\u30fa")
                content.append(nk >= 1 or (len(st) >= 2 and kata >= 2))
                kanji2.append(nk >= 2)
            self._tok_class = (content, kanji2)
        return self._tok_class

    def _regime_active(self) -> bool:
        return bool(self.direction_steer and self.steering_active and abs(self._direction()) > self.phase_thr)

    def _rhythm_params(self, prefix: str):
        """(period_bias, comma_bias, clause_token_cap) of the base 3-beat rhythm, or None when not applicable.

        Sentence mode cycles with the sentence index in the paragraph: short (1 clause, ~10-20 chars) -> medium
        (2 clauses, ~30-45 chars) -> long (3-4 clauses, ~65-95 chars; the period is held back, commas link the clauses).
        """
        if not self.rhythm or self._regime_active():
            return None
        n_sent = len(re.findall("[\u3002\uff01\uff1f!?]", prefix))
        tail = re.split("[\u3002\uff01\uff1f!?]", prefix)[-1]
        commas = tail.count("\u3001")
        mode = n_sent % 3
        if mode == 0:        # short: single clause, 8-18 chars: commas forbidden, '。' pushed
            return (8.0, -12.0, 13, True)
        if mode == 1:        # medium: two clauses
            return (-12.0, 4.0, 20, False) if commas < 1 else (8.0, -10.0, 20, True)
        # long (55-85 chars): '。' hard-sealed (-12) while the sentence is shorter than 55 chars (commas link the clauses;
        # at >= 5 commas the seal is released as a runaway guard); released after 55; pushed to close beyond 85
        slen = len(tail)
        if slen < 55 and commas < 5:
            return (-12.0, 4.0, 22, False)
        if slen >= 85:
            return (10.0, -10.0, 14, False)
        return (6.0, -6.0, 18, False)

    def _history_ngram_map(self, n: Optional[int] = None):
        """{(t1,..,t_{n-1}): {t_n,...}} over history_text (all previous paragraphs), cached by (text, n)."""
        txt = self.history_text
        n = self.ngram_ban if n is None else int(n)
        if self._hist_cache[0] != txt:
            self._hist_cache = (txt, {})
        cache = self._hist_cache[1]
        if n not in cache:
            ids = self.tokenizer.encode(txt, add_special_tokens=False) if txt else []
            m = {}
            for i in range(len(ids) - n + 1):
                m.setdefault(tuple(ids[i:i + n - 1]), set()).add(ids[i + n - 1])
            cache[n] = m
        return cache[n]

    def _char8_hits(self, prefix: str, cand_texts: Sequence[str]) -> np.ndarray:
        """1 where a candidate would create a (char8_max+1)-th occurrence of some 8-char substring.

        Existing text = history_text (all previous paragraphs) + current paragraph prefix. Whitespace, punctuation and
        quote marks are removed before counting (same normalisation as the max_8gram_count metric).
        """
        hits = np.zeros(len(cand_texts), dtype=bool)
        if self.char8_max <= 0:
            return hits
        norm = lambda s: re.sub(r"[\s、。！？!?「」『』（）()]", "", s)
        existing = norm(self.history_text + "\n" + prefix)
        counts: Dict[str, int] = {}
        for i in range(len(existing) - 7):
            g = existing[i:i + 8]
            counts[g] = counts.get(g, 0) + 1
        tail = existing[-7:]
        for k, c in enumerate(cand_texts):
            joined = tail + norm(c)
            own: Dict[str, int] = {}
            for i in range(len(joined) - 7):
                g = joined[i:i + 8]
                own[g] = own.get(g, 0) + 1
            for g, n_own in own.items():
                if counts.get(g, 0) + n_own > self.char8_max:
                    hits[k] = True
                    break
        return hits

    def _fragment_hits(self, cand_texts: Sequence[str]) -> np.ndarray:
        """Number of continuative-ending fragments ("...につれ。", "...て。") in each candidate."""
        h = np.zeros(len(cand_texts), dtype=np.float64)
        if not self.syntactic_gate:
            return h
        for i, c in enumerate(cand_texts):
            h[i] = len(_FRAGMENT_RE.findall(c))
        return h

    def _direction(self) -> float:
        """Shock-vs-Flux direction of the momentum vector, smooth in (-1, 1): tanh(kappa * (S - F) / (S + F))."""
        v = self.discharge_vec if self.discharge_vec is not None else self.integrator.p
        v = np.abs(np.asarray(v, dtype=np.float64).ravel())
        if v.size < 12:
            return 0.0
        s = float(np.mean(v[[3, 7, 11]]))
        f = float(np.mean(v[[0, 4, 8]]))
        return float(math.tanh(self.direction_kappa * (s - f) / (s + f + 1e-9)))

    def _row_period_bias(self, P: int) -> List[float]:
        """Per-row logit bias for "。" (and the opposite for "、"); 0 when inactive."""
        if P < 2 or not self.steering_active:
            return [0.0] * P
        if self.direction_steer:
            return [0.0] * P  # replaced by the uniform punctuation-phase modulation (_phase_bias)
        if not self.short_bias:
            return [0.0] * P
        mf = self._momentum_factor()
        return [float(self.period_bias_max * mf * i / (P - 1)) for i in range(P)]

    def _punct_ids(self):
        if self._punct_id_cache is None:
            tk = self.tokenizer

            def ids(ch):
                try:
                    e = tk.encode(ch, add_special_tokens=False)
                    return e if len(e) == 1 else []
                except Exception:
                    return []

            self._punct_id_cache = (ids("。"), ids("、"))
        return self._punct_id_cache

    def _temperature_scale(self) -> float:
        """Candidate-temperature multiplier tied to the momentum norm ||p|| of the integrator."""
        if self.temp_gain == 0.0 or not self.steering_active:
            return 1.0
        p_norm = float(np.linalg.norm(np.asarray(self.integrator.p, dtype=np.float64)))
        if p_norm <= 1e-9:
            return 1.0
        return float(np.clip(1.0 + self.temp_gain * math.log(p_norm / self.p_ref), self.temp_min, self.temp_max))

    def _generate_candidates_batched(self, prefix: str) -> List[str]:
        """Generates P candidate clauses in ONE batched generate() call with per-row temperatures.

        - chat_prefix (e.g. chat-template header) is prepended to the model input but never to the returned text.
        - Clause boundary = first punctuation in CLAUSE_PUNCT (ASCII + full-width variants).
        - Candidates that hit max_clause_tokens without punctuation are NOT given a fabricated comma;
          they are dropped when at least one complete clause exists (counted in self.stats).
        """
        device = next(self.model.parameters()).device
        self._cur_prefix = prefix
        text_in = self.chat_prefix + prefix
        if text_in:
            enc = self.tokenizer(text_in, return_tensors="pt", add_special_tokens=not self.chat_prefix)
            input_ids = enc["input_ids"].to(device)
        else:
            bos_id = getattr(self.tokenizer, "bos_token_id", None)
            input_ids = torch.tensor([[0 if bos_id is None else bos_id]], dtype=torch.long, device=device)
        P = self.n_particles
        input_ids = input_ids.expand(P, -1).contiguous()
        inp_len = input_ids.shape[-1]
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", None)
        temps = torch.tensor(self._candidate_temperatures(P), dtype=torch.float32, device=device)
        row_bias = torch.tensor(self._row_period_bias(P), dtype=torch.float32, device=device)
        period_ids, comma_ids = self._punct_ids()
        force_close = self._force_close
        close_boost = self.close_period_boost

        n_ctx = inp_len
        tokenizer_ref = self.tokenizer
        caps = self._row_caps(P)
        caps_t = torch.tensor(caps, dtype=torch.float32, device=device)
        rhythm_on = self._rhythm_params(prefix) is not None
        dyn_len = any(c != self.max_clause_tokens for c in caps) and not rhythm_on
        dir_d = self._direction() if dyn_len else 0.0
        ramp_gain = self.len_ramp
        use_mask = self.narration_mask
        if use_mask:
            bad_ids, post_ids = self._forbidden_token_ids()
            bad_t = torch.tensor(bad_ids, dtype=torch.long, device=device)
            post_t = torch.tensor(post_ids, dtype=torch.long, device=device)
        init_depth = text_in.count("「") + text_in.count("『") - text_in.count("」") - text_in.count("』")
        narr_bias = self.narration_bias

        class _ClausePunctStop(StoppingCriteria):
            """Per-row stop once a clause punctuation char has been generated (cheap vs. stop_strings)."""

            def __call__(self_inner, ids, scores, **kwargs):  # noqa: N805
                tails = tokenizer_ref.batch_decode(ids[:, n_ctx:], skip_special_tokens=True)
                return torch.tensor([bool(CLAUSE_PUNCT_RE.search(s)) for s in tails], device=ids.device)

        eos_pen = self.eos_penalty if self._suppress_eos else 0.0
        eos_tok_ids = [t for t in ([tokenizer_ref.eos_token_id] if not isinstance(getattr(tokenizer_ref, "eos_token_id", None), (list, tuple)) else list(tokenizer_ref.eos_token_id)) if t is not None]
        for extra in ("<|im_end|>", "<|endoftext|>"):
            try:
                tid = tokenizer_ref.convert_tokens_to_ids(extra)
                if isinstance(tid, int) and tid >= 0 and tid not in eos_tok_ids and tid != getattr(tokenizer_ref, "unk_token_id", None):
                    eos_tok_ids.append(tid)
            except Exception:
                pass

        class _EosPenalty(LogitsProcessor):
            def __call__(self_inner, ids, scores):  # noqa: N805
                if eos_pen > 0.0 and eos_tok_ids:
                    scores[:, eos_tok_ids] = scores[:, eos_tok_ids] - eos_pen
                return scores

        class _NarrationStyleLogitsMask(LogitsProcessor):
            """Outside 「」 (per row, tracked on the generated tail) polite auxiliaries / meta marks get a large negative logit bias."""

            def __call__(self_inner, ids, scores):  # noqa: N805
                tails = tokenizer_ref.batch_decode(ids[:, n_ctx:], skip_special_tokens=True)
                depth = torch.tensor(
                    [init_depth + t.count("「") + t.count("『") - t.count("」") - t.count("』") for t in tails],
                    device=scores.device,
                )
                narr = (depth <= 0).to(scores.dtype).unsqueeze(-1)
                v = scores.shape[-1]
                b = bad_t[bad_t < v]
                scores[:, b] = scores[:, b] + narr * narr_bias
                # right after a polite stem token (まし/でし/ませ) the completing token is suppressed as well
                stem = torch.tensor(
                    [bool(t) and t.endswith(_POLITE_STEMS) for t in tails], device=scores.device
                ).to(scores.dtype).unsqueeze(-1)
                if post_t.numel():
                    pt = post_t[post_t < v]
                    scores[:, pt] = scores[:, pt] + narr * stem * narr_bias
                return scores

        cr_pen = self.content_rep_penalty
        lex_dec = self.lex_decay
        need_cls = (cr_pen > 1.0) or lex_dec > 0 or (self.ngram_soft >= 2)
        if need_cls:
            content_l, kanji2_l = self._token_classes()
            content_t = torch.tensor(content_l, dtype=torch.bool, device=device)
            kanji2_t = torch.tensor(kanji2_l, dtype=torch.bool, device=device)
            para_ids_t = torch.tensor(self.tokenizer.encode(prefix, add_special_tokens=False) if prefix else [], dtype=torch.long, device=device)
        else:
            content_l = None

        class _ContentRep(LogitsProcessor):
            """Repetition penalty on content-word tokens only + lexical-diversity decay for repeated kanji words."""

            def __call__(self_inner, ids, scores):  # noqa: N805
                V = scores.shape[-1]
                ct, kt = content_t, kanji2_t
                if ct.numel() < V:  # logits are padded beyond len(tokenizer)
                    ct = torch.nn.functional.pad(ct, (0, V - ct.numel()))
                    kt = torch.nn.functional.pad(kt, (0, V - kt.numel()))
                ct, kt = ct[:V], kt[:V]
                for r in range(ids.shape[0]):
                    if cr_pen > 1.0:
                        u = torch.unique(ids[r])
                        u = u[u < V]
                        sel = u[ct[u]]
                        if sel.numel():
                            sc = scores[r, sel]
                            scores[r, sel] = torch.where(sc > 0, sc / cr_pen, sc * cr_pen)
                    if lex_dec > 0:
                        pids = torch.cat([para_ids_t, ids[r, n_ctx:]])
                        pids = pids[pids < V]
                        if pids.numel():
                            cnt = torch.bincount(pids, minlength=V)
                            idx = torch.nonzero((cnt >= 2) & kt).squeeze(-1)
                            if idx.numel():
                                scores[r, idx] = scores[r, idx] - lex_dec
                return scores

        ngram_n = self.ngram_ban
        hist_map = self._history_ngram_map() if ngram_n >= 2 else {}
        soft_n = self.ngram_soft if (self.ngram_soft >= 2 and (ngram_n < 2 or self.ngram_soft < ngram_n)) else 0
        soft_pen = self.ngram_soft_penalty
        soft_map = self._history_ngram_map(soft_n) if soft_n else {}
        first_ban = torch.tensor(self._first_ban_ids, dtype=torch.long, device=device) if self._first_ban_ids else None

        class _ContextNgramBan(LogitsProcessor):
            """-1e4 on any token that would re-create an n-gram already seen in history / prompt / this row."""

            def __call__(self_inner, ids, scores):  # noqa: N805
                if first_ban is not None and ids.shape[1] == n_ctx:
                    scores[:, first_ban] = scores[:, first_ban] - 1e4
                if ngram_n < 2 and not soft_n:
                    return scores
                for r in range(ids.shape[0]):
                    seq = ids[r].tolist()
                    if soft_n:
                        # moderate decay (-soft_pen) for the shorter n-gram; the hard ban below overrides
                        ks = tuple(seq[-(soft_n - 1):])
                        sb = set(soft_map.get(ks, ()))
                        for i in range(len(seq) - soft_n + 1):
                            if tuple(seq[i:i + soft_n - 1]) == ks:
                                sb.add(seq[i + soft_n - 1])
                        if sb:
                            sl = [t for t in sb if t < scores.shape[-1]]
                            scores[r, sl] = scores[r, sl] - soft_pen
                    if ngram_n < 2:
                        continue
                    n = ngram_n
                    key = tuple(seq[-(n - 1):])
                    banned = set(hist_map.get(key, ()))
                    for i in range(len(seq) - n + 1):
                        if tuple(seq[i:i + n - 1]) == key:
                            banned.add(seq[i + n - 1])
                    if banned:
                        bl = [t for t in banned if t < scores.shape[-1]]
                        scores[r, bl] = scores[r, bl] - 1e4
                return scores

        ph_period, ph_comma = self._phase_bias(prefix)
        gate_on = bool(self.syntactic_gate)
        gate_val = float(self.gate_bias)
        ctx_tail = text_in[-8:]
        _rp = self._rhythm_params(prefix)
        noun_req = bool(_rp is not None and _rp[3])
        sent_tail0 = re.split("[。！？!?]", prefix)[-1]
        _noun_re = re.compile("[一-鿿ァ-ヺ]")

        class _BoundaryBias(LogitsProcessor):
            """Momentum-scaled short-clause bias and end-of-paragraph sentence closing."""

            def __call__(self_inner, ids, scores):  # noqa: N805
                conn = term = noun = None
                if noun_req:
                    # a positive '。' bias fires only once the current sentence contains a noun (kanji / katakana)
                    _tl = tokenizer_ref.batch_decode(ids[:, n_ctx:], skip_special_tokens=True)
                    noun = torch.tensor([float(bool(_noun_re.search(sent_tail0 + t))) for t in _tl], device=scores.device, dtype=scores.dtype)
                if gate_on:
                    # Syntactic Finality Gate: classify the end of each row's text so far
                    tails = tokenizer_ref.batch_decode(ids[:, n_ctx:], skip_special_tokens=True)
                    ends = [(ctx_tail + t)[-8:] for t in tails]
                    conn = torch.tensor([float(_is_connective_end(e)) for e in ends], device=scores.device, dtype=scores.dtype)
                    term = torch.tensor([float(_is_terminal_end(e)) for e in ends], device=scores.device, dtype=scores.dtype)
                if ph_period and period_ids:
                    if ph_period > 0:
                        # Shock: '。' is mildly held back at the start of the clause (a sentence needs a subject and a
                        # predicate: 8-20 chars) and pushed up from ~40% of the clause token target on
                        t_rel = (float(ids.shape[1] - n_ctx)) / caps_t
                        scale = torch.clamp((t_rel - 0.2) / 0.4, 0.0, 1.0) * 1.6 - 0.6
                        pb = (ph_period * scale).to(scores.dtype)
                        if gate_on:
                            # positive '。' bias only after a terminal form / noun; never after a connective ending
                            pb = torch.where(term > 0, pb, torch.clamp(pb, max=0.0))
                        if noun is not None:
                            pb = torch.where(noun > 0, pb, torch.clamp(pb, max=0.0))
                        scores[:, period_ids] = scores[:, period_ids] + pb.unsqueeze(-1)
                    else:
                        scores[:, period_ids] = scores[:, period_ids] + ph_period
                if ph_comma and comma_ids:
                    scores[:, comma_ids] = scores[:, comma_ids] + ph_comma
                if dyn_len and (period_ids or comma_ids):
                    step = float(ids.shape[1] - n_ctx)
                    ramp = torch.clamp((step - 0.6 * caps_t) / (0.4 * caps_t), 0.0, 1.0) * ramp_gain
                    for grp in ((period_ids,) if dir_d > 0 else (period_ids, comma_ids)):
                        if grp:
                            r_g = ramp.to(scores.dtype)
                            if gate_on and grp is period_ids:
                                r_g = r_g * (1.0 - conn)
                            scores[:, grp] = scores[:, grp] + r_g.unsqueeze(-1)
                if period_ids and bool((row_bias != 0).any()):
                    scores[:, period_ids] = scores[:, period_ids] + row_bias.to(scores.dtype).unsqueeze(-1)
                    if comma_ids:
                        scores[:, comma_ids] = scores[:, comma_ids] - row_bias.to(scores.dtype).unsqueeze(-1)
                if force_close:
                    if comma_ids:
                        scores[:, comma_ids] = scores[:, comma_ids] - 30.0
                    if period_ids:
                        scores[:, period_ids] = scores[:, period_ids] + close_boost
                if gate_on and period_ids:
                    scores[:, period_ids] = scores[:, period_ids] + (conn * gate_val).unsqueeze(-1)
                return scores

        class _RowTemp(LogitsProcessor):
            def __call__(self_inner, ids, scores):  # noqa: N805
                return scores / temps.to(scores.dtype).unsqueeze(-1)

        min_p_val = self.min_p

        class _MinP(LogitsProcessor):
            """Min-p tail cut on the tempered distribution: p < min_p * p_max -> -inf."""

            def __call__(self_inner, ids, scores):  # noqa: N805
                probs = torch.softmax(scores.float(), dim=-1)
                pmax = probs.max(dim=-1, keepdim=True).values
                return scores.masked_fill(probs < min_p_val * pmax, float("-inf"))

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(max(caps)),
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 50,
            "attention_mask": torch.ones_like(input_ids),
            "logits_processor": ([_NarrationStyleLogitsMask()] if use_mask else []) + (
                [_ContextNgramBan()] if (ngram_n >= 2 or soft_n or first_ban is not None) else []) + ([_ContentRep()] if (cr_pen > 1.0 or lex_dec > 0) else []) + [_EosPenalty(), _BoundaryBias(), _RowTemp()] + ([_MinP()] if min_p_val > 0 else []),
            "stopping_criteria": StoppingCriteriaList([_ClausePunctStop()]),
        }
        if self.rep_penalty and self.rep_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = self.rep_penalty
        if pad_id is not None:
            gen_kwargs["pad_token_id"] = pad_id
        with torch.no_grad():
            out = self.model.generate(input_ids, **gen_kwargs)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        eos_ids = set(eos_id) if isinstance(eos_id, (list, tuple)) else ({eos_id} if eos_id is not None else set())
        self.eos_reached = False
        complete: List[str] = []
        incomplete: List[str] = []
        n_ended = 0
        for p in range(P):
            row = out[p, inp_len:]
            ended = bool(eos_ids) and any(int(t) in eos_ids for t in row.tolist())
            n_ended += int(ended)
            text = self.tokenizer.decode(row, skip_special_tokens=True)
            m = CLAUSE_PUNCT_RE.search(text)
            if m:
                complete.append(text[: m.end()])
            else:
                incomplete.append(text.strip())
        self.stats["candidates"] += P
        self.stats["incomplete"] += len(incomplete)
        self.stats["eos_rows"] = self.stats.get("eos_rows", 0) + n_ended
        non_empty = [c for c in complete if c.strip()]
        if n_ended * 2 >= P and not non_empty:
            # Majority of particles ended the sequence with nothing new to say: story is finished.
            self.eos_reached = True
            return [""]
        if complete:
            return complete
        return [c for c in incomplete if c] or ["\u3002"]

    def generate_candidate_clauses(
        self,
        prefix: str,
        target_end: Optional[str] = None,
    ) -> List[str]:
        """Generates P=4 candidate clauses from prefix."""
        if self.model is not None and self.tokenizer is not None and TORCH_AVAILABLE:
            return self._generate_candidates_batched(prefix)
        else:
            # Heuristic diverse candidate palette for CPU/offline testing
            palette = [
                "静かな風が木立を抜け、",
                "確かな論理の歩みを進め、",
                "冷涼な夜気が石畳を覆い、",
                "微かな光が遠くで揺らめき、",
                "遥かな山脈が稜線を現し、",
                "穏やかな川面が月影を映し、",
            ]
            if target_end:
                t_sub = target_end[:3] if len(target_end) >= 3 else target_end
                palette[2] = f"{t_sub}へと近づき、"
            cands = list(palette[:self.n_particles])
            while len(cands) < self.n_particles:
                cands.append(f"静けさが深まる節_{len(cands)+1}、")
            return cands

    def compute_particle_weights(
        self,
        prefix: str,
        candidates: Sequence[Union[str, Dict[str, Any]]],
        target_end: Optional[str] = None,
        target_q: Optional[np.ndarray] = None,
        resample_method: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Computes twisted importance weights w^{(p)} for P candidate clauses.

        w^{(p)} = ln P_LLM(c^{(p)} | w_{<=m})
                + gamma_reach * ln P_LLM(W_{end}^* | w_{<=m} o c^{(p)})
                - (beta_steer / 2) * ||hat{q}(w_{<=m} o c^{(p)}) - q_{m+1}^*||_{G_style}^2
        """
        if not candidates:
            raise ValueError("candidates sequence must not be empty")

        target = self.integrator.q if target_q is None else np.asarray(target_q, dtype=np.float64).ravel()
        method = self.resample_method if resample_method is None else str(resample_method).lower()

        term1_llm: List[float] = []
        term2_reach: List[float] = []
        term3_style: List[float] = []
        cand_texts: List[str] = []
        cand_q_list: List[np.ndarray] = []

        q_end_target = None
        if target_end:
            try:
                q_end_target = self.feature_extractor(target_end)
            except Exception:
                q_end_target = None

        for p_idx, cand in enumerate(candidates):
            if isinstance(cand, dict):
                c_text = str(cand.get("text", cand.get("clause", "")))
                explicit_llm = cand.get("llm_logprob", cand.get("logprob"))
                explicit_reach = cand.get("reach_logprob")
                explicit_q = cand.get("q")
            else:
                c_text = str(cand)
                explicit_llm = None
                explicit_reach = None
                explicit_q = None

            cand_texts.append(c_text)
            full_text = prefix + c_text

            # 1. Candidate style vector q_c
            if explicit_q is not None:
                q_c = np.asarray(explicit_q, dtype=np.float64).ravel()
            else:
                style_text = full_text[-self.style_window_chars:] if self.style_window_chars else full_text
                q_c = self.feature_extractor(style_text)
            cand_q_list.append(q_c)

            # 2. Term 1: ln P_LLM(c^{(p)} | w_{<=m})
            if explicit_llm is not None:
                ln_p_llm = float(explicit_llm)
            elif self.model is not None and self.tokenizer is not None and TORCH_AVAILABLE:
                ln_p_llm = self._compute_hf_logprob(prefix, c_text)
            else:
                # Heuristic fluency score based on clause length and punctuation
                c_len = len(c_text)
                len_penalty = -0.05 * abs(c_len - 14)
                punct_bonus = 0.5 if c_text.endswith(("、", "。")) else -0.5
                ln_p_llm = -1.0 + len_penalty + punct_bonus
            term1_llm.append(ln_p_llm)

            # 3. Term 2: gamma_reach * ln P_LLM(W_{end}^* | w_{<=m} o c^{(p)})
            if explicit_reach is not None:
                ln_p_reach = float(explicit_reach)
            elif target_end is None:
                ln_p_reach = 0.0
            elif self.model is not None and self.tokenizer is not None and TORCH_AVAILABLE:
                ln_p_reach = self._compute_hf_logprob(full_text, target_end)
            else:
                if q_end_target is not None:
                    d_diff = q_c - q_end_target
                    d_end = math.sqrt(max(0.0, float(d_diff.T @ self.G_style @ d_diff)))
                    ln_p_reach = -d_end
                else:
                    ln_p_reach = 0.0
            term2_reach.append(self.gamma_reach * ln_p_reach)

            # 4. Term 3: - (beta_steer / 2) * ||hat{q} - q_{m+1}^*||_{G_style}^2
            diff = q_c - target
            style_err = float(diff.T @ self.G_style @ diff)
            style_penalty = -0.5 * self.beta_steer * style_err
            term3_style.append(style_penalty)

        steer_gain = 1.0
        if self.auto_gain and self.steering_active and len(term1_llm) > 1:
            # term3_style = -0.5 * beta * err, so the beta-free part is term3 / beta.
            raw = np.array(term3_style) / (self.beta_steer if self.beta_steer != 0 else 1.0)
            sd_style = float(np.std(raw))
            sd_llm = float(np.std(np.array(term1_llm) + np.array(term2_reach)))
            if sd_style > 1e-9 and sd_llm > 1e-9:
                steer_gain = self.steer_balance * sd_llm / sd_style
                if self.gain_max is not None:
                    steer_gain = min(steer_gain, self.gain_max)
                term3_style = [float(steer_gain * r) for r in raw]

        # Term 4: local Shock/Flux potential gamma_dir * <p_hat, q_local(y_k) - mean_k q_local>, where q_local is the
        # feature vector of the sentence in progress completed by candidate y_k (not a sliding-window difference)
        term4_align = [0.0] * len(term1_llm)
        align_cos = [0.0] * len(term1_llm)
        dq_list = [None] * len(term1_llm)
        d_local = [0.0] * len(term1_llm)
        align_active = False
        dv = self.discharge_vec
        if self.direction_steer and self.steering_active and dv is not None and len(term1_llm) > 1:
            pv = np.asarray(dv, dtype=np.float64).ravel()[: self.dim]
            pn = float(np.linalg.norm(pv))
            if pn > 1e-9 and pv.size == self.dim:
                sent_tail = prefix.rsplit("\u3002", 1)[-1]
                ql = []
                for c_text in cand_texts:
                    try:
                        ql.append(np.asarray(self.feature_extractor(sent_tail + c_text), dtype=np.float64).ravel()[: self.dim])
                    except Exception:
                        ql.append(np.full(self.dim, 0.5))
                qbar = np.mean(ql, axis=0)
                phat = pv / pn
                align_active = True
                for ci, q_l in enumerate(ql):
                    dev = q_l - qbar
                    dq_list[ci] = [float(x) for x in dev]
                    term4_align[ci] = float(self.gamma_dir * (phat @ dev))
                    dn = float(np.linalg.norm(dev))
                    align_cos[ci] = float(phat @ dev) / dn if dn > 1e-12 else 0.0
                    aq = np.abs(q_l)
                    s_, f_ = float(np.mean(aq[[3, 7, 11]])), float(np.mean(aq[[0, 4, 8]]))
                    d_local[ci] = (s_ - f_) / (s_ + f_ + 1e-9)

        w = np.array(term1_llm) + np.array(term2_reach) + np.array(term3_style) + np.array(term4_align)

        # CV_L potential: sentence-length dispersion of the paragraph if candidate k closes its sentence
        if self.cv_gain > 0 and len(cand_texts) > 1:
            done = [x for x in re.split("(?<=[。！？!?])", prefix) if x.strip()]
            tail_open = bool(done) and not re.search(r"[。！？!?]\s*$", prefix)
            confirmed = [len(x.strip()) for x in (done[:-1] if tail_open else done)]
            cur = done[-1] if tail_open else ""
            cvp = np.zeros(len(cand_texts))
            for ci, c_text in enumerate(cand_texts):
                lens = list(confirmed)
                sent = cur + c_text
                if re.search(r"[。！？!?]\s*$", sent):
                    lens.append(len(sent.strip()))
                if len(lens) >= 3:
                    mu = float(np.mean(lens))
                    cv = float(np.std(lens) / mu) if mu > 0 else 0.0
                    cvp[ci] = -self.cv_gain * max(0.0, self.cv_floor - cv)
            if cvp.any():
                w = w + cvp
                self.stats["cv_potential_applied"] = self.stats.get("cv_potential_applied", 0) + 1

        # Register penalty (last line of defence behind the logit mask): always applied, no all-blocked pass-through
        reg_hits = self._register_hits(prefix, cand_texts)
        if reg_hits.any():
            w = w - self.register_penalty * reg_hits
            self.stats["register_hits"] = self.stats.get("register_hits", 0) + int(reg_hits.sum())

        blocked = self._blocked_candidates(prefix, cand_texts)
        if self.syntactic_gate:
            self.stats["fragment_cands"] = self.stats.get("fragment_cands", 0) + int((self._fragment_hits(cand_texts) > 0).sum())
        if self.char8_max > 0:
            self.stats["char8_cands"] = self.stats.get("char8_cands", 0) + int(self._char8_hits(prefix, cand_texts).sum())
        if blocked.any() and not blocked.all():
            w = np.where(blocked, -np.inf, w)
        elif blocked.all():
            # no pass-through: every candidate is rejected -> flat penalty recorded; step_smc_clause has already
            # tried temperature-boosted resampling with the repeated first tokens masked
            w = w - self.register_penalty
            self.stats["repeat_unresolved"] = self.stats.get("repeat_unresolved", 0) + 1

        # Numerical stabilization for softmax normalized weights
        max_w = np.max(w)
        exp_w = np.exp(w - max_w)
        sum_exp = np.sum(exp_w)
        if sum_exp > 0:
            norm_w = exp_w / sum_exp
        else:
            norm_w = np.full(len(w), 1.0 / len(w))

        # Effective Sample Size (ESS)
        ess = float(1.0 / max(1e-12, np.sum(norm_w**2)))

        # Particle selection (MAP or stochastic resampling)
        if method in ("multinomial", "systematic", "sample"):
            best_idx = int(self.rng.choice(len(candidates), p=norm_w))
        else:
            best_idx = int(np.argmax(w))

        return {
            "weights": w,
            "normalized_weights": norm_w,
            "ess": ess,
            "selected_idx": best_idx,
            "selected_clause": cand_texts[best_idx],
            "candidate_texts": cand_texts,
            "candidate_q": cand_q_list,
            "term1_llm": term1_llm,
            "term2_reach": term2_reach,
            "term3_style": term3_style,
            "term4_align": term4_align,
            "align_cos": align_cos,
            "d_local": d_local,
            "dq_list": dq_list,
            "align_active": align_active,
            "register_hits": reg_hits,
            "steer_gain": steer_gain,
        }

    def step_smc_clause(
        self,
        prefix: str,
        candidate_clauses: Optional[Sequence[Union[str, Dict[str, Any]]]] = None,
        target_end: Optional[str] = None,
        target_q: Optional[np.ndarray] = None,
        phi_bridge: Optional[np.ndarray] = None,
        h_syntax: Optional[float] = None,
        resample_method: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Runs one step of Clause Twisted SMC: generates/evaluates candidates, selects best, steps BAOAB."""
        if not candidate_clauses:
            cands: Sequence[Union[str, Dict[str, Any]]] = self.generate_candidate_clauses(prefix, target_end=target_end)
        else:
            cands = list(candidate_clauses)

        if not candidate_clauses and self.max_regen > 0 and (self.meta_guard or self.polite_guard):
            texts = [str(c.get("text", "")) if isinstance(c, dict) else str(c) for c in cands]
            tries = 0
            while tries < self.max_regen and (self._register_hits(prefix, texts) > 0).all():
                tries += 1
                self.stats["regen"] = self.stats.get("regen", 0) + 1
                cands = self.generate_candidate_clauses(prefix, target_end=target_end)
                texts = [str(c.get("text", "")) if isinstance(c, dict) else str(c) for c in cands]
            if (self._register_hits(prefix, texts) > 0).all():
                self.stats["meta_unresolved"] = self.stats.get("meta_unresolved", 0) + 1

        if not candidate_clauses and self.rep_regen > 0 and self.repeat_guard and self.model is not None:
            texts = [str(c.get("text", "")) if isinstance(c, dict) else str(c) for c in cands]
            tries = 0
            ban = []
            try:
                while tries < self.rep_regen and texts and self._blocked_candidates(prefix, texts).all():
                    tries += 1
                    self.stats["rep_regen"] = self.stats.get("rep_regen", 0) + 1
                    self._temp_boost += 0.3
                    for t in texts:
                        e = self.tokenizer.encode(t, add_special_tokens=False)
                        if e:
                            ban.append(int(e[0]))
                    self._first_ban_ids = sorted(set(ban))
                    cands = self.generate_candidate_clauses(prefix, target_end=target_end)
                    texts = [str(c.get("text", "")) if isinstance(c, dict) else str(c) for c in cands]
            finally:
                self._temp_boost = 0.0
                self._first_ban_ids = []

        weights_info = self.compute_particle_weights(
            prefix=prefix,
            candidates=cands,
            target_end=target_end,
            target_q=target_q,
            resample_method=resample_method,
        )

        chosen_clause = weights_info["selected_clause"]
        chosen_idx = weights_info["selected_idx"]
        next_prefix = prefix + chosen_clause

        # Extract measured style of next prefix to feed into outer loop
        q_meas = self.feature_extractor(next_prefix)

        # Estimate syntax entropy if not explicitly provided
        if h_syntax is None:
            h_syn = 0.3 if chosen_clause.endswith("。") and len(chosen_clause) < 6 else 1.2
        else:
            h_syn = float(h_syntax)

        # Step Adaptive-BAOAB Integrator
        baoab_step = self.integrator.step_clause(
            q_measured=q_meas,
            h_syntax=h_syn,
            phi_bridge=phi_bridge,
        )

        return {
            "selected_clause": chosen_clause,
            "selected_idx": chosen_idx,
            "next_prefix": next_prefix,
            "weights": weights_info["weights"],
            "normalized_weights": weights_info["normalized_weights"],
            "ess": weights_info["ess"],
            "term1_llm": weights_info["term1_llm"],
            "term2_reach": weights_info["term2_reach"],
            "term3_style": weights_info["term3_style"],
            "term4_align": weights_info["term4_align"],
            "align_cos": weights_info["align_cos"],
            "chosen_align_cos": float(weights_info["align_cos"][chosen_idx]),
            "chosen_dq": weights_info["dq_list"][chosen_idx],
            "chosen_d_local": weights_info["d_local"][chosen_idx],
            "align_active": weights_info["align_active"],
            "steer_gain": weights_info["steer_gain"],
            "greedy_agree": bool(chosen_idx == int(np.argmax(weights_info["term1_llm"]))),
            "candidate_texts": weights_info["candidate_texts"],
            "baoab_result": baoab_step,
            "q_next_target": baoab_step.q,
            "delta_q": baoab_step.delta_q,
            "q_shrunk": baoab_step.q_shrunk,
        }

    def generate_with_clause_twisted_smc(
        self,
        prefix: str,
        target_end: Optional[str] = None,
        max_clauses: int = 8,
        phi_bridge: Optional[np.ndarray] = None,
        candidate_generator: Optional[Callable[[str, int, float], str]] = None,
        early_stop_on_punct: bool = True,
        resample_method: Optional[str] = None,
        min_clauses: int = 0,
        min_chars: int = 0,
        max_chars: int = 0,
    ) -> Dict[str, Any]:
        """Full multi-clause generation using Clause-Level Twisted SMC and Adaptive-BAOAB.

        min_chars: EOS suppression and sentence closing are released only when the generated text has at least
        min_chars characters AND min_clauses clauses exist. max_chars (0 = off): once reached, closing is forced.
        """
        current_text = prefix
        telemetry: List[Dict[str, Any]] = []
        clauses: List[str] = []

        for m in range(max_clauses):
            # Min-clause / min-chars guard: EOS logits are penalised until both floors are met.
            self._clause_idx = m
            gen_chars = len(current_text) - len(prefix)
            chars_ok = gen_chars >= int(min_chars)
            self._suppress_eos = m < int(min_clauses) or not chars_ok
            self._force_close = (self.close_after is not None and m >= self.close_after and chars_ok) or (
                int(max_chars) > 0 and gen_chars >= int(max_chars) and m >= int(min_clauses))
            if candidate_generator is not None:
                cands: Optional[List[str]] = [candidate_generator(current_text, p, self.temperatures[p % len(self.temperatures)]) for p in range(self.n_particles)]
            else:
                cands = None

            step_res = self.step_smc_clause(
                prefix=current_text,
                candidate_clauses=cands,
                target_end=target_end,
                phi_bridge=phi_bridge,
                resample_method=resample_method,
            )
            if self.eos_reached:
                break
            telemetry.append(step_res)
            clauses.append(step_res["selected_clause"])
            current_text = step_res["next_prefix"]
            if self._force_close and step_res["selected_clause"].rstrip().endswith("。"):
                break  # the sentence has been closed: stop instead of generating further clauses
            if (int(min_chars) > 0 and m + 1 >= int(min_clauses) and len(current_text) - len(prefix) >= int(min_chars)
                    and step_res["selected_clause"].rstrip().endswith("\u3002")):
                break  # soft landing: char floor met -> close at the next natural sentence end (no forced cut)
            if self.verbose:
                # Streamed progress keeps long remote runs from looking idle to the gateway.
                sys.stdout.write(
                    f"  [smc] clause {m + 1}/{max_clauses} ess={step_res['ess']:.2f} "
                    f"t1_std={float(np.std(step_res['term1_llm'])):.3f} t3_std={float(np.std(step_res['term3_style'])):.4f} "
                    f"gain={step_res['steer_gain']:.1f} greedy={int(step_res['greedy_agree'])}\n"
                )
                sys.stdout.flush()

            if early_stop_on_punct and (
                (target_end is not None and target_end in current_text)
                or step_res["selected_clause"].endswith("。")
            ):
                break

        self._suppress_eos = False
        self._force_close = False
        if self.normalize_end:
            stripped = current_text.rstrip()
            if stripped and stripped[-1] in "、，,":
                current_text = stripped[:-1] + "。"
                if clauses:
                    clauses[-1] = clauses[-1].rstrip()[:-1] + "。" if clauses[-1].rstrip()[-1:] in ("、", "，", ",") else clauses[-1]
        return {
            "welded_text": current_text,
            "text": current_text,
            "clauses": clauses,
            "telemetry": telemetry,
            "final_q": self.integrator.q,
            "n_clauses": len(clauses),
        }


