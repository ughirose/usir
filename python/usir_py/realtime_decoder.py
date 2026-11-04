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
    ) -> None:
        self.chat_prefix = str(chat_prefix)
        self.eos_reached = False
        self.stats: Dict[str, int] = {"candidates": 0, "incomplete": 0, "logprob_errors": 0}
        self.dim = int(dim)
        self.n_particles = int(n_particles)
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

    def _generate_candidates_batched(self, prefix: str) -> List[str]:
        """Generates P candidate clauses in ONE batched generate() call with per-row temperatures.

        - chat_prefix (e.g. chat-template header) is prepended to the model input but never to the returned text.
        - Clause boundary = first punctuation in CLAUSE_PUNCT (ASCII + full-width variants).
        - Candidates that hit max_clause_tokens without punctuation are NOT given a fabricated comma;
          they are dropped when at least one complete clause exists (counted in self.stats).
        """
        device = next(self.model.parameters()).device
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
        temps = torch.tensor(self.temperatures[:P], dtype=torch.float32, device=device)

        n_ctx = inp_len
        tokenizer_ref = self.tokenizer

        class _ClausePunctStop(StoppingCriteria):
            """Per-row stop once a clause punctuation char has been generated (cheap vs. stop_strings)."""

            def __call__(self_inner, ids, scores, **kwargs):  # noqa: N805
                tails = tokenizer_ref.batch_decode(ids[:, n_ctx:], skip_special_tokens=True)
                return torch.tensor([bool(CLAUSE_PUNCT_RE.search(s)) for s in tails], device=ids.device)

        class _RowTemp(LogitsProcessor):
            def __call__(self_inner, ids, scores):  # noqa: N805
                return scores / temps.to(scores.dtype).unsqueeze(-1)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_clause_tokens,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 50,
            "attention_mask": torch.ones_like(input_ids),
            "logits_processor": [_RowTemp()],
            "stopping_criteria": StoppingCriteriaList([_ClausePunctStop()]),
        }
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
                q_c = self.feature_extractor(full_text)
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

        w = np.array(term1_llm) + np.array(term2_reach) + np.array(term3_style)

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
    ) -> Dict[str, Any]:
        """Full multi-clause generation using Clause-Level Twisted SMC and Adaptive-BAOAB."""
        current_text = prefix
        telemetry: List[Dict[str, Any]] = []
        clauses: List[str] = []

        for m in range(max_clauses):
            if candidate_generator is not None:
                cands: Optional[List[str]] = [candidate_generator(current_text, p, self.temperatures[p]) for p in range(self.n_particles)]
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

            if early_stop_on_punct and (
                (target_end is not None and target_end in current_text)
                or step_res["selected_clause"].endswith("。")
            ):
                break

        return {
            "welded_text": current_text,
            "text": current_text,
            "clauses": clauses,
            "telemetry": telemetry,
            "final_q": self.integrator.q,
            "n_clauses": len(clauses),
        }


