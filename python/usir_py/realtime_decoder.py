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

import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union

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
    from transformers import LogitsProcessor
except ImportError:
    class LogitsProcessor:  # type: ignore[no-redef]
        """Fallback base class when transformers is not installed."""
        def __call__(self, input_ids: Any, scores: Any) -> Any:
            return scores


# Canonical Japanese and punctuation characters for refractory period gating
PUNCTUATION_CHARS: Set[str] = {"、", "。", "！", "？", "!", "?", ",", ".", "…", "—", "」", "』"}


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
                steered[idx] += 15.0
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
        self.token_jacobians = token_jacobians
        self.embedding_weight: Optional[Any] = None
        self.v_bridge: Optional[Any] = None

        # Track per-batch sequence state
        self.batch_tokens_since_punctuation: Dict[int, int] = {}
        self.call_count: int = 0

    def register_bridge_embeddings(self, model: Any, tokenizer: Any, x_0: str, x_T: str) -> None:
        """Computes and registers embedding-space bridge vector v_bridge = normalize(e_xT - e_x0)."""
        if not TORCH_AVAILABLE:
            return
        try:
            emb_layer = model.get_input_embeddings()
            if emb_layer is None or not hasattr(emb_layer, "weight"):
                return
            self.embedding_weight = emb_layer.weight

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
        except Exception:
            pass

    def register_tokenizer_punctuation(self, tokenizer: Any) -> None:
        """Auto-detects and registers punctuation token IDs from HuggingFace tokenizer."""
        for punct in PUNCTUATION_CHARS:
            try:
                ids = tokenizer.encode(punct, add_special_tokens=False)
                for tid in ids:
                    self.punctuation_token_ids.add(tid)
            except Exception:
                pass

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        """Called at each generation step by HuggingFace generate()."""
        self.call_count += 1

        if not TORCH_AVAILABLE or not isinstance(scores, torch.Tensor):
            return scores

        device = scores.device
        dtype = scores.dtype
        is_2d = scores.ndim == 2
        batch_size = scores.shape[0] if is_2d else 1

        # Process each sequence in the batch
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

            # 4. Refractory Period Gating
            is_refractory_blocked = tokens_since_punct < self.refractory_min
            is_refractory_forced = tokens_since_punct > self.refractory_max

            if is_refractory_blocked or is_refractory_forced:
                # Find punctuation tokens within top-K
                for idx in top_indices:
                    tid = int(idx.item())
                    if tid in self.punctuation_token_ids:
                        if is_refractory_blocked:
                            curr_scores[tid] -= 1000.0
                        elif is_refractory_forced:
                            curr_scores[tid] += 3.5

            # 5. Backward Attraction Steering on Top-K
            if effective_beta > 0.0:
                if self.v_bridge is not None and self.embedding_weight is not None:
                    # True embedding-space bridge attraction
                    top_emb = self.embedding_weight[top_indices].to(device=device, dtype=torch.float32)
                    v_b = self.v_bridge.to(device=device, dtype=torch.float32)
                    top_norm = torch.norm(top_emb, p=2, dim=-1).clamp(min=1e-7)
                    v_norm = torch.norm(v_b, p=2).clamp(min=1e-7)
                    cos_sims = torch.matmul(top_emb, v_b) / (top_norm * v_norm)
                    curr_scores[top_indices] += (effective_beta * cos_sims * 0.5).to(dtype=curr_scores.dtype)
                elif self.backward_attraction is not None:
                    norm_attract = float(np.linalg.norm(self.backward_attraction))
                    if norm_attract > 1e-5:
                        attract_vec = self.backward_attraction
                        for idx in top_indices:
                            tid = int(idx.item())
                            if self.token_jacobians and tid in self.token_jacobians:
                                j_vec = self.token_jacobians[tid][:len(attract_vec)]
                                dot_val = float(np.dot(attract_vec, j_vec))
                                curr_scores[tid] += effective_beta * dot_val * 0.5

        return scores
