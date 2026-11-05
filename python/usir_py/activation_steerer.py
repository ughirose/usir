"""USIR v4.0 Level-2 Residual Stream 12-Element Activation Steering Engine.

Implements direct potential injection into LLM intermediate residual stream (L_mid = floor(0.65 * N_layers)):
    h'_t = h_t + alpha_act * ||h_t||_2 * (V_steer * Delta q*)

Where:
    - V_steer in R^{d_model x 12} with orthonormal columns (V_steer^T V_steer = I_12)
    - Delta q* in R^12 represents desired shift along the 12 USIR canonical symplectic axes
    - alpha_act: Adaptive scaling factor tied to instantaneous hidden state L2-norm
"""

from __future__ import annotations

import contextlib
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Conditional PyTorch imports
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False

from .extractor_24d import extract_usir_24d


class SymplecticActivationSteerer:
    """Manages 12-element activation steering in transformer residual stream.

    Supports:
    1. calibrate_12_element_axes: Extracts or fits orthonormal steering matrix V_steer in R^{d_model x 12}.
    2. register_forward_hook / steering_context: Direct in-flight potential injection.
    3. steer_tensor: Pure functional steering operator for PyTorch tensors and NumPy arrays.
    """

    def __init__(
        self,
        d_model: int = 1536,
        layer_idx: Optional[int] = None,
        num_layers: int = 28,
        layer_ratio: float = 0.65,
        v_steer: Optional[Union[np.ndarray, Any]] = None,
        inject_positions: str = "all",
    ):
        self._v_steer: Optional[np.ndarray] = None
        self._v_version = 0
        self._steer_cache: Optional[Dict[str, Any]] = None
        if inject_positions not in ("all", "last"):
            raise ValueError(f"inject_positions must be 'all' or 'last', got {inject_positions!r}")
        self.inject_positions = inject_positions
        self.calibration_info: Optional[Dict[str, Any]] = None
        self.q_mean: Optional[np.ndarray] = None
        self.q_std: Optional[np.ndarray] = None
        self.d_model = int(d_model)
        if self.d_model < 12:
            raise ValueError(f"d_model must be at least 12 to embed 12-element symplectic space, got {self.d_model}")
        self.num_layers = int(num_layers)
        self.layer_ratio = float(layer_ratio)

        # L_mid = floor(layer_ratio * num_layers) e.g. floor(0.65 * 28) = 18
        if layer_idx is not None:
            self.layer_idx = int(layer_idx)
        else:
            self.layer_idx = int(math.floor(self.layer_ratio * self.num_layers))

        if v_steer is not None:
            self.v_steer = self._ensure_orthonormal(v_steer)
        else:
            self.v_steer = self._initialize_random_orthonormal(self.d_model, 12)

        self._active_hook_handle: Any = None
        self._current_delta_q: Optional[np.ndarray] = None
        self._current_alpha: float = 0.15

    @property
    def v_steer(self) -> np.ndarray:
        return self._v_steer  # type: ignore[return-value]

    @v_steer.setter
    def v_steer(self, value: np.ndarray) -> None:
        # Any change of V invalidates the device-side steering-vector cache.
        self._v_steer = value
        self._v_version += 1
        self._steer_cache = None

    def _cached_steer_vec(self, device: Any, dq: np.ndarray, alpha: float) -> Any:
        """alpha * (V_steer @ dq) as a float32 torch tensor on `device`; rebuilt only when V, dq, alpha or device change.

        No host->device copy happens on a cache hit (audit item D-6).
        """
        dev = torch.device(device)
        dq_key = dq.tobytes()
        c = self._steer_cache
        if (c is not None and c["device"] == dev and c["v_version"] == self._v_version
                and c["dq"] == dq_key and c["alpha"] == float(alpha)):
            return c["vec"]
        vec = torch.as_tensor(float(alpha) * (self._v_steer @ dq), dtype=torch.float32).to(dev)
        self._steer_cache = {"device": dev, "v_version": self._v_version, "dq": dq_key, "alpha": float(alpha), "vec": vec}
        return vec

    def _ensure_orthonormal(self, v: Union[np.ndarray, Any]) -> np.ndarray:
        """Enforces strict column-orthonormality: V^T V = I_12 via QR factorization."""
        arr = np.asarray(v, dtype=np.float32)
        if arr.shape != (self.d_model, 12):
            if arr.shape[0] == 12 and arr.shape[1] == self.d_model:
                arr = arr.T
            else:
                raise ValueError(f"v_steer must have shape ({self.d_model}, 12), got {arr.shape}")
        # QR decomposition yields orthonormal columns Q
        q, r = np.linalg.qr(arr)
        # Ensure positive diagonal for uniqueness
        d = np.diag(r)
        ph = np.sign(d)
        ph[ph == 0] = 1.0
        q = q[:, :12] * ph[:12]
        return q.astype(np.float32)

    def _initialize_random_orthonormal(self, d_model: int, k: int = 12, seed: int = 42) -> np.ndarray:
        """Initializes Haar-orthonormal basis using Gaussian QR decomposition."""
        rng = np.random.RandomState(seed)
        gaussian = rng.randn(d_model, k).astype(np.float32)
        q, _ = np.linalg.qr(gaussian)
        return q[:, :k].astype(np.float32)

    def calibrate_12_element_axes(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        corpus: Optional[Sequence[str]] = None,
        hidden_states_list: Optional[Sequence[Any]] = None,
        ridge_lambda: float = 1e-3,
        axis_mode: str = "independent",
        return_info: bool = False,
        cv_folds: int = 5,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """Calibrates 12-element steering axes from corpus or paired hidden states.

        For each sentence m:
        - Extracts 12-element symplectic vector q_m from extract_usir_24d(m)['q_12d']
        - Extracts residual stream hidden state h_m in R^{d_model}
        - Solves ridge regression / cross-covariance W = (H^T H + lambda I)^{-1} H^T Q
        - axis_mode="independent" (default): each axis k is an independent ridge regression
          v_k = (H^T H + lam I)^{-1} H^T q_k (H column-centred, q_k standardised; dual form since N << d_model),
          then v_k / ||v_k||. No QR / Gram-Schmidt, so axis k depends on q_k only. Here lam = ridge_lambda *
          mean(diag(H H^T)) (relative ridge). Columns are unit-norm but not orthogonal.
        - axis_mode="qr": previous behaviour (absolute ridge, QR of the 12 columns, V^T V = I_12).
        Per-axis in-sample R^2 and `cv_folds`-fold CV R^2 are stored in self.calibration_info
        (and returned as the second element when return_info=True).
        """
        if axis_mode not in ("independent", "qr"):
            raise ValueError(f"axis_mode must be 'independent' or 'qr', got {axis_mode!r}")
        if corpus is None:
            corpus = [
                "静かな雨が庭の青苔を濡らし、かすかな湿気が古い板壁を包み込む。",
                "激しい閃光が大気を切り裂き、爆音とともに建物の屋根が激しく揺れた。",
                "二人は黙って互いを見つめ合い、言葉にならぬ合意が交わされた。",
                "遠い記憶の彼方から、懐かしい鐘の音が風に乗って微かに届いた。",
                "冷たい風が吹き抜け、枯れ葉が舗装道路の隅を舞い上がった。",
                "息を潜めて暗がりを進むと、微かな光の筋が廊下の先に現れた。",
                "突然の爆発に全員が息を呑み、静寂が一瞬にして破られた。",
                "川のせせらぎが絶え間なく続き、午後の日差しが水面を照らしていた。",
                "古い書物の頁をめくる乾いた音が、静まり返った部屋に響いた。",
                "見知らぬ街の路地裏に迷い込み、青い夕暮れが足元に忍び寄った。",
                "鋭い叫び声が夜空を震わせ、張り詰めた緊張が最高潮に達した。",
                "朝露に濡れた草花が光を浴びて、静かな一日が始まりを告げた。",
            ]

        # 1. Collect 12D physical coordinates Q in R^{M x 12}
        q_list: List[np.ndarray] = []
        for text in corpus:
            feat = extract_usir_24d(text)
            # Use q (first 12 dims) or p (second 12 dims) as 12-element coordinate
            q_12 = feat["q"][:12] if "q" in feat else feat["z_24d"][:12]
            q_list.append(np.asarray(q_12, dtype=np.float32))
        q_matrix = np.stack(q_list, axis=0)  # Shape (M, 12)

        m_samples = len(corpus)

        # 2. Collect hidden states H in R^{M x d_model}
        if hidden_states_list is not None and len(hidden_states_list) >= m_samples:
            h_matrix = np.stack([np.asarray(h, dtype=np.float32).reshape(-1)[:self.d_model] for h in hidden_states_list[:m_samples]], axis=0)
        elif TORCH_AVAILABLE and model is not None and tokenizer is not None and hasattr(model, "forward"):
            collected_h = []
            device = next(model.parameters()).device if hasattr(model, "parameters") else "cpu"
            for text in corpus:
                enc = tokenizer(text, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model(**enc, output_hidden_states=True)
                    # hidden_states[0] is the embedding output; the hook sits on the output of decoder layer
                    # `layer_idx`, which is hidden_states[layer_idx + 1].
                    l_idx = min(self.layer_idx + 1, len(out.hidden_states) - 1)
                    # Last token hidden state
                    h_last = out.hidden_states[l_idx][0, -1, :].cpu().numpy().astype(np.float32)
                    collected_h.append(h_last[:self.d_model])
            h_matrix = np.stack(collected_h, axis=0)
        else:
            # Synthetic / analytical pseudo-hidden states based on character n-gram hashing
            rng = np.random.RandomState(1234)
            h_matrix = np.zeros((m_samples, self.d_model), dtype=np.float32)
            for i, text in enumerate(corpus):
                base_vec = rng.randn(self.d_model).astype(np.float32) * 0.1
                # Project 12D signal into hidden space
                proj = q_matrix[i] @ rng.randn(12, self.d_model).astype(np.float32)
                h_matrix[i] = base_vec + proj

        # 3. Ridge Regression
        h_centered = h_matrix - np.mean(h_matrix, axis=0, keepdims=True)
        q_centered = q_matrix - np.mean(q_matrix, axis=0, keepdims=True)

        if axis_mode == "independent":
            q64 = q_matrix.astype(np.float64)
            q_std = np.std(q64, axis=0)
            q_z = (q64 - q64.mean(axis=0)) / np.where(q_std > 1e-12, q_std, 1.0)
            q_z[:, q_std <= 1e-12] = 0.0
            h64 = h_centered.astype(np.float64)
            w = self._ridge_dual(h64, q_z, ridge_lambda)  # (d_model, 12), one independent regression per column
            norms = np.linalg.norm(w, axis=0)
            v_calib = (w / np.where(norms > 1e-12, norms, 1.0)).astype(np.float32)
            r2_in = self._r2(q_z, h64 @ w)
            r2_cv = self._cv_r2(h64, q_z, ridge_lambda, cv_folds)
            self.q_mean = q64.mean(axis=0)
            self.q_std = q_std
            self.calibration_info = {
                "axis_mode": "independent", "n_samples": int(m_samples), "ridge_lambda_relative": float(ridge_lambda),
                "r2_in_sample": [float(x) for x in r2_in], "r2_cv": [float(x) for x in r2_cv], "cv_folds": int(cv_folds),
                "q_std": [float(x) for x in q_std],
            }
            self.v_steer = v_calib
        else:
            # Dual Ridge Regression: W = H^T (H H^T + lambda I)^{-1} Q
            if ridge_lambda > 0.0 and m_samples > 1:
                gram_hh = h_centered @ h_centered.T  # (M, M)
                reg_eye = np.eye(m_samples, dtype=np.float32) * float(ridge_lambda)
                try:
                    inv_kernel = np.linalg.solve(gram_hh + reg_eye, q_centered)  # (M, 12)
                    w_ridge = h_centered.T @ inv_kernel  # (d_model, 12)
                except np.linalg.LinAlgError:
                    w_ridge = h_centered.T @ q_centered
            else:
                w_ridge = h_centered.T @ q_centered
            # Orthonormalize w_ridge columns via QR
            self.calibration_info = {"axis_mode": "qr", "n_samples": int(m_samples), "ridge_lambda_absolute": float(ridge_lambda)}
            self.v_steer = self._ensure_orthonormal(w_ridge)
        if return_info:
            return self.v_steer, self.calibration_info
        return self.v_steer

    @staticmethod
    def _ridge_dual(h: np.ndarray, q: np.ndarray, lam_rel: float) -> np.ndarray:
        """W = H^T (H H^T + lam I)^{-1} Q with lam = lam_rel * mean(diag(H H^T)); H (N, d) must be column-centred."""
        n = h.shape[0]
        g = h @ h.T
        lam = float(lam_rel) * float(np.trace(g)) / max(n, 1)
        a = np.linalg.solve(g + lam * np.eye(n), q)
        return h.T @ a

    @staticmethod
    def _r2(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        sst = np.sum((y - y.mean(axis=0)) ** 2, axis=0)
        sse = np.sum((y - yhat) ** 2, axis=0)
        return np.where(sst > 1e-12, 1.0 - sse / np.where(sst > 1e-12, sst, 1.0), 0.0)

    def _cv_r2(self, h: np.ndarray, q: np.ndarray, lam_rel: float, folds: int) -> np.ndarray:
        """k-fold CV R^2 per axis (strided folds; H and q re-centred on the training part of each fold)."""
        n = h.shape[0]
        folds = max(2, min(int(folds), n))
        pred = np.zeros_like(q)
        for f in range(folds):
            te = np.arange(n) % folds == f
            tr = ~te
            hm, qm = h[tr].mean(axis=0), q[tr].mean(axis=0)
            w = self._ridge_dual(h[tr] - hm, q[tr] - qm, lam_rel)
            pred[te] = (h[te] - hm) @ w + qm
        return self._r2(q, pred)

    def steer_tensor(
        self,
        h: Union[np.ndarray, Any],
        delta_q_star: Union[np.ndarray, Sequence[float]],
        alpha_act: float = 0.15,
        positions: Optional[str] = None,
    ) -> Any:
        r"""Applies direct activation steering injection:

            h' = h + alpha_act * ||h||_2 * (V_steer * Delta q*)

        Supports both PyTorch tensors [B, T, d_model] and NumPy arrays.
        positions: "all" (every position) or "last" (only the last sequence position); default self.inject_positions.
        """
        dq = np.asarray(delta_q_star, dtype=np.float32).reshape(12)
        pos = positions or self.inject_positions
        if pos not in ("all", "last"):
            raise ValueError(f"positions must be 'all' or 'last', got {pos!r}")

        if TORCH_AVAILABLE and isinstance(h, torch.Tensor):
            dtype = h.dtype
            # cached alpha * (V @ dq) on h.device (no per-call host->device copy)
            inj_dir = self._cached_steer_vec(h.device, dq, alpha_act)
            h_fp32 = h.to(torch.float32)
            if pos == "last" and h.dim() >= 2:
                h_last = h_fp32[..., -1:, :]
                norm_h = torch.norm(h_last, p=2, dim=-1, keepdim=True) + 1e-8
                h_steered = torch.cat([h_fp32[..., :-1, :], h_last + norm_h * inj_dir], dim=-2)
            else:
                norm_h = torch.norm(h_fp32, p=2, dim=-1, keepdim=True) + 1e-8
                h_steered = h_fp32 + norm_h * inj_dir
            return h_steered.to(dtype)
        else:
            h_arr = np.asarray(h, dtype=np.float32)
            steer_dir = self.v_steer @ dq  # (d_model,)
            if pos == "last" and h_arr.ndim >= 2:
                out = h_arr.copy()
                last = h_arr[..., -1:, :]
                out[..., -1:, :] = last + float(alpha_act) * (np.linalg.norm(last, axis=-1, keepdims=True) + 1e-8) * steer_dir
                return out.astype(h_arr.dtype)
            norm_h = np.linalg.norm(h_arr, axis=-1, keepdims=True) + 1e-8
            injection = float(alpha_act) * norm_h * steer_dir
            return (h_arr + injection).astype(h_arr.dtype)

    def get_target_layer_module(self, model: Any) -> Optional[Any]:
        """Resolves target layer module from transformer model."""
        candidate_containers = [
            getattr(getattr(model, "model", None), "layers", None),
            getattr(model, "layers", None),
            getattr(getattr(model, "transformer", None), "h", None),
            getattr(getattr(getattr(model, "base_model", None), "model", None), "layers", None),
            getattr(getattr(model, "decoder", None), "layers", None),
        ]
        for c in candidate_containers:
            if c is not None:
                try:
                    num_l = len(c)
                    idx = self.layer_idx if self.layer_idx >= 0 else num_l + self.layer_idx
                    if 0 <= idx < num_l:
                        return c[idx]
                except Exception:
                    pass
        return None

    @contextlib.contextmanager
    def steering_context(
        self,
        delta_q_star: Union[np.ndarray, Sequence[float]],
        alpha_act: float = 0.15,
        model: Optional[Any] = None,
        inject_positions: Optional[str] = None,
    ):
        """Context manager for temporary forward-hook activation steering during inference.

        Example:
            with steerer.steering_context(delta_q_star=[0.0, 0.0, 0.5, ...], alpha_act=0.15, model=model):
                out = model.generate(...)
        """
        self._current_delta_q = np.asarray(delta_q_star, dtype=np.float32).reshape(12)
        self._current_alpha = float(alpha_act)
        _pos = inject_positions or self.inject_positions

        hook_handle = None
        if model is not None:
            target_layer = self.get_target_layer_module(model)
            if target_layer is not None and hasattr(target_layer, "register_forward_hook") and callable(target_layer.register_forward_hook):
                def hook_fn(module, input_args, output):
                    # Output can be a tensor, tuple (hidden_states, ...), or ModelOutput
                    if isinstance(output, tuple):
                        h_out = output[0]
                        steered_h = self.steer_tensor(h_out, self._current_delta_q, self._current_alpha, _pos)
                        return (steered_h,) + output[1:]
                    elif hasattr(output, "hidden_states") and output.hidden_states is not None:
                        output.hidden_states = self.steer_tensor(output.hidden_states, self._current_delta_q, self._current_alpha, _pos)
                        return output
                    else:
                        return self.steer_tensor(output, self._current_delta_q, self._current_alpha, _pos)

                hook_handle = target_layer.register_forward_hook(hook_fn)
                self._active_hook_handle = hook_handle

        try:
            yield self
        finally:
            if hook_handle is not None:
                hook_handle.remove()
                self._active_hook_handle = None
            self._current_delta_q = None

    def verify_orthonormality(self) -> Dict[str, Any]:
        """Verifies V_steer^T V_steer = I_12 and reports max off-diagonal discrepancy."""
        gram = self.v_steer.T @ self.v_steer
        eye = np.eye(12, dtype=np.float32)
        diff = np.abs(gram - eye)
        max_diag_err = float(np.max(np.abs(np.diag(gram) - 1.0)))
        max_offdiag_err = float(np.max(diff - np.diag(np.diag(diff))))
        is_orthonormal = bool(max_diag_err < 1e-4 and max_offdiag_err < 1e-4)
        return {
            "is_orthonormal": is_orthonormal,
            "max_diag_error": max_diag_err,
            "max_offdiag_error": max_offdiag_err,
            "condition_number": float(np.linalg.cond(self.v_steer)),
        }
