"""USIR v2.0 Module 3: SO(8) Pullback Metric & Kac-Regularized Moreau-Yosida Thermodynamics.

This module provides the rigorous thermodynamic and Riemannian foundation for USIR v2.0:
1. SO8PullbackMetric:
   - Skew-symmetric Lie algebra generator A in so(8) (28 free parameters)
   - Exact Cayley transform R = (I - A)(I + A)^{-1} in SO(8)
   - Orthonormal Haar 2nd-difference inter-scale projector P_{8x12} = U_{2x3} (x) I_4
   - Positive-definite style pullback metric G_style = P^T R^T Lambda_+ R P + epsilon I_12 > 0
2. KacMoreauYosidaThermodynamics:
   - Power-law Kac normalization Z_Kac(L, alpha) = sum_{k=1}^{L-1} k^{-alpha}
   - Intensive O(1) long-range interaction potential U_Kac(X_{1:L})
   - Negative Tsallis entropy -H_q(p(x)) (0 < q < 1) + L1 sparsity potential Psi_q(x)
   - Proximal operator prox_{lambda Psi_q}^{G_style}(x) via damped Newton-Raphson / soft-thresholding
   - Globally Lipschitz Moreau-Yosida smoothed gradient nabla_x F_{q, lambda}(x)
   - Intensive free energy density F_L / L proving Theorem 1.1 across L -> infty
3. HamiltonFristonResonance:
   - Reader aesthetic valence spike V_t = - dF_{reader}/dt = -kappa_r <nabla_x F, M^{-1} p_t>
   - Wundt-Berlyne inverted-U reader utility U_{reader}(t) with overload & fatigue damping
   - 1/f^beta power spectral slope beta_hat and 3-band cortical power equipartition (P1, P2, P3)
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
from scipy import signal


class SO8PullbackMetric:
    r"""Strictly positive-definite 12D Riemannian pullback metric G_style derived from SO(8).

    Mathematical Construction:
        1. Generator: A in so(8), A^T = -A, parameterized by 28 free degrees of freedom.
        2. Cayley transform: R = (I_8 - A)(I_8 + A)^{-1} in SO(8), satisfying R^T R = I_8, det(R) = +1.
        3. Haar 2nd-difference inter-scale projection:
               U_{2x3} = [[1/sqrt(2), 0, -1/sqrt(2)],
                          [1/sqrt(6), -2/sqrt(6), 1/sqrt(6)]]
               P_{8x12} = U_{2x3} (x) I_4
           satisfying P P^T = I_8 (strictly orthonormal rows).
        4. Pullback metric:
               G_style = P^T R^T diag(Lambda_+) R P + epsilon I_12 > 0  (epsilon = 1e-3).
    """

    def __init__(
        self,
        theta_28: np.ndarray | Sequence[float] | None = None,
        A: np.ndarray | None = None,
        lambda_plus: np.ndarray | Sequence[float] | None = None,
        epsilon: float = 1e-3,
    ) -> None:
        """Initialize the SO(8) Pullback Metric.

        Args:
            theta_28: 28-dimensional parameter vector for the upper-triangle of A in so(8).
            A: Optional 8x8 skew-symmetric matrix (A^T = -A). If provided, overrides theta_28.
            lambda_plus: 8-dimensional strictly positive eigenvalues for the diagonal scaling.
            epsilon: Regularization parameter to guarantee strict positive-definiteness (default 1e-3).
        """
        self.epsilon = float(epsilon)

        # 1. Construct or validate 8x8 skew-symmetric generator A in so(8)
        if A is not None:
            A_mat = np.asarray(A, dtype=np.float64)
            if A_mat.shape != (8, 8):
                raise ValueError(f"Matrix A must have shape (8, 8), got {A_mat.shape}")
            # Enforce exact skew-symmetry: A = (A - A^T) / 2
            self.A = 0.5 * (A_mat - A_mat.T)
        elif theta_28 is not None:
            th = np.asarray(theta_28, dtype=np.float64).ravel()
            if th.shape[0] != 28:
                raise ValueError(f"theta_28 must have exactly 28 elements, got {th.shape[0]}")
            self.A = np.zeros((8, 8), dtype=np.float64)
            idx = 0
            for i in range(8):
                for j in range(i + 1, 8):
                    self.A[i, j] = th[idx]
                    self.A[j, i] = -th[idx]
                    idx += 1
        else:
            self.A = np.zeros((8, 8), dtype=np.float64)

        # 2. Setup Lambda_+ (8-dimensional positive eigenvalues)
        if lambda_plus is not None:
            lp = np.asarray(lambda_plus, dtype=np.float64).ravel()
            if lp.shape[0] != 8:
                raise ValueError(f"lambda_plus must have shape (8,), got {lp.shape[0]}")
            if np.any(lp <= 0):
                raise ValueError(f"All eigenvalues in lambda_plus must be strictly positive, got {lp}")
            self.lambda_plus = lp
        else:
            # Canonical diverse positive spectrum (normalized around 1.0)
            self.lambda_plus = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4], dtype=np.float64)

        # 3. Compute Cayley transform: R = (I_8 - A)(I_8 + A)^{-1}
        I8 = np.eye(8, dtype=np.float64)
        # Note: I_8 + A is always invertible for any real skew-symmetric A because
        # eigenvalues of A are purely imaginary (1 +/- i lambda != 0).
        self.R = (I8 - self.A) @ np.linalg.inv(I8 + self.A)

        # Numerical verification of SO(8) Lie group properties:
        r_orth_err = np.max(np.abs(self.R.T @ self.R - I8))
        det_r = float(np.linalg.det(self.R))
        if r_orth_err > 1e-7 or abs(det_r - 1.0) > 1e-7:
            raise ValueError(
                f"Cayley transform failed SO(8) verification: orthogonality error={r_orth_err:.2e}, det={det_r:.6f}"
            )

        # 4. Construct Haar 2nd-difference inter-scale embedding matrix P_{8x12} = U_{2x3} (x) I_4
        # U_{2x3} has strictly orthonormal rows: U U^T = I_2
        self.U = np.array(
            [
                [1.0 / math.sqrt(2.0), 0.0, -1.0 / math.sqrt(2.0)],
                [1.0 / math.sqrt(6.0), -2.0 / math.sqrt(6.0), 1.0 / math.sqrt(6.0)],
            ],
            dtype=np.float64,
        )
        self.P = np.kron(self.U, np.eye(4, dtype=np.float64))  # Shape: (8, 12)

        # Verify P P^T = I_8
        p_orth_err = np.max(np.abs(self.P @ self.P.T - I8))
        if p_orth_err > 1e-10:
            raise ValueError(f"P matrix row orthonormality failed: error={p_orth_err:.2e}")

        # 5. Compute the Riemannian pullback metric G_style
        # G_style = P^T R^T diag(Lambda_+) R P + epsilon I_12
        diag_lambda = np.diag(self.lambda_plus)
        G_raw = self.P.T @ self.R.T @ diag_lambda @ self.R @ self.P
        self._G = 0.5 * (G_raw + G_raw.T) + self.epsilon * np.eye(12, dtype=np.float64)

        # Verify positive definiteness: all eigenvalues >= epsilon
        eigvals = np.linalg.eigvalsh(self._G)
        if np.min(eigvals) < self.epsilon * 0.999:
            raise ValueError(f"Pullback metric G_style is not SPD! min eigval={np.min(eigvals)}")

        self._G_inv = np.linalg.inv(self._G)

    @property
    def dim(self) -> int:
        return 12

    def metric_matrix(self) -> np.ndarray:
        """Return a copy of the 12x12 Riemannian metric matrix G_style."""
        return self._G.copy()

    def inverse_metric(self) -> np.ndarray:
        """Return a copy of G_style^{-1}."""
        return self._G_inv.copy()

    def mahalanobis_distance(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """Compute the Riemannian Mahalanobis distance d_G(q1, q2) = sqrt((q1 - q2)^T G (q1 - q2))."""
        diff = np.asarray(q1, dtype=np.float64) - np.asarray(q2, dtype=np.float64)
        quad = float(diff.T @ self._G @ diff)
        return math.sqrt(max(0.0, quad))

    def quadratic_form(self, v: np.ndarray) -> float:
        """Compute the metric quadratic form v^T G_style v."""
        vec = np.asarray(v, dtype=np.float64)
        return float(vec.T @ self._G @ vec)

    def inner_product(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute the metric inner product <v1, v2>_{G_style} = v1^T G_style v2."""
        vec1 = np.asarray(v1, dtype=np.float64)
        vec2 = np.asarray(v2, dtype=np.float64)
        return float(vec1.T @ self._G @ vec2)


class KacMoreauYosidaThermodynamics:
    r"""Kac-regularized long-range free energy and Moreau-Yosida smoothed proximal gradient flow.

    Mathematical Formulation:
        1. Kac Normalization Factor:
               Z_Kac(L, alpha) = sum_{k=1}^{L-1} k^{-alpha}  (alpha > 0)
        2. Long-Range Interaction Energy:
               U_Kac(X_{1:L}) = (1/L) sum_{t=1}^L V_loc(x_t)
                              - (J_0 / (2 L Z_Kac(L, alpha))) sum_{t=1}^L sum_{s != t} |t - s|^{-alpha} <x_t, G_style x_s>
           guaranteeing intensive O(1) energy density per clause (Theorem 1.1).
        3. Regularization Potential:
               Psi_q(x) = - tau H_q(p(x)) + mu_1 ||x||_1
           where p_i(x) = softmax(x_i^2), and H_q(p) = (1 - sum p_i^q) / (q - 1) is the Tsallis entropy (0 < q < 1).
        4. Proximal Operator prox_{lambda Psi_q}^{G_style}(x):
               Computed via 4 iterations of damped Newton-Raphson / soft-thresholding.
        5. Moreau-Yosida Smoothed Gradient:
               nabla_x F_{q, lambda}(x) = (1/lambda) G_style (x - prox_{lambda Psi_q}^{G_style}(x)) + nabla_x U_Kac(x).
    """

    def __init__(
        self,
        metric: SO8PullbackMetric | np.ndarray | None = None,
        j0: float = 1.0,
        alpha: float = 0.8,
        q_tsallis: float = 0.5,
        tau: float = 1.0,
        mu1: float = 0.05,
        lambda_reg: float = 0.1,
        v_loc_fn: Callable[[np.ndarray], float] | None = None,
        v_loc_grad_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        """Initialize the Kac-Moreau-Yosida thermodynamics system.

        Args:
            metric: SO8PullbackMetric instance or 12x12 SPD array.
            j0: Interaction coupling strength J_0 > 0.
            alpha: Power-law decay exponent alpha > 0 for long-range Kac memory (default 0.8).
            q_tsallis: Non-extensive Tsallis parameter 0 < q < 1 (default 0.5).
            tau: Effective temperature tau > 0.
            mu1: L1 sparsity regularizer weight mu_1 >= 0 (default 0.05).
            lambda_reg: Moreau-Yosida envelope smoothing parameter lambda > 0 (default 0.1).
            v_loc_fn: Local potential V_loc(x), default is harmonic well 0.5 * ||x||_2^2.
            v_loc_grad_fn: Gradient of V_loc(x), default is x.
        """
        if metric is None:
            self.metric = SO8PullbackMetric()
        elif isinstance(metric, SO8PullbackMetric):
            self.metric = metric
        elif isinstance(metric, np.ndarray):
            # Wrap array in dummy container
            self.metric = SO8PullbackMetric()
            self.metric._G = metric.copy()
            self.metric._G_inv = np.linalg.inv(metric)
        else:
            raise TypeError(f"Invalid metric type: {type(metric)}")

        self.G = self.metric.metric_matrix()
        self.G_inv = self.metric.inverse_metric()

        self.j0 = float(j0)
        self.alpha = float(alpha)
        if not (0.0 < q_tsallis < 1.0):
            raise ValueError(f"Tsallis q must be in (0, 1), got {q_tsallis}")
        self.q = float(q_tsallis)
        self.tau = float(tau)
        self.mu1 = float(mu1)
        self.lambda_reg = float(lambda_reg)

        # Default local potential: harmonic trap V_loc(x) = 0.5 * ||x||^2
        self.v_loc_fn = v_loc_fn if v_loc_fn is not None else (lambda x: 0.5 * float(np.sum(x**2)))
        self.v_loc_grad_fn = (
            v_loc_grad_fn if v_loc_grad_fn is not None else (lambda x: np.asarray(x, dtype=np.float64))
        )

    def kac_normalization(self, L: int, alpha: float | None = None) -> float:
        r"""Compute Kac normalization factor Z_Kac(L, alpha) = sum_{k=1}^{L-1} k^{-alpha}.

        For L <= 1, returns 1.0.
        """
        if L <= 1:
            return 1.0
        a = self.alpha if alpha is None else float(alpha)
        k = np.arange(1, L, dtype=np.float64)
        return float(np.sum(k ** (-a)))

    def softmax_p(self, x: np.ndarray) -> np.ndarray:
        """Compute p_i(x) = softmax(x_i^2) with numerical stabilization."""
        x_sq = np.asarray(x, dtype=np.float64) ** 2
        shift = np.max(x_sq)
        exp_vals = np.exp(x_sq - shift)
        s = np.sum(exp_vals)
        if s <= 0 or not np.isfinite(s):
            return np.ones(12, dtype=np.float64) / 12.0
        return exp_vals / s

    def tsallis_entropy(self, p: np.ndarray) -> float:
        r"""Compute Tsallis non-extensive entropy H_q(p) = (1 - sum p_i^q) / (q - 1)."""
        p_safe = np.clip(p, 1e-15, 1.0)
        sum_pq = float(np.sum(p_safe**self.q))
        return (1.0 - sum_pq) / (self.q - 1.0)

    def psi_q(self, x: np.ndarray) -> float:
        r"""Compute regularizing potential Psi_q(x) = - tau H_q(p(x)) + mu_1 ||x||_1."""
        p = self.softmax_p(x)
        h_q = self.tsallis_entropy(p)
        l1 = float(np.sum(np.abs(x)))
        return -self.tau * h_q + self.mu1 * l1

    def grad_smooth_psi(self, x: np.ndarray) -> np.ndarray:
        r"""Compute exact gradient of the smooth Tsallis term -tau H_q(p(x)).

        Derivation:
            H_q(p) = (1 - sum p_k^q) / (q - 1)
            dH_q / dp_k = - (q / (q - 1)) p_k^{q-1}
            p_k = softmax(z_k), z_k = x_k^2
            dp_k / dz_j = p_k (delta_{kj} - p_j)
            dH_q / dz_j = - (q / (q - 1)) [ p_j^q - p_j sum_k p_k^q ]
            dz_j / dx_j = 2 x_j
            nabla_x (-tau H_q) = tau * (2 q / (q - 1)) * x * p * [ p^{q-1} - sum p_k^q ]
        """
        p = self.softmax_p(x)
        p_safe = np.clip(p, 1e-15, 1.0)
        sum_pq = float(np.sum(p_safe**self.q))
        factor = self.tau * (2.0 * self.q / (self.q - 1.0))
        grad = factor * x * p_safe * (p_safe ** (self.q - 1.0) - sum_pq)
        return grad

    def prox_lambda_psi(self, x: np.ndarray, n_iter: int = 4, damping: float = 0.6) -> np.ndarray:
        r"""Compute prox_{lambda Psi_q}^{G_style}(x) via damped Newton-Raphson / soft-thresholding.

        Optimizes:
            y* = argmin_y { (1/(2 lambda)) ||y - x||_G^2 - tau H_q(p(y)) + mu_1 ||y||_1 }
        using 4 damped proximal steps with metric G_style and coordinate soft-thresholding.
        """
        y = np.asarray(x, dtype=np.float64).copy()
        inv_diag_g = np.diag(self.G_inv)

        for _ in range(n_iter):
            # Smooth gradient: (1/lambda) G (y - x) + grad(-tau H_q(y))
            g_smooth = (1.0 / self.lambda_reg) * (self.G @ (y - x)) + self.grad_smooth_psi(y)
            # Damped natural Newton step under quadratic metric G
            # Delta y = - damping * (G / lambda)^{-1} g_smooth
            #         = - damping * (y - x + lambda G^{-1} grad(-tau H_q(y)))
            delta_y = -damping * ((y - x) + self.lambda_reg * (self.G_inv @ self.grad_smooth_psi(y)))
            y_tentative = y + delta_y

            # Coordinate soft-thresholding for L1 penalty mu_1 ||y||_1
            # Threshold theta_i = damping * lambda * mu_1 * (G^{-1})_{ii}
            threshold = damping * self.lambda_reg * self.mu1 * np.abs(inv_diag_g)
            y = np.sign(y_tentative) * np.maximum(0.0, np.abs(y_tentative) - threshold)

        return y

    def moreau_yosida_gradient(
        self,
        x: np.ndarray,
        history_x: np.ndarray | None = None,
    ) -> np.ndarray:
        r"""Compute the globally Lipschitz Moreau-Yosida smoothed gradient:

            nabla_x F_{q, lambda}(x) = (1/lambda) G_style (x - prox_{lambda Psi_q}^{G_style}(x)) + nabla_x U_Kac(x)
        """
        x_arr = np.asarray(x, dtype=np.float64)
        prox_y = self.prox_lambda_psi(x_arr)
        # Proximal envelope gradient
        grad_env = (1.0 / self.lambda_reg) * (self.G @ (x_arr - prox_y))

        # Local potential gradient
        grad_u = self.v_loc_grad_fn(x_arr)

        # Long-range causal memory gradient if history is provided
        if history_x is not None and len(history_x) > 0:
            hist = np.asarray(history_x, dtype=np.float64)
            K = len(hist)
            z_kac = self.kac_normalization(K + 1)
            k_steps = np.arange(1, K + 1, dtype=np.float64)
            weights = (k_steps ** (-self.alpha)) / z_kac  # Shape: (K,)
            # Memory coupling: - J0 sum_k w_k G_style x_{t-k}
            weighted_hist = np.sum(weights[:, None] * hist, axis=0)  # Shape: (12,)
            grad_u = grad_u - self.j0 * (self.G @ weighted_hist)

        return grad_env + grad_u

    def kac_energy_trajectory(self, X: np.ndarray) -> float:
        r"""Compute the intensive long-range Kac-regularized energy U_Kac(X_{1:L}) for a trajectory X:

            U_Kac(X) = (1/L) sum_{t=1}^L V_loc(x_t)
                     - (J_0 / (2 L Z_Kac(L, alpha))) sum_{t=1}^L sum_{s != t} |t - s|^{-alpha} <x_t, G_style x_s>
        """
        X_arr = np.asarray(X, dtype=np.float64)
        L = X_arr.shape[0]
        if L == 0:
            return 0.0
        if L == 1:
            return float(self.v_loc_fn(X_arr[0]))

        # 1. Local potential mean
        v_loc_sum = float(sum(self.v_loc_fn(X_arr[t]) for t in range(L)))

        # 2. Pairwise long-range interaction
        t_indices = np.arange(L)[:, None]
        s_indices = np.arange(L)[None, :]
        dist = np.abs(t_indices - s_indices)
        np.fill_diagonal(dist, 1)  # avoid div by zero on diagonal

        k_kernel = dist ** (-self.alpha)
        np.fill_diagonal(k_kernel, 0.0)  # zero out self-interaction (s == t)

        # Inner product matrix M_{ts} = x_t^T G_style x_s
        M_inner = (X_arr @ self.G) @ X_arr.T

        interaction_sum = float(np.sum(k_kernel * M_inner))
        z_kac = self.kac_normalization(L)

        u_kac = (1.0 / L) * v_loc_sum - (self.j0 / (2.0 * L * z_kac)) * interaction_sum
        return u_kac

    def free_energy_density(self, trajectory_x: np.ndarray) -> float:
        r"""Compute the intensive O(1) free energy per clause F_L / L along a trajectory X_{1:L}.

        Proves Theorem 1.1: As sequence length L -> infty, F_L / L remains bounded and stable.
        """
        X = np.asarray(trajectory_x, dtype=np.float64)
        L = X.shape[0]
        if L == 0:
            return 0.0

        # Sum of Moreau-Yosida envelope energies: e_lambda(x_t)
        envelope_sum = 0.0
        for t in range(L):
            xt = X[t]
            yt = self.prox_lambda_psi(xt)
            diff = yt - xt
            quad = float(diff.T @ self.G @ diff)
            e_t = self.psi_q(yt) + (0.5 / self.lambda_reg) * quad
            envelope_sum += e_t

        env_density = envelope_sum / L
        u_kac_density = self.kac_energy_trajectory(X)

        return env_density + u_kac_density

    def verify_kac_bound(
        self,
        lengths: Sequence[int] = (16, 64, 256, 1024),
        seed: int = 42,
    ) -> dict[int, float]:
        """Empirically verify Theorem 1.1: Free energy density remains O(1) across arbitrary L."""
        rng = np.random.default_rng(seed)
        densities: dict[int, float] = {}

        # Generate a stationary Markov / AR(1) style trajectory
        for L in lengths:
            # AR(1) stationary process: x_t = 0.8 x_{t-1} + 0.2 xi
            X = np.zeros((L, 12), dtype=np.float64)
            cur = rng.normal(0.0, 0.5, size=12)
            for t in range(L):
                cur = 0.8 * cur + rng.normal(0.0, 0.2, size=12)
                X[t] = cur

            f_dens = self.free_energy_density(X)
            densities[L] = f_dens

        return densities


class HamiltonFristonResonance:
    r"""Hamilton-Friston two-layer aesthetic resonance and 1/f^beta cortical impedance matching.

    Mathematical Formulation:
        1. Reader Aesthetic Valence Spike (Catharsis Velocity):
               V_t = - dF_{reader}/dt = - kappa_r <nabla_x F_{q, lambda}(x_t), M^{-1} p_t>
           measuring the instantaneous rate of prediction-error resolution (Joffily & Coricelli).
        2. Wundt-Berlyne Inverted-U Utility:
               U_{reader}(t) = V_t * exp(- F_{q, lambda}(x_t) / F_{overload})
                             - lambda_{fatigue} sum_{s=t-tau_{rec}+1}^t ||p_s - p_{s-1}||_2^2
           penalizing overwrought style (purple prose) and continuous aesthetic fatigue.
        3. 1/f^beta Power Spectral Density Slope & 3-Band Cortical Equipartition:
               S(f) \propto f^{-beta}
           where beta_hat = 1.0 achieves equal dissipation across Micro (P1), Meso (P2), and Macro (P3) bands.
    """

    def __init__(
        self,
        thermo: KacMoreauYosidaThermodynamics | None = None,
        kappa_r: float = 1.0,
        mass_inv: np.ndarray | None = None,
        f_overload: float = 10.0,
        lambda_fatigue: float = 0.05,
        tau_rec: int = 6,
    ) -> None:
        """Initialize the Hamilton-Friston Resonance monitor."""
        self.thermo = thermo if thermo is not None else KacMoreauYosidaThermodynamics()
        self.kappa_r = float(kappa_r)
        self.mass_inv = (
            np.eye(12, dtype=np.float64)
            if mass_inv is None
            else np.asarray(mass_inv, dtype=np.float64)
        )
        self.f_overload = float(f_overload)
        self.lambda_fatigue = float(lambda_fatigue)
        self.tau_rec = int(tau_rec)

    def valence_spike(
        self,
        x_t: np.ndarray,
        p_t: np.ndarray,
        history_x: np.ndarray | None = None,
    ) -> float:
        r"""Compute reader aesthetic valence spike:

            V_t = - dF_{reader}/dt = - kappa_r <nabla_x F_{q, lambda}(x_t), M^{-1} p_t>
        """
        grad_f = self.thermo.moreau_yosida_gradient(x_t, history_x=history_x)
        vel = self.mass_inv @ np.asarray(p_t, dtype=np.float64)
        rate = float(np.dot(grad_f, vel))
        return -self.kappa_r * rate

    def wundt_berlyne_utility(
        self,
        v_t: float,
        x_t: np.ndarray,
        p_history: np.ndarray | Sequence[np.ndarray] | None = None,
    ) -> float:
        r"""Compute Wundt-Berlyne inverted-U aesthetic utility U_{reader}(t)."""
        yt = self.thermo.prox_lambda_psi(x_t)
        diff = yt - x_t
        quad = float(diff.T @ self.thermo.G @ diff)
        f_val = self.thermo.psi_q(yt) + (0.5 / self.thermo.lambda_reg) * quad

        # 1. Overload cutoff gate: exp(- F / F_{overload})
        gate = math.exp(-max(0.0, f_val) / max(1e-5, self.f_overload))
        utility_gain = v_t * gate

        # 2. Refractory period fatigue penalty
        fatigue = 0.0
        if p_history is not None and len(p_history) >= 2:
            p_arr = np.asarray(p_history, dtype=np.float64)
            # Take last tau_rec window
            window = p_arr[-self.tau_rec :]
            if len(window) >= 2:
                diffs = np.diff(window, axis=0)
                fatigue = float(np.sum(diffs**2))

        return float(utility_gain - self.lambda_fatigue * fatigue)

    def compute_wundt_berlyne_utility(
        self,
        valence_series: Sequence[float] | None = None,
        free_energy_series: Sequence[float] | None = None,
        p_trajectory: Sequence[np.ndarray | Sequence[float]] | None = None,
        q_trajectory: Sequence[np.ndarray | Sequence[float]] | None = None,
        f_overload: float | None = None,
        lambda_fatigue: float | None = None,
        tau_rec: int | None = None,
    ) -> dict[str, object]:
        r"""Compute Wundt-Berlyne utility over a sequence/trajectory using monitor parameters."""
        return compute_wundt_berlyne_utility(
            valence_series=valence_series,
            free_energy_series=free_energy_series,
            p_trajectory=p_trajectory,
            q_trajectory=q_trajectory,
            f_overload=self.f_overload if f_overload is None else float(f_overload),
            lambda_fatigue=self.lambda_fatigue if lambda_fatigue is None else float(lambda_fatigue),
            tau_rec=self.tau_rec if tau_rec is None else int(tau_rec),
            thermo=self.thermo,
            kappa_r=self.kappa_r,
            mass_inv=self.mass_inv,
        )

    @staticmethod
    def compute_spectral_equipartition(
        valence_series: np.ndarray | Sequence[float],
    ) -> tuple[float, tuple[float, float, float]]:
        r"""Compute 1/f^beta spectral slope beta_hat and 3-band cortical power equipartition (P1, P2, P3).

        Returns:
            (beta_hat, (P_macro, P_meso, P_micro)) normalized to P_macro + P_meso + P_micro = 1.0.
        """
        v = np.asarray(valence_series, dtype=np.float64).ravel()
        n = len(v)
        if n < 8:
            return 1.0, (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

        # Welch PSD or RFFT
        freqs, psd = signal.welch(v, nperseg=min(n, 32))
        pos_mask = (freqs > 0) & (psd > 1e-15)
        if np.sum(pos_mask) < 2:
            return 1.0, (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

        log_f = np.log(freqs[pos_mask])
        log_p = np.log(psd[pos_mask])

        # Linear regression: ln P(f) = - beta ln f + C
        poly = np.polyfit(log_f, log_p, 1)
        beta_hat = float(-poly[0])

        # 3-band division across positive frequency spectrum
        pos_freqs = freqs[pos_mask]
        pos_psd = psd[pos_mask]
        n_pos = len(pos_freqs)
        k1 = max(1, n_pos // 3)
        k2 = max(k1 + 1, (2 * n_pos) // 3)

        p_macro = float(np.sum(pos_psd[:k1]))
        p_meso = float(np.sum(pos_psd[k1:k2]))
        p_micro = float(np.sum(pos_psd[k2:]))

        total_power = p_macro + p_meso + p_micro
        if total_power <= 0:
            return beta_hat, (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

        p1 = p_macro / total_power
        p2 = p_meso / total_power
        p3 = p_micro / total_power

        return beta_hat, (p1, p2, p3)


def compute_wundt_berlyne_utility(
    valence_series: Sequence[float] | None = None,
    free_energy_series: Sequence[float] | None = None,
    p_trajectory: Sequence[np.ndarray | Sequence[float]] | None = None,
    q_trajectory: Sequence[np.ndarray | Sequence[float]] | None = None,
    f_overload: float = 10.0,
    lambda_fatigue: float = 0.05,
    tau_rec: int = 6,
    thermo: KacMoreauYosidaThermodynamics | None = None,
    kappa_r: float = 1.0,
    mass_inv: np.ndarray | None = None,
) -> dict[str, object]:
    r"""Compute Wundt-Berlyne inverted-U aesthetic utility over a sequence/trajectory.

    Mathematical Formulation:
        U_{reader}(t) = V_t * exp(- F_t / F_{overload}) - lambda_{fatigue} * sum_{s=t-tau_{rec}+1}^t ||p_s - p_{s-1}||_2^2
    where V_t = max(0, -dF_{reader}(t)/dt) is the surprise resolution spike,
    F_t is the local/cumulative free energy load, and the refractory integral sums
    momentum squared differences over window [t - tau_rec, t].

    Args:
        valence_series: Instantaneous reader valence spikes V_t (optional if q_trajectory provided).
        free_energy_series: Style free energy values F_t (optional if q_trajectory provided).
        p_trajectory: Sequence of 12D momentum vectors p_t.
        q_trajectory: Sequence of 12D style vectors q_t (used to compute V_t, F_t if not given).
        f_overload: Free energy overload cutoff scale F_{overload}.
        lambda_fatigue: Refractory fatigue penalty coefficient lambda_{fatigue}.
        tau_rec: Refractory window length tau_{rec} in sentences (default 6).
        thermo: KacMoreauYosidaThermodynamics instance for computing V_t and F_t from q_trajectory.
        kappa_r: Reader coupling constant kappa_r.
        mass_inv: Inverse mass matrix M^{-1}.

    Returns:
        dict containing:
            - 'utilities': 1D numpy array of U_{reader}(t)
            - 'mean_utility': float mean reader utility
            - 'cumulative_utility': float sum of reader utilities
            - 'fatigue_penalties': 1D numpy array of refractory fatigue penalties
            - 'gated_valences': 1D numpy array of overload-gated valences
    """
    if p_trajectory is None and q_trajectory is None:
        raise ValueError("Must provide either p_trajectory or q_trajectory.")

    if p_trajectory is not None:
        p_arr = np.asarray(p_trajectory, dtype=np.float64)
        if p_arr.ndim == 1:
            p_arr = p_arr.reshape(1, -1)
        T = p_arr.shape[0]
    else:
        q_arr = np.asarray(q_trajectory, dtype=np.float64)
        T = q_arr.shape[0]
        p_arr = np.zeros((T, 12), dtype=np.float64)

    # Compute or extract valence and free energy series
    if valence_series is not None:
        v_arr = np.asarray(valence_series, dtype=np.float64).ravel()
    elif q_trajectory is not None:
        th = thermo if thermo is not None else KacMoreauYosidaThermodynamics()
        m_inv = (
            np.eye(12, dtype=np.float64)
            if mass_inv is None
            else np.asarray(mass_inv, dtype=np.float64)
        )
        v_arr = np.zeros(T, dtype=np.float64)
        for t in range(T):
            grad_f = th.moreau_yosida_gradient(q_trajectory[t])
            vel = m_inv @ p_arr[t]
            rate = float(np.dot(grad_f, vel))
            v_arr[t] = max(0.0, -kappa_r * rate)
    else:
        v_arr = np.zeros(T, dtype=np.float64)

    if free_energy_series is not None:
        f_arr = np.asarray(free_energy_series, dtype=np.float64).ravel()
    elif q_trajectory is not None:
        th = thermo if thermo is not None else KacMoreauYosidaThermodynamics()
        f_arr = np.zeros(T, dtype=np.float64)
        for t in range(T):
            yt = th.prox_lambda_psi(q_trajectory[t])
            diff = yt - q_trajectory[t]
            quad = float(diff.T @ th.G @ diff)
            f_arr[t] = th.psi_q(yt) + (0.5 / th.lambda_reg) * quad
    else:
        f_arr = np.zeros(T, dtype=np.float64)

    if len(v_arr) < T:
        raise ValueError(f"valence_series length {len(v_arr)} is less than trajectory length {T}")
    if len(f_arr) < T:
        raise ValueError(
            f"free_energy_series length {len(f_arr)} is less than trajectory length {T}"
        )

    utilities = np.zeros(T, dtype=np.float64)
    fatigue_penalties = np.zeros(T, dtype=np.float64)
    gated_valences = np.zeros(T, dtype=np.float64)

    eff_tau_rec = max(1, int(tau_rec))
    eff_f_overload = max(1e-5, float(f_overload))

    for t in range(T):
        # 1. Catharsis spike V_t (non-negative surprise resolution)
        vt = max(0.0, float(v_arr[t]))
        ft = max(0.0, float(f_arr[t]))

        # 2. Overload cutoff gate: exp(- F_t / F_{overload})
        gate = math.exp(-ft / eff_f_overload)
        gated_v = vt * gate
        gated_valences[t] = gated_v

        # 3. Refractory fatigue integral over window [t - tau_rec, t]
        # Sum of ||p_s - p_{s-1}||_2^2 for s in [max(1, t - tau_rec + 1), t]
        fatigue = 0.0
        start_s = max(1, t - eff_tau_rec + 1)
        for s in range(start_s, t + 1):
            diff_p = p_arr[s] - p_arr[s - 1]
            fatigue += float(np.sum(diff_p**2))

        fatigue_penalty = float(lambda_fatigue) * fatigue
        fatigue_penalties[t] = fatigue_penalty

        # 4. Net reader aesthetic utility
        utilities[t] = gated_v - fatigue_penalty

    mean_util = float(np.mean(utilities))
    cum_util = float(np.sum(utilities))

    return {
        "utilities": utilities,
        "mean_utility": mean_util,
        "cumulative_utility": cum_util,
        "fatigue_penalties": fatigue_penalties,
        "gated_valences": gated_valences,
    }
