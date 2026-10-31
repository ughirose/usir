"""USIR v2.0 Module 4: Adaptive-BAOAB Symplectic Integrator, Clause-Level Twisted SMC & Jacobian Logit Projector.

This module provides the hierarchical dual-rate generation control engines for USIR v2.0:
1. AdaptiveBAOABIntegrator:
   - 5-stage symmetric splitting integrator (B - A - O - A - B) on (q, p) in R^12 x R^12
   - Syntax-entropy-dependent anisotropic friction Gamma_t(H_{syntax}) = gamma_0 exp(eta(H_0 - H_{syntax})) I_12
   - Exact Ornstein-Uhlenbeck update in Stage O preserving N(0, tau M)
   - verify_invariant_momentum_measure(...) verifying Theorem 5.1 under 1,000x random friction shifts
2. ClauseTwistedSMC:
   - Outer loop clause-level 4-particle Twisted Sequential Monte Carlo rollout
   - Schrodinger Bridge terminal anchor guidance potential psi_{T-t}(z_t; z_{end}^*)
   - Systematic resampling when ESS < N_{part} / 2
3. JacobianLogitProjector:
   - Inner loop KV-cache non-polluting Top-K=64 candidate token Jacobian logit steering
   - Syntax protection gate: if H_{LLM} < eps_{syn}, beta_{steer} = 0 (bypasses deterministic syntax boundaries)
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

from .thermodynamics_so8 import SO8PullbackMetric


class AdaptiveBAOABIntegrator:
    r"""5-stage symmetric splitting BAOAB integrator with state-dependent anisotropic friction.

    Mathematical Splitting:
        1. Stage B:  p_{t+1/2}^{(-)} = p_t - (Delta t / 2) nabla_q F_{q, lambda}(q_t)
        2. Stage A:  q_{t+1/2}      = q_t + (Delta t / 2) M^{-1} p_{t+1/2}^{(-)}
        3. Stage O:  p_{t+1/2}^{(+)} = Theta_t p_{t+1/2}^{(-)} + sqrt{tau (I - Theta_t^2)} M^{1/2} xi_t
                     where Theta_t = exp(- Gamma_t Delta t), xi_t ~ N(0, I_{12}),
                     and Gamma_t(H_syntax) = gamma_0 exp(eta (H_0 - H_syntax)) I_{12}.
        4. Stage A:  q_{t+1}        = q_{t+1/2} + (Delta t / 2) M^{-1} p_{t+1/2}^{(+)}
        5. Stage B:  p_{t+1}        = p_{t+1/2}^{(+)} - (Delta t / 2) nabla_q F_{q, lambda}(q_{t+1})

    Theorem 5.1 (Invariant Measure Preservation):
        Even when Gamma_t jumps randomly by 1,000x every step (varying wildly between 10^{-2} and 10^{+1}),
        the stationary momentum distribution under Stage O is strictly invariant and equals N(0, tau M).
    """

    def __init__(
        self,
        mass_matrix: np.ndarray | None = None,
        gamma0: float = 1.0,
        eta: float = 2.0,
        h0: float = 2.5,
        gamma_min: float = 1e-3,
        gamma_max: float = 1e3,
    ) -> None:
        """Initialize the Adaptive-BAOAB Integrator.

        Args:
            mass_matrix: 12x12 SPD mass matrix M (defaults to I_12).
            gamma0: Base friction coefficient gamma_0.
            eta: Syntax entropy sensitivity exponent eta.
            h0: Baseline syntax entropy threshold H_0.
            gamma_min: Numerical lower bound for friction coefficient.
            gamma_max: Numerical upper bound for friction coefficient.
        """
        self.dim = 12
        if mass_matrix is not None:
            M = np.asarray(mass_matrix, dtype=np.float64)
            if M.shape != (12, 12):
                raise ValueError(f"mass_matrix must have shape (12, 12), got {M.shape}")
            self.M = 0.5 * (M + M.T)
        else:
            self.M = np.eye(12, dtype=np.float64)

        self.M_inv = np.linalg.inv(self.M)
        # Compute M^{1/2} via eigendecomposition: M = V diag(d) V^T => M^{1/2} = V diag(sqrt(d)) V^T
        eigvals, eigvecs = np.linalg.eigh(self.M)
        if np.any(eigvals <= 0):
            raise ValueError("Mass matrix M must be strictly positive definite.")
        self.M_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

        self.gamma0 = float(gamma0)
        self.eta = float(eta)
        self.h0 = float(h0)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)

    def compute_friction(
        self,
        h_syntax: float | None = None,
        gamma_override: float | np.ndarray | None = None,
    ) -> np.ndarray:
        r"""Compute the anisotropic friction tensor Gamma_t:

            Gamma_t(H_{syntax}) = gamma_0 exp(eta * (H_0 - H_{syntax})) I_{12}
        """
        if gamma_override is not None:
            if np.isscalar(gamma_override):
                val = float(gamma_override)
                return np.eye(self.dim, dtype=np.float64) * val
            arr = np.asarray(gamma_override, dtype=np.float64)
            if arr.shape == (self.dim,):
                return np.diag(arr)
            elif arr.shape == (self.dim, self.dim):
                return arr
            raise ValueError(f"Invalid gamma_override shape: {arr.shape}")

        if h_syntax is not None:
            # High syntax entropy (freedom) => low friction
            # Low syntax entropy (rigid structure) => high friction (brake)
            val = self.gamma0 * math.exp(self.eta * (self.h0 - float(h_syntax)))
            val = float(np.clip(val, self.gamma_min, self.gamma_max))
            return np.eye(self.dim, dtype=np.float64) * val

        return np.eye(self.dim, dtype=np.float64) * self.gamma0

    def step(
        self,
        q: np.ndarray,
        p: np.ndarray,
        grad_fn: Callable[[np.ndarray], np.ndarray],
        h_syntax: float | None = None,
        gamma_override: float | np.ndarray | None = None,
        dt: float = 0.1,
        tau: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform one 5-stage BAOAB integration step.

        Args:
            q: Current position vector in R^12.
            p: Current momentum vector in R^12.
            grad_fn: Gradient oracle returning nabla_q F(q) in R^12.
            h_syntax: Current syntax entropy H_syntax.
            gamma_override: Direct override for friction tensor/coefficient.
            dt: Integration step size Delta t.
            tau: Thermodynamic temperature tau.
            rng: Optional numpy RandomGenerator.

        Returns:
            (q_next, p_next): Next phase space coordinates in R^12 x R^12.
        """
        if rng is None:
            rng = np.random.default_rng()

        q_curr = np.asarray(q, dtype=np.float64).copy()
        p_curr = np.asarray(p, dtype=np.float64).copy()
        dt_half = 0.5 * dt

        # 1. Stage B: half-step momentum kick
        grad_0 = np.asarray(grad_fn(q_curr), dtype=np.float64)
        p_half_minus = p_curr - dt_half * grad_0

        # 2. Stage A: half-step position drift
        q_half = q_curr + dt_half * (self.M_inv @ p_half_minus)

        # 3. Stage O: exact Ornstein-Uhlenbeck thermal bath step
        Gamma = self.compute_friction(h_syntax, gamma_override)
        # For diagonal / scalar Gamma = gamma * I
        if np.allclose(Gamma, np.diag(np.diag(Gamma))):
            gamma_diag = np.diag(Gamma)
            theta = np.exp(-gamma_diag * dt)  # Shape (12,)
            # sqrt(tau * (1 - theta^2))
            sigma_noise = np.sqrt(np.maximum(0.0, tau * (1.0 - theta**2)))
            xi = rng.normal(0.0, 1.0, size=self.dim)
            noise_term = sigma_noise * (self.M_sqrt @ xi)
            p_half_plus = theta * p_half_minus + noise_term
        else:
            # Full matrix Ornstein-Uhlenbeck: Theta = expm(- Gamma * dt)
            import scipy.linalg
            Theta = scipy.linalg.expm(-Gamma * dt)
            I12 = np.eye(self.dim, dtype=np.float64)
            diff_mat = I12 - Theta @ Theta.T
            diff_mat = 0.5 * (diff_mat + diff_mat.T)
            eig_d, eig_v = np.linalg.eigh(diff_mat)
            sqrt_diff = eig_v @ np.diag(np.sqrt(np.maximum(0.0, eig_d))) @ eig_v.T
            xi = rng.normal(0.0, 1.0, size=self.dim)
            noise_term = math.sqrt(tau) * (sqrt_diff @ (self.M_sqrt @ xi))
            p_half_plus = Theta @ p_half_minus + noise_term

        # 4. Stage A: half-step position drift
        q_next = q_half + dt_half * (self.M_inv @ p_half_plus)

        # 5. Stage B: half-step momentum kick
        grad_next = np.asarray(grad_fn(q_next), dtype=np.float64)
        p_next = p_half_plus - dt_half * grad_next

        return q_next, p_next

    def verify_invariant_momentum_measure(
        self,
        n_steps: int = 20000,
        tau: float = 1.0,
        seed: int = 42,
        tol_cov: float = 0.10,
        tol_mean: float = 0.06,
    ) -> dict[str, object]:
        r"""Verify Theorem 5.1: Momentum Gaussian invariant measure under 1,000x random friction shifts.

        In the O-step / free-particle limit (grad_q F = 0), switches Gamma_t randomly between
        10^{-2} and 10^{+1} (a factor of 1,000x jump) at every step and asserts that
        p_t follows N(0, tau M) with zero drift and exact target covariance.
        """
        rng = np.random.default_rng(seed)
        p_samples = np.zeros((n_steps, self.dim), dtype=np.float64)

        # Initial momentum sampled from exact target distribution N(0, tau M)
        xi_init = rng.normal(0.0, 1.0, size=self.dim)
        p_curr = math.sqrt(tau) * (self.M_sqrt @ xi_init)
        dt = 0.1

        for step_idx in range(n_steps):
            # 1,000x random friction switch: log10(gamma) in [-2.0, 1.0] => gamma in [0.01, 10.0]
            log_gamma = rng.uniform(-2.0, 1.0)
            gamma_k = 10.0**log_gamma

            # Exact Stage O update
            theta = math.exp(-gamma_k * dt)
            sigma_noise = math.sqrt(max(0.0, tau * (1.0 - theta**2)))
            xi = rng.normal(0.0, 1.0, size=self.dim)
            p_curr = theta * p_curr + sigma_noise * (self.M_sqrt @ xi)
            p_samples[step_idx] = p_curr

        # Statistical analysis against N(0, tau M)
        mean_p = np.mean(p_samples, axis=0)
        target_cov = tau * self.M
        emp_cov = np.cov(p_samples, rowvar=False)

        # Relative Frobenius norm error on covariance
        cov_frob_err = float(np.linalg.norm(emp_cov - target_cov) / np.linalg.norm(target_cov))
        diag_err = float(np.mean(np.abs(np.diag(emp_cov) - np.diag(target_cov))))
        max_abs_mean = float(np.max(np.abs(mean_p)))
        passed = bool(cov_frob_err < tol_cov and max_abs_mean < tol_mean)

        return {
            "n_steps": n_steps,
            "max_abs_mean": max_abs_mean,
            "cov_relative_error": cov_frob_err,
            "diag_mean_error": diag_err,
            "passed": passed,
            "empirical_mean": mean_p,
            "empirical_cov": emp_cov,
            "target_cov": target_cov,
        }


class ClauseTwistedSMC:
    r"""Outer loop clause-level 4-particle Twisted Sequential Monte Carlo (SMC) controller.

    Mathematical Formulation:
        1. Particle rollouts: P = 4 candidate clauses {c_{m, k}^{(p)}} (p = 1..4)
        2. Schrödinger Bridge Terminal Guidance Potential:
               psi_{T-t}(z_t; z_{end}^*) = exp( - (1 / (2 sigma_{bridge}^2 (T - t + eps))) ||q(c) - q_{end}^*||_{G_{style}}^2 )
        3. Twisted SMC Weight:
               w^{(p)} = ln P_{LLM}(c^{(p)} | w_{<k})
                       + ln psi_{T-t}(hat{q}(c^{(p)}); q_{end}^*)
                       - (beta_{steer} / 2) ||hat{q}(c^{(p)}) - q_t^*||_{G_{style}}^2
        4. Systematic Resampling:
               Executed whenever ESS = 1 / sum (W^{(p)})^2 < N_{part} / 2.
    """

    def __init__(
        self,
        metric: SO8PullbackMetric | np.ndarray | None = None,
        n_particles: int = 4,
        beta_steer: float = 1.0,
        sigma_bridge: float = 1.0,
        eps_time: float = 0.1,
    ) -> None:
        """Initialize ClauseTwistedSMC."""
        if metric is None:
            self.metric = SO8PullbackMetric()
        elif isinstance(metric, SO8PullbackMetric):
            self.metric = metric
        elif isinstance(metric, np.ndarray):
            self.metric = SO8PullbackMetric()
            self.metric._G = metric.copy()
            self.metric._G_inv = np.linalg.inv(metric)
        else:
            raise TypeError(f"Invalid metric: {type(metric)}")

        self.G = self.metric.metric_matrix()
        self.n_particles = int(n_particles)
        self.beta_steer = float(beta_steer)
        self.sigma_bridge = float(sigma_bridge)
        self.eps_time = float(eps_time)

    def schrodinger_bridge_log_potential(
        self,
        q_cand: np.ndarray,
        q_end: np.ndarray | None,
        t: int,
        T: int,
    ) -> float:
        r"""Compute terminal anchor guidance log potential:

            ln psi_{T-t}(q; q_{end}^*) = - (1 / (2 sigma_{bridge}^2 (T - t + eps))) ||q - q_{end}^*||_G^2
        """
        if q_end is None or t >= T:
            return 0.0

        remaining_time = max(self.eps_time, float(T - t))
        diff = np.asarray(q_cand, dtype=np.float64) - np.asarray(q_end, dtype=np.float64)
        dist_sq = float(diff.T @ self.G @ diff)

        coeff = 1.0 / (2.0 * (self.sigma_bridge**2) * remaining_time)
        return -coeff * dist_sq

    def evaluate_particle_weights(
        self,
        candidate_q: Sequence[np.ndarray],
        llm_logprobs: Sequence[float],
        target_q: np.ndarray,
        terminal_anchor_q: np.ndarray | None = None,
        t: int = 0,
        T: int = 10,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute unnormalized and normalized twisted weights across all particles.

        Returns:
            (log_weights, normalized_weights, ess)
        """
        P = len(candidate_q)
        log_w = np.zeros(P, dtype=np.float64)

        for p_idx in range(P):
            q_cand = np.asarray(candidate_q[p_idx], dtype=np.float64)
            llm_lp = float(llm_logprobs[p_idx])

            # 1. Target USIR trajectory tracking penalty
            diff_target = q_cand - np.asarray(target_q, dtype=np.float64)
            dist_sq_target = float(diff_target.T @ self.G @ diff_target)
            penalty_target = -0.5 * self.beta_steer * dist_sq_target

            # 2. Schrödinger Bridge terminal reachability potential
            bridge_lp = self.schrodinger_bridge_log_potential(
                q_cand, terminal_anchor_q, t, T
            )

            log_w[p_idx] = llm_lp + bridge_lp + penalty_target

        # Normalize via log-sum-exp
        max_log = np.max(log_w)
        exp_w = np.exp(log_w - max_log)
        sum_exp = np.sum(exp_w)
        norm_w = exp_w / sum_exp

        # Effective Sample Size
        ess = 1.0 / float(np.sum(norm_w**2))
        return log_w, norm_w, ess

    def systematic_resample(
        self,
        weights: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> list[int]:
        r"""Perform low-variance Systematic Resampling on normalized particle weights."""
        if rng is None:
            rng = np.random.default_rng()

        P = len(weights)
        cumsum = np.cumsum(weights)
        cumsum[-1] = 1.0  # Avoid roundoff errors

        u0 = rng.uniform(0.0, 1.0 / P)
        positions = u0 + np.arange(P) / float(P)

        indices: list[int] = []
        j = 0
        for pos in positions:
            while j < P and cumsum[j] < pos:
                j += 1
            indices.append(min(j, P - 1))

        return indices

    def step(
        self,
        candidate_clauses: Sequence[dict[str, object]],
        target_q: np.ndarray,
        terminal_anchor_q: np.ndarray | None = None,
        t: int = 0,
        T: int = 10,
        rng: np.random.Generator | None = None,
    ) -> dict[str, object]:
        """Execute one clause-level SMC filtering step over candidate clauses.

        Each candidate dict must contain:
            - 'q': 12D array representing style vector
            - 'llm_logprob': float representing base LLM generation log-probability
            - 'text': clause text or tokens (optional)

        Returns:
            dict containing selected candidate, weights, ESS, and resampling status.
        """
        candidate_q = [np.asarray(c["q"], dtype=np.float64) for c in candidate_clauses]
        llm_lp = [float(c.get("llm_logprob", 0.0)) for c in candidate_clauses]

        log_w, norm_w, ess = self.evaluate_particle_weights(
            candidate_q, llm_lp, target_q, terminal_anchor_q, t, T
        )

        resampled = False
        resample_indices: list[int] | None = None
        # Systematic resampling trigger: ESS < N_part / 2
        if ess < (self.n_particles / 2.0):
            resample_indices = self.systematic_resample(norm_w, rng=rng)
            resampled = True

        best_idx = int(np.argmax(norm_w))
        selected_clause = candidate_clauses[best_idx]

        return {
            "selected_index": best_idx,
            "selected_clause": selected_clause,
            "normalized_weights": norm_w,
            "log_weights": log_w,
            "ess": ess,
            "resampled": resampled,
            "resample_indices": resample_indices,
        }


class JacobianLogitProjector:
    r"""Inner loop KV-cache non-polluting Top-K=64 candidate token Jacobian logit steering.

    Mathematical Formulation:
        1. Steering rule:
               \tilde{\ell}_k = \ell_k - \beta_{steer}(H_{LLM}) <nabla_q F(q_t), \Delta \hat{q}(w_k)>
        2. Syntax Protection Gate:
               If H_{LLM} < eps_{syn}:
                   \beta_{steer} = 0  (100% preserves deterministic grammar / closing tokens).
               Else:
                   \beta_{steer}(H_{LLM}) = \beta_0 * sigmoid((H_{LLM} - eps_{syn}) / tau_{gate}).
    """

    def __init__(
        self,
        metric: SO8PullbackMetric | np.ndarray | None = None,
        beta_0: float = 1.0,
        eps_syn: float = 0.5,
        tau_gate: float = 0.2,
        top_k: int = 64,
    ) -> None:
        """Initialize Jacobian Logit Projector."""
        if metric is None:
            self.metric = SO8PullbackMetric()
        elif isinstance(metric, SO8PullbackMetric):
            self.metric = metric
        elif isinstance(metric, np.ndarray):
            self.metric = SO8PullbackMetric()
            self.metric._G = metric.copy()
            self.metric._G_inv = np.linalg.inv(metric)
        else:
            raise TypeError(f"Invalid metric: {type(metric)}")

        self.G = self.metric.metric_matrix()
        self.beta_0 = float(beta_0)
        self.eps_syn = float(eps_syn)
        self.tau_gate = float(tau_gate)
        self.top_k = int(top_k)

    def compute_entropy_and_gate(self, logits: np.ndarray) -> tuple[float, float, bool]:
        """Compute base LLM next-token entropy and adaptive steering gain beta_steer.

        Returns:
            (H_LLM, beta_steer, is_gated)
        """
        # Softmax probabilities
        shift = np.max(logits)
        exp_vals = np.exp(logits - shift)
        probs = exp_vals / np.sum(exp_vals)

        # Shannon entropy in nats
        p_safe = np.clip(probs, 1e-15, 1.0)
        h_llm = float(-np.sum(p_safe * np.log(p_safe)))

        # Syntax protection gate:
        if h_llm < self.eps_syn:
            return h_llm, 0.0, True

        # Smooth sigmoidal ramp
        gate_arg = (h_llm - self.eps_syn) / max(1e-5, self.tau_gate)
        # Avoid overflow
        gate_arg = max(-30.0, min(30.0, gate_arg))
        sigmoid_val = 1.0 / (1.0 + math.exp(-gate_arg))
        beta = self.beta_0 * sigmoid_val
        return h_llm, beta, False

    def project_logits(
        self,
        logits_topk: np.ndarray,
        grad_f: np.ndarray,
        delta_q_matrix: np.ndarray,
        h_llm_override: float | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        r"""Steer candidate token logits without polluting base LLM KV-cache.

        Args:
            logits_topk: Raw LLM logits for Top-K candidate tokens (shape: (K,)).
            grad_f: Style free energy gradient nabla_q F(q_t) (shape: (12,)).
            delta_q_matrix: Displacement vectors Delta hat{q}(w_k) for each token k (shape: (K, 12)).
            h_llm_override: Optional explicit syntax entropy override.

        Returns:
            (steered_logits, telemetry_dict)
        """
        logits = np.asarray(logits_topk, dtype=np.float64).ravel()
        K = min(self.top_k, len(logits))
        logits = logits[:K]

        grad = np.asarray(grad_f, dtype=np.float64).ravel()
        delta_q = np.asarray(delta_q_matrix, dtype=np.float64)[:K]

        # 1. Compute entropy & steering gain
        if h_llm_override is not None:
            h_llm = float(h_llm_override)
            if h_llm < self.eps_syn:
                beta = 0.0
                is_gated = True
            else:
                gate_arg = (h_llm - self.eps_syn) / max(1e-5, self.tau_gate)
                sigmoid_val = 1.0 / (1.0 + math.exp(-gate_arg))
                beta = self.beta_0 * sigmoid_val
                is_gated = False
        else:
            h_llm, beta, is_gated = self.compute_entropy_and_gate(logits)

        # 2. Compute inner products: <nabla_q F, Delta hat{q}(w_k)>
        # Shape: (K,)
        inner_prods = delta_q @ grad

        # 3. Steer logits
        steered_logits = logits - beta * inner_prods

        # Compute steered probabilities for telemetry
        shift_s = np.max(steered_logits)
        exp_s = np.exp(steered_logits - shift_s)
        steered_probs = exp_s / np.sum(exp_s)

        telemetry = {
            "h_llm": h_llm,
            "beta_steer": beta,
            "syntax_gate_active": is_gated,
            "steered_probs": steered_probs,
            "max_logit_delta": float(np.max(np.abs(steered_logits - logits))),
        }

        return steered_logits, telemetry
