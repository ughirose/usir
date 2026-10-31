"""Universal Style Genome (USIR v2.0 / Nishiki)
Gap 5: Style Chemistry - The Four Canonical Style Molecules & Unexplored Style Discovery Operator (usir-genesis).

This module implements:
1. evaluate_style_molecules:
   Evaluates the activation scores in [0, 1] of the 4 canonical aesthetic molecules:
   - Mol_Yugen: emerges from Sg-03 Pr (idx 2) x Sg-08 Lp (idx 7) x Sg-11 Ma (idx 10),
     amplified by so(8) metric coupling between (2, 7, 10).
   - Mol_Wabi: emerges from Sg-04 Jt (idx 3) x Sg-07 Cs (idx 6) x Sg-11 Ma (idx 10),
     amplified by so(8) metric coupling between (3, 6, 10).
   - Mol_QiYun: emerges from Sg-01 Dn/Gr (idx 0) x Sg-06 Mp/Sy (idx 5) x Sg-10 At (idx 9),
     gated by active symplectic momentum circulation tanh(||p||_{G_style}) > 0.
   - Mol_Iki: emerges from Sg-02 Cr (idx 1) x Sg-07 Cs (idx 6) x Sg-12 Ru (idx 11),
     amplified by so(8) metric coupling between (1, 6, 11).
2. discover_voronoi_genesis (usir_genesis):
   The 4th Style Algebra Operator for finding unexplored aesthetic coordinates
   in the Voronoi void of known reference particles within the 95% Mahalanobis trust region
   d_M^2 <= chi^2_{24, 0.95} = 36.4150.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from usir_py.v2.thermodynamics_so8 import SO8PullbackMetric

# Triplet index definitions for the 4 canonical style molecules
MOLECULE_TRIPLETS: Dict[str, Tuple[int, int, int]] = {
    "Mol_Yugen": (2, 7, 10),  # Sg-03 Pr, Sg-08 Lp, Sg-11 Ma
    "Mol_Wabi": (3, 6, 10),   # Sg-04 Jt, Sg-07 Cs, Sg-11 Ma
    "Mol_QiYun": (0, 5, 9),   # Sg-01 Gr, Sg-06 Sy, Sg-10 At
    "Mol_Iki": (1, 6, 11),    # Sg-02 Cr, Sg-07 Cs, Sg-12 Ru
}

# Chi-squared critical value for 24 degrees of freedom at alpha = 0.05 (95% confidence)
CHI2_24_095: float = 36.415028


@dataclass
class StyleMoleculesResult:
    """Activation scores and coupling diagnostics of the 4 canonical aesthetic molecules."""
    Mol_Yugen: float
    Mol_Wabi: float
    Mol_QiYun: float
    Mol_Iki: float
    dominantMolecule: str
    synestheticCouplingEnergy: float

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VoronoiGenesisResult:
    """Result of the 4th Style Algebra Operator (usirGenesis / Voronoi Void Discovery)."""
    z_genesis: np.ndarray
    mahalanobis_d2: float
    min_riemannian_distance: float
    inside_trust_region: bool
    molecules: StyleMoleculesResult
    initial_centroid_min_dist: float

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["z_genesis"] = self.z_genesis.tolist()
        return d


def _compute_triplet_coupling(g: np.ndarray, i: int, j: int, k: int) -> float:
    """Compute normalized off-diagonal Riemannian metric coupling between indices (i, j, k)."""
    # Normalized correlation coupling c_ab = |G_ab| / sqrt(G_aa * G_bb)
    denom_ij = math.sqrt(max(1e-12, g[i, i] * g[j, j]))
    c_ij = abs(float(g[i, j])) / denom_ij

    denom_jk = math.sqrt(max(1e-12, g[j, j] * g[k, k]))
    c_jk = abs(float(g[j, k])) / denom_jk

    denom_ki = math.sqrt(max(1e-12, g[k, k] * g[i, i]))
    c_ki = abs(float(g[k, i])) / denom_ki

    return (c_ij + c_jk + c_ki) / 3.0


def evaluate_style_molecules(
    z: Sequence[float] | np.ndarray,
    g_style: np.ndarray | None = None,
) -> StyleMoleculesResult:
    r"""Compute activation scores in [0, 1] of the 4 canonical aesthetic molecules.

    Args:
        z: 24D symplectic state vector (q, p) in R^24, or 12D configuration vector q in [0, 1]^12.
        g_style: Optional 12x12 Riemannian metric G_style. Default: SO8PullbackMetric().metric_matrix().

    Returns:
        StyleMoleculesResult containing Mol_Yugen, Mol_Wabi, Mol_QiYun, Mol_Iki,
        dominantMolecule, and synestheticCouplingEnergy.
    """
    z_arr = np.asarray(z, dtype=np.float64).ravel()
    if z_arr.shape[0] == 12:
        q = np.clip(z_arr, 0.0, 1.0)
        p = np.zeros(12, dtype=np.float64)
    elif z_arr.shape[0] == 24:
        q = np.clip(z_arr[:12], 0.0, 1.0)
        p = z_arr[12:]
    else:
        raise ValueError(f"evaluate_style_molecules expects 12D or 24D vector, got length {z_arr.shape[0]}")

    if g_style is None:
        g = SO8PullbackMetric().metric_matrix()
    else:
        g = np.asarray(g_style, dtype=np.float64)
        if g.shape != (12, 12):
            raise ValueError(f"g_style must have shape (12, 12), got {g.shape}")

    # 1. Total synesthetic coupling energy in q induced by off-diagonal SO(8) metric components:
    # E_coupling = sum_{i < j} |G_ij| * q_i * q_j
    coupling_energy = 0.0
    for i in range(12):
        for j in range(i + 1, 12):
            coupling_energy += abs(float(g[i, j])) * float(q[i]) * float(q[j])

    # 2. Mol_Yugen (幽玄): (idx 2, 7, 10)
    # Sg-03 Pr (idx 2) x Sg-08 Lp (idx 7) x Sg-11 Ma (idx 10)
    # Amplified by SO(8) metric coupling between (2, 7, 10)
    i_y, j_y, k_y = MOLECULE_TRIPLETS["Mol_Yugen"]
    base_yugen = (float(q[i_y]) * float(q[j_y]) * float(q[k_y])) ** (1.0 / 3.0)
    coup_yugen = _compute_triplet_coupling(g, i_y, j_y, k_y)
    score_yugen = min(1.0, max(0.0, base_yugen * (1.0 + coup_yugen)))

    # 3. Mol_Wabi (侘び・数寄): (idx 3, 6, 10)
    # Sg-04 Jt (idx 3) x Sg-07 Cs (idx 6) x Sg-11 Ma (idx 10)
    # Amplified by metric coupling between (3, 6, 10)
    i_w, j_w, k_w = MOLECULE_TRIPLETS["Mol_Wabi"]
    base_wabi = (float(q[i_w]) * float(q[j_w]) * float(q[k_w])) ** (1.0 / 3.0)
    coup_wabi = _compute_triplet_coupling(g, i_w, j_w, k_w)
    score_wabi = min(1.0, max(0.0, base_wabi * (1.0 + coup_wabi)))

    # 4. Mol_QiYun (気韻生動): (idx 0, 5, 9)
    # Sg-01 Dn/Gr (idx 0) x Sg-06 Mp/Sy (idx 5) x Sg-10 At (idx 9)
    # Gated by active symplectic momentum circulation tanh(||p||_{G_style}) > 0
    i_q, j_q, k_q = MOLECULE_TRIPLETS["Mol_QiYun"]
    base_qiyun = (float(q[i_q]) * float(q[j_q]) * float(q[k_q])) ** (1.0 / 3.0)
    coup_qiyun = _compute_triplet_coupling(g, i_q, j_q, k_q)
    # Symplectic momentum Riemannian norm ||p||_{G_style} = sqrt(p^T G p)
    p_norm_g = math.sqrt(max(0.0, float(p @ g @ p)))
    momentum_gate = math.tanh(p_norm_g)
    score_qiyun = min(1.0, max(0.0, base_qiyun * (1.0 + coup_qiyun) * momentum_gate))

    # 5. Mol_Iki (粋・いなせ): (idx 1, 6, 11)
    # Sg-02 Cr (idx 1) x Sg-07 Cs (idx 6) x Sg-12 Ru (idx 11)
    # Amplified by metric coupling between (1, 6, 11)
    i_ik, j_ik, k_ik = MOLECULE_TRIPLETS["Mol_Iki"]
    base_iki = (float(q[i_ik]) * float(q[j_ik]) * float(q[k_ik])) ** (1.0 / 3.0)
    coup_iki = _compute_triplet_coupling(g, i_ik, j_ik, k_ik)
    score_iki = min(1.0, max(0.0, base_iki * (1.0 + coup_iki)))

    # Determine dominant molecule
    molecules_dict = {
        "Mol_Yugen": score_yugen,
        "Mol_Wabi": score_wabi,
        "Mol_QiYun": score_qiyun,
        "Mol_Iki": score_iki,
    }
    dominant = max(molecules_dict, key=molecules_dict.get)  # type: ignore

    return StyleMoleculesResult(
        Mol_Yugen=score_yugen,
        Mol_Wabi=score_wabi,
        Mol_QiYun=score_qiyun,
        Mol_Iki=score_iki,
        dominantMolecule=dominant,
        synestheticCouplingEnergy=coupling_energy,
    )


def _build_g24(g_style: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Construct 24D Riemannian metric G_24 = block_diag(G_style, I_12) and its inverse."""
    g_inv = np.linalg.inv(g_style)
    i12 = np.eye(12, dtype=np.float64)
    g24 = np.block([
        [g_style, np.zeros((12, 12), dtype=np.float64)],
        [np.zeros((12, 12), dtype=np.float64), i12],
    ])
    g24_inv = np.block([
        [g_inv, np.zeros((12, 12), dtype=np.float64)],
        [np.zeros((12, 12), dtype=np.float64), i12],
    ])
    return g24, g24_inv


def discover_voronoi_genesis(
    known_particles: Sequence[Sequence[float] | np.ndarray],
    g_style: np.ndarray | None = None,
    options: Optional[Dict[str, Any]] = None,
) -> VoronoiGenesisResult:
    r"""Discover the optimal unexplored aesthetic coordinate z_genesis^* in the Voronoi void.

    Solves:
        z_genesis^* = argmax_{z \in E_{0.95}} min_{k \in {1,...,K}} d_{G_24}(z, z_k)
    Subject to:
        d_M^2(z, \bar{z}) = (z - \bar{z})^T \hat{\Sigma}_{24}^{-1} (z - \bar{z}) <= chi^2_{24, 0.95} = 36.4150
        q \in [0, 1]^{12}, p >= 0.

    Args:
        known_particles: Collection of K >= 2 reference style particles in R^24 (or 12D padded).
        g_style: 12x12 Riemannian metric. Defaults to SO8PullbackMetric().metric_matrix().
        options: Dictionary with optional tuning parameters:
            - lambda_shrinkage: Shrinkage factor lambda for covariance (default 0.15).
            - chi2_threshold: Confidence ellipsoid threshold (default 36.4150).
            - n_restarts: Multi-start restart count (default 8).
            - n_steps: Gradient ascent steps per restart (default 45).
            - learning_rate: Step size eta (default 0.08).
            - softmin_beta: Softmin smoothing parameter (default 8.0).

    Returns:
        VoronoiGenesisResult with z_genesis, mahalanobis_d2, min_riemannian_distance,
        inside_trust_region, molecules, and initial_centroid_min_dist.
    """
    if len(known_particles) < 2:
        raise ValueError(f"discover_voronoi_genesis requires at least 2 reference particles, got {len(known_particles)}")

    opts = options or {}
    lambda_shrinkage = float(opts.get("lambda_shrinkage", 0.15))
    chi2_threshold = float(opts.get("chi2_threshold", CHI2_24_095))
    n_restarts = int(opts.get("n_restarts", 8))
    n_steps = int(opts.get("n_steps", 45))
    learning_rate = float(opts.get("learning_rate", 0.08))
    softmin_beta = float(opts.get("softmin_beta", 8.0))

    # Standardize particles to (K, 24)
    particles_list: List[np.ndarray] = []
    for item in known_particles:
        arr = np.asarray(item, dtype=np.float64).ravel()
        if arr.shape[0] == 12:
            arr = np.pad(arr, (0, 12), mode="constant")
        elif arr.shape[0] != 24:
            raise ValueError(f"Particle must be 12D or 24D, got {arr.shape[0]}")
        particles_list.append(arr)

    particles = np.array(particles_list, dtype=np.float64)  # (K, 24)
    k_count = particles.shape[0]

    # Metrics
    if g_style is None:
        g12 = SO8PullbackMetric().metric_matrix()
    else:
        g12 = np.asarray(g_style, dtype=np.float64)
        if g12.shape != (12, 12):
            raise ValueError(f"g_style must have shape (12, 12), got {g12.shape}")

    g24, g24_inv = _build_g24(g12)

    # 1. Centroid and Regularized Shrinkage Covariance
    z_bar = np.mean(particles, axis=0)  # (24,)
    if k_count > 1:
        diffs_emp = particles - z_bar
        sigma_emp = (diffs_emp.T @ diffs_emp) / (k_count - 1.0)
    else:
        sigma_emp = np.zeros((24, 24), dtype=np.float64)

    # Regularized covariance: \hat{\Sigma}_{24} = (1 - \lambda)\Sigma_{emp} + \lambda G_{24}^{-1}
    sigma_hat = (1.0 - lambda_shrinkage) * sigma_emp + lambda_shrinkage * g24_inv
    # Guarantee strict symmetry and invertibility
    sigma_hat = 0.5 * (sigma_hat + sigma_hat.T) + 1e-6 * np.eye(24, dtype=np.float64)
    sigma_inv = np.linalg.inv(sigma_hat)

    # Calculate initial distance from centroid
    diff_init = z_bar - particles
    init_dists = [math.sqrt(max(0.0, float(diff_init[k] @ g24 @ diff_init[k]))) for k in range(k_count)]
    d0 = min(init_dists)

    def project_onto_feasible_set(z_cand: np.ndarray) -> np.ndarray:
        """Projects z into [0, 1]^12 x [0, inf)^12 and ellipsoid d_M^2 <= chi2_threshold."""
        z_proj = z_cand.copy()
        for _ in range(5):
            # Box bounds: q in [0, 1], p in [0, 2.0]
            z_proj[:12] = np.clip(z_proj[:12], 0.0, 1.0)
            z_proj[12:] = np.clip(z_proj[12:], 0.0, 2.0)

            # Ellipsoid constraint
            diff = z_proj - z_bar
            d2 = float(diff @ sigma_inv @ diff)
            if d2 > chi2_threshold:
                z_proj = z_bar + diff * math.sqrt((chi2_threshold - 1e-4) / d2)
            else:
                break
        return z_proj

    # 2. Multi-Start Projected Gradient Ascent
    best_z = z_bar.copy()
    best_min_dist = -1.0

    # Build initial candidate seeds
    seeds: List[np.ndarray] = [z_bar.copy()]
    # Pairwise midpoints or anti-points
    for k in range(min(k_count, n_restarts - 1)):
        # Point pushed away from particle k relative to centroid
        anti_seed = z_bar + 0.6 * (z_bar - particles[k])
        seeds.append(anti_seed)

    rng = np.random.default_rng(42)
    while len(seeds) < n_restarts:
        # Gaussian perturbation in coordinate space
        rand_dir = rng.normal(0.0, 0.25, size=24)
        seeds.append(z_bar + rand_dir)

    for seed_idx, raw_seed in enumerate(seeds[:n_restarts]):
        z_curr = project_onto_feasible_set(raw_seed)

        for step in range(n_steps):
            diffs = z_curr - particles  # (K, 24)
            dists = np.array([
                math.sqrt(max(1e-12, float(diffs[k] @ g24 @ diffs[k])))
                for k in range(k_count)
            ])

            # Smooth softmin approximation to minimum distance
            min_d = np.min(dists)
            weights = np.exp(-softmin_beta * (dists - min_d))
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                weights /= weight_sum
            else:
                weights = np.ones(k_count) / k_count

            # Ascent gradient: sum_k w_k * G_24 * (z - z_k) / dist_k
            grad = np.zeros(24, dtype=np.float64)
            for k in range(k_count):
                grad += weights[k] * (g24 @ diffs[k]) / dists[k]

            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-6:
                step_vec = (grad / grad_norm) * learning_rate
            else:
                step_vec = grad * learning_rate

            z_next = z_curr + step_vec
            z_curr = project_onto_feasible_set(z_next)

        # Evaluate exact minimum Riemannian distance
        cur_diffs = z_curr - particles
        cur_dists = [
            math.sqrt(max(0.0, float(cur_diffs[k] @ g24 @ cur_diffs[k])))
            for k in range(k_count)
        ]
        cur_min_dist = min(cur_dists)

        if cur_min_dist > best_min_dist:
            best_min_dist = cur_min_dist
            best_z = z_curr.copy()

    # Final projection to ensure absolute strict compliance with trust region
    best_z = project_onto_feasible_set(best_z)
    final_diff = best_z - z_bar
    final_d2 = float(final_diff @ sigma_inv @ final_diff)

    final_diffs = best_z - particles
    final_dists = [
        math.sqrt(max(0.0, float(final_diffs[k] @ g24 @ final_diffs[k])))
        for k in range(k_count)
    ]
    final_min_dist = min(final_dists)

    molecules = evaluate_style_molecules(best_z, g12)

    return VoronoiGenesisResult(
        z_genesis=best_z,
        mahalanobis_d2=final_d2,
        min_riemannian_distance=final_min_dist,
        inside_trust_region=(final_d2 <= chi2_threshold + 1e-4),
        molecules=molecules,
        initial_centroid_min_dist=d0,
    )


# Aliases
usir_genesis = discover_voronoi_genesis
evaluateStyleMolecules = evaluate_style_molecules
discoverVoronoiGenesis = discover_voronoi_genesis
usirGenesis = discover_voronoi_genesis
