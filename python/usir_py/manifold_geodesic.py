"""USIR v4.0 Manifold Geodesic Engine & 3-Phase QDA Asymmetric Double-Well Barrier.

Implements:
1. Asymmetric Double-Well 3-Phase QDA Functional:
   D_3phase(z) = 0.5 * (z - mu_crit)^T Sigma_crit^-1 (z - mu_crit)
                 + gamma_freeze * max(0, sigma_low - ||sigma||_2)^2
                 + gamma_chaos * max(0, ||sigma||_2 - sigma_high)^2
2. Semantic Space Trajectory Geodesic Curvature & Brownian Motion Detector:
   kappa_{sem, m} = 1 - cos(Delta e_{m+1}, Delta e_m)
   Torsion tau_m and Brownian ratio eta_brownian = sum ||Delta e_m||^2 / ||e_M - e_1||^2
   M_geodesic(y) in [0, 1]
3. 4-Fold Multiplicative GRPO Barrier Reward:
   R_GRPO(y | x_0, x_T) = M_syntax(y) * M_geodesic(y) * M_bridge(y | x_0, x_T) * exp(-0.5 * D_3phase(z(y)))
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .extractor_24d import PHYSICAL_SENSE_INTERVALS, extract_usir_24d
from .grpo_reward import (
    _FEATURE_STD,
    BRACKET_PAIRS,
    HARD_AI_CLICHE_FRAMES,
    META_GOODHART_TERMS,
    MODERN_LOANWORDS,
    PHASE_CENTROIDS_24D,
    compute_cleanroom_multiplier,
)

# Canonical 24D Critical Phase Centroid
_MU_CRIT = PHASE_CENTROIDS_24D.get("II_Critical", np.zeros(24))
_SIGMA_CRIT_DIAG = _FEATURE_STD.copy()
_SIGMA_CRIT_DIAG[_SIGMA_CRIT_DIAG < 1e-4] = 1e-4
_SIGMA_CRIT_INV = 1.0 / (_SIGMA_CRIT_DIAG ** 2)

# Domain-conditioned 3-Phase QDA target profiles
_MU_ARTICLE = _MU_CRIT.copy()
_MU_ARTICLE[9:12] = np.array([0.70, 0.40, 0.35])
_MU_COT = _MU_CRIT.copy()
_MU_COT[0] = 0.60
_MU_COT[2] = 0.08
_MU_COT[4] = 0.45
_MU_COT[8] = 0.50
_MU_COT[9:12] = np.array([0.85, 0.20, 0.25])

DOMAIN_QDA_PROFILES: Dict[str, Dict[str, Any]] = {
    "novel": {
        "mu_crit": _MU_CRIT,
        "sigma_crit_low": 0.35,
        "sigma_crit_high": 1.25,
    },
    "article": {
        "mu_crit": _MU_ARTICLE,
        "sigma_crit_low": 0.30,
        "sigma_crit_high": 1.40,
    },
    "cot_reasoning": {
        "mu_crit": _MU_COT,
        "sigma_crit_low": 1.35,
        "sigma_crit_high": 2.65,
    },
}


# ==============================================================================
# 1. 3-Phase QDA Asymmetric Double-Well Barrier Functional
# ==============================================================================

def compute_three_phase_qda_distance(
    z: Union[np.ndarray, List[float], Dict[str, Any], str],
    domain: str = "novel",
    mu_crit: Optional[np.ndarray] = None,
    sigma_crit_inv: Optional[np.ndarray] = None,
    sigma_norm: Optional[float] = None,
    gamma_freeze: float = 12.0,
    gamma_chaos: float = 12.0,
    sigma_crit_low: Optional[float] = None,
    sigma_crit_high: Optional[float] = None,
) -> Dict[str, Any]:
    r"""Computes the domain-conditioned 3-phase QDA asymmetric double-well barrier distance functional:

    \mathcal{D}_{\text{3-phase}}(\mathbf{z}; \text{domain}) =
        \frac{1}{2}(\mathbf{z} - \boldsymbol{\mu}_{\text{crit}}^{(\text{domain})})^\top
        \Sigma_{\text{crit}}^{-1} (\mathbf{z} - \boldsymbol{\mu}_{\text{crit}}^{(\text{domain})})
        + \gamma_{\text{freeze}} \max(0, \sigma_{\text{crit}}^{\text{low}} - \|\boldsymbol{\sigma}\|_2)^2
        + \gamma_{\text{chaos}} \max(0, \|\boldsymbol{\sigma}\|_2 - \sigma_{\text{crit}}^{\text{high}})^2

    Rejects both Phase I Deadzone (under-excited freeze) and Phase III Chaos (purple prose / punctuation spam).
    """
    profile = DOMAIN_QDA_PROFILES.get(domain, DOMAIN_QDA_PROFILES["novel"])
    eff_sigma_low = float(sigma_crit_low) if sigma_crit_low is not None else float(profile["sigma_crit_low"])
    eff_sigma_high = float(sigma_crit_high) if sigma_crit_high is not None else float(profile["sigma_crit_high"])

    if isinstance(z, str):
        extracted = extract_usir_24d(z)
        z_vec = np.asarray(extracted["z_24d"], dtype=float)
        p_vec = np.asarray(extracted["p"], dtype=float)
        calc_sigma_norm = float(np.linalg.norm(p_vec))
    elif isinstance(z, dict):
        z_vec = np.asarray(z.get("z_24d", np.zeros(24)), dtype=float)
        p_vec = np.asarray(z.get("p", z_vec[12:24]), dtype=float)
        calc_sigma_norm = float(np.linalg.norm(p_vec))
    else:
        z_vec = np.asarray(z, dtype=float)
        if len(z_vec) >= 24:
            p_vec = z_vec[12:24]
            calc_sigma_norm = float(np.linalg.norm(p_vec))
        else:
            calc_sigma_norm = 0.5

    if sigma_norm is not None:
        calc_sigma_norm = float(sigma_norm)

    target_mu = np.asarray(profile["mu_crit"], dtype=float) if mu_crit is None else np.asarray(mu_crit, dtype=float)
    inv_cov = _SIGMA_CRIT_INV if sigma_crit_inv is None else np.asarray(sigma_crit_inv, dtype=float)

    diff = z_vec - target_mu
    if inv_cov.ndim == 1:
        d_mahalanobis = float(0.5 * np.sum((diff ** 2) * inv_cov) / 24.0)
    else:
        d_mahalanobis = float(0.5 * (diff.T @ inv_cov @ diff) / 24.0)

    # Freeze barrier penalty (Left wall)
    freeze_defect = max(0.0, eff_sigma_low - calc_sigma_norm)
    penalty_freeze = gamma_freeze * (freeze_defect ** 2)

    # Chaos barrier penalty (Right wall)
    chaos_excess = max(0.0, calc_sigma_norm - eff_sigma_high)
    penalty_chaos = gamma_chaos * (chaos_excess ** 2)

    total_d_3phase = float(d_mahalanobis + penalty_freeze + penalty_chaos)

    if freeze_defect > 0.05:
        regime = "Phase_I_Deadzone"
    elif chaos_excess > 0.05:
        regime = "Phase_III_Chaos"
    else:
        regime = "Phase_II_Critical"

    return {
        "d_3phase": total_d_3phase,
        "d_mahalanobis": d_mahalanobis,
        "penalty_freeze": penalty_freeze,
        "penalty_chaos": penalty_chaos,
        "sigma_norm": calc_sigma_norm,
        "phase_regime": regime,
        "is_healthy_critical": (freeze_defect == 0.0 and chaos_excess == 0.0),
    }


# ==============================================================================
# 2. Semantic Space Trajectory Geodesic Curvature & Brownian Motion Detector
# ==============================================================================

def _split_into_trajectory_units(text: str) -> List[str]:
    """Splits text into fine-grained sentences or clauses for discrete geodesic tracking."""
    # Split by standard sentence delimiters
    raw_sents = [s.strip() for s in re.split(r'[。！？!?\n]+', text) if len(s.strip()) >= 2]
    if len(raw_sents) >= 3:
        return raw_sents

    # If fewer than 3 sentences, split by punctuation clauses to enable discrete geometry
    clauses = [c.strip() for c in re.split(r'[、,;；]+', text) if len(c.strip()) >= 3]
    if len(clauses) >= 3:
        return clauses

    # Fallback: slice into fixed-width units
    stride = max(10, len(text) // 5)
    return [text[i:i+stride] for i in range(0, len(text), stride) if len(text[i:i+stride]) >= 3]


def _embed_sentence_unit(sentence: str, dim: int = 24) -> np.ndarray:
    """Computes a deterministic semantic and structural state vector e_m in R^dim.

    Combines:
    1. Sensory / concrete physical grounding (Kanji radical intervals)
    2. Stylistic syntactic frame (clause count, length, terminal markers)
    3. Content concept hash projection (orthogonal random projection of kanji stems)
    """
    vec = np.zeros(dim, dtype=float)
    if not sentence:
        return vec

    L = max(len(sentence), 1)

    # 1. Sensory concrete density
    sensory_hits = 0
    for ch in sentence:
        cp = ord(ch)
        for lo, hi in PHYSICAL_SENSE_INTERVALS:
            if lo <= cp <= hi:
                sensory_hits += 1
                break
    sensory_density = sensory_hits / L
    vec[0] = np.clip(sensory_density * 4.0, 0.0, 1.0)

    # 2. Abstract bloat density
    abst_count = sum(sentence.count(f) for f in HARD_AI_CLICHE_FRAMES)
    vec[1] = np.clip(abst_count / max(L / 20.0, 1.0), 0.0, 1.0)

    # 3. Kanji ratio
    kanji_count = sum(1 for ch in sentence if '\u4e00' <= ch <= '\u9fff')
    vec[2] = kanji_count / L

    # 4. Length norm
    vec[3] = np.clip(L / 50.0, 0.0, 1.0)

    # 5. Clause density
    vec[4] = np.clip(sentence.count('、') / 3.0, 0.0, 1.0)

    # 6. Exclamation / rhetorical intensity
    vec[5] = np.clip((sentence.count('！') + sentence.count('？') + sentence.count('!')) / 1.5, 0.0, 1.0)

    # 7-23. Deterministic Concept Hash Projection of Kanji Stems
    kanji_stems = re.findall(r'[\u4e00-\u9fff]+', sentence)
    for stem in kanji_stems:
        # Pseudo-orthogonal hashing across dimensions 6..dim-1
        h = 2166136261
        for ch in stem:
            h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
        for k in range(6, dim):
            bit = (h >> (k - 6)) & 1
            sign = 1.0 if bit == 1 else -0.5
            weight = len(stem) / 3.0
            vec[k] += sign * weight

    # L2 normalize the semantic state vector
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        vec = vec / norm
    return vec


def compute_semantic_geodesic_curvature(
    text_or_units: Union[str, List[str]],
    embeddings: Optional[np.ndarray] = None,
    alpha_kappa: float = 1.8,
    alpha_brownian: float = 1.2,
    is_coordinates: bool = False,
) -> Dict[str, Any]:
    r"""Computes semantic space trajectory geodesic curvature, torsion, and Brownian diffusion.

    Given sentence sequence s_1, ..., s_M with progress vectors \Delta \mathbf{e}_m:
    1. Discrete curvature:
       \kappa_{\text{sem}, m} = 1 - \cos(\Delta \mathbf{e}_{m+1}, \Delta \mathbf{e}_m)
    2. Brownian motion / catalog jump ratio:
       \eta_{\text{brownian}} = \frac{M \sum_{m=1}^M \|\Delta \mathbf{e}_m\|_2^2}{\|\sum_{m=1}^M \Delta \mathbf{e}_m\|_2^2 + \epsilon}
    3. Geodesic multiplier:
       \mathcal{M}_{\text{geodesic}}(y) = \exp(-\alpha_\kappa \bar{\kappa}_{\text{sem}} - \alpha_{\text{brownian}} \max(0, \eta_{\text{brownian}} - 1.5))

    Detects 'props cataloging' (Brownian jumping across uncoordinated nouns) where surface 24D
    looks high in sensory words, but semantic curvature and diffusion spike uncontrollably.
    """
    if isinstance(text_or_units, str):
        units = _split_into_trajectory_units(text_or_units)
    else:
        units = list(text_or_units)

    if embeddings is not None:
        raw_e = np.asarray(embeddings, dtype=float)
        M = raw_e.shape[0]
        if is_coordinates and raw_e.ndim == 2 and M >= 2:
            # If coordinates are explicitly given, steps are consecutive differences
            diffs = raw_e[1:] - raw_e[:-1]
        else:
            diffs = raw_e
    else:
        M = len(units)
        if M < 2:
            return {
                "curvatures": [],
                "mean_curvature": 0.0,
                "max_curvature": 0.0,
                "mean_torsion": 0.0,
                "brownian_ratio": 1.0,
                "m_geodesic": 1.0,
                "is_brownian_catalog": False,
                "units_count": M,
            }
        diffs = np.vstack([_embed_sentence_unit(u) for u in units])

    diff_norms = np.linalg.norm(diffs, axis=1)

    # Curvatures between consecutive steps
    curvatures: List[float] = []
    torsions: List[float] = []

    for i in range(len(diffs) - 1):
        v1 = diffs[i]
        v2 = diffs[i + 1]
        n1 = diff_norms[i]
        n2 = diff_norms[i + 1]
        if n1 > 1e-7 and n2 > 1e-7:
            cos_theta = float(np.dot(v1, v2) / (n1 * n2))
            cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
            kappa = 1.0 - cos_theta
        else:
            kappa = 0.0
        curvatures.append(kappa)

    # Discrete Torsion (deviation from 2D osculating plane)
    for i in range(len(diffs) - 2):
        v1 = diffs[i]
        v2 = diffs[i + 1]
        v3 = diffs[i + 2]
        # Normal to (v1, v2)
        n1 = v1 - (np.dot(v1, v2) / (np.dot(v2, v2) + 1e-7)) * v2
        n2 = v3 - (np.dot(v3, v2) / (np.dot(v2, v2) + 1e-7)) * v2
        len_n1 = np.linalg.norm(n1)
        len_n2 = np.linalg.norm(n2)
        if len_n1 > 1e-7 and len_n2 > 1e-7:
            cos_tau = float(np.dot(n1, n2) / (len_n1 * len_n2))
            cos_tau = float(np.clip(cos_tau, -1.0, 1.0))
            torsions.append(1.0 - cos_tau)
        else:
            torsions.append(0.0)

    mean_curv = float(np.mean(curvatures)) if curvatures else 0.0
    max_curv = float(np.max(curvatures)) if curvatures else 0.0
    mean_tor = float(np.mean(torsions)) if torsions else 0.0

    # Brownian Diffusion Ratio:
    # Ratio of sum of squared steps to net displacement squared
    sum_steps_sq = float(np.sum(diff_norms ** 2))
    net_step_vec = np.sum(diffs, axis=0)
    net_disp_sq = float(np.linalg.norm(net_step_vec) ** 2) + 1e-5
    brownian_ratio = float((len(diffs) * sum_steps_sq) / net_disp_sq)

    # Geodesic multiplier M_geodesic in [0, 1]
    brownian_excess = max(0.0, brownian_ratio - 1.5)
    exponent = -(alpha_kappa * mean_curv + alpha_brownian * brownian_excess)
    m_geodesic = float(np.clip(math.exp(exponent), 0.0, 1.0))

    # Flag if text is cataloging props without narrative trajectory
    is_brownian_catalog = bool(brownian_ratio > 2.0 or mean_curv > 0.50)

    return {
        "curvatures": curvatures,
        "mean_curvature": mean_curv,
        "max_curvature": max_curv,
        "mean_torsion": mean_tor,
        "brownian_ratio": brownian_ratio,
        "m_geodesic": m_geodesic,
        "is_brownian_catalog": is_brownian_catalog,
        "units_count": M,
    }


# ==============================================================================
# 3. 4-Fold Multiplicative GRPO Barrier Reward Engine
# ==============================================================================

def compute_punctuation_ratio_gate(text: str) -> float:
    """Computes punctuation ratio gate in normal literary / reasoned bounds [0.015, 0.080].

    Returns multiplier in [0.0, 1.0]. Sharp cliff to 0 if text spams or lacks punctuation.
    """
    if not text:
        return 0.0
    L = len(text)
    punct_count = sum(text.count(p) for p in ("、", "。", "！", "？", "!", "?", ",", "."))
    ratio = punct_count / float(L)

    # Normal range: [0.015, 0.080]
    if ratio < 0.010:
        # Run-on text without punctuation
        return 0.0
    if ratio > 0.120:
        # Runaway punctuation spam
        return 0.0

    if ratio < 0.015:
        # Smooth ramp up
        return float((ratio - 0.010) / 0.005)
    if ratio > 0.080:
        # Smooth ramp down
        return float(max(0.0, (0.120 - ratio) / 0.040))
    return 1.0


def compute_multiplicative_grpo_reward(
    text: str,
    x_0: Optional[Union[str, np.ndarray]] = None,
    x_T: Optional[Union[str, np.ndarray]] = None,
    domain: str = "novel",
    target: str = "CRITICAL",
    is_authentic_masterwork: bool = False,
    masterwork_texts: Optional[List[str]] = None,
    alpha_metric: float = 1.0,
) -> Dict[str, Any]:
    r"""Computes v4.0 4-factor non-hackable multiplicative GRPO barrier reward:

    R_GRPO(y | x_0, x_T) = \mathcal{M}_{\text{syntax}}(y)
                         * \mathcal{M}_{\text{geodesic}}(y)
                         * \mathcal{M}_{\text{bridge}}(y | x_0, x_T)
                         * \exp(-0.5 * \mathcal{D}_{\text{3-phase}}(\mathbf{z}(y)))

    If any structural gate collapses (syntax gate, Brownian jumping, bridge disconnection),
    the reward strictly drops to 0.
    """
    if not text or len(text.strip()) == 0:
        return {
            "reward": 0.0,
            "m_syntax": 0.0,
            "m_geodesic": 0.0,
            "m_bridge": 0.0,
            "m_cleanroom": 0.0,
            "d_3phase": 99.0,
            "exp_barrier": 0.0,
            "is_rejected": True,
        }

    # 1. Syntax Multiplier with Strict Punctuation Gate & Bracket Matching
    punct_gate = compute_punctuation_ratio_gate(text)
    if punct_gate <= 0.0:
        # Strict zero-tolerance gate
        m_syntax = 0.0
    else:
        # Bracket matching & ending integrity
        open_brackets = set(BRACKET_PAIRS.keys())
        close_brackets = {v: k for k, v in BRACKET_PAIRS.items()}
        stack: List[str] = []
        matched = 0
        unmatched = 0
        for ch in text:
            if ch in open_brackets:
                stack.append(ch)
            elif ch in close_brackets:
                expected = close_brackets[ch]
                if stack and stack[-1] == expected:
                    stack.pop()
                    matched += 1
                else:
                    unmatched += 1
        unmatched += len(stack)
        tot_b = (2 * matched) + unmatched
        b_score = 1.0 if tot_b == 0 else (2.0 * matched) / float(tot_b)

        # Repetition penalty
        rep_pen = 0.0
        if len(text) >= 16:
            for i in range(len(text) - 10):
                quad = text[i:i+5]
                if text.count(quad) >= 4:
                    rep_pen = 0.50
                    break

        m_syntax = float(np.clip(punct_gate * b_score * (1.0 - rep_pen), 0.0, 1.0))

    # 2. Semantic Geodesic Multiplier (Brownian Motion & Curvature Gate)
    geodesic_res = compute_semantic_geodesic_curvature(text)
    m_geodesic = geodesic_res["m_geodesic"]

    # 3. Bidirectional Bridge Multiplier M_bridge(y | x_0, x_T)
    m_bridge = 1.0
    if x_0 is not None or x_T is not None:
        units = _split_into_trajectory_units(text)
        if units:
            tip_fwd = _embed_sentence_unit(units[0])
            tip_bwd = _embed_sentence_unit(units[-1])

            cos_0 = 1.0
            if x_0 is not None:
                vec_0 = _embed_sentence_unit(x_0) if isinstance(x_0, str) else np.asarray(x_0, dtype=float)
                if np.linalg.norm(vec_0) > 1e-5:
                    cos_0 = float(np.dot(tip_fwd, vec_0) / (np.linalg.norm(tip_fwd) * np.linalg.norm(vec_0) + 1e-7))
                    cos_0 = float(np.clip(cos_0, 0.0, 1.0))

            cos_T = 1.0
            if x_T is not None:
                vec_T = _embed_sentence_unit(x_T) if isinstance(x_T, str) else np.asarray(x_T, dtype=float)
                if np.linalg.norm(vec_T) > 1e-5:
                    cos_T = float(np.dot(tip_bwd, vec_T) / (np.linalg.norm(tip_bwd) * np.linalg.norm(vec_T) + 1e-7))
                    cos_T = float(np.clip(cos_T, 0.0, 1.0))

            m_bridge = float(np.clip(0.5 * (cos_0 + cos_T), 0.0, 1.0))

    # 4. Cleanroom Multiplier
    m_cleanroom = compute_cleanroom_multiplier(
        text,
        is_authentic_masterwork=is_authentic_masterwork,
        masterwork_texts=masterwork_texts,
    )

    # 5. 3-Phase QDA Asymmetric Barrier Distance
    qda_res = compute_three_phase_qda_distance(text, domain=domain)
    d_3phase = qda_res["d_3phase"]
    exp_barrier = float(math.exp(-0.5 * (d_3phase / max(alpha_metric, 0.1))))

    # Multiplicative Product: any zero shuts down total reward
    raw_reward = m_syntax * m_geodesic * m_bridge * m_cleanroom * exp_barrier
    r_grpo = float(np.clip(raw_reward, 0.0, 1.0))

    is_rejected = bool(r_grpo < 0.005 or m_syntax < 0.1 or m_geodesic < 0.1)

    return {
        "reward": r_grpo,
        "m_syntax": m_syntax,
        "m_geodesic": m_geodesic,
        "m_bridge": m_bridge,
        "m_cleanroom": m_cleanroom,
        "d_3phase": d_3phase,
        "d_mahalanobis": qda_res["d_mahalanobis"],
        "penalty_freeze": qda_res["penalty_freeze"],
        "penalty_chaos": qda_res["penalty_chaos"],
        "sigma_norm": qda_res["sigma_norm"],
        "phase_regime": qda_res["phase_regime"],
        "brownian_ratio": geodesic_res["brownian_ratio"],
        "mean_curvature": geodesic_res["mean_curvature"],
        "is_brownian_catalog": geodesic_res["is_brownian_catalog"],
        "exp_barrier": exp_barrier,
        "is_rejected": is_rejected,
    }
