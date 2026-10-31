"""GRPO Style Reward Engine using USIR 24D Symplectic State and Anti-Slop Penalty."""

import numpy as np
from .extractor_24d import extract_usir_24d
from .corpus_40_benchmark import CORPUS

# Precompute Phase Centroids from the Golden Corpus
_PROCESSED_CORPUS = [{**item, **extract_usir_24d(item["text"])} for item in CORPUS]

PHASE_CENTROIDS_24D = {}
for ph in ["I_Crystalline", "II_Critical", "III_Void", "IV_Turbulent", "V_Deadzone"]:
    sub = [r for r in _PROCESSED_CORPUS if r["phase"] == ph]
    if sub:
        PHASE_CENTROIDS_24D[ph] = np.mean([r["z_24d"] for r in sub], axis=0)

# Precompute empirical diagonal variances across all 40 passages for Mahalanobis scaling
_ALL_Z = np.array([r["z_24d"] for r in _PROCESSED_CORPUS])
_FEATURE_STD = np.std(_ALL_Z, axis=0) + 1e-4

USIR_CANONICAL_TARGETS = {
    "AMBER_CRITICAL": {
        "phase": "II_Critical",
        "centroid": PHASE_CENTROIDS_24D.get("II_Critical"),
        "description": "Phase II Self-Organized Criticality (1/f rhythm volatility, dynamic leap)",
    },
    "OBSIDIAN_VOID": {
        "phase": "III_Void",
        "centroid": PHASE_CENTROIDS_24D.get("III_Void"),
        "description": "Phase III Superfluid Void (Wago brevity, stillness operators, clean ending cut)",
    },
    "PORCELAIN_CRYSTAL": {
        "phase": "I_Crystalline",
        "centroid": PHASE_CENTROIDS_24D.get("I_Crystalline"),
        "description": "Phase I Crystalline Order (High formal structure, axiomatic symmetry)",
    },
}


def compute_grpo_reward(
    text: str,
    target: str = "OBSIDIAN_VOID",
    beta: float = 0.08,
    slop_penalty_weight: float = 1.20,
) -> dict:
    """Computes GRPO style reinforcement learning reward R_USIR(y).

    Args:
        text: Input generated text candidate y.
        target: Target particle or phase name ('OBSIDIAN_VOID', 'AMBER_CRITICAL', or a phase name).
        beta: Distance scaling factor.
        slop_penalty_weight: Weight of the narrative AI slop penalty.

    Returns:
        dict containing 'reward', 'distance', 'slop_deadness', 'phase_detected', and 'extracted'.
    """
    extracted = extract_usir_24d(text)
    z = extracted["z_24d"]

    if target in USIR_CANONICAL_TARGETS:
        target_mu = USIR_CANONICAL_TARGETS[target]["centroid"]
    elif target in PHASE_CENTROIDS_24D:
        target_mu = PHASE_CENTROIDS_24D[target]
    else:
        raise ValueError(f"Unknown target '{target}'. Valid targets: {list(USIR_CANONICAL_TARGETS.keys())} or {list(PHASE_CENTROIDS_24D.keys())}")

    # Normalized precision-weighted Euclidean distance
    diff_norm = (z - target_mu) / _FEATURE_STD
    dist_sq = float(np.sum(diff_norm**2))
    dist = float(np.sqrt(dist_sq))

    # Base similarity reward in [0, 1]
    r_sim = float(np.exp(-beta * dist))

    # Slop deadness penalty derived from cadence alive
    # cadence_alive is in [0.08, 1.0], 1.0 being alive and <= 0.32 being dead
    slop_deadness = float(1.0 - extracted["cadence_alive"])
    abst_penalty = float(extracted["abstract_b_mean"] * 4.0)
    total_penalty = float(slop_penalty_weight * (slop_deadness * 0.70 + abst_penalty * 0.30))

    total_reward = float(r_sim - total_penalty)

    return {
        "reward": total_reward,
        "similarity_score": r_sim,
        "distance": dist,
        "slop_penalty": total_penalty,
        "cadence_alive": extracted["cadence_alive"],
        "len_cv": extracted["len_cv"],
        "extracted": extracted,
    }
