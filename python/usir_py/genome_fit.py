"""Genome fit metrics (Phase 21): how well a text fits a Tier-M style genome.

Reuses the Phase 20 style-atlas primitives unchanged: 600/300-char windows (``make_windows``), the
24-D extractor (``window_features``) and the 3-mode GMM log-likelihood (``Gmm.loglik``).
No author / work names are used here; genomes are anonymous 4096-byte binaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from .style_atlas import Gmm, make_windows, prepare_text, window_features
from .usir_forge import load_tier_m_binary, precision_matrix_from_packed_cholesky_24d

GenomeLike = Union[bytes, bytearray, str, Path]


def to_genome_bytes(genome: GenomeLike) -> bytes:
    """Accepts raw bytes or a path (str / Path) to a .usir Tier-M binary."""
    if isinstance(genome, (bytes, bytearray)):
        return bytes(genome)
    return Path(genome).read_bytes()


def genome_centroid(genome: GenomeLike) -> np.ndarray:
    """Weight-averaged 24-D centroid over the valid modes (weight > 1e-9) of a Tier-M genome."""
    d = load_tier_m_binary(to_genome_bytes(genome))
    w = np.asarray(d["weights"], dtype=float)
    c = np.asarray(d["centroids"], dtype=float)
    valid = w > 1e-9
    if not valid.any():
        raise ValueError("genome has no valid mode (all weights ~ 0)")
    return (w[valid, None] * c[valid]).sum(axis=0) / w[valid].sum()


def genome_loglik(text: str, genome_bytes: GenomeLike) -> Dict[str, Any]:
    """Mean per-window GMM log-likelihood of ``text`` under a genome.

    Windows follow the atlas (600 chars, stride 300). A text shorter than one window (or whose
    prepared form is under 20 chars, where ``make_windows`` returns nothing) is scored as one window.
    ``n_windows`` is 0 and ``mean_ll`` is NaN only for an empty text.
    """
    gmm = Gmm.from_binary(to_genome_bytes(genome_bytes))
    wins = make_windows(text)
    if not wins:
        t = prepare_text(text) or str(text).strip()
        wins = [t] if t else []
    if not wins:
        return {"mean_ll": float("nan"), "n_windows": 0}
    Z = np.array([window_features(w)["z"] for w in wins]).reshape(-1, 24)
    ll = gmm.loglik(Z)
    return {"mean_ll": float(np.mean(ll)), "n_windows": int(len(wins))}


def delta_ll(text: str, genome_a: GenomeLike, genome_b: GenomeLike) -> float:
    """mean_ll under genome A minus mean_ll under genome B (positive = text fits A better than B)."""
    return genome_loglik(text, genome_a)["mean_ll"] - genome_loglik(text, genome_b)["mean_ll"]


def genome_modes(genome: GenomeLike, dim: int = 12) -> list:
    """Per-mode steering data of a Tier-M genome (Phase 26).

    For every mode k with weight > 1e-9: {"mode": k, "weight": w_k, "centroid": mu_k (24-D),
    "G": trace-normalised position block of the precision matrix, (L_k L_k^T)[:dim, :dim] * dim / trace}.
    The trace normalisation keeps the overall scale of the identity metric it replaces (trace = dim), so only the
    shape of the author's covariance structure (which deviations are penalised more) changes the steering."""
    d = load_tier_m_binary(to_genome_bytes(genome))
    out = []
    for k, (w, c, packed) in enumerate(zip(d["weights"], d["centroids"], d["cholesky_factors"])):
        if float(w) <= 1e-9:
            continue
        lam = precision_matrix_from_packed_cholesky_24d(packed)
        G = np.asarray(lam[:dim, :dim], dtype=float)
        G = 0.5 * (G + G.T)
        tr = float(np.trace(G))
        if not np.isfinite(tr) or tr <= 0:
            continue
        out.append({"mode": k, "weight": float(w), "centroid": np.asarray(c, dtype=float), "G": G * dim / tr})
    return out
