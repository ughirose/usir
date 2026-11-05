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
from .usir_forge import load_tier_m_binary

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
