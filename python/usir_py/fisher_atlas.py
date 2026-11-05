"""Class-balanced Fisher discriminant metric over style windows (Phase 30).

W solves S_B w = lambda (S_W + reg I) w, where
  S_W = mean over classes of the per-class covariance (each author weighs the same, regardless of window count),
  S_B = mean over classes of (mu_c - mu_bar)(mu_c - mu_bar)^T, mu_bar = unweighted mean of class means.
Features are standardised first (class-balanced pooled std).  In the projected space the within-class covariance is
the identity (eigh normalisation W^T (S_W + reg I) W = I), so a work is scored per class by the summed squared
distance of its windows to the projected class mean, with equal class priors.

Pure numpy / scipy; no labels other than the caller's opaque class keys.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, Optional

import numpy as np
from scipy.linalg import eigh


@dataclass
class FisherMetric:
    center: np.ndarray        # (d,) standardisation centre
    scale: np.ndarray         # (d,) standardisation scale
    W: np.ndarray             # (d, k) projection
    eigvals: np.ndarray       # (k,)
    means: Dict[Hashable, np.ndarray]   # projected class means (k,)
    mu_bar: np.ndarray        # projected unweighted mean of class means (k,)

    def project(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return ((X - self.center) / self.scale) @ self.W

    def scores(self, X: np.ndarray) -> Dict[Hashable, float]:
        """Summed log-likelihood (up to a shared constant) of the windows X under each class."""
        Y = self.project(X)
        return {c: float(-0.5 * np.sum((Y - m) ** 2)) for c, m in self.means.items()}

    def predict(self, X: np.ndarray) -> Hashable:
        s = self.scores(X)
        return max(s, key=s.get)

    def direction(self, c: Hashable) -> np.ndarray:
        """Unit vector, in standardised feature space, from the class-mean centroid toward class c.

        It is W (m_c - m_bar) mapped back through the projection (W W^T acts as the discriminant metric)."""
        v = self.W @ (self.means[c] - self.mu_bar)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v


def fit_fisher(X_by_class: Dict[Hashable, np.ndarray], reg: float = 1e-2, n_components: Optional[int] = None,
               ) -> FisherMetric:
    classes = list(X_by_class)
    if len(classes) < 2:
        raise ValueError("need at least two classes")
    Xs = {c: np.asarray(X_by_class[c], dtype=float) for c in classes}
    d = next(iter(Xs.values())).shape[1]
    raw_means = np.stack([Xs[c].mean(axis=0) for c in classes])
    center = raw_means.mean(axis=0)
    var = np.mean([Xs[c].var(axis=0) for c in classes], axis=0)
    scale = np.sqrt(np.maximum(var, 1e-12))
    Zs = {c: (Xs[c] - center) / scale for c in classes}
    mus = np.stack([Zs[c].mean(axis=0) for c in classes])
    mbar = mus.mean(axis=0)
    S_W = np.mean([np.cov(Zs[c], rowvar=False, bias=True) if len(Zs[c]) > 1 else np.zeros((d, d))
                   for c in classes], axis=0)
    D = mus - mbar
    S_B = D.T @ D / len(classes)
    reg_abs = reg * float(np.trace(S_W)) / d
    vals, vecs = eigh(S_B, S_W + reg_abs * np.eye(d))
    order = np.argsort(vals)[::-1]
    k = n_components or min(len(classes) - 1, d)
    W = vecs[:, order[:k]]
    proj = {c: (Zs[c] @ W).mean(axis=0) for c in classes}
    return FisherMetric(center=center, scale=scale, W=W, eigvals=vals[order[:k]], means=proj,
                        mu_bar=np.mean(np.stack(list(proj.values())), axis=0))
