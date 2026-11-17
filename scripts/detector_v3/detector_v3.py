"""DETV3 detector v3 (definitions: docs/handoff/2026-11-18_usir_v5_detector_v3_design.md, pre-registered).

Pure functions on numeric arrays. Raw per-scene input matrix has columns NAMES = [q1,q2,q3,LS,DR,S1b,S2a,S2b].
Derived signals (7): q1,q2,q3,S1a,S1b,S2a,S2b with direction SGN (-1 = decrease is anomalous).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

SIG = ["q1", "q2", "q3", "S1a", "S1b", "S2a", "S2b"]
SGN = np.array([-1, -1, -1, -1, +1, -1, +1], float)
SURF, S1, S2 = [0, 1, 2], [3, 4], [5, 6]
FLOOR = 1e-6


@dataclass
class Cfg:
    name: str = "full"
    sig: list = field(default_factory=lambda: list(range(7)))
    work_norm: bool = True
    m0: float = 8.0


def design(raw):
    return np.column_stack([np.ones(len(raw)), np.log(np.maximum(raw[:, 3], 1.0)), raw[:, 4]])


def derive(coef, raw):
    """raw (n,8) -> derived (n,7). S1a = q1 - (1, log LS, DR) @ coef."""
    s1a = raw[:, 0] - design(raw) @ coef
    return np.column_stack([raw[:, 0], raw[:, 1], raw[:, 2], s1a, raw[:, 5], raw[:, 6], raw[:, 7]])


def loo_normalize(x, ref, mu_g, sig_w, m0):
    """Within-work LOO empirical-Bayes z for scenes of ONE work. x, ref: (n_w,7); row i of ref is scene i's own reference value,
    excluded from its own baseline. Returns sgn-signed z (n_w,7). Uses only rows of `ref` other than i."""
    n = len(ref)
    k = n - 1
    out = np.empty_like(x, dtype=float)
    if k <= 0:
        mu, var = np.tile(mu_g, (n, 1)), np.tile(sig_w ** 2, (n, 1))
    else:
        S, Sq = ref.sum(0), (ref ** 2).sum(0)
        m = (S - ref) / k
        if k >= 2:
            s2 = np.maximum((Sq - ref ** 2 - k * m ** 2) / (k - 1), 0.0)
        else:
            s2 = np.zeros_like(m)
        mu = (k * m + m0 * mu_g) / (k + m0) if (k + m0) > 0 else np.tile(mu_g, (n, 1))
        w = max(k - 1, 0)
        den = w + m0
        var = (w * s2 + m0 * sig_w ** 2) / den if den > 0 else np.tile(sig_w ** 2, (n, 1))
    out = (x - mu) / np.maximum(np.sqrt(var), FLOOR)
    return out * SGN


@dataclass
class Model:
    cfg: Cfg
    coef: np.ndarray
    mu_g: np.ndarray
    sig_tot: np.ndarray
    sig_w: np.ndarray
    cal: list  # sorted calibration arrays per signal


def _by_work(widx):
    return [np.where(widx == w)[0] for w in np.unique(widx)]


def normalize(model, D_x, D_ref, widx):
    """Normalize derived signals D_x against reference D_ref per work (row-aligned; same work index array)."""
    c = model.cfg
    if not c.work_norm:
        return SGN * (D_x - model.mu_g) / np.maximum(model.sig_tot, FLOOR)
    Z = np.empty_like(D_x, dtype=float)
    for ii in _by_work(widx):
        Z[ii] = loo_normalize(D_x[ii], D_ref[ii], model.mu_g, model.sig_w, c.m0)
    return Z


def fit(raw_train, widx_train, cfg: Cfg) -> Model:
    """Everything (regression, global moments, calibration distributions) from training-fold unperturbed scenes only."""
    X = design(raw_train)
    coef, *_ = np.linalg.lstsq(X, raw_train[:, 0], rcond=None)
    D = derive(coef, raw_train)
    mu_g = D.mean(0)
    sig_tot = np.maximum(D.std(0, ddof=1), FLOOR)
    num, den = np.zeros(D.shape[1]), 0.0
    for ii in _by_work(widx_train):
        if len(ii) >= 2:
            num += (len(ii) - 1) * D[ii].var(0, ddof=1)
            den += len(ii) - 1
    sig_w = np.sqrt(np.maximum(num / den, FLOOR ** 2))
    m = Model(cfg, coef, mu_g, sig_tot, sig_w, [])
    Z = normalize(m, D, D, widx_train)
    m.cal = [np.sort(Z[:, j]) for j in range(Z.shape[1])]
    return m


def score_from_z(model, Z):
    """-log p per signal (empirical upper tail of calibration distribution), aggregated by max over cfg.sig."""
    u = np.empty_like(Z, dtype=float)
    for j in range(Z.shape[1]):
        F = model.cal[j]
        cnt = len(F) - np.searchsorted(F, Z[:, j], side="left")
        u[:, j] = -np.log((1.0 + cnt) / (len(F) + 1.0))
    return u[:, model.cfg.sig].max(1), u


def score(model, raw_x, raw_ref, widx):
    """Score scenes raw_x against same-work reference scenes raw_ref (row-aligned)."""
    Dx = derive(model.coef, raw_x)
    Dr = derive(model.coef, raw_ref)
    return score_from_z(model, normalize(model, Dx, Dr, widx))[0]
