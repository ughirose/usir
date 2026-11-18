"""DETV4 detector v4 (definitions: docs/handoff/2026-11-18_usir_v5_detector_v4_design.md, pre-registered).

Raw per-scene matrix (9 cols, from signals.py): [q1c,q2s,q3e,LS,DR,S1b,S2a,S2b,L_narr]
(v3 naming: q1c=comma density, q2s=whole-text sensory rate, q3e=ending diversity).
v4 signals (design §1): q1=q3e (ending, down), q2=q1c (comma, down), A=S2a (narration sensory density, down),
B=S2b (longest run of non-sensory narration sentences, up), v3q2 = q2s (ablation ii only, down).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

ORDER = ["q1", "q2", "A", "B", "v3q2"]
COLS = [2, 0, 6, 7, 1]
SGN = np.array([-1, -1, -1, +1, -1], float)
IQ1, IQ2, IA, IB, IV3 = range(5)
RATE_COL, LNARR_COL = 6, 8
FLOOR = 1e-6
GRID = np.arange(80.0, 97.0 + 1e-9, 0.5)
Z_SE = 1.28
FPR_BOUND = 0.10


@dataclass
class Cfg:
    name: str = "v4"
    hurdle: bool = True
    tau: float = 2.0
    q3mode: str = "v4"  # "v4" (A,B composite * hurdle) or "v3q2" (ablation ii)
    m0: float = 8.0


def derive(raw):
    return raw[:, COLS].astype(float)


def hurdle_weight(e_count, tau):
    """w = 1 - exp(-E/tau). w(0)=0, monotone increasing, <1."""
    return 1.0 - np.exp(-np.maximum(e_count, 0.0) / tau)


def loo_normalize(x, ref, mu_g, sig_w, m0):
    """Within-work LOO EB z (sgn-signed) for ONE work; identical to detector_v3.loo_normalize with v4 SGN."""
    n = len(ref)
    k = n - 1
    if k <= 0:
        mu, var = np.tile(mu_g, (n, 1)), np.tile(sig_w ** 2, (n, 1))
    else:
        S, Sq = ref.sum(0), (ref ** 2).sum(0)
        m = (S - ref) / k
        s2 = np.maximum((Sq - ref ** 2 - k * m ** 2) / (k - 1), 0.0) if k >= 2 else np.zeros_like(m)
        mu = (k * m + m0 * mu_g) / (k + m0) if (k + m0) > 0 else np.tile(mu_g, (n, 1))
        w = max(k - 1, 0)
        den = w + m0
        var = (w * s2 + m0 * sig_w ** 2) / den if den > 0 else np.tile(sig_w ** 2, (n, 1))
    return (x - mu) / np.maximum(np.sqrt(var), FLOOR) * SGN


def loo_mean_rate(rate_ref, mu_rate_g):
    """mu_LOO,i = mean over other scenes of per-scene rate; fallback global mean if no other scene."""
    n = len(rate_ref)
    if n <= 1:
        return np.full(n, mu_rate_g)
    return (rate_ref.sum() - rate_ref) / (n - 1)


@dataclass
class Model:
    cfg: Cfg
    mu_g: np.ndarray
    sig_w: np.ndarray
    mu_rate_g: float
    cal: list
    comp_cal: np.ndarray = field(default_factory=lambda: np.zeros(0))


def _by_work(widx):
    return [np.where(widx == w)[0] for w in np.unique(widx)]


def _normalize(D_x, D_ref, widx, mu_g, sig_w, m0):
    Z = np.empty_like(D_x, dtype=float)
    for ii in _by_work(widx):
        Z[ii] = loo_normalize(D_x[ii], D_ref[ii], mu_g, sig_w, m0)
    return Z


def tail_p(F, z):
    cnt = len(F) - np.searchsorted(F, z, side="left")
    return (1.0 + cnt) / (len(F) + 1.0)


def composite_r(pA, pB):
    """Rank average of the two component calibration ranks r=1-p (design §4.2)."""
    return ((1.0 - pA) + (1.0 - pB)) / 2.0


def fit(raw_train, widx_train, cfg: Cfg) -> Model:
    D = derive(raw_train)
    mu_g = D.mean(0)
    num, den = np.zeros(D.shape[1]), 0.0
    for ii in _by_work(widx_train):
        if len(ii) >= 2:
            num += (len(ii) - 1) * D[ii].var(0, ddof=1)
            den += len(ii) - 1
    sig_w = np.sqrt(np.maximum(num / den, FLOOR ** 2))
    m = Model(cfg, mu_g, sig_w, float(raw_train[:, RATE_COL].mean() * 1000.0), [])
    Z = _normalize(D, D, widx_train, mu_g, sig_w, cfg.m0)
    m.cal = [np.sort(Z[:, j]) for j in range(Z.shape[1])]
    rbar = composite_r(tail_p(m.cal[IA], Z[:, IA]), tail_p(m.cal[IB], Z[:, IB]))
    m.comp_cal = np.sort(rbar)
    return m


def score_parts(model, raw_x, raw_ref, widx):
    """Per-signal scores for scenes raw_x with same-work reference scenes raw_ref (row-aligned).
    Returns dict: u (n,5) per-signal -log p, u_c composite, w hurdle weight, u3 final q3', score (n,)."""
    c = model.cfg
    Dx, Dr = derive(raw_x), derive(raw_ref)
    Z = _normalize(Dx, Dr, widx, model.mu_g, model.sig_w, c.m0)
    u = np.column_stack([-np.log(tail_p(model.cal[j], Z[:, j])) for j in range(Z.shape[1])])
    rbar = composite_r(tail_p(model.cal[IA], Z[:, IA]), tail_p(model.cal[IB], Z[:, IB]))
    u_c = -np.log(tail_p(model.comp_cal, rbar))
    mu_loo = np.empty(len(raw_x))
    rate_ref = raw_ref[:, RATE_COL] * 1000.0
    for ii in _by_work(widx):
        mu_loo[ii] = loo_mean_rate(rate_ref[ii], model.mu_rate_g)
    e_count = mu_loo * raw_x[:, LNARR_COL] / 1000.0
    w = hurdle_weight(e_count, c.tau) if c.hurdle else np.ones(len(raw_x))
    if c.q3mode == "v4":
        u3 = w * u_c
    else:
        u3 = u[:, IV3]
    score = np.maximum(np.maximum(u[:, IQ1], u[:, IQ2]), u3)
    return {"u": u, "u_c": u_c, "w": w, "e_count": e_count, "u3": u3, "score": score}


def score(model, raw_x, raw_ref, widx):
    return score_parts(model, raw_x, raw_ref, widx)["score"]


def p_op_procedure(scores, widx, grid=GRID, z=Z_SE, bound=FPR_BOUND):
    """Design §5: leave-one-work-out threshold = p-th percentile of other works' unperturbed scores;
    FPR_w = share of w's scenes with score >= threshold. p_OP = largest p with mean_w FPR + z*SE <= bound; else grid[0]."""
    works = np.unique(widx)
    rows = []
    for p in grid:
        f = []
        for w in works:
            thr = np.percentile(scores[widx != w], p)
            f.append(float(np.mean(scores[widx == w] >= thr)))
        f = np.array(f)
        se = float(f.std(ddof=1) / np.sqrt(len(f))) if len(f) > 1 else 0.0
        rows.append({"p": float(p), "mean_fpr": float(f.mean()), "se": se, "bound_value": float(f.mean() + z * se),
                     "ok": bool(f.mean() + z * se <= bound)})
    ok = [r["p"] for r in rows if r["ok"]]
    return {"rows": rows, "p_op": float(max(ok)) if ok else float(grid[0]), "fallback_to_floor": not ok}
