"""P-4a v13 detector, Python reference implementation (design: docs/handoff/2026-11-18_usir_v5_p4_design.md 3.1-3.3, on stage/p4-design).

Raw per-scene matrix (signals.py): [q1 ending diversity, q2 comma density, q3 whole-scene sensory rate, L_total]. All three signals: lower = anomalous.
Pipeline: within-work LOO empirical-Bayes z (q1, q2) / sensory variance stabilization (q3) -> one-sided calibration tail_p -> u=-ln p -> score=max_k u.
Removed vs v4: hurdle weight, A/B composite, narration-only sensory rate, span length.
p_OP: minimum-p rule (design 3.2; Orion J6: no feasible p => failure, never a silent floor).
q3 modes: "poisson" (A: lower-tail Poisson p, lambda = mu_hat * L_total; mu_hat shrunk or not), "anscombe" (B: Anscombe sqrt transform vs expected),
"rate" (diagnostic only: plain EB z of the rate, v3-style).
Written dtype-agnostic where possible (arrays keep their dtype) so epsilon_py can be measured."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

ORDER = ["q1", "q2", "q3"]
SGN = np.array([-1.0, -1.0, -1.0])
FLOOR = 1e-6
LAM_FLOOR = 1e-6
GRID = np.arange(80.0, 97.0 + 1e-9, 0.5)
Z_SE = 1.28
FPR_BOUND = 0.10


@dataclass
class Cfg:
    name: str = "A_poisson_shrink"
    q3mode: str = "poisson"   # poisson | anscombe | rate
    shrink: bool = True       # mu_hat shrinkage for q3 (poisson / anscombe); q1, q2 always use m0 shrinkage
    m0: float = 8.0
    rule: str = "min"         # p_OP rule: "min" (v13) | "max" (v4 legacy, only for the reproduction check)


class PopFailure(Exception):
    """No grid p satisfies mean FPR + 1.28 SE <= 0.10 under the minimum-p rule (Orion J6)."""


def derive(raw):
    return raw[:, :3]


def loo_normalize(x, ref, mu_g, sig_w, m0, sgn=SGN):
    """Within-work LOO EB z for ONE work. 1-pass variance formula kept identical to detector_v3/v4 (design 3.2, do not switch to Welford)."""
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
    return (x - mu) / np.maximum(np.sqrt(var), FLOOR) * sgn


def loo_rate(rate_ref, mu_rate_g, m0, shrink):
    """mu_hat_i for each scene of ONE work from the other scenes' per-scene rates. k=0 -> global mean."""
    n = len(rate_ref)
    k = n - 1
    if k <= 0:
        return np.full(n, mu_rate_g, dtype=rate_ref.dtype)
    m = (rate_ref.sum() - rate_ref) / k
    return (k * m + m0 * mu_rate_g) / (k + m0) if shrink else m


def count_of(raw_x):
    return np.rint(raw_x[:, 2] * np.maximum(raw_x[:, 3], 1.0))


def log_factorial_table(cmax):
    return np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, cmax + 1, dtype=float)))])


def poisson_lower_logp(c, lam):
    """ln P(X <= c | lam) by direct log-space sum of exp(-lam + i ln lam - lnGamma(i+1)), i=0..c (same method as the planned Rust code)."""
    cmax = int(c.max()) if len(c) else 0
    lf = log_factorial_table(cmax)
    ll = np.log(lam)
    acc = np.full(len(c), -np.inf)
    for i in range(cmax + 1):
        t = -lam + i * ll - lf[i]
        acc = np.where(c >= i, np.logaddexp(acc, t), acc)
    return acc


def q3_z(cfg, raw_x, raw_ref, widx, mu_rate_g):
    """Anomaly-oriented q3 statistic (higher = more anomalous) for scenes raw_x against same-work reference scenes raw_ref (row-aligned)."""
    c = count_of(raw_x)
    L = np.maximum(raw_x[:, 3], 1.0)
    lam = np.empty(len(raw_x))
    for ii in _by_work(widx):
        lam[ii] = loo_rate(raw_ref[ii, 2], mu_rate_g, cfg.m0, cfg.shrink)
    lam = np.maximum(lam * L, LAM_FLOOR)
    if cfg.q3mode == "poisson":
        return -poisson_lower_logp(c, lam)
    return 2.0 * np.sqrt(lam + 0.375) - 2.0 * np.sqrt(c + 0.375)  # anscombe: unit-variance, lower count than expected -> positive


@dataclass
class Model:
    cfg: Cfg
    mu_g: np.ndarray
    sig_w: np.ndarray
    cal: list


def _by_work(widx):
    return [np.where(widx == w)[0] for w in np.unique(widx)]


def _z_all(model, raw_x, raw_ref, widx):
    c = model.cfg
    Dx, Dr = derive(raw_x), derive(raw_ref)
    Z = np.empty((len(raw_x), 3))
    for ii in _by_work(widx):
        Z[ii] = loo_normalize(Dx[ii], Dr[ii], model.mu_g, model.sig_w, c.m0)
    if c.q3mode != "rate":
        Z[:, 2] = q3_z(c, raw_x, raw_ref, widx, float(model.mu_g[2]))
    return Z


def tail_p(F, z):
    cnt = len(F) - np.searchsorted(F, z, side="left")
    return (1.0 + cnt) / (len(F) + 1.0)


def fit(raw_train, widx_train, cfg: Cfg) -> Model:
    D = derive(raw_train)
    mu_g = D.mean(0)
    num, den = np.zeros(3), 0.0
    for ii in _by_work(widx_train):
        if len(ii) >= 2:
            num += (len(ii) - 1) * D[ii].var(0, ddof=1)
            den += len(ii) - 1
    sig_w = np.sqrt(np.maximum(num / den, FLOOR ** 2))
    m = Model(cfg, mu_g, sig_w, [])
    Z = _z_all(m, raw_train, raw_train, widx_train)
    m.cal = [np.sort(Z[:, j]) for j in range(3)]
    return m


def score_parts(model, raw_x, raw_ref, widx):
    Z = _z_all(model, raw_x, raw_ref, widx)
    p = np.column_stack([tail_p(model.cal[j], Z[:, j]) for j in range(3)])
    u = -np.log(p)
    return {"Z": Z, "tail_p": p, "u": u, "score": u.max(1)}


def score(model, raw_x, raw_ref, widx):
    return score_parts(model, raw_x, raw_ref, widx)["score"]


def p_op_procedure(scores, widx, grid=GRID, z=Z_SE, bound=FPR_BOUND, rule="min"):
    """Leave-one-work-out threshold = p-th percentile (numpy linear) of other works' unperturbed scores; FPR_w = share of w's scenes with score >= threshold.
    rule "min": smallest p with mean_w FPR + z*SE <= bound; none -> failed (p_op None). rule "max": v4 legacy (largest p, else grid[0])."""
    works = np.unique(widx)
    rows = []
    for p in grid:
        f = np.array([float(np.mean(scores[widx == w] >= np.percentile(scores[widx != w], p))) for w in works])
        se = float(f.std(ddof=1) / np.sqrt(len(f))) if len(f) > 1 else 0.0
        rows.append({"p": float(p), "mean_fpr": float(f.mean()), "se": se, "bound_value": float(f.mean() + z * se),
                     "ok": bool(f.mean() + z * se <= bound)})
    ok = [r["p"] for r in rows if r["ok"]]
    if rule == "max":
        return {"rows": rows, "p_op": float(max(ok)) if ok else float(grid[0]), "failed": False, "fallback_to_floor": not ok}
    return {"rows": rows, "p_op": float(min(ok)) if ok else None, "failed": not ok, "fallback_to_floor": False}


def pop(scores, widx, cfg):
    """Adapter used by measure.py."""
    return p_op_procedure(scores, widx, rule=cfg.rule)
