"""Reproduction of the v4 configuration inside the P-4a framework (design 5 gate: DETV4 headline check).
Re-implements v4's hurdle + A/B composite q3' on the 9-col detv4 layout, reusing only the shared P-4a primitives
(loo_normalize, tail_p, p_op_procedure with rule="max"). Not part of the v13 detector."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import detector_p4 as P

COLS = [2, 0, 6, 7, 1]
SGN5 = np.array([-1, -1, -1, +1, -1], float)
IQ1, IQ2, IA, IB, IV3 = range(5)
RATE_COL, LNARR_COL = 6, 8


@dataclass
class Cfg:
    name: str = "v4"
    hurdle: bool = True
    tau: float = 2.0
    q3mode: str = "v4"
    m0: float = 8.0
    rule: str = "max"


def derive(raw):
    return raw[:, COLS].astype(float)


def _by_work(widx):
    return [np.where(widx == w)[0] for w in np.unique(widx)]


def _normalize(Dx, Dr, widx, mu_g, sig_w, m0):
    Z = np.empty_like(Dx, dtype=float)
    for ii in _by_work(widx):
        Z[ii] = P.loo_normalize(Dx[ii], Dr[ii], mu_g, sig_w, m0, sgn=SGN5)
    return Z


def _rbar(pA, pB):
    return ((1.0 - pA) + (1.0 - pB)) / 2.0


def fit(raw_train, widx_train, cfg):
    D = derive(raw_train)
    mu_g = D.mean(0)
    num, den = np.zeros(5), 0.0
    for ii in _by_work(widx_train):
        if len(ii) >= 2:
            num += (len(ii) - 1) * D[ii].var(0, ddof=1)
            den += len(ii) - 1
    sig_w = np.sqrt(np.maximum(num / den, P.FLOOR ** 2))
    m = {"cfg": cfg, "mu_g": mu_g, "sig_w": sig_w, "mu_rate_g": float(raw_train[:, RATE_COL].mean() * 1000.0)}
    Z = _normalize(D, D, widx_train, mu_g, sig_w, cfg.m0)
    m["cal"] = [np.sort(Z[:, j]) for j in range(5)]
    m["comp_cal"] = np.sort(_rbar(P.tail_p(m["cal"][IA], Z[:, IA]), P.tail_p(m["cal"][IB], Z[:, IB])))
    return m


def score(m, raw_x, raw_ref, widx):
    c = m["cfg"]
    Z = _normalize(derive(raw_x), derive(raw_ref), widx, m["mu_g"], m["sig_w"], c.m0)
    u = np.column_stack([-np.log(P.tail_p(m["cal"][j], Z[:, j])) for j in range(5)])
    u_c = -np.log(P.tail_p(m["comp_cal"], _rbar(P.tail_p(m["cal"][IA], Z[:, IA]), P.tail_p(m["cal"][IB], Z[:, IB]))))
    rate_ref = raw_ref[:, RATE_COL] * 1000.0
    mu_loo = np.empty(len(raw_x))
    for ii in _by_work(widx):
        r = rate_ref[ii]
        mu_loo[ii] = np.full(len(r), m["mu_rate_g"]) if len(r) <= 1 else (r.sum() - r) / (len(r) - 1)
    e_count = mu_loo * raw_x[:, LNARR_COL] / 1000.0
    w = 1.0 - np.exp(-np.maximum(e_count, 0.0) / c.tau) if c.hurdle else np.ones(len(raw_x))
    u3 = w * u_c if c.q3mode == "v4" else u[:, IV3]
    return np.maximum(np.maximum(u[:, IQ1], u[:, IQ2]), u3)


def pop(scores, widx, cfg):
    return P.p_op_procedure(scores, widx, rule=cfg.rule)
