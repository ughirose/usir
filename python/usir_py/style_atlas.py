"""24-D style atlas primitives (Phase 20).

Windowing -> 24-D (q, p) extraction -> 5-phase diagnosis -> per-author Tier-M genome (existing
OSWEStreamingCompiler) -> GMM scoring and mixture-Wasserstein (MW2) distances between genomes.

Invariant: no author / character / work name is used by any function here.  Labels are carried only
by the caller (atlas.json, file names).  The feature extractor is the unchanged ``extract_usir_24d``.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog

from .extractor_24d import extract_usir_24d
from .grpo_reward import PHASE_CENTROIDS_24D, _FEATURE_STD
from .hsmm_longform_director import compute_cv_l
from .usir_forge import (
    OSWEStreamingCompiler,
    classify_mode_24d,
    load_tier_m_binary,
    unpack_cholesky_24d,
)

WINDOW_CHARS = 600
STRIDE_CHARS = 300
PHASES = ["I_Crystalline", "II_Critical", "III_Void", "IV_Turbulent", "V_Deadzone"]
_TERMINALS = "。！？!?"

# Banned target keywords of packages/usir-core/src/cleanroom/threeRoomGate.ts (kept identical).
BANNED_TARGET_KEYWORDS = [
    "メロス", "李徴", "島村", "下人", "芥川", "太宰", "中島", "梶井",
    "志賀", "川端", "芭蕉", "漱石", "羅生門", "山月記", "雪国", "檸檬",
]


# ---------------------------------------------------------------------------
# Windowing and per-window features
# ---------------------------------------------------------------------------

def prepare_text(text: str) -> str:
    """Join lines into one string; a line that does not end with a sentence terminal gets '。'.

    extract_usir_24d splits sentences only on 。！？, so paragraph ends without a terminal
    (dialogue ending in 」, headings) would otherwise be glued to the next paragraph.
    """
    parts: List[str] = []
    for ln in text.split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        parts.append(ln if ln[-1] in _TERMINALS else ln + "。")
    return "".join(parts)


def make_windows(text: str, size: int = WINDOW_CHARS, stride: int = STRIDE_CHARS) -> List[str]:
    """Fixed windows over the prepared text; a text shorter than ``size`` gives one window."""
    t = prepare_text(text)
    if len(t) < 20:
        return []
    if len(t) <= size:
        return [t]
    out = [t[i:i + size] for i in range(0, len(t) - size + 1, stride)]
    return out


def classify_phase(z: np.ndarray) -> str:
    """Nearest golden-corpus phase centroid under the feature-std scaled distance (as grpo_reward)."""
    best, best_d = PHASES[0], float("inf")
    for ph in PHASES:
        mu = PHASE_CENTROIDS_24D.get(ph)
        if mu is None:
            continue
        d = float(np.sum(((z - mu) / _FEATURE_STD) ** 2))
        if d < best_d:
            best, best_d = ph, d
    return best


def window_features(window: str) -> Dict[str, Any]:
    ext = extract_usir_24d(window)
    z = np.asarray(ext["z_24d"], dtype=float)
    g = ext["groups"]
    return {
        "z": z,
        "flux": float(g["I_Flux"]),
        "shock": float(g["IV_Shock"]),
        "cv_l": float(compute_cv_l(window)),
        "phase": classify_phase(z),
    }


def featurize_text(text: str) -> Dict[str, Any]:
    feats = [window_features(w) for w in make_windows(text)]
    return {
        "Z": np.array([f["z"] for f in feats]).reshape(-1, 24),
        "flux": np.array([f["flux"] for f in feats]),
        "shock": np.array([f["shock"] for f in feats]),
        "cv_l": np.array([f["cv_l"] for f in feats]),
        "phase": [f["phase"] for f in feats],
    }


def profile_from_features(F: Dict[str, Any]) -> Dict[str, Any]:
    Z = F["Z"]
    n = int(len(Z))
    if n == 0:
        raise ValueError("no windows")
    phases = F["phase"]
    return {
        "n_windows": n,
        "mean_q": [float(v) for v in Z[:, :12].mean(axis=0)],
        "mean_p": [float(v) for v in Z[:, 12:].mean(axis=0)],
        "p_norm_mean": float(np.linalg.norm(Z[:, 12:], axis=1).mean()),
        "phase_occupancy": {ph: float(sum(1 for x in phases if x == ph)) / n for ph in PHASES},
        "cv_l_mean": float(np.mean(F["cv_l"])),
        "cv_l_std": float(np.std(F["cv_l"])),
        "shock_mean": float(np.mean(F["shock"])),
        "flux_mean": float(np.mean(F["flux"])),
    }


def concat_features(parts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "Z": np.concatenate([p["Z"] for p in parts], axis=0),
        "flux": np.concatenate([p["flux"] for p in parts]),
        "shock": np.concatenate([p["shock"] for p in parts]),
        "cv_l": np.concatenate([p["cv_l"] for p in parts]),
        "phase": [x for p in parts for x in p["phase"]],
    }


# ---------------------------------------------------------------------------
# Genome distillation (existing forge) and GMM view of a Tier-M binary
# ---------------------------------------------------------------------------

def distill_genome(works: Sequence[Tuple[str, np.ndarray]], shrinkage: str = "scaled") -> bytes:
    """Distill one Tier-M binary from per-work window matrices [(work_id, Z[n,24]), ...].

    The K>=5 cluster anonymity gate is disabled: an atlas genome is deliberately attributed to a
    single (public-domain) author.  Windows are added in text order; work boundaries reset the
    transition tracker.  Mode assignment is the forge's own classify_mode_24d.
    """
    comp = OSWEStreamingCompiler()
    for work_id, Z in works:
        for z in Z:
            comp.add_sample(z, mode=None, cluster_id=work_id)
    return comp.compile_tier_m_binary(enforce_anonymity=False, shrinkage=shrinkage)


class Gmm:
    """3-mode Gaussian mixture decoded from a Tier-M binary (float32 precision of the file)."""

    def __init__(self, weights: np.ndarray, means: np.ndarray, covs: np.ndarray, chols: np.ndarray):
        self.weights, self.means, self.covs, self.chols = weights, means, covs, chols
        self.active = [k for k in range(len(weights)) if weights[k] > 1e-9]
        self._logdet_prec = np.array([2.0 * np.sum(np.log(np.abs(np.diag(chols[k])))) for k in range(len(weights))])

    @classmethod
    def from_binary(cls, data: bytes) -> "Gmm":
        d = load_tier_m_binary(data)
        w = np.array(d["weights"], dtype=float)
        mu = np.array(d["centroids"], dtype=float)
        Ls = np.array([unpack_cholesky_24d(c) for c in d["cholesky_factors"]])
        covs = np.array([np.linalg.inv(L @ L.T) for L in Ls])
        covs = 0.5 * (covs + np.transpose(covs, (0, 2, 1)))
        return cls(w, mu, covs, Ls)

    def loglik(self, Z: np.ndarray) -> np.ndarray:
        """Per-row log p(z) = logsumexp_k [log w_k + log N(z; mu_k, Sigma_k)]."""
        Z = np.atleast_2d(Z)
        comps = []
        for k in self.active:
            y = (Z - self.means[k]) @ self.chols[k]  # (z-mu)^T L ; Mahalanobis^2 = |y|^2
            maha = np.sum(y * y, axis=1)
            comps.append(math.log(self.weights[k]) + 0.5 * self._logdet_prec[k] - 0.5 * maha - 12.0 * math.log(2 * math.pi))
        M = np.stack(comps, axis=0)
        mx = M.max(axis=0)
        return mx + np.log(np.exp(M - mx).sum(axis=0))


def score_authors(Z: np.ndarray, gmms: Dict[str, Gmm]) -> Dict[str, Dict[str, float]]:
    return {a: {"sum_loglik": float(g.loglik(Z).sum()), "mean_loglik": float(g.loglik(Z).mean())} for a, g in gmms.items()}


# ---------------------------------------------------------------------------
# Distances: Gaussian W2, mixture W2 (Delon-Desolneux), moment-matched single Gaussian
# ---------------------------------------------------------------------------

def _psd_sqrt(S: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(0.5 * (S + S.T))
    return (V * np.sqrt(np.clip(w, 0.0, None))) @ V.T


def gaussian_w2_sq(m1: np.ndarray, S1: np.ndarray, m2: np.ndarray, S2: np.ndarray) -> float:
    r2 = _psd_sqrt(S2)
    inner = r2 @ S1 @ r2
    ev = np.clip(np.linalg.eigvalsh(0.5 * (inner + inner.T)), 0.0, None)
    val = float(np.sum((m1 - m2) ** 2) + np.trace(S1) + np.trace(S2) - 2.0 * np.sum(np.sqrt(ev)))
    return max(val, 0.0)


def mixture_w2_sq(a: Gmm, b: Gmm) -> Tuple[float, np.ndarray]:
    """MW2^2 = min_pi sum_ij pi_ij W2^2(N_i^a, N_j^b), pi with marginals (w^a, w^b); solved as an LP."""
    ia, ib = a.active, b.active
    C = np.array([[gaussian_w2_sq(a.means[i], a.covs[i], b.means[j], b.covs[j]) for j in ib] for i in ia])
    wa = np.array([a.weights[i] for i in ia]); wa = wa / wa.sum()
    wb = np.array([b.weights[j] for j in ib]); wb = wb / wb.sum()
    na, nb = len(ia), len(ib)
    A_eq, b_eq = [], []
    for i in range(na):
        row = np.zeros(na * nb); row[i * nb:(i + 1) * nb] = 1.0
        A_eq.append(row); b_eq.append(wa[i])
    for j in range(nb):
        row = np.zeros(na * nb); row[j::nb] = 1.0
        A_eq.append(row); b_eq.append(wb[j])
    res = linprog(C.ravel(), A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError(f"MW2 LP failed: {res.message}")
    return float(res.fun), res.x.reshape(na, nb)


def moment_matched(g: Gmm) -> Tuple[np.ndarray, np.ndarray]:
    w = np.array([g.weights[k] for k in g.active]); w = w / w.sum()
    mu = sum(w[i] * g.means[k] for i, k in enumerate(g.active))
    S = sum(w[i] * (g.covs[k] + np.outer(g.means[k] - mu, g.means[k] - mu)) for i, k in enumerate(g.active))
    return mu, S


def single_gaussian_w2_sq(a: Gmm, b: Gmm) -> float:
    ma, Sa = moment_matched(a)
    mb, Sb = moment_matched(b)
    return gaussian_w2_sq(ma, Sa, mb, Sb)


def distance_matrices(gmms: Dict[str, Gmm]) -> Dict[str, Any]:
    ids = list(gmms)
    n = len(ids)
    mw2 = np.zeros((n, n)); w2 = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v, _ = mixture_w2_sq(gmms[ids[i]], gmms[ids[j]])
            mw2[i, j] = mw2[j, i] = math.sqrt(max(v, 0.0))
            w2[i, j] = w2[j, i] = math.sqrt(max(single_gaussian_w2_sq(gmms[ids[i]], gmms[ids[j]]), 0.0))
    return {"ids": ids, "mw2": mw2.tolist(), "w2_single_gaussian": w2.tolist()}


# ---------------------------------------------------------------------------
# Clean-room gate on a binary (port of auditCleanRoomBinary, stricter decoders added)
# ---------------------------------------------------------------------------

def audit_cleanroom_binary(binary: bytes, source_text: str, n: int = 7) -> Dict[str, Any]:
    """True iff no >= n-char substring of the binary (decoded as UTF-8 / UTF-16LE at both byte
    offsets / CP932) occurs in ``source_text`` and no banned target keyword is present."""
    decoded = [
        binary.decode("utf-8", errors="replace"),
        binary.decode("utf-16le", errors="replace"),
        binary[1:].decode("utf-16le", errors="replace"),
        binary.decode("cp932", errors="replace"),
    ]
    flat = re.sub(r"\s+", "", source_text)
    leaked, banned = [], []
    for s in decoded:
        for i in range(len(s) - n + 1):
            g = s[i:i + n]
            if "�" in g:
                continue
            if g in flat and g not in leaked:
                leaked.append(g)
    for w in BANNED_TARGET_KEYWORDS:
        if any(w in s for s in decoded):
            banned.append(w)
    load_tier_m_binary(binary)  # raises on bad magic / CRC / length
    return {"passed": not leaked and not banned, "leaked_7grams": leaked, "banned_words": banned, "byte_length": len(binary)}


# ---------------------------------------------------------------------------
# Atlas build / load (labels are caller-supplied; nothing here knows author names)
# ---------------------------------------------------------------------------

def build_atlas(
    author_works: Dict[str, List[Tuple[str, str]]],
    out_dir,
    shrinkages: Sequence[str] = ("scaled", "legacy"),
    features: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Distill genomes for {author_id: [(work_id, text), ...]} into ``out_dir`` (scaled) and
    ``out_dir/legacy``; returns per-author profile + audit info.  ``features`` may supply cached
    {author_id: {work_id: featurize_text() result}}."""
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Any] = {}
    for aid, works in author_works.items():
        feats = {}
        for wid, text in works:
            feats[wid] = features[aid][wid] if features and aid in features and wid in features[aid] else featurize_text(text)
        per_work = [(wid, feats[wid]["Z"]) for wid, _ in works if len(feats[wid]["Z"])]
        allF = concat_features([feats[wid] for wid, _ in works if len(feats[wid]["Z"])])
        source = "".join(t for _, t in works)
        entry: Dict[str, Any] = {"profile": profile_from_features(allF), "audit": {}}
        for sh in shrinkages:
            blob = distill_genome(per_work, shrinkage=sh)
            d = out_dir if sh == "scaled" else out_dir / sh
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{aid}.usir").write_bytes(blob)
            entry["audit"][sh] = audit_cleanroom_binary(blob, source)
        result[aid] = entry
    return result


def load_atlas_gmms(atlas_dir) -> Dict[str, Gmm]:
    from pathlib import Path

    gm = {}
    for p in sorted(Path(atlas_dir).glob("*.usir")):
        gm[p.stem] = Gmm.from_binary(p.read_bytes())
    return gm
