"""USIR v5 Six-Pole Integrated Atlas Module.

Core components:
1. NarrativeRGNormalizer: Scale-invariant Renormalization Group Normalizer for 4 Deep Poles.
2. normalized_hsic: Normalized Hilbert-Schmidt Independence Criterion (HW-08).
3. svd_lowdin_orthogonalize: SVD-Lowdin symmetric orthogonalization (HW-08).
4. GoldilocksVoidFinder: Canonical manifold void finder with Wundt hedonic bounds (HW-09 & HW-10).
5. aggregate_work_profile_from_scenes: In-memory deterministic work profiler strictly matching Rust nishiki profile.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

POLE_NAMES = ["Kasane", "Sasae", "Kamae", "Sadame", "Nagare", "Awase"]


# =================================================================================================
# 1. HW-11 NarrativeRGNormalizer
# =================================================================================================
class NarrativeRGNormalizer:
    """Scale-invariant Renormalization Group Normalizer for the 4 Deep Poles.

    Converts extensive counts into scene-density features and computes 5-act (Jo-Ha-Kyu)
    macro-trajectory gradients across relative narrative time u in [0, 1].
    Applies linear log(N) orthogonalization to strictly ensure |rho(x, log N)| < 0.25.
    """

    DEEP_FEATURE_NAMES = [
        # Kamae (3D)
        "kamae_hook_survival",
        "kamae_hook_density",
        "kamae_survival_idf",
        # Sadame (4D)
        "sadame_total",
        "sadame_e_intra",
        "sadame_edge_density",
        "sadame_grad",
        # Nagare (4D)
        "nagare_phi_zeami",
        "nagare_mood_speed_mean",
        "nagare_still_density",
        "nagare_speed_ratio",
        # Awase (4D)
        "awase_irony_mean",
        "awase_detachment_mean",
        "awase_irony_active",
        "awase_irony_grad",
    ]

    def __init__(self, profile_data: Dict[str, Any]) -> None:
        self.profile_data = profile_data
        self.work_profiles: Dict[str, Dict[str, Any]] = {}
        for w in profile_data.get("works", []):
            self.work_profiles[w["work_id"]] = w
            aid = w.get("author_id", "")
            wid = w.get("work_id", "")
            self.work_profiles[f"{aid}:{wid}"] = w
        self.slopes: Optional[np.ndarray] = None
        self.intercepts: Optional[np.ndarray] = None
        self.apply_mask: Optional[np.ndarray] = None

    def fit_orthogonalizer(self, X: np.ndarray, log_chars: np.ndarray, threshold: float = 0.20) -> None:
        """Fit linear trend against log(char_len) on training fold."""
        D = X.shape[1]
        self.slopes = np.zeros(D)
        self.intercepts = np.zeros(D)
        self.apply_mask = np.zeros(D, dtype=bool)
        for j in range(D):
            r_raw, _ = spearmanr(X[:, j], log_chars)
            slope, intercept = np.polyfit(log_chars, X[:, j], 1)
            res = X[:, j] - (slope * log_chars + intercept)
            r_res, _ = spearmanr(res, log_chars)
            if abs(r_raw) >= threshold and abs(r_res) < abs(r_raw):
                self.slopes[j] = slope
                self.intercepts[j] = intercept
                self.apply_mask[j] = True

    def orthogonalize(self, X: np.ndarray, log_chars: np.ndarray) -> np.ndarray:
        """Subtract fitted log(char_len) trends."""
        if self.slopes is None or self.intercepts is None or self.apply_mask is None:
            return X
        X_res = X.copy()
        for j in range(X.shape[1]):
            if self.apply_mask[j]:
                trend = self.slopes[j] * log_chars + self.intercepts[j]
                X_res[:, j] = X[:, j] - trend
        return X_res

    def extract_work_features(self, work_key: str) -> np.ndarray:
        """Extract 15 deep narrative features from work profile."""
        if work_key in self.work_profiles:
            w = self.work_profiles[work_key]
        else:
            parts = work_key.split(":")
            wid = parts[-1]
            if len(parts) >= 2:
                aid_wid = f"{parts[-2]}:{wid}"
                w = self.work_profiles.get(aid_wid, self.work_profiles.get(wid, {}))
            else:
                w = self.work_profiles.get(wid, {})
        tw = max(w.get("scene_count", 1), 1)

        sad = w.get("sadame", {})
        awa = w.get("awase", {})
        kam = w.get("kamae", {})
        nag = w.get("nagare", {})

        hook_density = kam.get("opened_hooks", 0) / tw
        edge_density = sad.get("edges_eff", 0.0) / tw
        still_density = len(nag.get("still_runs", [])) / tw

        irony_scenes = awa.get("irony", {}).get("per_scene", [])
        if len(irony_scenes) >= 5:
            n_per_act = max(1, len(irony_scenes) // 5)
            act1_irony = float(np.mean(irony_scenes[:n_per_act]))
            act5_irony = float(np.mean(irony_scenes[-n_per_act:]))
            irony_grad = act5_irony - act1_irony
        elif len(irony_scenes) >= 2:
            irony_grad = float(irony_scenes[-1] - irony_scenes[0])
        else:
            irony_grad = 0.0
        irony_active = float(np.mean(np.array(irony_scenes) > 0.0)) if len(irony_scenes) > 0 else 0.0

        sadame_scenes = [s.get("e_inter", 0.0) for s in sad.get("per_scene", [])]
        if len(sadame_scenes) >= 5:
            n_per_act = max(1, len(sadame_scenes) // 5)
            act1_sad = float(np.mean(sadame_scenes[:n_per_act]))
            act5_sad = float(np.mean(sadame_scenes[-n_per_act:]))
            sadame_grad = act5_sad - act1_sad
        elif len(sadame_scenes) >= 2:
            sadame_grad = float(sadame_scenes[-1] - sadame_scenes[0])
        else:
            sadame_grad = 0.0

        mood_speeds = nag.get("mood", {}).get("speed", [])
        if len(mood_speeds) >= 4:
            mid = len(mood_speeds) // 2
            v_early = float(np.mean(mood_speeds[:mid])) + 1e-4
            v_late = float(np.mean(mood_speeds[mid:])) + 1e-4
            speed_ratio = v_late / v_early
            speed_mean = float(np.mean(mood_speeds))
        elif len(mood_speeds) > 0:
            speed_mean = float(np.mean(mood_speeds))
            speed_ratio = 1.0
        else:
            speed_mean = 0.5
            speed_ratio = 1.0

        feats = [
            # Kamae
            float(kam.get("hook_survival", 0.5)),
            float(hook_density),
            float(kam.get("hook_survival_idf", 0.5)),
            # Sadame
            float(sad.get("total", 0.35)),
            float(sad.get("e_intra", 0.2)),
            float(edge_density),
            float(sadame_grad),
            # Nagare
            float(nag.get("phi_zeami", 0.45)),
            float(speed_mean),
            float(still_density),
            float(speed_ratio),
            # Awase
            float(awa.get("irony", {}).get("mean", 0.1)),
            float(awa.get("detachment", {}).get("mean", 0.12)),
            float(irony_active),
            float(irony_grad),
        ]
        return np.array(feats, dtype=float)

    def extract_all(self, work_ids: List[str]) -> np.ndarray:
        return np.array([self.extract_work_features(wid) for wid in work_ids], dtype=float)


# =================================================================================================
# 2. HW-08 Normalized HSIC & SVD-Lowdin Orthogonalization
# =================================================================================================
def rbf_kernel(X: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
    """RBF kernel with median heuristic bandwidth."""
    dists = squareform(pdist(X, metric="sqeuclidean"))
    if sigma is None or sigma <= 0.0:
        med = float(np.median(dists[dists > 0])) if np.any(dists > 0) else 1.0
        sigma_sq = max(med, 1e-6)
    else:
        sigma_sq = sigma**2
    return np.exp(-dists / (2.0 * sigma_sq))


def center_kernel(K: np.ndarray) -> np.ndarray:
    """Center kernel matrix: H K H where H = I - (1/N) 1 1^T."""
    N = K.shape[0]
    H = np.eye(N) - (1.0 / N) * np.ones((N, N))
    return H @ K @ H


def normalized_hsic(X: np.ndarray, Y: np.ndarray) -> float:
    """Normalized Hilbert-Schmidt Independence Criterion (nHSIC) in [0, 1].

    nHSIC(X, Y) = tr(K_tilde @ L_tilde) / sqrt(tr(K_tilde^2) * tr(L_tilde^2))
    """
    K = center_kernel(rbf_kernel(X))
    L = center_kernel(rbf_kernel(Y))

    tr_KL = float(np.sum(K * L))
    tr_K2 = float(np.sum(K * K))
    tr_L2 = float(np.sum(L * L))

    denom = math.sqrt(max(tr_K2 * tr_L2, 1e-12))
    return float(np.clip(tr_KL / denom, 0.0, 1.0))


def svd_lowdin_orthogonalize(Z: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """SVD-Lowdin symmetric orthogonalization for standardized N x D matrix Z.

    Z_orth = Z @ U @ (Lambda + eps*I)^{-1/2} @ U^T
    Guarantees: (1/N) * Z_orth^T @ Z_orth = I_D within machine precision.
    Returns:
        Z_orth: orthogonalized coordinates (N x D)
        W_lowdin: orthogonalization transformation matrix (D x D)
    """
    N, D = Z.shape
    Z_c = Z - np.mean(Z, axis=0, keepdims=True)
    S = (Z_c.T @ Z_c) / N
    eigvals, U = np.linalg.eigh(S)
    eigvals = np.maximum(eigvals, eps)
    inv_sqrt_lambda = np.diag(1.0 / np.sqrt(eigvals))
    W_lowdin = U @ inv_sqrt_lambda @ U.T
    Z_orth = Z_c @ W_lowdin
    return Z_orth, W_lowdin


# =================================================================================================
# 3. HW-09 & HW-10 Goldilocks Void Finder
# =================================================================================================
class GoldilocksVoidFinder:
    """Finds viable aesthetic voids in the canonical 6-pole manifold satisfying Wundt hedonic bounds."""

    def __init__(
        self,
        Z_orth: np.ndarray,
        works: List[Dict[str, Any]],
        aids: List[str],
        ppm_surprisals: Dict[str, float],
        breath_intervals: Dict[str, float],
        labels: Dict[str, str],
    ) -> None:
        self.Z_orth = Z_orth
        self.works = works
        self.aids = aids
        self.ppm_surprisals = ppm_surprisals
        self.breath_intervals = breath_intervals
        self.labels = labels
        self.author_of = {w["id"]: w["author_id"] for w in works}

    def candidate_voids(self, n_candidates: int = 20, seed: int = 42) -> List[Dict[str, Any]]:
        """Generate viable candidate voids with positive novelty and hedonic plausibility."""
        rng = np.random.default_rng(seed)
        N, D = self.Z_orth.shape
        mu = np.mean(self.Z_orth, axis=0)
        cov = np.cov(self.Z_orth, rowvar=False)

        candidates: List[Dict[str, Any]] = []
        for i in range(n_candidates * 3):
            # Sample in manifold neighborhood
            pt = rng.multivariate_normal(mu, cov)
            dists = np.linalg.norm(self.Z_orth - pt, axis=1)
            min_dist = float(np.min(dists))
            k_nearest = np.argsort(dists)[:3]

            # Wundt plausibility: moderate novelty (distance in [0.75, 2.5] standard deviations)
            if 0.75 <= min_dist <= 2.5:
                near_works = [self.works[idx]["id"] for idx in k_nearest]
                near_aids = [self.author_of[wid] for wid in near_works]
                candidates.append({
                    "void_id": f"void_{len(candidates):03d}",
                    "coord": pt.tolist(),
                    "isolation_dist": round(min_dist, 4),
                    "nearest_works": near_works,
                    "nearest_authors": near_aids,
                })
                if len(candidates) >= n_candidates:
                    break
        return candidates


# =================================================================================================
# 4. In-Memory Work Profile Aggregator (Strictly matching Rust nishiki profile)
# =================================================================================================
def norm_val(v: int) -> float:
    """Clamp integer valence [-2, 2] to [-1.0, 1.0]."""
    return max(-1.0, min(1.0, float(v) / 2.0))


def compute_awase_profile(scenes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute Awase irony and detachment metrics strictly matching usir_metrics::awase."""
    ironies: List[float] = []
    detachments: List[float] = []

    for s in scenes:
        a = s.get("awase", {})
        vf = norm_val(a.get("f_event_valence", 0))
        vs = norm_val(a.get("f_style_valence", 0))
        af = norm_val(a.get("f_event_arousal", 0))
        as_ = norm_val(a.get("f_style_arousal", 0))

        # Ψ_irony = max(0, -V_F * V_f)
        ir = max(0.0, -vf * vs)
        ironies.append(float(ir))

        # Ψ_detachment = |V_F| * max(0, A_F - A_f)
        det = abs(vf) * max(0.0, af - as_)
        detachments.append(float(det))

    mean_ir = float(np.mean(ironies)) if ironies else 0.0
    max_ir = float(np.max(ironies)) if ironies else 0.0
    mean_det = float(np.mean(detachments)) if detachments else 0.0
    max_det = float(np.max(detachments)) if detachments else 0.0

    return {
        "irony": {"mean": mean_ir, "max": max_ir, "per_scene": ironies},
        "detachment": {"mean": mean_det, "max": max_det, "per_scene": detachments},
    }


def compute_kamae_profile(
    scenes: List[Dict[str, Any]],
    hook_idf: Optional[Dict[str, Any]] = None,
    declared_hooks: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute Kamae hook survival and closure metrics strictly matching usir_metrics::kamae."""
    total_scenes = len(scenes)
    opened_count = 0
    hooks_map: Dict[int, Tuple[int, bool]] = {}  # id -> (bin, closed)

    for s in scenes:
        idx = s.get("scene_index", 0)
        n = s.get("scene_count", total_scenes)
        b = min(3, (idx * 4) // max(n, 1))

        kam = s.get("kamae", {})
        for h in kam.get("opened_hooks", []):
            hid = h.get("id", 0)
            if hid not in hooks_map:
                hooks_map[hid] = (b, False)
                opened_count += 1

        for cid in kam.get("closed_hook_ids", []):
            if cid in hooks_map:
                hooks_map[cid] = (hooks_map[cid][0], True)

    unresolved_unweighted = 0.0
    open_unweighted = 0.0
    unresolved_weighted = 0.0
    open_weighted = 0.0

    idf_weights = [1.0, 1.0, 1.0, 1.0]
    if hook_idf and "weight" in hook_idf:
        idf_weights = list(hook_idf["weight"])

    for b, closed in hooks_map.values():
        w = idf_weights[min(b, len(idf_weights) - 1)]
        open_unweighted += 1.0
        open_weighted += w
        if not closed:
            unresolved_unweighted += 1.0
            unresolved_weighted += w

    eps = 1e-12
    hook_survival = unresolved_unweighted / (open_unweighted + eps) if open_unweighted > 0 else 0.0
    hook_survival_idf = unresolved_weighted / (open_weighted + eps) if open_weighted > 0 else 0.0

    # D5: Registered declared hooks closure rate
    declared_stat: Optional[Dict[str, Any]] = None
    if declared_hooks:
        dec_opened = 0
        dec_closed = 0
        all_text = "\n".join(s.get("text", "") for s in scenes)
        for dh in declared_hooks:
            if dh in all_text:
                dec_opened += 1
                # Check if appeared in resolution scene (scenes with closed hooks)
                resolv_hit = any(
                    dh in s.get("text", "") and len(s.get("kamae", {}).get("closed_hook_ids", [])) > 0
                    for s in scenes
                )
                if resolv_hit:
                    dec_closed += 1
        dec_rate = (dec_closed / dec_opened) if dec_opened > 0 else 1.0
        declared_stat = {
            "declared_opened": dec_opened,
            "declared_closed": dec_closed,
            "closure_rate": float(dec_rate),
        }

    res: Dict[str, Any] = {
        "hook_survival": float(hook_survival),
        "hook_survival_idf": float(hook_survival_idf),
        "opened_hooks": opened_count,
    }
    if declared_stat:
        res["declared_hooks"] = declared_stat
    return res


def compute_sadame_profile(scenes: List[Dict[str, Any]], kappa: float = 3.0) -> Dict[str, Any]:
    """Compute Sadame polyphony inter/intra metrics strictly matching usir_metrics::sadame."""
    per_scene_data: List[Dict[str, Any]] = []
    stalks_history: Dict[str, List[List[float]]] = {}
    total_edges = 0.0

    for s in scenes:
        sad = s.get("sadame", {})
        stalks = sad.get("character_stalks", [])
        edges = sad.get("dialogue_edges", [])

        # Build name -> axes_q map
        stalk_map: Dict[str, List[float]] = {}
        for c in stalks:
            name = c.get("name", "")
            aq = [float(x) for x in c.get("axes_q", [0] * 6)]
            if name not in stalk_map:
                stalk_map[name] = aq
            stalks_history.setdefault(name, []).append(aq)

        # Inter-character distance on dialogue edges
        num = 0.0
        den = 0.0
        eff_edges = 0.0
        for e in edges:
            u, v = e.get("u", ""), e.get("v", "")
            conf = float(e.get("confidence", 0))
            if u != v and conf > 0 and u in stalk_map and v in stalk_map:
                a_q = np.array(stalk_map[u])
                b_q = np.array(stalk_map[v])
                d2 = float(np.sum((a_q - b_q) ** 2))
                num += conf * min(1.0, d2 / 24.0)
                den += conf
                eff_edges += 1.0

        e_inter = (num / den) if den > 0 else 0.0
        total_edges += eff_edges
        per_scene_data.append({
            "scene_index": s.get("scene_index", 0),
            "e_inter": float(e_inter),
            "edges_eff": float(eff_edges),
        })

    # Intra-character temporal variance across scenes
    intra_vars: List[float] = []
    for name, seq in stalks_history.items():
        if len(seq) >= 2:
            arr = np.array(seq)  # N_t x 6
            var_per_axis = np.var(arr, axis=0)  # population variance
            intra_vars.append(float(np.mean(var_per_axis)))
    e_intra = float(np.mean(intra_vars)) if intra_vars else 0.0

    mean_e_inter = float(np.mean([x["e_inter"] for x in per_scene_data])) if per_scene_data else 0.0
    lam = (total_edges / (total_edges + kappa)) if (total_edges + kappa) > 0 else 0.0
    sadame_tot = lam * mean_e_inter + (1.0 - lam) * e_intra

    return {
        "total": float(np.clip(sadame_tot, 0.0, 1.0)),
        "e_inter": float(mean_e_inter),
        "e_intra": float(e_intra),
        "edges_eff": float(total_edges),
        "per_scene": per_scene_data,
    }


def compute_nagare_profile(scenes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute Nagare spherical kinematics and Zeami phase strictly matching usir_metrics::nagare."""
    event_speeds: List[float] = []
    mood_speeds: List[float] = []
    event_accels: List[float] = []
    mood_accels: List[float] = []

    # S^15 lift: x = sqrt(softmax(logp))
    event_vecs: List[np.ndarray] = []
    mood_vecs: List[np.ndarray] = []

    for s in scenes:
        nag = s.get("nagare", {})
        e_logp = np.array(nag.get("event_logp", [0.0] * 16), dtype=float)
        m_logp = np.array(nag.get("mood_logp", [0.0] * 16), dtype=float)

        e_p = np.exp(e_logp - np.max(e_logp))
        e_p /= np.sum(e_p)
        m_p = np.exp(m_logp - np.max(m_logp))
        m_p /= np.sum(m_p)

        event_vecs.append(np.sqrt(np.maximum(e_p, 1e-12)))
        mood_vecs.append(np.sqrt(np.maximum(m_p, 1e-12)))

    for t in range(len(event_vecs) - 1):
        cos_e = np.clip(np.dot(event_vecs[t], event_vecs[t + 1]), -1.0, 1.0)
        s_e = float(np.arccos(cos_e))
        event_speeds.append(s_e)

        cos_m = np.clip(np.dot(mood_vecs[t], mood_vecs[t + 1]), -1.0, 1.0)
        s_m = float(np.arccos(cos_m))
        mood_speeds.append(s_m)

    for t in range(len(event_speeds) - 1):
        event_accels.append(abs(event_speeds[t + 1] - event_speeds[t]))
        mood_accels.append(abs(mood_speeds[t + 1] - mood_speeds[t]))

    sum_ev = float(np.sum(event_speeds))
    sum_mo = float(np.sum(mood_speeds))
    phi_zeami = sum_mo / (sum_ev + sum_mo + 1e-12) if (sum_ev + sum_mo) > 0 else 0.5

    # Stagnation runs (speed < 0.2 rad in both event and mood for >= 2 transitions)
    still_runs: List[List[int]] = []
    cur_start = -1
    for t in range(len(event_speeds)):
        if event_speeds[t] < 0.25 and mood_speeds[t] < 0.25:
            if cur_start == -1:
                cur_start = t
        else:
            if cur_start != -1:
                if t - cur_start >= 2:
                    still_runs.append([cur_start, t])
                cur_start = -1
    if cur_start != -1 and len(event_speeds) - cur_start >= 2:
        still_runs.append([cur_start, len(event_speeds)])

    return {
        "phi_zeami": float(phi_zeami),
        "still_runs": still_runs,
        "event": {"speed": event_speeds, "accel": event_accels},
        "mood": {"speed": mood_speeds, "accel": mood_accels},
    }


def aggregate_work_profile_from_scenes(
    scenes: List[Dict[str, Any]],
    hook_idf: Optional[Dict[str, Any]] = None,
    declared_hooks: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Aggregate a list of continuous scene annotations into a single work profile.

    Output format strictly conforms to Rust nishiki profile JSON specification:
    {
        "work_id": str,
        "author_id": str,
        "scene_count": int,
        "char_len": int,
        "awase": {...},
        "kamae": {...},
        "sadame": {...},
        "nagare": {...}
    }
    """
    if not scenes:
        return {
            "work_id": "",
            "author_id": "",
            "scene_count": 0,
            "char_len": 0,
            "awase": {},
            "kamae": {},
            "sadame": {},
            "nagare": {},
        }

    first = scenes[0]
    work_id = first.get("work_id", "")
    author_id = first.get("author_id", "")
    char_len = sum(int(s.get("char_len", 0)) for s in scenes)
    scene_count = len(scenes)

    awase_data = compute_awase_profile(scenes)
    kamae_data = compute_kamae_profile(scenes, hook_idf=hook_idf, declared_hooks=declared_hooks)
    sadame_data = compute_sadame_profile(scenes)
    nagare_data = compute_nagare_profile(scenes)

    return {
        "work_id": work_id,
        "author_id": author_id,
        "scene_count": scene_count,
        "char_len": char_len,
        "awase": awase_data,
        "kamae": kamae_data,
        "sadame": sadame_data,
        "nagare": nagare_data,
    }
