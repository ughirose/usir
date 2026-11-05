"""OSWE Polar-Fourier Phase-Aligned usir-forge & 4KB 3-Mode GMM Binary Compiler.

Phase D WS-D1 Reference Implementation based on Blueprint Section 3:
1. align_polar_fourier_phase:
   Calculates climax peak phase theta_peak = arg(c_1) from fundamental Fourier
   coefficient c_1 = 1/M sum_{m=0}^{M-1} E(m) exp(-i 2pi m / M), and circularly
   aligns chunks to target phase theta_0 = 3pi/4.
   Prevents destructive interference / phase cancellation Deadzone.
2. OSWEStreamingCompiler:
   - Welford online update for 3 modes (Mode 1: 地の文, Mode 2: 会話, Mode 3: クライマックス):
     means mu_k in R^{24}, covariance Sigma_k in R^{24x24}, and lower-triangular
     Cholesky precision factors L_k in R^{300} (Sigma_k^{-1} = L_k L_k^T).
   - K >= 5 cluster anonymity gate (prevents single-author de-anonymization).
   - compile_tier_m_binary() -> bytes: strictly 4,096 bytes .usir binary.
   - load_tier_m_binary(data: bytes) -> dict: validates CRC32 and unpacks 4,096 bytes binary.
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import zlib

import numpy as np


USIR_MAGIC = 0x55534952  # ASCII "USIR" (little-endian: b'RISU')
TIER_M_BYTE_LENGTH = 4096


# ==============================================================================
# 1. POLAR-FOURIER PHASE ALIGNMENT
# ==============================================================================

def compute_polar_fourier_phase(energy: Union[Sequence[float], np.ndarray]) -> float:
    """Computes peak phase angle arg(c_1) of fundamental Fourier component c_1.

    c_1 = (1/M) sum_{m=0}^{M-1} E(m) exp(-i 2pi m / M)
    """
    arr = np.asarray(energy, dtype=np.float64)
    M = len(arr)
    if M <= 1 or not np.all(np.isfinite(arr)):
        return 0.0

    m = np.arange(M, dtype=np.float64)
    angles = -2.0 * np.pi * m / float(M)
    c1 = np.sum(arr * np.exp(1j * angles)) / float(M)

    if not np.isfinite(c1) or abs(c1) < 1e-12:
        return 0.0
    return float(np.angle(c1))


def align_polar_fourier_phase(
    energy_trajectory: Union[Sequence[float], Sequence[Sequence[float]], np.ndarray],
    target_phase: float = 3.0 * math.pi / 4.0,
    return_shift: bool = False,
) -> Union[
    List[float],
    List[List[float]],
    np.ndarray,
    Tuple[Union[List[float], List[List[float]], np.ndarray], int],
]:
    """Circularly aligns energy trajectory peak phase to target_phase (default 3pi/4).

    Prevents destructive interference / phase cancellation Deadzone when aggregating
    literary/narrative chunks.
    Supports both 1D energy sequences and 2D trajectories (M, D).
    """
    arr = np.asarray(energy_trajectory, dtype=np.float64)
    if arr.size == 0 or len(arr) <= 1 or not np.all(np.isfinite(arr)):
        res = list(energy_trajectory) if isinstance(energy_trajectory, list) else arr.copy()
        return (res, 0) if return_shift else res

    is_1d = (arr.ndim == 1)
    if is_1d:
        energy = arr
    else:
        # For 2D trajectory (M, D), extract 1D energy envelope
        M, D = arr.shape
        if D >= 24:
            # Hamiltonian energy: 0.5 * ||p||^2 + G_shock (Micro q3, Meso q7, Macro q11)
            p = arr[:, 12:24]
            q_shock = (arr[:, 3] + arr[:, 7] + arr[:, 11]) / 3.0
            energy = 0.5 * np.sum(p**2, axis=1) + q_shock
        else:
            energy = np.linalg.norm(arr, axis=1)

    M = len(energy)
    theta_peak = compute_polar_fourier_phase(energy)

    # We want arg(c_1') = theta_peak - 2*pi*s/M = target_phase (mod 2*pi)
    # => 2*pi*s/M = theta_peak - target_phase (mod 2*pi)
    diff = theta_peak - target_phase
    diff_wrapped = (diff + np.pi) % (2.0 * np.pi) - np.pi
    # Deterministic half-up rounding (eliminates banker's rounding divergence with JS Math.round)
    s = int(math.floor(float(M) * diff_wrapped / (2.0 * np.pi) + 0.5)) % M

    if s == 0:
        if isinstance(energy_trajectory, list):
            res = [list(r) if isinstance(r, (list, np.ndarray)) else r for r in energy_trajectory]  # type: ignore
        else:
            res = arr.copy()
        return (res, 0) if return_shift else res

    if is_1d:
        aligned = np.roll(arr, s)
        res = aligned.tolist() if isinstance(energy_trajectory, list) else aligned
    else:
        aligned = np.roll(arr, s, axis=0)
        res = aligned.tolist() if isinstance(energy_trajectory, list) else aligned

    return (res, s) if return_shift else res


# ==============================================================================
# 2. 24D CHOLESKY FACTOR SERIALIZATION (300 Lower-Triangular Elements)
SHRINKAGE_ALPHA: float = 0.75  # weight of the sample covariance in the Tier-M precision estimate


# ==============================================================================

def compute_packed_cholesky_24d(A: np.ndarray) -> List[float]:
    """Computes Cholesky decomposition A = L L^T of a 24x24 symmetric positive definite matrix.

    Returns 300 lower-triangular elements packed row-major (j <= i).
    Includes numerical floor to guarantee stability.
    """
    D = 24
    L = np.zeros((D, D), dtype=np.float64)
    for i in range(D):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s += L[i, k] * L[j, k]
            if i == j:
                val = A[i, i] - s
                L[i, j] = math.sqrt(max(val, 1e-4))
            else:
                diag = L[j, j]
                L[i, j] = (A[i, j] - s) / (diag if abs(diag) > 1e-12 else 1e-12)

    packed: List[float] = []
    for i in range(D):
        for j in range(i + 1):
            packed.append(float(L[i, j]))
    return packed


def unpack_cholesky_24d(packed: Sequence[float]) -> np.ndarray:
    """Unpacks 300 lower-triangular elements into a 24x24 lower-triangular matrix L."""
    D = 24
    L = np.zeros((D, D), dtype=np.float64)
    idx = 0
    for i in range(D):
        for j in range(i + 1):
            L[i, j] = packed[idx]
            idx += 1
    return L


def precision_matrix_from_packed_cholesky_24d(packed: Sequence[float]) -> np.ndarray:
    """Reconstructs 24x24 precision matrix Lambda = L L^T from packed 300 Cholesky elements."""
    L = unpack_cholesky_24d(packed)
    return L @ L.T


# ==============================================================================
# 3. 4,096 BYTES TIER M BINARY PACK / UNPACK
# ==============================================================================

def compile_tier_m_binary(profile: Dict[str, Any]) -> bytes:
    """Encodes a 3-Mode HSMM Style Profile into a strictly 4,096-byte .usir binary.

    Layout (100% compliant with usirBinarySchema.ts and Blueprint Section 24.1):
    - 0..3: Magic 0x55534952 (uint32 LE)
    - 4..5: Version (uint16 LE) = 1
    - 6: Tier (uint8) = 2
    - 7: Flags (uint8) = 0
    - 8..11: Reserved (uint32 LE) = 0
    - 12..47: 3x3 Transition Matrix (9 x float32 LE) = 36 bytes
    - 48..59: 3 Mode Weights (3 x float32 LE) = 12 bytes
    - 60..347: 3 Mode Centroids (3 x 24 = 72 x float32 LE) = 288 bytes
    - 348..3947: 3 Cholesky Factors L_k (3 x 300 = 900 x float32 LE) = 3600 bytes
    - 3948..4091: Auxiliary Alignment / Padding (144 bytes)
    - 4092..4095: CRC32 of bytes 0..4091 (uint32 LE) = 4 bytes
    Total: 4096 bytes.
    """
    buf = bytearray(TIER_M_BYTE_LENGTH)

    # Header
    struct.pack_into("<I", buf, 0, USIR_MAGIC)
    struct.pack_into("<H", buf, 4, int(profile.get("version", 1)))
    struct.pack_into("<B", buf, 6, 2)  # Tier M = 2
    struct.pack_into("<B", buf, 7, int(profile.get("flags", 0)))
    struct.pack_into("<I", buf, 8, 0)  # Reserved

    # Transition Matrix (3x3 = 9 floats, offset 12)
    t_mat = profile.get("transition_matrix", [[1.0 / 3.0] * 3 for _ in range(3)])
    for r in range(3):
        for c in range(3):
            val = float(t_mat[r][c]) if r < len(t_mat) and c < len(t_mat[r]) else (0.9 if r == c else 0.05)
            struct.pack_into("<f", buf, 12 + (r * 3 + c) * 4, val)

    # Mode Weights (3 floats, offset 48)
    weights = profile.get("weights", [1.0 / 3.0] * 3)
    for k in range(3):
        val = float(weights[k]) if k < len(weights) else 1.0 / 3.0
        struct.pack_into("<f", buf, 48 + k * 4, val)

    # Centroids (3 x 24 = 72 floats, offset 60)
    centroids = profile.get("centroids", [[0.5] * 24 for _ in range(3)])
    for k in range(3):
        c_k = centroids[k] if k < len(centroids) else [0.5] * 24
        for d in range(24):
            val = float(c_k[d]) if d < len(c_k) else 0.5
            struct.pack_into("<f", buf, 60 + (k * 24 + d) * 4, val)

    # Cholesky Factors (3 x 300 = 900 floats, offset 348)
    cholesky = profile.get("cholesky_factors", [[0.0] * 300 for _ in range(3)])
    for k in range(3):
        cf_k = cholesky[k] if k < len(cholesky) else [0.0] * 300
        for idx in range(300):
            val = float(cf_k[idx]) if idx < len(cf_k) else 0.0
            struct.pack_into("<f", buf, 348 + (k * 300 + idx) * 4, val)

    # Auxiliary Padding (3948..4091) remains zeros

    # CRC32 Checksum over 0..4091 at offset 4092
    crc = zlib.crc32(buf[:4092]) & 0xFFFFFFFF
    struct.pack_into("<I", buf, 4092, crc)

    return bytes(buf)


def load_tier_m_binary(data: bytes) -> Dict[str, Any]:
    """Validates and unpacks a strictly 4,096-byte .usir Tier M binary."""
    if len(data) != TIER_M_BYTE_LENGTH:
        raise ValueError(f"Invalid Tier M length: expected {TIER_M_BYTE_LENGTH} bytes, got {len(data)}")

    magic = struct.unpack_from("<I", data, 0)[0]
    if magic != USIR_MAGIC:
        raise ValueError(f"Invalid USIR magic in Tier M: 0x{magic:08x}")

    expected_crc = zlib.crc32(data[:4092]) & 0xFFFFFFFF
    stored_crc = struct.unpack_from("<I", data, 4092)[0]
    if expected_crc != stored_crc:
        raise ValueError(f"CRC32 mismatch in Tier M: computed 0x{expected_crc:08x}, stored 0x{stored_crc:08x}")

    version = struct.unpack_from("<H", data, 4)[0]
    tier = struct.unpack_from("<B", data, 6)[0]
    flags = struct.unpack_from("<B", data, 7)[0]

    transition_matrix = [
        [struct.unpack_from("<f", data, 12 + (r * 3 + c) * 4)[0] for c in range(3)]
        for r in range(3)
    ]
    weights = [struct.unpack_from("<f", data, 48 + k * 4)[0] for k in range(3)]
    centroids = [
        [struct.unpack_from("<f", data, 60 + (k * 24 + d) * 4)[0] for d in range(24)]
        for k in range(3)
    ]
    cholesky_factors = [
        [struct.unpack_from("<f", data, 348 + (k * 300 + idx) * 4)[0] for idx in range(300)]
        for k in range(3)
    ]

    return {
        "version": version,
        "tier": tier,
        "flags": flags,
        "transition_matrix": transition_matrix,
        "weights": weights,
        "centroids": centroids,
        "cholesky_factors": cholesky_factors,
        "crc32": stored_crc,
        "sha256": hashlib.sha256(data[:3948]).hexdigest(),
    }


# ==============================================================================
# 4. OSWE STREAMING COMPILER & K >= 5 ANONYMITY GATE
# ==============================================================================

def classify_mode_24d(z: Union[Sequence[float], np.ndarray]) -> int:
    """Classifies a 24D vector into one of 3 narrative modes:

    Mode 0 (地の文 / Narration / Setup): High Lattice & Void, lower kinetic momentum
    Mode 1 (会話 / Dialogue / Polyphony): High Flux & communicative momentum
    Mode 2 (クライマックス / Climax / Critical): High Shock & peak momentum
    """
    arr = np.asarray(z, dtype=np.float64)
    if len(arr) < 24:
        return 0

    q = arr[:12]
    p = arr[12:24]
    g_flux = float((q[0] + q[4] + q[8]) / 3.0)
    g_lattice = float((q[1] + q[5] + q[9]) / 3.0)
    g_void = float((q[2] + q[6] + q[10]) / 3.0)
    g_shock = float((q[3] + q[7] + q[11]) / 3.0)
    p_norm = float(np.linalg.norm(p))

    score_narration = g_lattice * 1.3 + g_void * 1.2
    score_dialogue = g_flux * 1.5 + p_norm * 0.4
    score_climax = g_shock * 1.8 + p_norm * 1.5

    if score_climax > score_narration and score_climax > score_dialogue:
        return 2  # Climax
    elif score_dialogue > score_narration:
        return 1  # Dialogue
    else:
        return 0  # Narration


class OSWEStreamingCompiler:
    """OSWE Streaming Style Genome Compiler with Polar-Fourier Alignment & K>=5 Anonymity Gate.

    Features:
    - Welford online update for 3 modes (Narration, Dialogue, Climax).
    - Polar-Fourier phase alignment to target climax phase (default 3pi/4).
    - Differential privacy / cluster anonymity gate requiring >= 5 distinct clusters/authors.
    - Strict 4,096-byte Tier M binary compiler and loader.
    """

    def __init__(
        self,
        min_clusters: int = 5,
        target_phase: float = 3.0 * math.pi / 4.0,
    ) -> None:
        self.min_clusters = int(min_clusters)
        self.target_phase = float(target_phase)
        self.clusters: Set[str] = set()

        # 3 modes: 0: 地の文, 1: 会話, 2: クライマックス
        self.counts: List[int] = [0, 0, 0]
        self.means: List[np.ndarray] = [np.zeros(24, dtype=np.float64) for _ in range(3)]
        self.M2: List[np.ndarray] = [np.zeros((24, 24), dtype=np.float64) for _ in range(3)]

        self.transition_counts: np.ndarray = np.zeros((3, 3), dtype=np.float64)
        self.prev_mode: Optional[int] = None
        self.last_cluster: Optional[str] = None
        self.sample_count: int = 0

    @property
    def cluster_count(self) -> int:
        return len(self.clusters)

    def reset_boundary(self) -> None:
        """Resets the previous mode tracker across independent chunks or authors."""
        self.prev_mode = None

    def register_cluster(self, cluster_id: Union[str, int]) -> None:
        """Explicitly registers a cluster or author identity."""
        cid = str(cluster_id).strip()
        if not cid:
            raise ValueError("cluster_id cannot be empty or whitespace")
        self.clusters.add(cid)

    def _normalize_mode(self, mode: Union[int, str]) -> int:
        """Normalizes mode representation to 0, 1, or 2."""
        if isinstance(mode, int):
            if mode in (0, 1, 2):
                return mode
            if mode == 3:
                return 2  # 1-based Mode 3 -> index 2
            raise ValueError(f"Invalid integer mode {mode}; expected 0, 1, 2 or 3")

        s = str(mode).strip().lower()
        if s in ("0", "narration", "narrative", "地の文", "setup", "mode 0", "mode_0", "mode0", "mode 1", "mode_1", "mode1"):
            return 0
        if s in ("1", "dialogue", "dialog", "会話", "polyphony", "mode 2", "mode_2", "mode2"):
            return 1
        if s in ("2", "3", "climax", "クライマックス", "shock", "mode 3", "mode_3", "mode3"):
            return 2
        raise ValueError(f"Unrecognized mode identifier: {mode}")

    def add_sample(
        self,
        vector: Sequence[float],
        mode: Optional[Union[int, str]] = None,
        cluster_id: Optional[Union[str, int]] = None,
    ) -> int:
        """Adds a single 24D sample using Welford online update."""
        z = np.asarray(vector, dtype=np.float64)
        if len(z) != 24:
            raise ValueError(f"Expected 24-dimensional vector, got {len(z)}")

        if cluster_id is not None:
            cid = str(cluster_id).strip()
            if not cid:
                raise ValueError("cluster_id cannot be empty or whitespace")
            self.clusters.add(cid)
            if self.last_cluster is not None and cid != self.last_cluster:
                self.reset_boundary()
            self.last_cluster = cid

        mode_idx = self._normalize_mode(mode) if mode is not None else classify_mode_24d(z)

        # Welford online update for mode_idx
        self.counts[mode_idx] += 1
        n = self.counts[mode_idx]
        delta = z - self.means[mode_idx]
        self.means[mode_idx] += delta / n
        delta2 = z - self.means[mode_idx]
        self.M2[mode_idx] += np.outer(delta, delta2)
        # Symmetrize M2 against numerical drift
        self.M2[mode_idx] = 0.5 * (self.M2[mode_idx] + self.M2[mode_idx].T)

        # Transition tracking
        if self.prev_mode is not None:
            self.transition_counts[self.prev_mode, mode_idx] += 1.0
        self.prev_mode = mode_idx
        self.sample_count += 1

        return mode_idx

    def add_trajectory(
        self,
        trajectory: Union[Sequence[Sequence[float]], np.ndarray],
        modes: Optional[Sequence[Union[int, str]]] = None,
        cluster_id: Optional[Union[str, int]] = None,
        align_phase: bool = True,
    ) -> List[int]:
        """Adds a trajectory of 24D vectors with optional Fourier phase alignment."""
        traj = np.asarray(trajectory, dtype=np.float64)
        if traj.ndim != 2 or traj.shape[1] != 24:
            raise ValueError(f"Expected trajectory of shape (M, 24), got {traj.shape}")

        active_modes = list(modes) if modes is not None else None

        if align_phase and len(traj) > 1:
            traj, s = align_polar_fourier_phase(traj, self.target_phase, return_shift=True)  # type: ignore
            # Circularly shift modes in lockstep with trajectory vectors to preserve 1:1 pairing
            if active_modes is not None and len(active_modes) == len(traj) and s != 0:
                active_modes = list(active_modes[-s:]) + list(active_modes[:-s])

        assigned_modes: List[int] = []
        for i, vec in enumerate(traj):
            m = active_modes[i] if (active_modes is not None and i < len(active_modes)) else None
            assigned = self.add_sample(vec, mode=m, cluster_id=cluster_id)
            assigned_modes.append(assigned)

        return assigned_modes

    def add_chunk(
        self,
        text: str,
        cluster_id: Optional[Union[str, int]] = None,
        align_phase: bool = True,
    ) -> List[int]:
        """Extracts 24D vectors from text sentences, aligns phase, and compiles."""
        from usir_py.extractor_24d import extract_usir_24d, split_sentences

        self.reset_boundary()
        sentences = split_sentences(text)
        if not sentences:
            return []

        vectors = []
        modes = []
        for sent in sentences:
            s_clean = sent.strip()
            if not s_clean:
                continue
            ext = extract_usir_24d(s_clean)
            vectors.append(ext.z_24d)
            if "「" in s_clean or "『" in s_clean:
                modes.append(1)  # Dialogue
            else:
                modes.append(None)  # Auto-classify

        if not vectors:
            return []

        return self.add_trajectory(vectors, modes=modes, cluster_id=cluster_id, align_phase=align_phase)

    def compile_tier_m_binary(self, enforce_anonymity: bool = True, shrinkage: str = "scaled") -> bytes:
        """Validates K >= min_clusters anonymity gate and outputs strict 4,096-byte binary.

        shrinkage="scaled" (default): Sigma = a*S + (1-a)*tr(S)/24*I with a = SHRINKAGE_ALPHA.
        shrinkage="legacy": the pre-3f288f0 formula Sigma = 0.2*S + 0.8*I (kept for comparison).
        """
        if shrinkage not in ("scaled", "legacy"):
            raise ValueError(f"Unknown shrinkage '{shrinkage}'; expected 'scaled' or 'legacy'")
        if enforce_anonymity and len(self.clusters) < self.min_clusters:
            raise ValueError(
                f"Cluster anonymity gate failed: K={len(self.clusters)} < {self.min_clusters}. "
                f"At least {self.min_clusters} distinct clusters/authors required to prevent single-author de-anonymization."
            )

        # 1. Weights
        total_samples = sum(self.counts)
        if total_samples > 0:
            weights = [float(c) / float(total_samples) for c in self.counts]
        else:
            weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

        # 2. Centroids
        centroids: List[List[float]] = []
        for k in range(3):
            if self.counts[k] > 0:
                centroids.append([float(v) for v in self.means[k]])
            else:
                centroids.append([0.5] * 24)

        # 3. Transition Matrix
        trans = np.zeros((3, 3), dtype=np.float64)
        for r in range(3):
            for c in range(3):
                prior = 3.0 if r == c else 1.0
                trans[r, c] = self.transition_counts[r, c] + prior
            row_sum = float(np.sum(trans[r]))
            if row_sum > 0:
                trans[r] /= row_sum
            else:
                trans[r] = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        transition_matrix = [[float(trans[r, c]) for c in range(3)] for r in range(3)]

        # 4. Cholesky Factors
        cholesky_factors: List[List[float]] = []
        for k in range(3):
            Nk = self.counts[k]
            if Nk > 1:
                S_k = self.M2[k] / (float(Nk) - 1.0)
            else:
                S_k = np.zeros((24, 24), dtype=np.float64)

            # Scaled-identity shrinkage: Sigma_k = a * S_k + (1 - a) * (tr(S_k)/24) * I, a = 0.75.
            # The target is scaled to the data variance, so the sample correlations stay dominant.
            # For PSD S_k, cond(Sigma_k) <= (24a + 1 - a)/(1 - a) = 73 < 500. Modes with no usable
            # variance (N_k <= 1 or identical samples) keep the legacy 0.8 * I.
            mean_var = float(np.trace(S_k)) / 24.0
            if shrinkage == "legacy":
                Sigma_k = 0.20 * S_k + 0.80 * np.eye(24, dtype=np.float64)
            elif mean_var < 1e-8:
                Sigma_k = 0.80 * np.eye(24, dtype=np.float64)
            else:
                Sigma_k = SHRINKAGE_ALPHA * S_k + (1.0 - SHRINKAGE_ALPHA) * mean_var * np.eye(24, dtype=np.float64)

            # Precision matrix Lambda_k = Sigma_k^-1
            Lambda_k = np.linalg.inv(Sigma_k)
            Lambda_k = 0.5 * (Lambda_k + Lambda_k.T)

            # Cholesky factor L_k: Lambda_k = L_k L_k^T
            packed_L = compute_packed_cholesky_24d(Lambda_k)
            cholesky_factors.append(packed_L)

        profile = {
            "version": 1,
            "flags": 0,
            "transition_matrix": transition_matrix,
            "weights": weights,
            "centroids": centroids,
            "cholesky_factors": cholesky_factors,
        }

        return compile_tier_m_binary(profile)

    @staticmethod
    def load_tier_m_binary(data: bytes) -> Dict[str, Any]:
        """Static accessor to load_tier_m_binary."""
        return load_tier_m_binary(data)
