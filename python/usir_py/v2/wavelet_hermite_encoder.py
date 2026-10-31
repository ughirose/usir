"""Wavelet-Hermite, Fourier-Poisson MCA & RQA 12-Element Encoder.

Implements Module 2 of USIR v2.0 Paper-Grade Architecture:
1. Extraction of 4 base structural signals s_m = (s_{m,1}, s_{m,2}, s_{m,3}, s_{m,4}) per clause.
2. OPEN-08: Ghost History Warm-Start & stationary padding to power-of-2 length >= 16 with James-Stein shrinkage.
3. Orthogonal Haar / D4 Wavelet filter bank decomposing signal into 3 strictly orthogonal subspaces:
   Micro (j=1), Meso (j=2), Macro (j=3), with <s^{(j)}, s^{(j')}> = 0 to machine precision (< 1e-12) (Theorem 2.1).
4. Morphological Component Analysis (MCA): 4th-order Fourier basis D_F + Poisson-Soliton basis D_S via ISTA L1 separation.
5. Wiener-Hermite orthogonal polynomials (He_2, He_3, He_4) + Recurrence Quantification Analysis (RQA DET & I_arch).
6. 24D Symplectic state z = (q, p) in [0, 1]^12 x R_{>=0}^12.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from usir_py.v2.hth_lexicon import (
    CLOSED_ADVERSATIVE_CONJUNCTIONS,
    CLOSED_LOGICAL_MARKERS,
    GEMINATES,
    VOICED_OBSTRUENTS,
    HTHEntry,
    HTHLexicon,
    MusicNoteEvent,
    TokenAnalysis,
    clz30,
    compute_sensory_bridge,
    discrete_jump_distance,
    lca_depth_30bit,
    poincare_hyperbolic_distance,
)


# ==============================================================================
# Result Containers
# ==============================================================================

@dataclass
class WaveletDecompositionResult:
    """3 strictly orthogonal scale components in R^N satisfying <s^(j), s^(j')> = 0."""
    micro: np.ndarray  # j=1 detail reconstruction in R^N
    meso: np.ndarray   # j=2 detail reconstruction in R^N
    macro: np.ndarray  # j=3 detail + scaling reconstruction in R^N
    coeffs: np.ndarray # Full wavelet coefficients in R^N


@dataclass
class MCAResult:
    """Morphological Component Analysis separation into Fourier and Soliton bases."""
    fourier_comp: np.ndarray  # D_F c_F (smooth periodic)
    soliton_comp: np.ndarray  # D_S c_S (sparse impulse spikes)
    c_F: np.ndarray
    c_S: np.ndarray
    residual: np.ndarray


@dataclass
class RQAResult:
    """Recurrence Quantification Analysis metrics for Macro scale."""
    recurrence_matrix: np.ndarray
    det: float     # Determinism (fraction of recurrence points on diagonal lines >= 2)
    i_arch: float  # Structural return arch index (Sg-10 At)
    lam: float     # Laminarity (vertical line density)


# ==============================================================================
# Wiener-Hermite Polynomials
# ==============================================================================

def wiener_hermite_polynomials(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wiener-Hermite polynomials He_2(z), He_3(z), He_4(z) for standardized signals."""
    he2 = z**2 - 1.0
    he3 = z**3 - 3.0 * z
    he4 = z**4 - 6.0 * z**2 + 3.0
    return he2, he3, he4


# ==============================================================================
# Strictly Orthogonal Haar Wavelet Filter Bank (Theorem 2.1)
# ==============================================================================

def build_haar_orthonormal_matrix(n: int) -> np.ndarray:
    """Construct an exact N x N orthonormal Haar wavelet transform matrix W.
    Rows 0 .. N/2 - 1: Level 1 details (Micro, j=1)
    Rows N/2 .. 3N/4 - 1: Level 2 details (Meso, j=2)
    Rows 3N/4 .. N - 1: Level 3 details + Level 3 scaling (Macro, j=3)
    Because W is orthonormal (W^T W = I_N), the reconstructed components
    s^(1), s^(2), s^(3) satisfy <s^(j), s^(j')> = 0 to machine precision (< 1e-15).
    """
    if n < 8 or (n & (n - 1)) != 0:
        raise ValueError(f"Wavelet length N must be a power of 2 >= 8, got {n}")

    w = np.zeros((n, n), dtype=np.float64)

    # 1. Level 1 details (Micro, j=1): N/2 wavelets of support 2
    for k in range(n // 2):
        w[k, 2 * k] = 1.0 / math.sqrt(2.0)
        w[k, 2 * k + 1] = -1.0 / math.sqrt(2.0)

    # 2. Level 2 details (Meso, j=2): N/4 wavelets of support 4
    offset = n // 2
    for k in range(n // 4):
        w[offset + k, 4 * k] = 0.5
        w[offset + k, 4 * k + 1] = 0.5
        w[offset + k, 4 * k + 2] = -0.5
        w[offset + k, 4 * k + 3] = -0.5

    # 3. Level 3 details (Macro detail): N/8 wavelets of support 8
    offset = (3 * n) // 4
    for k in range(n // 8):
        scale = 1.0 / math.sqrt(8.0)
        w[offset + k, 8 * k : 8 * k + 4] = scale
        w[offset + k, 8 * k + 4 : 8 * k + 8] = -scale

    # 4. Level 3 scaling approximations (Macro coarse): N/8 wavelets of support 8
    offset = (7 * n) // 8
    for k in range(n // 8):
        scale = 1.0 / math.sqrt(8.0)
        w[offset + k, 8 * k : 8 * k + 8] = scale

    return w


def orthogonal_wavelet_decompose_1d(signal: np.ndarray) -> WaveletDecompositionResult:
    """Strictly orthogonal 3-scale decomposition in R^N.
    Returns micro (j=1), meso (j=2), and macro (j=3).
    """
    n = len(signal)
    w = build_haar_orthonormal_matrix(n)
    coeffs = w @ signal

    # Project onto mutually orthogonal subspaces
    c_micro = np.zeros(n, dtype=np.float64)
    c_micro[: n // 2] = coeffs[: n // 2]
    micro_rec = w.T @ c_micro

    c_meso = np.zeros(n, dtype=np.float64)
    c_meso[n // 2 : (3 * n) // 4] = coeffs[n // 2 : (3 * n) // 4]
    meso_rec = w.T @ c_meso

    c_macro = np.zeros(n, dtype=np.float64)
    c_macro[(3 * n) // 4 :] = coeffs[(3 * n) // 4 :]
    macro_rec = w.T @ c_macro

    return WaveletDecompositionResult(
        micro=micro_rec,
        meso=meso_rec,
        macro=macro_rec,
        coeffs=coeffs,
    )


# ==============================================================================
# Morphological Component Analysis (MCA) via ISTA
# ==============================================================================

def solve_mca_fourier_soliton(
    signal: np.ndarray,
    num_harmonics: int = 4,
    mu_soliton: float = 0.15,
    max_iter: int = 25,
) -> MCAResult:
    """Morphological Component Analysis:
    Separates signal s into 4th-order Fourier D_F and Poisson-Soliton D_S
    via Iterative Shrinkage-Thresholding Algorithm (ISTA).
    """
    n = len(signal)
    t = np.arange(n, dtype=np.float64)

    # 1. Build Fourier Dictionary D_F (N, 2 * num_harmonics + 1)
    fourier_cols: List[np.ndarray] = [np.ones(n, dtype=np.float64) / math.sqrt(n)]
    for k in range(1, num_harmonics + 1):
        cos_col = np.cos(2.0 * math.pi * k * t / float(n))
        sin_col = np.sin(2.0 * math.pi * k * t / float(n))
        cos_norm = np.linalg.norm(cos_col)
        sin_norm = np.linalg.norm(sin_col)
        fourier_cols.append(cos_col / (cos_norm if cos_norm > 1e-7 else 1.0))
        fourier_cols.append(sin_col / (sin_norm if sin_norm > 1e-7 else 1.0))
    d_f = np.column_stack(fourier_cols)

    # 2. Build Soliton Kernel Dictionary D_S (N, N * 2)
    soliton_cols: List[np.ndarray] = []
    for tau in range(n):
        for omega in (1.5, 2.5):
            arg = np.clip(omega * (t - float(tau)), -20.0, 20.0)
            col = 1.0 / np.cosh(arg)
            col_norm = np.linalg.norm(col)
            soliton_cols.append(col / (col_norm if col_norm > 1e-7 else 1.0))
    d_s = np.column_stack(soliton_cols)

    # Step sizes (Lipschitz constants)
    gram_f = d_f.T @ d_f
    l_f = float(np.linalg.norm(gram_f, ord=2))
    eta_f = 0.95 / max(l_f, 1e-5)

    # Compute Lipschitz constant for Soliton dictionary via 8 power-iterations (O(N^2) instead of O(N^3) SVD)
    v = np.ones(d_s.shape[1], dtype=np.float64) / math.sqrt(d_s.shape[1])
    l_s = 1.0
    for _ in range(8):
        u = d_s @ v
        v = d_s.T @ u
        nv = float(np.linalg.norm(v))
        if nv > 1e-12:
            v /= nv
        l_s = nv
    eta_s = 0.95 / max(l_s, 1e-5)

    c_f = np.zeros(d_f.shape[1], dtype=np.float64)
    c_s = np.zeros(d_s.shape[1], dtype=np.float64)

    # ISTA Iterations
    for _ in range(max_iter):
        residual = signal - d_f @ c_f - d_s @ c_s
        c_f += eta_f * (d_f.T @ residual)

        residual = signal - d_f @ c_f - d_s @ c_s
        grad_s = c_s + eta_s * (d_s.T @ residual)
        thresh = eta_s * mu_soliton
        c_s = np.sign(grad_s) * np.maximum(0.0, np.abs(grad_s) - thresh)

    fourier_comp = d_f @ c_f
    soliton_comp = d_s @ c_s
    final_residual = signal - fourier_comp - soliton_comp

    return MCAResult(
        fourier_comp=fourier_comp,
        soliton_comp=soliton_comp,
        c_F=c_f,
        c_S=c_s,
        residual=final_residual,
    )


# ==============================================================================
# Recurrence Quantification Analysis (RQA)
# ==============================================================================

def compute_rqa_metrics(signal_1d: np.ndarray) -> RQAResult:
    """Recurrence Quantification Analysis:
    Calculates recurrence matrix, determinism DET, and long-range return arch I_arch (Sg-10 At).
    """
    n = len(signal_1d)
    diff = np.abs(signal_1d[:, None] - signal_1d[None, :])
    if n > 1:
        triu_dists = diff[np.triu_indices(n, k=1)]
        eps = float(np.percentile(triu_dists, 25))
        eps = max(eps, 1e-4)
    else:
        eps = 1e-4

    rec_mat = (diff <= eps).astype(np.float64)

    # Determinism: diagonal lines parallel to main diagonal of length >= 2
    diag_lengths: List[int] = []
    for k in range(1, n - 1):
        diag = np.diag(rec_mat, k=k)
        current_len = 0
        for val in diag:
            if val > 0.5:
                current_len += 1
            else:
                if current_len >= 2:
                    diag_lengths.append(current_len)
                current_len = 0
        if current_len >= 2:
            diag_lengths.append(current_len)

    total_diag_points = sum(diag_lengths)
    total_rec_points = max(float(np.sum(rec_mat)) - n, 1.0)
    det = float(np.clip(total_diag_points / total_rec_points, 0.0, 1.0))

    # Long-range return arch index: recurrence between early quarter and late quarter
    q1 = max(n // 4, 1)
    q3 = (3 * n) // 4
    arch_block = rec_mat[:q1, q3:]
    i_arch = float(np.mean(arch_block)) if arch_block.size > 0 else 0.0

    return RQAResult(
        recurrence_matrix=rec_mat,
        det=det,
        i_arch=float(np.clip(i_arch * 2.5, 0.0, 1.0)),
        lam=0.0,
    )


# ==============================================================================
# WaveletHermiteEncoder Class
# ==============================================================================

class WaveletHermiteEncoder:
    """Universal 12-Element & 24D Symplectic Wavelet-Hermite Encoder."""

    def __init__(self, lexicon: Optional[HTHLexicon] = None) -> None:
        if lexicon is None:
            lexicon = HTHLexicon()
        self.lexicon = lexicon

    # --------------------------------------------------------------------------
    # Base Structural Signals Extraction (s_m \in R^4)
    # --------------------------------------------------------------------------

    def extract_clause_signals(self, clause_text: str) -> np.ndarray:
        """Extract the 4 base structural signals for a single clause."""
        tokens = self.lexicon.analyze_clause(clause_text)
        if not tokens:
            return np.array([0.1, 0.5, 0.5, 0.0], dtype=np.float64)

        n_tok = len(tokens)
        content_tokens = [t for t in tokens if t.pos_major in ("名詞", "動詞", "形容詞", "副詞", "Mimetic")]
        particle_tokens = [t for t in tokens if t.pos_major in ("助詞", "助動詞")]

        # ----------------------------------------------------------------------
        # s_1 (Flux): Radial entropy + Transition surprise + Effective sensory density
        # ----------------------------------------------------------------------
        lengths = [len(t.surface) for t in tokens]
        tot_chars = sum(lengths)
        p_len = np.array([l / float(tot_chars) for l in lengths], dtype=np.float64) if tot_chars > 0 else np.ones(n_tok, dtype=np.float64) / max(n_tok, 1)
        if tot_chars > 0 and n_tok > 1:
            h_radial = -float(np.sum(p_len * np.log2(p_len + 1e-12))) / math.log2(max(n_tok, 2))
        else:
            h_radial = 0.2

        # Character bigram conditional transition surprise
        clean = re.sub(r"\s+", "", clause_text)
        if len(clean) >= 3:
            bigrams = [clean[i : i + 2] for i in range(len(clean) - 1)]
            counts: Dict[str, int] = {}
            for bg in bigrams:
                counts[bg] = counts.get(bg, 0) + 1
            bg_entropy = -sum((c / len(bigrams)) * math.log2(c / len(bigrams)) for c in counts.values())
            surprise = bg_entropy / math.log2(max(len(bigrams), 2))
        else:
            surprise = 0.1

        # Effective sensory density under OPEN-01 polysemy damping
        sensory_sum = 0.0
        neg_momentum = 0.0
        if content_tokens:
            for idx in range(len(content_tokens)):
                tok_curr = content_tokens[idx]
                tok_prev = content_tokens[idx - 1] if idx > 0 else tok_curr
                damped_s, _, _ = self.lexicon.apply_polysemy_damping(tok_prev.hth, tok_curr.hth)
                if tok_curr.hth.is_negated:
                    # OPEN-02: alpha_mod = -0.8, mean zeroed, routed to contrast momentum
                    neg_momentum += 0.8 * float(np.linalg.norm(damped_s))
                else:
                    sensory_sum += float(np.sum(damped_s))
            sensory_density = sensory_sum / (len(content_tokens) * 8.0)
        else:
            sensory_density = 0.05

        s_1 = float(np.clip(0.35 * h_radial + 0.35 * surprise + 0.30 * sensory_density, 0.0, 1.0))

        # ----------------------------------------------------------------------
        # s_2 (Lattice): Mora beat autocorrelation + symmetry + closed logical density
        # ----------------------------------------------------------------------
        # Mora counting
        mora_count = sum(1 for c in clean if c not in "ゃゅょぁぃぅぇぉャュョァィゥェォっッー")
        # Deviation from canonical Japanese cadences (5, 7, 12, 17)
        cadence_dist = min(abs(mora_count - 5), abs(mora_count - 7), abs(mora_count - 12), abs(mora_count - 17))
        mora_resonance = max(0.0, 1.0 - cadence_dist / 8.0)

        # Graph Laplacian eigenvalue symmetry proxy (clause token length symmetry)
        if n_tok >= 3:
            sym_diff = np.abs(p_len - p_len[::-1])
            lap_sym = 1.0 - float(np.mean(sym_diff))
        else:
            lap_sym = 0.5

        logical_count = sum(1 for t in tokens if t.surface in CLOSED_LOGICAL_MARKERS or t.pos_major in ("助詞", "助動詞"))
        logical_density = logical_count / float(max(n_tok, 1))

        s_2 = float(np.clip(0.40 * mora_resonance + 0.30 * lap_sym + 0.30 * logical_density, 0.0, 1.0))

        # ----------------------------------------------------------------------
        # s_3 (Void [間]): OPEN-05 Wago concreteness * particle entropy * unornamented subtraction rate
        # ----------------------------------------------------------------------
        if content_tokens:
            wago_concreteness = float(np.mean([t.hth.concreteness for t in content_tokens]))
            apophatic_stillness = float(np.mean([t.hth.sensory_vector[7] for t in content_tokens]))
        else:
            wago_concreteness = 0.1
            apophatic_stillness = 0.5

        # Particle transition entropy
        if particle_tokens:
            p_counts: Dict[str, int] = {}
            for pt in particle_tokens:
                p_counts[pt.surface] = p_counts.get(pt.surface, 0) + 1
            n_pt = len(particle_tokens)
            pt_entropy = -sum((c / n_pt) * math.log2(c / n_pt) for c in p_counts.values())
            pt_entropy_norm = pt_entropy / math.log2(max(n_pt, 2)) if n_pt > 1 else 0.5
        else:
            pt_entropy_norm = 0.2

        # Unornamented subtraction rate exp(-gamma * N_adj_adv)
        n_adj_adv = sum(1 for t in tokens if t.pos_major in ("形容詞", "副詞") and not t.hth.mimetic_immunity)
        subtraction_rate = math.exp(-0.5 * float(n_adj_adv))

        s_3 = float(np.clip(wago_concreteness * (0.5 + 0.5 * pt_entropy_norm) * subtraction_rate * (0.6 + 0.4 * apophatic_stillness), 0.0, 1.0))

        # ----------------------------------------------------------------------
        # s_4 (Shock): Hyperbolic jump * G_bridge + Negation momentum + Phonological edge
        # ----------------------------------------------------------------------
        jump_sum = 0.0
        jump_count = 0
        if len(content_tokens) >= 2:
            for i in range(len(content_tokens) - 1):
                u = content_tokens[i].hth
                v = content_tokens[i + 1].hth
                d_jump = discrete_jump_distance(u.h_code, v.h_code)
                _, g_bridge_eff, _ = self.lexicon.apply_polysemy_damping(u, v)
                jump_sum += d_jump * g_bridge_eff
                jump_count += 1
            jump_val = jump_sum / float(jump_count)
        else:
            jump_val = 0.0

        # Phonological edge (voiced obstruents + geminates)
        voiced_count = sum(1 for c in clean if c in VOICED_OBSTRUENTS)
        geminate_count = sum(1 for c in clean if c in GEMINATES)
        phonological_edge = min(1.0, (voiced_count * 0.15 + geminate_count * 0.35))

        # Adversative conjunction impulse
        has_adversative = any(t.surface in CLOSED_ADVERSATIVE_CONJUNCTIONS for t in tokens)
        adversative_impulse = 0.6 if has_adversative else 0.0

        s_4 = float(np.clip(0.35 * jump_val + 0.25 * neg_momentum + 0.20 * phonological_edge + 0.20 * adversative_impulse, 0.0, 1.0))

        return np.array([s_1, s_2, s_3, s_4], dtype=np.float64)

    def extract_text_signals(self, text: str) -> np.ndarray:
        """Extract sequence of base signals s_m across all clauses in narrative text."""
        # OPEN-04: Separate dialogue, extract base signals from narrative prose
        narrative, _ = self.lexicon.separate_dialogue_and_narrative(text)
        if not narrative:
            narrative = text

        # Split into clauses by punctuation (句点・読点・改行)
        clause_splits = [c.strip() for c in re.split(r"[。！？!?\n、,]+", narrative) if len(c.strip()) >= 2]
        if not clause_splits:
            clause_splits = [narrative]

        signals = np.array([self.extract_clause_signals(c) for c in clause_splits], dtype=np.float64)
        return signals

    # --------------------------------------------------------------------------
    # OPEN-08: Ghost History Warm-Start & Stationary Power-of-2 Padding
    # --------------------------------------------------------------------------

    @staticmethod
    def apply_open08_ghost_padding(signals: np.ndarray, min_len: int = 16, max_len: int = 1024) -> np.ndarray:
        """Apply OPEN-08 Ghost History Warm-Start with James-Stein shrinkage."""
        m = signals.shape[0]
        if m == 0:
            neutral = np.array([[0.1, 0.5, 0.5, 0.0]], dtype=np.float64)
            return np.repeat(neutral, min_len, axis=0)

        # Target length is nearest power of 2 in [min_len, max_len]
        target_len = max(min_len, min(max_len, 1 << int(math.ceil(math.log2(max(m, 1))))))

        # Compute global stationary mean
        stat_mean = np.mean(signals, axis=0, keepdims=True)

        if m < target_len:
            # Reflection padding
            pad_needed = target_len - m
            padded_list = [signals]
            while pad_needed > 0:
                take = min(pad_needed, m)
                padded_list.append(signals[-take:][::-1])
                pad_needed -= take
            padded = np.vstack(padded_list)[:target_len]
        else:
            padded = signals[:target_len]

        # Apply James-Stein shrinkage alpha_k = k / (k + 5)
        k_indices = np.arange(1, target_len + 1, dtype=np.float64)[:, None]
        alpha = k_indices / (k_indices + 5.0)
        shrunk = stat_mean + alpha * (padded - stat_mean)
        return shrunk

    # --------------------------------------------------------------------------
    # Full Symplectic 24D Encoding
    # --------------------------------------------------------------------------

    def encode_signals(self, raw_signals: np.ndarray, use_mca: bool = True) -> np.ndarray:
        """Encode (M, 4) base structural signals into 24D symplectic vector z = (q, p)."""
        signals = self.apply_open08_ghost_padding(raw_signals, min_len=16)
        n = signals.shape[0]

        # Wavelet decomposition for all 4 signal channels
        micro_list = []
        meso_list = []
        macro_list = []

        for a in range(4):
            dec = orthogonal_wavelet_decompose_1d(signals[:, a])
            micro_list.append(dec.micro)
            meso_list.append(dec.meso)
            macro_list.append(dec.macro)

        micro_arr = np.column_stack(micro_list)  # (N, 4)
        meso_arr = np.column_stack(meso_list)    # (N, 4)
        macro_arr = np.column_stack(macro_list)  # (N, 4)

        # Compute position coordinates q \in [0, 1]^12
        q = np.zeros(12, dtype=np.float64)

        # Micro (j=1): Elements 0..3 (Sg-01 Dn, Sg-02 Cr, Sg-03 Pr, Sg-04 Jt)
        for a in range(4):
            rms_val = float(np.sqrt(np.mean(micro_arr[:, a] ** 2)))
            std_val = float(np.std(micro_arr[:, a]))
            z_norm = micro_arr[:, a] / (std_val + 1e-8)
            he2, he3, he4 = wiener_hermite_polynomials(z_norm)
            moment_mod = 0.05 * float(np.mean(he3)) + 0.02 * float(np.mean(he4))
            q[a] = float(np.clip(rms_val * 2.5 + moment_mod, 0.0, 1.0))

        # Meso (j=2): Elements 4..7 (Sg-05 Cp, Sg-06 Mp, Sg-07 Cs, Sg-08 Lp)
        for a in range(4):
            rms_val = float(np.sqrt(np.mean(meso_arr[:, a] ** 2)))
            std_val = float(np.std(meso_arr[:, a]))
            z_norm = meso_arr[:, a] / (std_val + 1e-8)
            he2, he3, he4 = wiener_hermite_polynomials(z_norm)
            moment_mod = 0.05 * float(np.mean(he3))
            q[4 + a] = float(np.clip(rms_val * 3.5 + moment_mod, 0.0, 1.0))

        # Macro (j=3): Elements 8..11 (Sg-09, Sg-10 At, Sg-11 Ma, Sg-12 Ru)
        q[8] = float(np.clip(np.mean(macro_arr[:, 0]), 0.0, 1.0))  # Sg-09: Flux Macro

        # Element 9 (Sg-10 At): RQA determinism and return arch on Macro Lattice
        rqa = compute_rqa_metrics(macro_arr[:, 1])
        q[9] = float(np.clip(0.5 * rqa.det + 0.5 * rqa.i_arch, 0.0, 1.0))

        # Element 10 (Sg-11 Ma): Void Macro
        q[10] = float(np.clip(np.mean(macro_arr[:, 2]), 0.0, 1.0))

        # Element 11 (Sg-12 Ru): Shock Macro via Poisson-Soliton MCA
        if use_mca:
            mca = solve_mca_fourier_soliton(macro_arr[:, 3], num_harmonics=4, mu_soliton=0.15)
            f_norm = float(np.linalg.norm(mca.fourier_comp))
            s_l1 = float(np.sum(np.abs(mca.c_S)))
            q[11] = float(np.clip(s_l1 / (f_norm + 1e-4), 0.0, 1.0))
        else:
            q[11] = float(np.clip(np.mean(macro_arr[:, 3]), 0.0, 1.0))

        q_norm = np.clip(q, 0.0, 1.0)

        # Compute conjugate momentum coordinates p \in R_{>=0}^12
        p = np.zeros(12, dtype=np.float64)
        for a in range(4):
            # Micro momentum
            p[a] = float(np.std(micro_arr[:, a])) + float(np.sqrt(np.mean(np.diff(micro_arr[:, a]) ** 2)))
            # Meso momentum
            p[4 + a] = float(np.std(meso_arr[:, a])) + float(np.sqrt(np.mean(np.diff(meso_arr[:, a]) ** 2)))
            # Macro momentum
            p[8 + a] = float(np.std(macro_arr[:, a])) + float(np.sqrt(np.mean(np.diff(macro_arr[:, a]) ** 2)))

        p = np.clip(p, 0.0, None)

        z = np.concatenate([q_norm, p])
        return z

    def encode(self, text: str, use_mca: bool = True) -> np.ndarray:
        """Encode raw text into 24D symplectic state."""
        signals = self.extract_text_signals(text)
        return self.encode_signals(signals, use_mca=use_mca)


# ==============================================================================
# Public API Convenience Function
# ==============================================================================

_DEFAULT_ENCODER: Optional[WaveletHermiteEncoder] = None


def extract_24d_v2(text_or_signals: Union[str, np.ndarray, List[MusicNoteEvent]], use_mca: bool = True) -> np.ndarray:
    """Extract 24-dimensional symplectic state z = (q, p) in R^24."""
    global _DEFAULT_ENCODER
    if _DEFAULT_ENCODER is None:
        _DEFAULT_ENCODER = WaveletHermiteEncoder()

    if isinstance(text_or_signals, str):
        return _DEFAULT_ENCODER.encode(text_or_signals, use_mca=use_mca)
    elif isinstance(text_or_signals, np.ndarray):
        return _DEFAULT_ENCODER.encode_signals(text_or_signals, use_mca=use_mca)
    elif isinstance(text_or_signals, list) and (len(text_or_signals) == 0 or isinstance(text_or_signals[0], MusicNoteEvent)):
        from usir_py.v2.hth_lexicon import extract_music_functor_signals
        signals = extract_music_functor_signals(text_or_signals)
        return _DEFAULT_ENCODER.encode_signals(signals, use_mca=use_mca)
    else:
        raise TypeError(f"Unsupported input type for extract_24d_v2: {type(text_or_signals)}")
