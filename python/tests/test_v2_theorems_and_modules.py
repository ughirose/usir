"""Unit Tests for USIR v2.0 Mathematical Theorems & Modules 1..4."""

from __future__ import annotations

import math
import numpy as np
import pytest

from usir_py.v2 import (
    AdaptiveBAOABIntegrator,
    HTHEntry,
    HTHLexicon,
    HamiltonFristonResonance,
    KacMoreauYosidaThermodynamics,
    MusicNoteEvent,
    SO8PullbackMetric,
    clz30,
    compute_rqa_metrics,
    discrete_jump_distance,
    extract_24d_v2,
    extract_music_functor_signals,
    lca_depth_30bit,
    orthogonal_wavelet_decompose_1d,
    poincare_hyperbolic_distance,
    solve_mca_fourier_soliton,
)
from usir_py.v2.wavelet_hermite_encoder import WaveletHermiteEncoder


# --- 1. Theorem 1.1: Kac-Regularized Free Energy Intensivity ---

def test_theorem_1_1_kac_regularized_free_energy_intensivity():
    """Verify Theorem 1.1: Free energy density |F_L / L| remains O(1) bounded for L in {16, 64, 256, 1024}."""
    thermo = KacMoreauYosidaThermodynamics(alpha=1.5, j0=1.0)
    densities = thermo.verify_kac_bound(lengths=(16, 64, 256, 1024), seed=42)
    for L in (16, 64, 256, 1024):
        assert 0.1 <= abs(densities[L]) <= 20.0, f"Density at L={L} diverged: {densities[L]}"
    # Asymptotic stability: relative deviation between L=256 and L=1024 is tight O(1)
    diff = abs(densities[1024] - densities[256])
    assert diff < 2.0, f"Free energy density diverged as L -> infty: diff={diff:.4f}"


# --- 2. Theorem 2.1: Orthogonal Wavelet Subspace Decomposition ---

def test_theorem_2_1_orthogonal_wavelet_decomposition():
    """Verify Theorem 2.1: Under Gaussian null P_0, cross-scale inner products < 1e-11 and reconstruction holds."""
    rng = np.random.default_rng(2026)
    N = 64
    for _ in range(200):
        s = rng.normal(0.0, 1.0, size=N)
        dec = orthogonal_wavelet_decompose_1d(s)
        assert abs(float(np.dot(dec.micro, dec.meso))) < 1e-11
        assert abs(float(np.dot(dec.micro, dec.macro))) < 1e-11
        assert abs(float(np.dot(dec.meso, dec.macro))) < 1e-11
        rec_err = float(np.max(np.abs((dec.micro + dec.meso + dec.macro) - s)))
        assert rec_err < 1e-11, f"Reconstruction failed: error={rec_err:.2e}"


# --- 3. Theorem 5.1: Adaptive-BAOAB Invariant Measure under 1,000x Friction ---

def test_theorem_5_1_adaptive_baoab_invariant_measure():
    """Verify Theorem 5.1: Adaptive-BAOAB preserves N(0, tau M) under 1,000x random friction shifts."""
    baoab = AdaptiveBAOABIntegrator()
    res = baoab.verify_invariant_momentum_measure(n_steps=20000, tau=1.0, seed=42, tol_cov=0.12, tol_mean=0.08)
    assert res["passed"] is True
    assert float(res["max_abs_mean"]) < 0.08
    assert float(res["cov_relative_error"]) < 0.12


# --- 4. Module 1: 64-bit HTH, OPEN-01..04, and Music Functor ---

def test_module_1_bitops_and_poincare_distance():
    """Test 30-bit bitwise primitives and continuous Poincare distance."""
    assert clz30(0) == 30 and clz30(1) == 29 and clz30(1 << 29) == 0
    assert lca_depth_30bit(0x12345678, 0x12345678) == 10 and lca_depth_30bit(0x0, 0x3FFFFFFF) == 0
    assert discrete_jump_distance(0x12345678, 0x12345678) == 0.0 and discrete_jump_distance(0x0, 0x3FFFFFFF) == 1.0

    entry_a = HTHEntry(token="a", pos="N", h_code=0x12345678, sensory_vector=np.ones(8)*0.5, concreteness=0.8)
    entry_b = HTHEntry(token="b", pos="N", h_code=0x12345678, sensory_vector=np.ones(8)*0.5, concreteness=0.8)
    entry_c = HTHEntry(token="c", pos="N", h_code=0x0, sensory_vector=np.zeros(8), concreteness=0.2)
    assert poincare_hyperbolic_distance(entry_a, entry_b) == 0.0
    assert 0.0 < poincare_hyperbolic_distance(entry_a, entry_c) <= 1.0


def test_module_1_open01_polysemy_damping_three_regimes():
    """Test OPEN-01 contextual polysemy damping gate 3 regimes + mimetic immunity."""
    s_dense = np.array([0.7, 0.6, 0.5, 0.8, 0.4, 0.6, 0.1, 0.2], dtype=np.float64)
    # 1. Literal: high concreteness (C >= 0.45) & shared octant (lca >= 2)
    head_lit = HTHEntry("石", "N", 0x12340000, s_dense, concreteness=0.80)
    mod_lit = HTHEntry("硬い", "Adj", 0x12350000, s_dense, concreteness=0.75)
    s_out, _, mode = HTHLexicon.apply_polysemy_damping(head_lit, mod_lit)
    assert mode == "literal" and np.allclose(s_out, s_dense)

    # 2. Poetic Metaphor: concrete head (C >= 0.45) & cross-octant jump (lca < 2, ||s_v|| > 0.3)
    mod_poet = HTHEntry("燃える", "V", 0x20000000, s_dense, concreteness=0.70)
    _, g_eff, mode = HTHLexicon.apply_polysemy_damping(head_lit, mod_poet)
    assert mode == "poetic_metaphor" and g_eff > 0.5

    # 3. Dead Metaphor: abstract head (C < 0.38) -> damped sensory vector
    head_abs = HTHEntry("概念", "N", 0x00010000, np.ones(8)*0.1, concreteness=0.20)
    s_damped, _, mode = HTHLexicon.apply_polysemy_damping(head_abs, mod_lit)
    assert mode == "dead_metaphor" and np.allclose(s_damped, 0.20 * s_dense)


def test_module_1_open02_negation_scope_damping_and_momentum():
    """Test OPEN-02 negation detection damping mean and boosting contrast momentum."""
    enc = WaveletHermiteEncoder()
    assert all(not t.hth.is_negated for t in enc.lexicon.analyze_clause("花が咲く"))
    tokens_neg = enc.lexicon.analyze_clause("花が咲かない")
    assert any(t.hth.is_negated for t in tokens_neg if t.pos_major in ("名詞", "動詞"))
    s_pos = enc.extract_clause_signals("花が咲く")
    s_neg = enc.extract_clause_signals("花が咲かない")
    assert s_neg[3] > s_pos[3]  # Shock/momentum boosted


def test_module_1_open03_onomatopoeia_and_mimetic_immunity():
    """Test OPEN-03 onomatopoeia sound-symbolic synthesis and mimetic immunity."""
    lex = HTHLexicon()
    for word in ("きらきら", "ざわざわ", "ふわり", "しんと"):
        assert lex.is_onomatopoeia(word, word, "副詞", "*")
        entry = lex.lookup_token(word, word, "副詞", "*")
        assert entry.mimetic_immunity is True and entry.concreteness >= 0.45
        # Immunity preserves sensory vector even under abstract head
        head_abs = HTHEntry("概念", "N", 0, np.zeros(8), concreteness=0.10)
        s_out, _, mode = HTHLexicon.apply_polysemy_damping(head_abs, entry)
        assert mode == "mimetic_immune" and np.allclose(s_out, entry.sensory_vector)


def test_module_1_open04_dialogue_narrative_separation():
    """Test OPEN-04 clean orthogonal separation of narrative prose and bracketed dialogue."""
    nar, dia = HTHLexicon.separate_dialogue_and_narrative("少年は「先へ行こう」と叫び、『決して止まるな』と促した。")
    assert "先へ行こう" not in nar and "決して止まるな" not in nar
    assert dia == ["先へ行こう", "決して止まるな"]


def test_module_1_music_functor_and_24d_pipeline():
    """Test M1 cross-modal music functor and 24D symplectic pipeline."""
    notes = [
        MusicNoteEvent(pitch=60 + i * 4, duration=0.5, velocity=80 + i * 5, ioi=0.5, onset=i * 0.5)
        for i in range(8)
    ]
    signals = extract_music_functor_signals(notes, num_bins=32)
    assert signals.shape == (32, 4) and np.all((signals >= 0.0) & (signals <= 1.0))
    z = extract_24d_v2(notes, use_mca=True)
    assert z.shape == (24,)
    assert np.all((z[:12] >= 0.0) & (z[:12] <= 1.0)) and np.all(z[12:] >= 0.0)


# --- 5. Modules 2 & 3: MCA, RQA, SO(8) Pullback Metric & Moreau-Yosida Prox ---

def test_module_2_mca_fourier_soliton_separation():
    """Test MCA ISTA separation of smooth Fourier component and localized Poisson-soliton spike."""
    t = np.arange(32, dtype=np.float64)
    fourier_gt = 2.0 * np.cos(2.0 * math.pi * 1.0 * t / 32.0)
    soliton_gt = 3.0 / np.cosh(2.0 * (t - 16.0))
    mca = solve_mca_fourier_soliton(fourier_gt + soliton_gt, num_harmonics=4, mu_soliton=0.15, max_iter=40)
    assert float(np.linalg.norm(mca.residual)) < 0.5
    assert np.corrcoef(mca.fourier_comp, fourier_gt)[0, 1] > 0.98
    assert np.corrcoef(mca.soliton_comp, soliton_gt)[0, 1] > 0.98
    assert int(np.argmax(mca.soliton_comp)) == 16


def test_module_2_rqa_metrics():
    """Test Recurrence Quantification Analysis DET and long-range return arch I_arch."""
    periodic = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), 8)
    rqa_per = compute_rqa_metrics(periodic)
    assert rqa_per.det > 0.4 and rqa_per.i_arch > 0.4
    rng = np.random.default_rng(123)
    rqa_noise = compute_rqa_metrics(rng.normal(0.0, 1.0, size=32))
    assert rqa_per.det > rqa_noise.det


def test_module_3_so8_pullback_metric_and_haar_projector():
    """Test SO(8) Cayley transform R^T R = I_8, det(R)=+1, P P^T = I_8, and G_style > 0."""
    metric = SO8PullbackMetric()
    assert float(np.max(np.abs(metric.R.T @ metric.R - np.eye(8)))) < 1e-7
    assert abs(float(np.linalg.det(metric.R)) - 1.0) < 1e-7
    assert float(np.max(np.abs(metric.P @ metric.P.T - np.eye(8)))) < 1e-10
    assert np.min(np.linalg.eigvalsh(metric.metric_matrix())) >= 1e-3 * 0.999


def test_module_3_moreau_yosida_prox_and_valence_resonance():
    """Test Moreau-Yosida proximal operator, Hamilton-Friston valence spike, and 1/f^beta slope."""
    thermo = KacMoreauYosidaThermodynamics(mu1=0.01)
    x = np.array([0.01, 0.02, 0.5, -0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    prox_y = thermo.prox_lambda_psi(x)
    assert abs(prox_y[0]) < 1e-6 and abs(prox_y[1]) < 1e-6  # Sparse thresholding

    hf = HamiltonFristonResonance(thermo=thermo)
    assert isinstance(hf.valence_spike(np.ones(12) * 0.5, np.ones(12) * 0.2), float)

    # 1/f pink noise spectral equipartition test
    rng = np.random.default_rng(42)
    N = 256
    freqs = np.fft.rfftfreq(N)
    white_fft = rng.normal(0, 1, size=len(freqs)) + 1j * rng.normal(0, 1, size=len(freqs))
    pink = np.fft.irfft(white_fft / np.sqrt(np.maximum(freqs, 1e-5)), n=N)

    beta_hat, (p1, p2, p3) = hf.compute_spectral_equipartition(pink)
    assert 0.6 <= beta_hat <= 1.4
    assert abs((p1 + p2 + p3) - 1.0) < 1e-6
