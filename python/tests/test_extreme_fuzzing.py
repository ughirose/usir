"""USIR v2.0 Extreme Fuzzing & Numerical Robustness Test Suite.

Tests extreme boundary conditions across python/usir_py/v2:
1. >100k-char structured text
2. Text containing only punctuation / symbols (e.g. 「……！？……」, empty, whitespace)
3. Kanji-only classical Chinese
4. Hiragana-only continuous stream
5. Fullwidth Katakana-only continuous stream
6. Single very long text (>100k chars) without newline, space, or punctuation
7. MIDI sequences that are all rests (velocity=0, large gaps, empty list) or a single note
8. Downstream thermodynamic / Hamiltonian stability on extreme states
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from usir_py.v2 import (
    AdaptiveBAOABIntegrator,
    HTHLexicon,
    HamiltonFristonResonance,
    KacMoreauYosidaThermodynamics,
    MusicNoteEvent,
    SO8PullbackMetric,
    WaveletHermiteEncoder,
    extract_24d_v2,
    extract_music_functor_signals,
)


def assert_valid_symplectic_24d(z: np.ndarray, context: str = ""):
    """Verify symplectic state invariants:
    - Shape is exactly (24,)
    - All elements are finite (no NaN, no +/-Inf)
    - Position coordinates q \in [0.0, 1.0]^12
    - Momentum coordinates p \in [0.0, \infty)^12
    """
    assert isinstance(z, np.ndarray), f"{context}: Result is not numpy array"
    assert z.shape == (24,), f"{context}: Expected shape (24,), got {z.shape}"
    assert np.all(np.isfinite(z)), f"{context}: Found NaN or Inf in state z: {z}"
    q = z[:12]
    p = z[12:]
    assert np.all((q >= 0.0) & (q <= 1.0)), f"{context}: q coordinates out of [0, 1]: {q}"
    assert np.all(p >= 0.0), f"{context}: Conjugate momenta p negative: {p}"


def assert_downstream_numerical_stability(z: np.ndarray, context: str = ""):
    """Verify state z can undergo downstream Riemannian / thermodynamic dynamics
    without NaN, Infinity, or zero division.
    """
    q = z[:12]
    p = z[12:]

    # 1. SO(8) Pullback metric computation
    metric = SO8PullbackMetric()
    G = metric.metric_matrix()
    assert np.all(np.isfinite(G)), f"{context}: Pullback metric matrix contains NaN/Inf"
    assert np.min(np.linalg.eigvalsh(G)) > 0.0, f"{context}: Metric matrix not positive definite"

    # 2. Kac-Moreau-Yosida potential & gradient
    thermo = KacMoreauYosidaThermodynamics(lambda_reg=0.1, mu1=0.01)
    grad_F = thermo.moreau_yosida_gradient(q)
    assert np.all(np.isfinite(grad_F)), f"{context}: Thermodynamic gradient contains NaN/Inf: {grad_F}"

    # 3. Hamilton-Friston valence spike
    hf = HamiltonFristonResonance(thermo=thermo)
    val = hf.valence_spike(q, p)
    assert math.isfinite(val), f"{context}: Valence spike is not finite: {val}"

    # 4. Adaptive-BAOAB 1-step integration
    baoab = AdaptiveBAOABIntegrator()
    q_next, p_next = baoab.step(q, p, grad_fn=thermo.moreau_yosida_gradient, dt=0.05, tau=1.0)
    assert np.all(np.isfinite(q_next)), f"{context}: BAOAB step produced NaN/Inf in q_next"
    assert np.all(np.isfinite(p_next)), f"{context}: BAOAB step produced NaN/Inf in p_next"


# ==============================================================================
# 1. Text of Only Punctuation and Symbols
# ==============================================================================

@pytest.mark.parametrize(
    "punct_text,desc",
    [
        ("「……！？……」", "brackets and ellipsis/exclamation"),
        ("！？！？！？！？！？！？！？！？", "repeated exclamation and question marks"),
        ("…・、。！？「」『』（）―〜", "comprehensive Japanese punctuation set"),
        ("", "empty string"),
        ("   \n\t\r   \n   ", "whitespace and newlines only"),
        ("。。。。。。。。。。", "repeated full-stops"),
        ("、、、、、、、、、、", "repeated commas"),
        ("!@#$%^&*()_+=-[]{};':\",./<>?", "ascii symbols only"),
    ],
)
def test_extreme_text_only_punctuation_and_symbols(punct_text: str, desc: str):
    """Verify pure punctuation / symbol inputs do not crash or yield NaN/Inf."""
    z = extract_24d_v2(punct_text, use_mca=True)
    assert_valid_symplectic_24d(z, f"Punctuation ({desc})")
    assert_downstream_numerical_stability(z, f"Punctuation ({desc})")


# ==============================================================================
# 2. Kanji-Only Classical Chinese
# ==============================================================================

def test_extreme_kanji_only_classical_chinese():
    """Verify unpunctuated and punctuated classical Chinese (pure CJK ideographs)."""
    # Excerpt from Analects (論語)
    classical = (
        "子曰學而時習之不亦說乎有朋自遠方來不亦樂乎人不知而不慍不亦君子乎"
        "有子曰其為人也孝弟而好犯上者鮮矣不好犯上而好作亂者未知之也"
        "君子務本本立而道生孝弟也者其為仁之本與"
        "子曰巧言令色鮮矣仁"
        "曾子曰吾日三省吾身為人謀而不忠乎與朋友交而不信乎傳不習乎"
    ) * 10  # ~750 characters pure kanji
    z = extract_24d_v2(classical, use_mca=True)
    assert_valid_symplectic_24d(z, "Kanji-only classical Chinese")
    assert_downstream_numerical_stability(z, "Kanji-only classical Chinese")


# ==============================================================================
# 3. Hiragana-Only Continuous Stream
# ==============================================================================

def test_extreme_hiragana_only():
    """Verify 100% Hiragana stream with no kanji, whitespace, or punctuation."""
    hira_stream = (
        "いろはにほへとちりぬるをわかよたれそつねならむうゐのおくやまけふこえてあさきゆめみしゑひもせす"
        "あめつちほしそらかないきひつるみあめつちをさむるひとのたみ"
    ) * 15  # ~1500 chars pure hiragana
    z = extract_24d_v2(hira_stream, use_mca=True)
    assert_valid_symplectic_24d(z, "Hiragana-only continuous stream")
    assert_downstream_numerical_stability(z, "Hiragana-only continuous stream")


# ==============================================================================
# 4. Fullwidth Katakana-Only Continuous Stream
# ==============================================================================

def test_extreme_katakana_only():
    """Verify 100% Fullwidth Katakana stream (acoustic onomatopoeic bias)."""
    kata_stream = (
        "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン"
        "ガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポヴッー"
    ) * 15  # ~1500 chars pure katakana
    z = extract_24d_v2(kata_stream, use_mca=True)
    assert_valid_symplectic_24d(z, "Katakana-only continuous stream")
    assert_downstream_numerical_stability(z, "Katakana-only continuous stream")


# ==============================================================================
# 5. Single Very Long Text Without Delimiters (>100k chars)
# ==============================================================================

def test_extreme_single_very_long_text_without_delimiters():
    """Verify single contiguous text >100,000 characters without newline/space/punctuation."""
    # 100,005 chars of single kana character
    single_kana_100k = "あ" * 100005
    z_kana = extract_24d_v2(single_kana_100k, use_mca=True)
    assert_valid_symplectic_24d(z_kana, "Single kana >100k chars")
    assert_downstream_numerical_stability(z_kana, "Single kana >100k chars")

    # 100,005 chars of single kanji character
    single_kanji_100k = "龍" * 100005
    z_kanji = extract_24d_v2(single_kanji_100k, use_mca=True)
    assert_valid_symplectic_24d(z_kanji, "Single kanji >100k chars")
    assert_downstream_numerical_stability(z_kanji, "Single kanji >100k chars")


# ==============================================================================
# 6. >100k-Char Structured Natural Text
# ==============================================================================

def test_extreme_structured_text_over_100k_chars():
    """Verify realistic multi-clause text >100,000 characters."""
    passage = (
        "山路を登りながら、こう考えた。智に働けば角が立つ。情に棹させば流される。"
        "意地を通せば窮屈だ。とかくに人の世は住みにくい。住みにくさが高じると、"
        "安い所へ引き越したくなる。どこへ越しても住みにくいと悟った時、詩が生れて、画が出来る。"
    )
    repeats = 100000 // len(passage) + 2
    text_100k = passage * repeats
    assert len(text_100k) > 100000

    z = extract_24d_v2(text_100k, use_mca=True)
    assert_valid_symplectic_24d(z, ">100k-char structured text")
    assert_downstream_numerical_stability(z, ">100k-char structured text")


# ==============================================================================
# 7. MIDI Sequences: All Rests or Single Note
# ==============================================================================

def test_extreme_midi_single_note():
    """Verify single note event."""
    notes = [
        MusicNoteEvent(pitch=60, duration=1.0, velocity=64.0, ioi=1.0, onset=0.0)
    ]
    signals = extract_music_functor_signals(notes)
    assert signals.shape == (32, 4)
    assert np.all(np.isfinite(signals))

    z = extract_24d_v2(notes, use_mca=True)
    assert_valid_symplectic_24d(z, "MIDI single note")
    assert_downstream_numerical_stability(z, "MIDI single note")


def test_extreme_midi_all_rests():
    """Verify MIDI sequences representing all rests in various forms."""
    # 1. Notes with velocity = 0 (silent events / rest markers)
    silent_notes = [
        MusicNoteEvent(pitch=0, duration=1.0, velocity=0.0, ioi=1.0, onset=float(i))
        for i in range(8)
    ]
    z_silent = extract_24d_v2(silent_notes, use_mca=True)
    assert_valid_symplectic_24d(z_silent, "MIDI velocity=0 rests")
    assert_downstream_numerical_stability(z_silent, "MIDI velocity=0 rests")

    # 2. Large rest gaps between brief notes (mostly rests)
    sparse_notes = [
        MusicNoteEvent(pitch=60, duration=0.05, velocity=40.0, ioi=20.0, onset=0.0),
        MusicNoteEvent(pitch=64, duration=0.05, velocity=40.0, ioi=20.0, onset=20.0),
    ]
    z_sparse = extract_24d_v2(sparse_notes, use_mca=True)
    assert_valid_symplectic_24d(z_sparse, "MIDI large rest gaps")
    assert_downstream_numerical_stability(z_sparse, "MIDI large rest gaps")

    # 3. Empty list of notes (all rests / pure void)
    z_empty = extract_24d_v2([], use_mca=True)
    assert_valid_symplectic_24d(z_empty, "MIDI empty list (pure rests)")
    assert_downstream_numerical_stability(z_empty, "MIDI empty list (pure rests)")


# ==============================================================================
# 8. Ghost Padding Guard on 0-Length Signals Array
# ==============================================================================

def test_extreme_empty_signals_array():
    """Verify encode_signals on empty (0, 4) array returns valid state without infinite loop."""
    enc = WaveletHermiteEncoder()
    empty_sig = np.zeros((0, 4), dtype=np.float64)
    padded = enc.apply_open08_ghost_padding(empty_sig, min_len=16)
    assert padded.shape == (16, 4)
    assert np.all(np.isfinite(padded))

    z = enc.encode_signals(empty_sig, use_mca=True)
    assert_valid_symplectic_24d(z, "Empty (0, 4) signals array")
    assert_downstream_numerical_stability(z, "Empty (0, 4) signals array")
