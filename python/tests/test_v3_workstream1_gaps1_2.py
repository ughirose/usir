"""Workstream 1 Test Suite: Gap 1 & Gap 2.

Tests:
1. Gap 1: 128-bit Dual-Sense HTH Lexicon & Dynamic Contextual Resolution.
   - DualSenseHthEntry dataclass with primary_physical_hth (64-bit) & secondary_abstract_hth (64-bit).
   - All 20 polysemous words registered and validated.
   - Dynamic resolution based on context concreteness threshold C(u) = 0.52:
     - Concrete context (C >= 0.52: 氷, 瓦, 馬, 山, 石) -> primary_physical slot.
     - Abstract context (C < 0.52: 制度, 論理, 態度, 単位, 計画) -> secondary_abstract slot.
   - Return dictionary / tuple formats.
2. Gap 2: Logarithmic Self-Similar Renormalization & Multiscale Wavelet Filters.
   - compute_logarithmic_scales across arbitrary lengths M in [3, 2048].
   - renormalize_decimate_signals RG decimation for M > 1024:
     - Mean field coarse-graining on channels 0..2.
     - Peak/jump envelope preservation on channel 3.
   - Ultra-long sequence (M = 2048) Beginning-Middle-End return arch preservation (Sg-10 At / i_arch).
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from usir_py.v2.hth_lexicon import (
    DUAL_SENSE_LEXICON,
    DualSenseHthEntry,
    HTHEntry,
    HTHLexicon,
    get_dual_sense_entry,
    pack_64bit_hth,
    pack_lower32_sensory,
    resolve_dual_sense_hth,
    unpack_64bit_hth,
    unpack_lower32_sensory,
)
from usir_py.v2.wavelet_hermite_encoder import (
    WaveletHermiteEncoder,
    compute_logarithmic_scales,
    compute_rqa_metrics,
    extract_24d_v2,
    renormalize_decimate_signals,
)


POLYSEMOUS_20_WORDS = [
    "冷たい", "落とす", "走る", "深い", "高い",
    "重い", "光る", "闇", "切る", "立つ",
    "熱い", "甘い", "固い", "開く", "閉じる",
    "浮かぶ", "沈む", "揺れる", "響く", "掴む",
]


# ==============================================================================
# Gap 1: 128-bit Dual-Sense HTH Tests
# ==============================================================================

def test_gap1_all_20_polysemous_words_registered():
    """Verify all 20 specified polysemous words have 128-bit DualSenseHthEntry."""
    assert len(DUAL_SENSE_LEXICON) == 20
    for word in POLYSEMOUS_20_WORDS:
        entry = get_dual_sense_entry(word)
        assert entry is not None, f"Word '{word}' missing from DUAL_SENSE_LEXICON"
        assert isinstance(entry, DualSenseHthEntry)
        assert entry.word == word

        # 64-bit integer properties
        pri_hth = entry.primary_physical_hth
        sec_hth = entry.secondary_abstract_hth
        assert isinstance(pri_hth, int) and 0 <= pri_hth < (1 << 64)
        assert isinstance(sec_hth, int) and 0 <= sec_hth < (1 << 64)
        assert pri_hth != sec_hth, f"Primary and secondary HTH must differ for '{word}'"

        # Concreteness separation
        assert entry.primary_entry.concreteness >= 0.70, (
            f"Primary physical entry for '{word}' must have high concreteness >= 0.70, "
            f"got {entry.primary_entry.concreteness}"
        )
        assert entry.secondary_entry.concreteness < 0.40, (
            f"Secondary abstract entry for '{word}' must have low concreteness < 0.40, "
            f"got {entry.secondary_entry.concreteness}"
        )

        # Sensory vector dimension
        assert entry.primary_entry.sensory_vector.shape == (8,)
        assert entry.secondary_entry.sensory_vector.shape == (8,)


def test_gap1_bit_packing_and_unpacking_roundtrip():
    """Verify 64-bit pack/unpack roundtrip fidelity."""
    pos = 2
    octree = 0x20049249
    vec = np.array([0.3, 0.1, 0.9, 0.7, 0.2, 0.4, 0.0, 0.2], dtype=np.float64)

    lower32 = pack_lower32_sensory(vec)
    assert 0 <= lower32 <= 0xFFFFFFFF
    vec_unpacked = unpack_lower32_sensory(lower32)
    # Quantized to 4-bit nibbles (step = 1/15)
    assert np.all(np.abs(vec - vec_unpacked) <= (1.0 / 15.0 + 1e-6))

    hth64 = pack_64bit_hth(pos, octree, lower32)
    assert 0 <= hth64 < (1 << 64)
    p_out, oct_out, vec_out = unpack_64bit_hth(hth64)
    assert p_out == pos
    assert oct_out == (octree & 0x3FFFFFFF)
    assert np.allclose(vec_out, vec_unpacked)


def test_gap1_contextual_dynamic_resolution_concrete():
    """Verify dynamic selection of Primary Physical slot when C(context) >= 0.52."""
    lex = HTHLexicon()
    concrete_pairs = [
        ("冷たい", "氷"),  # 氷 (ice) is concrete thermal/nature
        ("落とす", "瓦"),  # 瓦 (roof tile) is concrete artifact
        ("走る", "馬"),    # 馬 (horse) is concrete bio/fauna
        ("高い", "山"),    # 山 (mountain) is concrete spatial
        ("固い", "石"),    # 石 (stone) is concrete solid material
    ]

    for word, ctx in concrete_pairs:
        res = resolve_dual_sense_hth(word, ctx, lexicon=lex)
        c_u = res["context_concreteness"]
        assert c_u >= 0.52, f"Expected concrete context '{ctx}' to have C >= 0.52, got {c_u}"
        assert res["selected_slot"] == "primary_physical"

        entry = get_dual_sense_entry(word)
        assert entry is not None
        assert res["selected_hth"] == entry.primary_physical_hth
        assert res["primary_64bit"] == entry.primary_physical_hth
        assert res["secondary_64bit"] == entry.secondary_abstract_hth
        assert res["full_128bit"] == (entry.primary_physical_hth, entry.secondary_abstract_hth)
        assert np.allclose(res["effective_sensory"], entry.primary_entry.sensory_vector)


def test_gap1_contextual_dynamic_resolution_abstract():
    """Verify dynamic selection of Secondary Abstract slot when C(context) < 0.52."""
    lex = HTHLexicon()
    abstract_pairs = [
        ("冷たい", "制度"),  # 制度 (institution)
        ("落とす", "論理"),  # 論理 (logic)
        ("冷たい", "態度"),  # 態度 (attitude)
        ("落とす", "単位"),  # 単位 (credit / academic unit)
        ("走る", "計画"),    # 計画 (plan)
    ]

    for word, ctx in abstract_pairs:
        res = resolve_dual_sense_hth(word, ctx, lexicon=lex)
        c_u = res["context_concreteness"]
        assert c_u < 0.52, f"Expected abstract context '{ctx}' to have C < 0.52, got {c_u}"
        assert res["selected_slot"] == "secondary_abstract"

        entry = get_dual_sense_entry(word)
        assert entry is not None
        assert res["selected_hth"] == entry.secondary_abstract_hth
        assert res["primary_64bit"] == entry.primary_physical_hth
        assert res["secondary_64bit"] == entry.secondary_abstract_hth
        assert res["full_128bit"] == (entry.primary_physical_hth, entry.secondary_abstract_hth)
        assert np.allclose(res["effective_sensory"], entry.secondary_entry.sensory_vector)


def test_gap1_hth_lexicon_class_method():
    """Verify resolve_dual_sense_hth and get_concreteness methods on HTHLexicon instance."""
    lex = HTHLexicon()
    assert lex.get_concreteness("氷") >= 0.52
    assert lex.get_concreteness("制度") < 0.52

    res_ice = lex.resolve_dual_sense_hth("冷たい", "氷")
    assert res_ice["selected_slot"] == "primary_physical"

    res_inst = lex.resolve_dual_sense_hth("冷たい", "制度")
    assert res_inst["selected_slot"] == "secondary_abstract"


def test_gap1_fallback_for_non_polysemous_word():
    """Verify resolve_dual_sense_hth gracefully handles words outside the 20-word list."""
    lex = HTHLexicon()
    res = resolve_dual_sense_hth("木", "森", lexicon=lex)
    assert res["selected_slot"] in ("primary_physical", "secondary_abstract")
    assert isinstance(res["selected_hth"], int)
    assert len(res["full_128bit"]) == 2
    assert res["effective_sensory"].shape == (8,)


# ==============================================================================
# Gap 2: Logarithmic Self-Similar Renormalization Tests
# ==============================================================================

def test_gap2_compute_logarithmic_scales():
    """Verify logarithmic scales and dynamic wavelet levels across sequence lengths."""
    # 1. Short haiku M = 3
    s3 = compute_logarithmic_scales(3)
    assert s3["m"] == 3
    assert math.isclose(s3["log_m"], math.log2(3), rel_tol=1e-5)
    assert s3["tertiles"] == (s3["scale_micro"], s3["scale_meso"], s3["scale_macro"])
    assert s3["scale_micro"] < s3["scale_meso"] < s3["scale_macro"]
    assert s3["micro_level"] >= 1 and s3["meso_level"] > s3["micro_level"] and s3["macro_level"] > s3["meso_level"]

    # 2. Medium text M = 64
    s64 = compute_logarithmic_scales(64)
    assert s64["log_m"] == 6.0
    assert math.isclose(s64["scale_micro"], 2.0)
    assert math.isclose(s64["scale_meso"], 4.0)
    assert math.isclose(s64["scale_macro"], 6.0)

    # 3. Ultra-long novel M = 2048
    s2048 = compute_logarithmic_scales(2048)
    assert s2048["log_m"] == 11.0
    assert math.isclose(s2048["scale_micro"], 11.0 / 3.0)
    assert math.isclose(s2048["scale_meso"], 22.0 / 3.0)
    assert math.isclose(s2048["scale_macro"], 11.0)


def test_gap2_renormalize_decimate_signals_invariance_for_short():
    """Verify signals with length <= 1024 are returned unchanged by decimation."""
    sig = np.random.RandomState(42).rand(512, 4)
    dec = renormalize_decimate_signals(sig, max_length=1024)
    assert np.array_equal(sig, dec)


def test_gap2_renormalize_decimate_signals_coarse_graining_rules():
    """Verify 2-clause coarse-graining: mean fields on 0..2 and peak envelope on 3."""
    # Construct exact 4-row signal
    sig = np.array([
        [0.2, 0.4, 0.6, 0.1],
        [0.4, 0.6, 0.8, 0.9],  # Pair 1: mean=[0.3, 0.5, 0.7], peak=0.9
        [0.1, 0.3, 0.5, 0.8],
        [0.3, 0.5, 0.7, 0.2],  # Pair 2: mean=[0.2, 0.4, 0.6], peak=0.8
    ], dtype=np.float64)

    # Decimate with max_length=2 (1 coarse-graining step)
    dec = renormalize_decimate_signals(sig, max_length=2)
    assert dec.shape == (2, 4)

    # Check pair 1
    assert np.allclose(dec[0, :3], [0.3, 0.5, 0.7])
    assert math.isclose(dec[0, 3], 0.9)  # Preserved local peak!

    # Check pair 2
    assert np.allclose(dec[1, :3], [0.2, 0.4, 0.6])
    assert math.isclose(dec[1, 3], 0.8)  # Preserved local peak!


def test_gap2_renormalize_decimate_large_odd_and_even_lengths():
    """Verify decimation loop handles large lengths (2048, 3000) down to <= 1024."""
    rng = np.random.RandomState(123)
    sig_2048 = rng.rand(2048, 4)
    dec_2048 = renormalize_decimate_signals(sig_2048, max_length=1024)
    assert dec_2048.shape == (1024, 4)

    sig_3000 = rng.rand(3000, 4)
    dec_3000 = renormalize_decimate_signals(sig_3000, max_length=1024)
    assert dec_3000.shape[0] <= 1024
    # 3000 -> 1500 -> 750
    assert dec_3000.shape == (750, 4)


def test_gap2_ultra_long_narrative_arch_preservation_m2048():
    """Verify that for M = 2048 with beginning-middle-end arch structure,
    renormalization decimation preserves the global narrative return arc (Sg-10 At),
    whereas naive truncation signals[:1024] destroys it!
    """
    m = 2048
    sig = np.zeros((m, 4), dtype=np.float64)

    # Lattice channel (col 1):
    # - Opening theme in early quarter (0..511): high recurrence pattern around 0.85
    # - Middle development (512..1535): divergent pattern around 0.15
    # - Return arch in late quarter (1536..2047): returns to opening theme around 0.85
    t = np.arange(512, dtype=np.float64)
    theme = 0.85 + 0.05 * np.sin(2.0 * math.pi * t / 64.0)
    contrast = 0.15 + 0.05 * np.cos(2.0 * math.pi * t / 32.0)

    sig[:512, 1] = theme
    sig[512:1024, 1] = contrast
    sig[1024:1536, 1] = contrast
    sig[1536:2048, 1] = theme  # Thematic return!

    # Channels 0, 2, 3 have baseline signals
    sig[:, 0] = 0.5
    sig[:, 2] = 0.4
    sig[:, 3] = 0.1
    # Place a shock spike in the resolution section
    sig[1800, 3] = 0.95

    # 1. Under naive truncation signals[:1024]
    sig_naive = sig[:1024]
    rqa_naive = compute_rqa_metrics(sig_naive[:, 1])
    # The late quarter of naive slice (768..1023) is middle contrast (0.15), not the theme (0.85)!
    # Therefore, return arch recurrence between early and late quarters is near zero
    assert rqa_naive.i_arch < 0.15, (
        f"Naive truncation should miss the narrative return, but got i_arch={rqa_naive.i_arch}"
    )

    # 2. Under renormalization decimation
    sig_decimated = renormalize_decimate_signals(sig, max_length=1024)
    assert sig_decimated.shape == (1024, 4)
    rqa_renorm = compute_rqa_metrics(sig_decimated[:, 1])

    # The late quarter of decimated signal (768..1023) contains the coarse-grained return theme!
    # Therefore, return arch recurrence between early and late quarters is strongly preserved
    assert rqa_renorm.i_arch >= 0.35, (
        f"Renormalization must preserve return arch (i_arch >= 0.35), got {rqa_renorm.i_arch}"
    )
    assert rqa_renorm.i_arch > rqa_naive.i_arch * 2.0

    # 3. Verify the local peak on channel 3 was preserved through decimation
    assert np.max(sig_decimated[:, 3]) == pytest.approx(0.95, abs=1e-3)

    # 4. Verify full 24D extraction operates on M = 2048 without error and yields valid state
    z = extract_24d_v2(sig, use_mca=False)
    assert z.shape == (24,)
    assert np.all((z[:12] >= 0.0) & (z[:12] <= 1.0))
    assert np.all(z[12:] >= 0.0)
    # Coordinate 9 (Sg-10 At: 0.5 * det + 0.5 * i_arch) reflects the preserved arch
    assert z[9] > 0.15
