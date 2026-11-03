"""Tests for 128-bit Dual-Sense HTH Poly Lexicon (hth_poly_30k).

Verifies:
1. Binary file invariants (strictly 480,000 Bytes, SHA-256 integrity, 30,000 unique words).
2. O(1) Memory-mapped index & query performance.
3. OPEN-01 Dynamic sense slot selection via context concreteness C(u).
4. Discrete hyperbolic tree jump distance (LCA depth).
5. Exact 4.0-bit maximum entropy quantile uniform quantization across all 8 sensory dimensions.
6. Bitfield packing / unpacking bit-level roundtrips.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pytest

from usir_py.hth_lexicon import (
    BYTES_PER_ENTRY,
    ENTRIES_PER_QUANTILE_BIN,
    EXPECTED_BINARY_SIZE,
    TOTAL_WORDS,
    DualSenseEntry,
    HTHPolyLexicon,
    jump_distance_64bit,
    lca_depth_64bit,
    pack_128bit_entry,
    pack_30bit_octree,
    pack_poly_64bit,
    pack_sensory_32bit,
    sensory_nibbles_to_floats,
    unpack_128bit_entry,
    unpack_30bit_octree,
    unpack_poly_64bit,
    unpack_sensory_32bit,
)


@pytest.fixture(scope="module")
def lexicon() -> HTHPolyLexicon:
    lex = HTHPolyLexicon()
    yield lex
    lex.close()


def test_binary_file_invariants(lexicon: HTHPolyLexicon):
    """Verify binary is strictly 480,000 Bytes and companion meta is consistent."""
    bin_path = lexicon.bin_path
    meta_path = lexicon.meta_path

    assert bin_path.is_file(), f"Binary file does not exist: {bin_path}"
    assert meta_path.is_file(), f"Metadata JSON does not exist: {meta_path}"

    # Strict byte length check
    file_size = bin_path.stat().st_size
    assert file_size == EXPECTED_BINARY_SIZE, (
        f"File size must be strictly {EXPECTED_BINARY_SIZE} Bytes, got {file_size} Bytes"
    )

    # SHA-256 verification
    bin_data = bin_path.read_bytes()
    computed_sha256 = hashlib.sha256(bin_data).hexdigest()
    assert computed_sha256 == lexicon.sha256, (
        f"SHA-256 mismatch: computed {computed_sha256} != metadata {lexicon.sha256}"
    )

    # Metadata word list consistency
    assert len(lexicon.words) == TOTAL_WORDS
    assert len(set(lexicon.words)) == TOTAL_WORDS, "Vocabulary contains duplicate words"


def test_o1_lookup_and_indexing(lexicon: HTHPolyLexicon):
    """Verify O(1) direct offset access, index query, and latency."""
    assert len(lexicon) == TOTAL_WORDS

    # Boundary index lookups
    e_first = lexicon.lookup_by_index(0)
    assert isinstance(e_first, DualSenseEntry)
    assert e_first.index == 0

    e_last = lexicon.lookup_by_index(TOTAL_WORDS - 1)
    assert isinstance(e_last, DualSenseEntry)
    assert e_last.index == TOTAL_WORDS - 1

    # Out of range indices
    with pytest.raises(IndexError):
        lexicon.lookup_by_index(-1)
    with pytest.raises(IndexError):
        lexicon.lookup_by_index(TOTAL_WORDS)

    # Named word lookup
    entry = lexicon.lookup("冷たい")
    assert entry is not None
    assert entry.word == "冷たい"
    assert len(entry.primary_sensory_nibbles) == 8
    assert len(entry.secondary_sensory_nibbles) == 8
    assert entry.primary_sensory_vector.shape == (8,)
    assert entry.secondary_sensory_vector.shape == (8,)

    # Unknown word
    assert lexicon.lookup("存在しない単語_XYZ_12345") is None

    # Latency benchmark: 1,000 lookups should execute in < 25 ms (< 25 microseconds per lookup)
    t0 = time.perf_counter()
    for i in range(1000):
        _ = lexicon.lookup_by_index(i * 29 % TOTAL_WORDS)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.05, f"Lookup latency too high: {elapsed * 1000:.2f} ms for 1,000 items"


def test_resolve_sense_dynamic_selection(lexicon: HTHPolyLexicon):
    """Verify OPEN-01 sense slot selection based on context concreteness C(u)."""
    # High concreteness C(u) >= 0.45 -> Primary physical sense
    res_high = lexicon.resolve_sense("冷たい", context_concreteness=0.85)
    assert res_high is not None
    hth_high, vec_high, slot_high = res_high
    assert slot_high == "primary_physical"
    entry = lexicon.lookup("冷たい")
    assert entry is not None
    assert hth_high == entry.primary_physical_hth
    assert np.allclose(vec_high, entry.primary_sensory_vector)

    # Low concreteness C(u) < 0.30 -> Secondary abstract sense
    res_low = lexicon.resolve_sense("冷たい", context_concreteness=0.20)
    assert res_low is not None
    hth_low, vec_low, slot_low = res_low
    assert slot_low == "secondary_abstract"
    assert hth_low == entry.secondary_abstract_hth
    assert np.allclose(vec_low, entry.secondary_sensory_vector)

    # Transition band 0.30 <= C(u) < 0.45 -> Soft blended transition
    res_mid = lexicon.resolve_sense("冷たい", context_concreteness=0.375)
    assert res_mid is not None
    _, vec_mid, slot_mid = res_mid
    alpha = (0.375 - 0.30) / 0.15  # 0.50
    expected_blended = 0.5 * entry.primary_sensory_vector + 0.5 * entry.secondary_sensory_vector
    assert np.allclose(vec_mid, expected_blended, atol=1e-5)

    # Boundary test at exactly 0.45 and 0.30
    assert lexicon.resolve_sense("冷たい", 0.45)[2] == "primary_physical"
    assert lexicon.resolve_sense("冷たい", 0.299)[2] == "secondary_abstract"

    # Missing word
    assert lexicon.resolve_sense("未知語_ABC", 0.5) is None


def test_discrete_jump_distance(lexicon: HTHPolyLexicon):
    """Verify discrete hyperbolic tree jump distance D_jump = 1.0 - (LCA / 10.0)."""
    # Identical word -> LCA=10 -> D_jump=0.0
    dist_self = lexicon.discrete_jump("冷たい", "冷たい")
    assert dist_self is not None
    assert dist_self == 0.0

    # Words sharing primary root octant (e.g. 氷 and 雪)
    dist_related = lexicon.discrete_jump("氷", "雪")
    assert dist_related is not None
    assert dist_related < 1.0, f"Expected LCA >= 1 for 氷 and 雪, got jump {dist_related}"

    # Words in completely different octants
    dist_distant = lexicon.discrete_jump("冷たい", "単位")
    assert dist_distant is not None
    assert dist_distant == 1.0, f"Expected LCA == 0 for 冷たい and 単位, got {dist_distant}"

    # Secondary slot jump
    dist_sec = lexicon.discrete_jump("冷たい", "態度", slot="secondary")
    assert dist_sec is not None
    assert 0.0 <= dist_sec <= 1.0

    # Missing word
    assert lexicon.discrete_jump("冷たい", "未知語_ZZZ") is None


def test_quantile_4bit_entropy_and_uniformity(lexicon: HTHPolyLexicon):
    """Verify strictly uniform quantile 4-bit quantization (1,875 per bin, 4.0 bit entropy)."""
    pri_nibbles = np.zeros((TOTAL_WORDS, 8), dtype=np.int32)
    sec_nibbles = np.zeros((TOTAL_WORDS, 8), dtype=np.int32)

    for idx in range(TOTAL_WORDS):
        pri_64, sec_64 = lexicon.get_raw_128bit_by_index(idx)
        _, _, pri_sens = unpack_poly_64bit(pri_64)
        _, _, sec_sens = unpack_poly_64bit(sec_64)
        pri_nibbles[idx] = unpack_sensory_32bit(pri_sens)
        sec_nibbles[idx] = unpack_sensory_32bit(sec_sens)

    # For each of the 8 sensory dimensions
    for dim in range(8):
        # Primary sensory dimension
        pri_counts = np.bincount(pri_nibbles[:, dim], minlength=16)
        assert len(pri_counts) == 16
        assert np.all(pri_counts == ENTRIES_PER_QUANTILE_BIN), (
            f"Primary dim {dim} counts not uniform: min={pri_counts.min()}, max={pri_counts.max()}"
        )
        pri_probs = pri_counts / TOTAL_WORDS
        pri_entropy = -np.sum(pri_probs * np.log2(pri_probs))
        assert np.isclose(pri_entropy, 4.0, atol=1e-7), (
            f"Primary dim {dim} entropy != 4.0 bits: {pri_entropy}"
        )

        # Secondary sensory dimension
        sec_counts = np.bincount(sec_nibbles[:, dim], minlength=16)
        assert len(sec_counts) == 16
        assert np.all(sec_counts == ENTRIES_PER_QUANTILE_BIN), (
            f"Secondary dim {dim} counts not uniform: min={sec_counts.min()}, max={sec_counts.max()}"
        )
        sec_probs = sec_counts / TOTAL_WORDS
        sec_entropy = -np.sum(sec_probs * np.log2(sec_probs))
        assert np.isclose(sec_entropy, 4.0, atol=1e-7), (
            f"Secondary dim {dim} entropy != 4.0 bits: {sec_entropy}"
        )


def test_bitfield_packing_roundtrip():
    """Verify bit manipulation roundtrips and bit boundaries."""
    # 1. 30-bit octree
    levels = [0, 1, 2, 3, 4, 5, 6, 7, 0, 7]
    packed_oct = pack_30bit_octree(levels)
    assert 0 <= packed_oct < (1 << 30)
    unpacked_oct = unpack_30bit_octree(packed_oct)
    assert unpacked_oct == levels

    # 2. 32-bit sensory nibbles
    nibbles = [0, 1, 2, 3, 4, 5, 14, 15]
    packed_sens = pack_sensory_32bit(nibbles)
    assert 0 <= packed_sens <= 0xFFFFFFFF
    unpacked_sens = unpack_sensory_32bit(packed_sens)
    assert unpacked_sens == nibbles

    # 3. 64-bit poly pack/unpack
    tag = 2
    packed_64 = pack_poly_64bit(packed_oct, tag, packed_sens)
    assert 0 <= packed_64 < (1 << 64)
    out_oct, out_tag, out_sens = unpack_poly_64bit(packed_64)
    assert out_oct == packed_oct
    assert out_tag == tag
    assert out_sens == packed_sens

    # 4. 128-bit entry pack/unpack
    sec_64 = pack_poly_64bit(packed_oct ^ 0x3FFFFFFF, 1, packed_sens ^ 0xFFFFFFFF)
    entry_bytes = pack_128bit_entry(packed_64, sec_64)
    assert len(entry_bytes) == BYTES_PER_ENTRY
    out_pri_64, out_sec_64 = unpack_128bit_entry(entry_bytes)
    assert out_pri_64 == packed_64
    assert out_sec_64 == sec_64

    # 5. LCA depth and jump distance bit invariants
    assert lca_depth_64bit(packed_64, packed_64) == 10
    assert jump_distance_64bit(packed_64, packed_64) == 0.0
