"""128-bit Dual-Sense HTH (Hyperbolic-Thermodynamic Hash) Lexicon (hth_poly_30k).

Strictly 480,000 Bytes (30,000 words x 16 Bytes).
Architecture Reference:
- 10_SPEC_PHASE_A_HEXAGONAL_AND_HTH_LEXICON.md Section 1.1
- usir_adversarial_stress_test_and_feasibility_audit.md Section 24.2

Layout per entry (16 Bytes / uint64 x 2, Little-Endian):
├── High 64 bits (Bytes 0..7): Primary Physical Sense (第一語義：字義通りの身体・物理感覚)
│   ├── [bit 63..34 (30b)]: Octree Node ID 1 (物理空間クラスタ階層 10階層 x 3bit)
│   ├── [bit 33..32 (2b)] : POS Tag (0b00:名詞, 0b01:動詞, 0b10:形容詞, 0b11:副詞・その他)
│   └── [bit 31..00 (32b)]: 8次元感覚アフォーダンス q_1..q_8 (各4bit: 0x0..0xF)
└── Low 64 bits (Bytes 8..15): Secondary Abstract / Figurative Sense (第二語義：抽象転用・制度・心理)
    ├── [bit 63..34 (30b)]: Octree Node ID 2 (抽象概念クラスタ階層 10階層 x 3bit)
    ├── [bit 33..32 (2b)] : Contextual Flag (0b00:多義なし, 0b01:感情転用, 0b10:制度転用, 0b11:機能語化)
    └── [bit 31..00 (32b)]: 抽象感覚アフォーダンス (通常は低ノルム、心理情動のみ保持)
"""

from __future__ import annotations

import json
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Total entries and binary constants
TOTAL_WORDS = 30_000
BYTES_PER_ENTRY = 16
EXPECTED_BINARY_SIZE = TOTAL_WORDS * BYTES_PER_ENTRY  # Exactly 480,000 Bytes
ENTRIES_PER_QUANTILE_BIN = TOTAL_WORDS // 16         # Exactly 1,875 words per bin


# ==============================================================================
# Bit Manipulation and Tree Operations
# ==============================================================================

def pack_30bit_octree(levels_10: Sequence[int]) -> int:
    """Pack 10 octal digits (each in 0..7) into a 30-bit unsigned integer (MSB = level 1)."""
    code = 0
    for lvl in range(min(10, len(levels_10))):
        digit = int(levels_10[lvl]) & 0x7
        shift = 3 * (9 - lvl)
        code |= digit << shift
    return code & 0x3FFFFFFF


def unpack_30bit_octree(code: int) -> List[int]:
    """Unpack 30-bit unsigned integer into 10 octal digits (levels 1..10)."""
    code = int(code) & 0x3FFFFFFF
    levels: List[int] = []
    for lvl in range(10):
        shift = 3 * (9 - lvl)
        digit = (code >> shift) & 0x7
        levels.append(digit)
    return levels


def pack_sensory_32bit(nibbles: Sequence[int]) -> int:
    """Pack 8 x 4-bit nibbles (q_1..q_8, each 0x0..0xF) into lower 32 bits."""
    lower32 = 0
    for d in range(min(8, len(nibbles))):
        val = int(nibbles[d]) & 0xF
        lower32 |= val << (4 * d)
    return lower32 & 0xFFFFFFFF


def unpack_sensory_32bit(sensory_32bit: int) -> List[int]:
    """Unpack lower 32-bit integer into 8 x 4-bit nibbles (q_1..q_8)."""
    lower32 = int(sensory_32bit) & 0xFFFFFFFF
    nibbles: List[int] = []
    for d in range(8):
        nibble = (lower32 >> (4 * d)) & 0xF
        nibbles.append(nibble)
    return nibbles


def sensory_nibbles_to_floats(nibbles: Sequence[int]) -> np.ndarray:
    """Convert 8 nibbles (0..15) to float array normalized to [0.0, 1.0]."""
    return np.array([float(n) / 15.0 for n in nibbles], dtype=np.float64)


def pack_poly_64bit(octree_30bit: int, tag_2bit: int, sensory_32bit: int) -> int:
    """Pack 30-bit Octree (bits 63..34), 2-bit tag/flag (bits 33..32), and 32-bit sensory (bits 31..0).

    bit 63..34 (30b): Octree Node ID
    bit 33..32 (2b) : POS Tag (Primary) or Contextual Flag (Secondary)
    bit 31..00 (32b): 8D 4-bit sensory nibbles
    """
    oct_part = (int(octree_30bit) & 0x3FFFFFFF) << 34
    tag_part = (int(tag_2bit) & 0x3) << 32
    sens_part = int(sensory_32bit) & 0xFFFFFFFF
    return oct_part | tag_part | sens_part


def unpack_poly_64bit(val: int) -> Tuple[int, int, int]:
    """Unpack 64-bit HTH into (octree_30bit, tag_2bit, sensory_32bit)."""
    h = int(val) & 0xFFFFFFFFFFFFFFFF
    octree_30bit = (h >> 34) & 0x3FFFFFFF
    tag_2bit = (h >> 32) & 0x3
    sensory_32bit = h & 0xFFFFFFFF
    return octree_30bit, tag_2bit, sensory_32bit


def pack_128bit_entry(pri_64: int, sec_64: int) -> bytes:
    """Serialize 128-bit entry into 16 bytes little-endian (<QQ)."""
    return struct.pack('<QQ', int(pri_64) & 0xFFFFFFFFFFFFFFFF, int(sec_64) & 0xFFFFFFFFFFFFFFFF)


def unpack_128bit_entry(data: bytes) -> Tuple[int, int]:
    """Deserialize 16 bytes into (primary_physical_hth, secondary_abstract_hth)."""
    return struct.unpack('<QQ', data[:16])


def lca_depth_64bit(h1: int, h2: int) -> int:
    """Lowest Common Ancestor (LCA) depth in 10-level 3-bit octree using bits 63..34.

    Returns an integer in 0..10.
    """
    xor_diff = (int(h1) ^ int(h2)) & 0xFFFFFFFFFFFFFFFF
    if xor_diff == 0:
        return 10
    lzcnt64 = 64 - xor_diff.bit_length()
    matching_octree_bits = min(30, lzcnt64)
    return matching_octree_bits // 3


def jump_distance_64bit(h1: int, h2: int) -> float:
    """Discrete hyperbolic tree jump distance D_jump(u, v) = 1.0 - (LCA / 10.0) in [0.0, 1.0]."""
    return 1.0 - (lca_depth_64bit(h1, h2) / 10.0)


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass(frozen=True)
class DualSenseEntry:
    """128-bit Dual-Sense entry representation for a word."""
    word: str
    index: int
    primary_physical_hth: int
    secondary_abstract_hth: int
    primary_octree: int
    primary_pos: int
    primary_sensory_nibbles: List[int]
    primary_sensory_vector: np.ndarray
    secondary_octree: int
    secondary_flag: int
    secondary_sensory_nibbles: List[int]
    secondary_sensory_vector: np.ndarray

    @property
    def primary_entry(self) -> int:
        return self.primary_physical_hth

    @property
    def secondary_entry(self) -> int:
        return self.secondary_abstract_hth


# ==============================================================================
# HTHPolyLexicon Runtime Class
# ==============================================================================

class HTHPolyLexicon:
    """O(1) memory-mapped reader for 480KB 128-bit Dual-Sense Lexicon hth_poly_30k.bin."""

    def __init__(
        self,
        bin_path: Optional[Union[Path, str]] = None,
        meta_path: Optional[Union[Path, str]] = None,
    ) -> None:
        base_dir = Path(__file__).resolve().parent / "data"
        self.bin_path = Path(bin_path) if bin_path else (base_dir / "hth_poly_30k.bin")
        self.meta_path = Path(meta_path) if meta_path else (base_dir / "hth_poly_30k_meta.json")

        if not self.bin_path.is_file():
            raise FileNotFoundError(f"HTH poly binary file not found: {self.bin_path}")

        file_size = self.bin_path.stat().st_size
        if file_size != EXPECTED_BINARY_SIZE:
            raise ValueError(
                f"Binary size mismatch: expected strictly {EXPECTED_BINARY_SIZE} Bytes, got {file_size} Bytes"
            )

        # Open and memory map
        self._file_obj = open(self.bin_path, "rb")
        self._mmap = mmap.mmap(self._file_obj.fileno(), 0, access=mmap.ACCESS_READ)

        # Load metadata / word index
        self.word_to_index: Dict[str, int] = {}
        self.words: List[str] = []
        if self.meta_path.is_file():
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            self.words = meta.get("words", [])
            self.word_to_index = {w: i for i, w in enumerate(self.words)}
            self.sha256 = meta.get("sha256", "")
            self.version = meta.get("version", "1.0.0")
        else:
            self.sha256 = ""
            self.version = "1.0.0"

    def close(self) -> None:
        if hasattr(self, "_mmap") and self._mmap is not None:
            self._mmap.close()
        if hasattr(self, "_file_obj") and self._file_obj is not None:
            self._file_obj.close()

    def __enter__(self) -> HTHPolyLexicon:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __len__(self) -> int:
        return TOTAL_WORDS

    def __contains__(self, word: str) -> bool:
        return word in self.word_to_index

    def get_index(self, word: str) -> Optional[int]:
        return self.word_to_index.get(word)

    def get_raw_128bit_by_index(self, idx: int) -> Tuple[int, int]:
        """Read 128-bit entry at index in strictly O(1) time."""
        if not (0 <= idx < TOTAL_WORDS):
            raise IndexError(f"Index {idx} out of range [0, {TOTAL_WORDS})")
        offset = idx * BYTES_PER_ENTRY
        return unpack_128bit_entry(self._mmap[offset : offset + BYTES_PER_ENTRY])

    def get_raw_128bit(self, word: str) -> Optional[Tuple[int, int]]:
        idx = self.word_to_index.get(word)
        if idx is None:
            return None
        return self.get_raw_128bit_by_index(idx)

    def lookup_by_index(self, idx: int) -> DualSenseEntry:
        """Construct full DualSenseEntry for index in O(1) time."""
        pri_64, sec_64 = self.get_raw_128bit_by_index(idx)
        word = self.words[idx] if idx < len(self.words) else f"word_{idx}"

        pri_oct, pri_pos, pri_sens = unpack_poly_64bit(pri_64)
        pri_nibs = unpack_sensory_32bit(pri_sens)
        pri_vec = sensory_nibbles_to_floats(pri_nibs)

        sec_oct, sec_flag, sec_sens = unpack_poly_64bit(sec_64)
        sec_nibs = unpack_sensory_32bit(sec_sens)
        sec_vec = sensory_nibbles_to_floats(sec_nibs)

        return DualSenseEntry(
            word=word,
            index=idx,
            primary_physical_hth=pri_64,
            secondary_abstract_hth=sec_64,
            primary_octree=pri_oct,
            primary_pos=pri_pos,
            primary_sensory_nibbles=pri_nibs,
            primary_sensory_vector=pri_vec,
            secondary_octree=sec_oct,
            secondary_flag=sec_flag,
            secondary_sensory_nibbles=sec_nibs,
            secondary_sensory_vector=sec_vec,
        )

    def lookup(self, word: str) -> Optional[DualSenseEntry]:
        """O(1) dictionary query returning full DualSenseEntry."""
        if not word:
            return None
        idx = self.word_to_index.get(word)
        if idx is None:
            idx = self.word_to_index.get(word.strip())
        if idx is None:
            return None
        return self.lookup_by_index(idx)

    def resolve_sense(
        self,
        word: str,
        context_concreteness: float,
        apply_damping: bool = False,
    ) -> Optional[Tuple[int, np.ndarray, str]]:
        """Dynamic sense slot selection governed by context concreteness C(u) (OPEN-01).

        - If C(u) >= 0.45: select High 64-bit (Primary Physical Sense)
        - If C(u) < 0.30:  select Low 64-bit (Secondary Abstract Sense)
        - If 0.30 <= C(u) < 0.45: soft blended transition
        - If apply_damping is True: scales secondary sensory components by C(u) (OPEN-01 attenuation product).
        """
        entry = self.lookup(word)
        if entry is None:
            return None

        # Guard against NaN / Inf
        if np.isnan(context_concreteness) or np.isinf(context_concreteness):
            c_u = 0.50
        else:
            c_u = float(np.clip(context_concreteness, 0.0, 1.0))

        if c_u >= 0.45:
            return entry.primary_physical_hth, entry.primary_sensory_vector, "primary_physical"
        elif c_u < 0.30:
            vec = entry.secondary_sensory_vector * c_u if apply_damping else entry.secondary_sensory_vector
            return entry.secondary_abstract_hth, vec, "secondary_abstract"
        else:
            # Linear interpolation in transition band [0.30, 0.45)
            alpha = (c_u - 0.30) / 0.15
            sec_vec = entry.secondary_sensory_vector * c_u if apply_damping else entry.secondary_sensory_vector
            blended_vec = alpha * entry.primary_sensory_vector + (1.0 - alpha) * sec_vec
            chosen_hth = entry.primary_physical_hth if alpha >= 0.5 else entry.secondary_abstract_hth
            slot = "primary_physical" if alpha >= 0.5 else "secondary_abstract"
            return chosen_hth, blended_vec, slot

    def discrete_jump(
        self,
        word_u: str,
        word_v: str,
        slot: str = "primary",
        slot_v: Optional[str] = None,
    ) -> Optional[float]:
        """Calculate discrete hyperbolic tree jump distance between two words.

        Accepts slot='primary'/'primary_physical' or 'secondary'/'secondary_abstract'.
        """
        entry_u = self.lookup(word_u)
        entry_v = self.lookup(word_v)
        if entry_u is None or entry_v is None:
            return None

        def _pick_hth(entry: DualSenseEntry, s: str) -> int:
            s_clean = s.lower().strip()
            if s_clean in ("primary", "primary_physical", "physical", "pri", "1"):
                return entry.primary_physical_hth
            elif s_clean in ("secondary", "secondary_abstract", "abstract", "sec", "2"):
                return entry.secondary_abstract_hth
            raise ValueError(
                f"Unknown HTH sense slot '{s}'. Must be 'primary'/'primary_physical' or 'secondary'/'secondary_abstract'."
            )

        hu = _pick_hth(entry_u, slot)
        hv = _pick_hth(entry_v, slot_v if slot_v is not None else slot)
        return jump_distance_64bit(hu, hv)
