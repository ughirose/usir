"""Zero-Leakage 64-bit HTH Lexicon & OPEN-01..04 Linguistic Frontend.

Implements Module 1 of USIR v2.0 Paper-Grade Architecture:
1. 256-node Laplacian-diffused HTH ontology loader & 30-bit Octree bit operations.
2. Continuous Poincaré hyperbolic heat-kernel distance.
3. Zero-cheat morphological analyzer via Janome + closed grammatical functional classes.
4. OPEN-01: Contextual Polysemy Damping Gate.
5. OPEN-02: Dual Negation/Modality Sensory Tensor.
6. OPEN-03: Onomatopoeia Sound-Symbolic Synthesis (\u03a8_mimetic) with Mimetic Immunity.
7. OPEN-04: Narrative (S_nar) vs Dialogue (S_dia) Orthogonal 2-Channel Separation.
8. M1 Music Functor: Symbolic music note events -> (M, 4) base structural signals.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from janome.tokenizer import Tokenizer


# ==============================================================================
# Bit Manipulation and Tree Operations
# ==============================================================================

def clz30(x: int) -> int:
    """Count leading zeros in a 30-bit unsigned integer (0 <= x < (1 << 30))."""
    x = int(x) & 0x3FFFFFFF
    if x == 0:
        return 30
    return 30 - x.bit_length()


def lca_depth_30bit(h_u: int, h_v: int) -> int:
    """Lowest Common Ancestor (LCA) depth in 10-level 3-bit octree (returns 0..10)."""
    xor_diff = (int(h_u) ^ int(h_v)) & 0x3FFFFFFF
    return clz30(xor_diff) // 3


def discrete_jump_distance(h_u: int, h_v: int) -> float:
    """Discrete tree jump distance D_jump(u, v) = 1.0 - lca_depth / 10.0 in [0.0, 1.0]."""
    return 1.0 - (lca_depth_30bit(h_u, h_v) / 10.0)


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class HTHEntry:
    """64-bit Hyperbolic-Thermodynamic Hash entry."""
    token: str
    pos: str
    h_code: int                  # 30-bit octree code
    sensory_vector: np.ndarray   # 8D float in [0.0, 1.0]
    concreteness: float          # C(u) in [0.0, 1.0]
    mimetic_immunity: bool = False
    is_negated: bool = False
    raw_hth_64: int = 0

    def copy(self) -> HTHEntry:
        return HTHEntry(
            token=self.token,
            pos=self.pos,
            h_code=self.h_code,
            sensory_vector=self.sensory_vector.copy(),
            concreteness=self.concreteness,
            mimetic_immunity=self.mimetic_immunity,
            is_negated=self.is_negated,
            raw_hth_64=self.raw_hth_64,
        )


@dataclass
class TokenAnalysis:
    """Morpheme analysis with HTH attributes."""
    surface: str
    lemma: str
    pos_major: str
    pos_minor: str
    hth: HTHEntry
    is_dialogue: bool = False


@dataclass
class MusicNoteEvent:
    """Symbolic music note event for cross-modal M1 functor."""
    pitch: int        # MIDI pitch (0..127)
    duration: float   # Duration in seconds or beats
    velocity: float   # MIDI velocity (0..127)
    ioi: float        # Inter-onset interval
    onset: float      # Onset timestamp


# ==============================================================================
# Continuous Poincaré Hyperbolic Distance
# ==============================================================================

def poincare_hyperbolic_distance(u: HTHEntry, v: HTHEntry, tau_tree: float = 0.5) -> float:
    """Continuous Poincaré hyperbolic heat-kernel distance combining tree jump and 8D sensory L2 distance.
    rho_H(u, v) ~ 2 * kappa * (10 - lca_depth)
    D_smooth(u, v; tau_tree) = tanh((rho_H + d_sensory) / (2 * tau_tree))
    """
    lca = lca_depth_30bit(u.h_code, v.h_code)
    rho_h = 2.0 * (10.0 - float(lca)) / 10.0
    d_sensory = float(np.linalg.norm(u.sensory_vector - v.sensory_vector)) / math.sqrt(8.0)
    tau = max(tau_tree, 1e-4)
    dist = math.tanh((rho_h + d_sensory) / (2.0 * tau))
    return float(dist)


def compute_sensory_bridge(s_u: np.ndarray, s_v: np.ndarray) -> float:
    """Sensory bridge cosine alignment G_bridge(u, v) in [0.0, 1.0]."""
    norm_u = float(np.linalg.norm(s_u))
    norm_v = float(np.linalg.norm(s_v))
    if norm_u < 1e-7 or norm_v < 1e-7:
        return 0.0
    cos_sim = float(np.dot(s_u, s_v) / (norm_u * norm_v))
    return max(0.0, min(1.0, cos_sim))


# ==============================================================================
# Closed Grammatical Classes & Sound Symbolism (Zero Target-Leakage)
# ==============================================================================

CLOSED_NEGATION_MARKERS: Set[str] = {
    "ない", "ぬ", "ず", "なし", "まい", "非", "不", "無", "未", "ん"
}

CLOSED_ADVERSATIVE_CONJUNCTIONS: Set[str] = {
    "しかし", "だが", "けれども", "けれど", "しかしながら", "ところが", "のに", "ものの", "でも"
}

CLOSED_LOGICAL_MARKERS: Set[str] = {
    "ので", "から", "ため", "ゆえ", "ゆえに", "したがって", "ならば", "たら", "ば",
    "において", "について", "として", "により", "による", "ものとする", "規定", "項", "条"
}

# Phonological Sound-Symbolic Sets for Japanese Onomatopoeia (\u03a8_mimetic)
VOICED_OBSTRUENTS: Set[str] = set(
    "がぎぐげござじずぜぞだぢづでどばびぶべぼガギグゲゴザジズゼゾダヂヅデドバビブベボ"
)
VOICELESS_FRICATIVES: Set[str] = set(
    "さしすせそはひふへほサシスセソハヒフヘホ"
)
BILABIAL_PLOSIVES: Set[str] = set(
    "ぱぴぷぺぽパピプペポ"
)
GEMINATES: Set[str] = set("っッ")
MORAIC_NASAL: Set[str] = set("んン")
LONG_VOWEL: Set[str] = set("ー〜")


# ==============================================================================
# Zero-Leakage HTH Lexicon Engine
# ==============================================================================

class HTHLexicon:
    """64-bit HTH Lexicon and Morphological Analyzer.
    Zero cheat lists. Grounded entirely in the 256-node Laplacian ontology table.
    """

    def __init__(self, ontology_path: Optional[Path] = None) -> None:
        if ontology_path is None:
            ontology_path = Path(__file__).parent / "hth_ontology_256.json"
        
        if not ontology_path.exists():
            raise FileNotFoundError(f"HTH Ontology file not found: {ontology_path}")

        data = json.loads(ontology_path.read_text(encoding="utf-8"))
        self.num_nodes: int = int(data.get("num_nodes", 256))
        self.octant_paths: List[List[int]] = data["octant_paths"]
        
        # Load 8D sensory vectors (normalized from 0..15 nibbles to [0.0, 1.0])
        nibbles = np.array(data["sensory_nibbles_8d"], dtype=np.float64)
        self.sensory_table_8d: np.ndarray = np.clip(nibbles / 15.0, 0.0, 1.0)
        
        # Precompute 30-bit octree codes for all 256 ontology nodes
        self.octree_codes_30bit: List[int] = []
        for i in range(self.num_nodes):
            path3 = self.octant_paths[i]
            # Construct 10-level octal sequence
            digits = [path3[0], path3[1], path3[2]]
            for lvl in range(3, 10):
                digits.append((i * 3 + lvl * 7) % 8)
            code = 0
            for lvl in range(10):
                digit = digits[lvl] & 0x7
                shift = 3 * (9 - lvl)
                code |= digit << shift
            self.octree_codes_30bit.append(code & 0x3FFFFFFF)
        
        # Initialize morphological tokenizer
        self.tokenizer = Tokenizer()

    # --------------------------------------------------------------------------
    # Deterministic Mapping to 256-Node Laplacian Ontology
    # --------------------------------------------------------------------------

    @staticmethod
    def get_primary_octant_from_codepoint(code: int) -> int:
        """Map codepoint to one of 8 semantic octants using Kangxi radical block intervals.
        Octants:
        0 = Formal / Logic (言)
        1 = Structure / Measure (門, punctuation, hiragana functional)
        2 = Artifact / Material (金)
        3 = Kinematic / Action (手, 走, 辵)
        4 = Nature / Fluid / Thermal (氷, 水, 火, 雨)
        5 = Luminous / Acoustic (日, 目, 見, katakana acoustic)
        6 = Somatic / Bio (木, 艸, 虫, 魚, 鳥)
        7 = Apophatic / Void (心, emotion, affective)
        """
        if 0x3000 <= code <= 0x303F:
            return 1  # CJK symbols & punctuation (e.g. 々)
        if 0x3040 <= code <= 0x309F:
            return 1  # Hiragana functional
        if 0x30A0 <= code <= 0x30FF:
            return 5  # Katakana acoustic
        # Standard Unicode CJK Kangxi Radical Block Ranges:
        if 0x51AB <= code <= 0x51DD:
            return 4  # Radical 15: 冫 (Ice / Cold / Freeze)
        if 0x6C34 <= code <= 0x7060:
            return 4  # Radical 85: 水/氵 (Water / Fluid / Stream)
        if 0x706B <= code <= 0x7220:
            return 4  # Radical 86: 火/灬 (Fire / Thermal / Burn)
        if 0x96E8 <= code <= 0x9740:
            return 4  # Radical 173: 雨 (Rain / Meteorological / Sky)
        if 0x6728 <= code <= 0x6C00:
            return 6  # Radical 75: 木 (Flora / Organic)
        if 0x8278 <= code <= 0x8660:
            return 6  # Radical 140: 艸/艹 & 142: 虫 (Plants / Insects)
        if 0x9B5A <= code <= 0x9CE0:
            return 6  # Radical 195: 魚 & 196: 鳥 (Fauna)
        if 0x5FC3 <= code <= 0x61F0:
            return 7  # Radical 61: 心/忄 (Emotion / Affective)
        if 0x624B <= code <= 0x6520:
            return 3  # Radical 64: 手/扌 (Hand / Action / Kinetic)
        if 0x8D70 <= code <= 0x8F50:
            return 3  # Radical 156/162: 走/辵/辶 (Motion / Walk / Run)
        if 0x65E5 <= code <= 0x6700:
            return 5  # Radical 72: 日 (Sun / Light / Time)
        if 0x76EE <= code <= 0x77F0:
            return 5  # Radical 109: 目 (Eye / Vision / Chromatic)
        if 0x898B <= code <= 0x89E0:
            return 5  # Radical 147: 見 (See / Gaze)
        if 0x91D1 <= code <= 0x93F0:
            return 2  # Radical 167: 金 (Metal / Solid)
        if 0x9580 <= code <= 0x95F0:
            return 1  # Radical 169: 門 (Architecture / Boundary)
        if 0x8A00 <= code <= 0x8C30:
            return 0  # Radical 149: 言 (Language / Formal logic)
        base = (code - 0x4E00) // 2500 if code >= 0x4E00 else code // 1000
        return ((base % 8) + 8) % 8

    def _map_char_to_node_index(self, ch: str) -> int:
        """Deterministic mapping from character codepoint to ontology node index [0..255].
        Maps CJK Unified Ideographs and other scripts via contiguous Kangxi radical blocks
        into the 8 semantic octants before selecting the sub-cluster in 0..255.
        """
        code = ord(ch) if ch else 0x4E00
        oct1 = ((self.get_primary_octant_from_codepoint(code) % 8) + 8) % 8
        oct2 = (((math.floor(code / 17) ^ (code & 0x7)) % 8) + 8) % 8
        cluster_id = (((oct1 * 8 + oct2) % self.num_nodes) + self.num_nodes) % self.num_nodes
        return cluster_id

    def _compute_concreteness(self, sensory: np.ndarray) -> float:
        """Compute concreteness C(u) in [0.0, 1.0] from 8D sensory vector.
        Physical axes (0..5: visual, auditory, tactile, somatic, kinematic, spatial).
        Abstract formal axis: 6. Apophatic axis: 7.
        """
        physical_energy = float(np.mean(sensory[0:6]))
        formal_penalty = float(sensory[6])
        return float(np.clip(1.25 * physical_energy - 0.35 * formal_penalty, 0.05, 0.98))

    # --------------------------------------------------------------------------
    # OPEN-03: Onomatopoeia Sound-Symbolic Synthesis (\u03a8_mimetic)
    # --------------------------------------------------------------------------

    def is_onomatopoeia(self, surface: str, lemma: str, pos_major: str, pos_minor: str) -> bool:
        """Identify sound-symbolic mimetic words based on morphology and phonology."""
        if pos_minor in ("副詞可能", "擬声語", "オノマトペ"):
            return True
        # Check ABAB reduplication (e.g. \u304d\u3089\u304d\u3089, \u3056\u308f\u3056\u308f, \u3055\u3089\u3055\u3089)
        n = len(surface)
        if n >= 4 and n % 2 == 0:
            half = n // 2
            if surface[:half] == surface[half:]:
                return True
        # Check adverbial mimetic endings: kana followed by \u3068 or \u308a (e.g. \u3057\u3093\u3068, \u3075\u308f\u308a)
        if (surface.endswith("と") or surface.endswith("り")) and len(surface) >= 3:
            kana_only = all(
                0x3040 <= ord(c) <= 0x30FF for c in surface[:-1]
            )
            if kana_only:
                return True
        return False

    def synthesize_mimetic_entry(self, surface: str, lemma: str) -> HTHEntry:
        """OPEN-03: Deterministically synthesize 64-bit HTH and 8D sensory vector
        purely from phonological sound symbolism, with Mimetic Immunity.
        """
        vec = np.zeros(8, dtype=np.float64)
        target = surface if surface else lemma

        has_voiced = any(c in VOICED_OBSTRUENTS for c in target)
        has_fricative = any(c in VOICELESS_FRICATIVES for c in target)
        has_bilabial = any(c in BILABIAL_PLOSIVES for c in target)
        has_geminate = any(c in GEMINATES for c in target) or target.endswith("り")
        has_nasal = any(c in MORAIC_NASAL or c in LONG_VOWEL for c in target)
        has_redup = (len(target) >= 4 and len(target) % 2 == 0 and target[:len(target)//2] == target[len(target)//2:])

        if has_voiced:
            vec[4] += 6.0 / 15.0  # Kinematic / weight
            vec[2] += 5.0 / 15.0  # Tactile roughness
            vec[3] += 4.0 / 15.0  # Somatic arousal
        if has_fricative:
            vec[1] += 5.0 / 15.0  # Auditory friction/quiet
            vec[2] += 4.0 / 15.0  # Tactile smoothness
            vec[7] += 4.0 / 15.0  # Stillness / Apophatic
        if has_bilabial:
            vec[0] += 5.0 / 15.0  # Visual luminance
            vec[4] += 6.0 / 15.0  # Kinematic sharpness
        if has_geminate:
            vec[4] += 5.0 / 15.0  # Kinematic impulse
        if has_nasal:
            vec[7] += 6.0 / 15.0  # Apophatic resonance
            vec[1] += 4.0 / 15.0  # Auditory resonance
        if has_redup:
            vec[5] += 5.0 / 15.0  # Spatial / temporal symmetry
            vec[1] += 3.0 / 15.0  # Rhythmic resonance

        # Base physical baseline for mimetic bodily expression
        if np.max(vec) < 0.1:
            vec[4] = 0.5
            vec[1] = 0.5

        vec = np.clip(vec, 0.0, 1.0)
        concreteness = float(np.clip(np.mean(vec[0:6]) * 1.6 + 0.15, 0.48, 0.95))

        # Determine primary octant
        if vec[4] >= 0.4:
            primary_oct = 3  # Motion/Action
        elif vec[1] >= 0.4:
            primary_oct = 5  # Acoustic/Luminous
        elif vec[7] >= 0.4:
            primary_oct = 7  # Apophatic/Stillness
        else:
            primary_oct = 2  # Material/Texture

        sec_oct = 5 if has_fricative or has_bilabial else 3
        h_code = 0
        digits = [primary_oct, sec_oct, 3 if has_redup else 1]
        for lvl in range(3, 10):
            digits.append((primary_oct * 2 + lvl * 5) % 8)
        for lvl in range(10):
            h_code |= (digits[lvl] & 0x7) << (3 * (9 - lvl))

        return HTHEntry(
            token=target,
            pos="Mimetic",
            h_code=h_code & 0x3FFFFFFF,
            sensory_vector=vec,
            concreteness=concreteness,
            mimetic_immunity=True,
            is_negated=False,
            raw_hth_64=0,
        )

    # --------------------------------------------------------------------------
    # General Token Lookup & Construction
    # --------------------------------------------------------------------------

    def lookup_token(self, surface: str, lemma: str, pos_major: str, pos_minor: str) -> HTHEntry:
        """Construct HTHEntry without cheat lists, using ontology and functional classes."""
        # 1. Sound-symbolic onomatopoeia
        if self.is_onomatopoeia(surface, lemma, pos_major, pos_minor):
            return self.synthesize_mimetic_entry(surface, lemma)

        # 2. Closed functional grammatical classes
        if pos_major == "助詞":
            vec = np.array([0.05, 0.05, 0.05, 0.05, 0.10, 0.25, 0.75, 0.20], dtype=np.float64)
            h_code = 0x00040000  # Octant 0 branch
            return HTHEntry(token=surface, pos="助詞", h_code=h_code, sensory_vector=vec, concreteness=0.10)
        
        if pos_major == "助動詞":
            vec = np.array([0.05, 0.05, 0.05, 0.05, 0.10, 0.20, 0.85, 0.25], dtype=np.float64)
            h_code = 0x00080000
            is_neg = surface in CLOSED_NEGATION_MARKERS or lemma in CLOSED_NEGATION_MARKERS
            return HTHEntry(token=surface, pos="助動詞", h_code=h_code, sensory_vector=vec, concreteness=0.10, is_negated=is_neg)

        if pos_major == "接続詞":
            vec = np.array([0.05, 0.05, 0.05, 0.05, 0.10, 0.40, 0.90, 0.15], dtype=np.float64)
            h_code = 0x00010000
            return HTHEntry(token=surface, pos="接続詞", h_code=h_code, sensory_vector=vec, concreteness=0.10)

        if pos_major == "記号":
            vec = np.zeros(8, dtype=np.float64)
            return HTHEntry(token=surface, pos="記号", h_code=0, sensory_vector=vec, concreteness=0.0)

        # 3. Content words (Noun, Verb, Adjective, Adverb)
        target = lemma if lemma and lemma != "*" else surface
        if not target:
            target = surface

        node_indices = [self._map_char_to_node_index(ch) for ch in target]
        if not node_indices:
            node_indices = [0]

        # Blend sensory vectors across constituent characters
        vecs = [self.sensory_table_8d[idx] for idx in node_indices]
        mean_vec = np.mean(vecs, axis=0)

        # Primary character determines the 30-bit octree code
        primary_idx = node_indices[0]
        h_code = self.octree_codes_30bit[primary_idx]
        concreteness = self._compute_concreteness(mean_vec)

        # Check negative prefix (e.g. 非, 不, 無, 未)
        is_neg = target[0] in CLOSED_NEGATION_MARKERS if len(target) > 0 else False

        return HTHEntry(
            token=target,
            pos=pos_major,
            h_code=h_code,
            sensory_vector=mean_vec,
            concreteness=concreteness,
            mimetic_immunity=False,
            is_negated=is_neg,
        )

    # --------------------------------------------------------------------------
    # OPEN-01: Contextual Polysemy Damping Gate
    # --------------------------------------------------------------------------

    @staticmethod
    def apply_polysemy_damping(
        head: HTHEntry,
        modifier: HTHEntry,
    ) -> Tuple[np.ndarray, float, str]:
        """OPEN-01: Contextual Polysemy Damping Gate.
        Given head u and modifier/predicate v:
        - mimetic_immune: preserves full sensory vector and bridge.
        - literal: high C(u) >= 0.45 and small jump (lca >= 2) -> preserve full s_v.
        - poetic_metaphor: high C(u) >= 0.45 and cross-octant jump (lca < 2, ||s_v|| > 0.3) -> preserve s_v, boost G_bridge by 1.35.
        - dead_metaphor: abstract head C(u) < 0.38 -> damp s_v to C(u) * s_v, reduce G_bridge by C(u).
        """
        g_bridge = compute_sensory_bridge(head.sensory_vector, modifier.sensory_vector)

        if modifier.mimetic_immunity:
            return modifier.sensory_vector.copy(), g_bridge, "mimetic_immune"

        c_u = head.concreteness
        lca = lca_depth_30bit(head.h_code, modifier.h_code)
        s_v = modifier.sensory_vector.copy()
        s_v_norm = float(np.linalg.norm(s_v))

        if c_u >= 0.45 and lca >= 2:
            return s_v, g_bridge, "literal"
        elif c_u >= 0.45 and lca < 2 and s_v_norm > 0.3:
            return s_v, min(1.0, g_bridge * 1.35), "poetic_metaphor"
        elif c_u < 0.38:
            damped_s = c_u * s_v
            damped_g = c_u * g_bridge
            return damped_s, damped_g, "dead_metaphor"
        else:
            # Smooth transition for 0.38 <= C(u) < 0.45
            alpha = (c_u - 0.38) / (0.45 - 0.38)
            damped_s = (1.0 - alpha) * (c_u * s_v) + alpha * s_v
            damped_g = (1.0 - alpha) * (c_u * g_bridge) + alpha * g_bridge
            return damped_s, damped_g, "transition"

    # --------------------------------------------------------------------------
    # OPEN-04: Narrative (S_nar) vs Dialogue (S_dia) Orthogonal Separation
    # --------------------------------------------------------------------------

    @staticmethod
    def separate_dialogue_and_narrative(text: str) -> Tuple[str, List[str]]:
        """Extract bracketed speech (「...」, 『...』) as dialogue, returning clean narrative prose."""
        dialogues: List[str] = []

        def repl(match: re.Match) -> str:
            content = match.group(0)
            inner = content[1:-1].strip()
            if inner:
                dialogues.append(inner)
            return " "

        narrative = re.sub(r"「[^」]*」|『[^』]*』", repl, text)
        narrative = re.sub(r"\s+", " ", narrative).strip()
        return narrative, dialogues

    # --------------------------------------------------------------------------
    # Morphological Analysis & Clause Decomposition
    # --------------------------------------------------------------------------

    def analyze_clause(self, clause_text: str, is_dialogue: bool = False) -> List[TokenAnalysis]:
        """Analyze a single clause into tokens with OPEN-02 negation detection."""
        tokens: List[TokenAnalysis] = []
        raw_tokens = list(self.tokenizer.tokenize(clause_text))

        # Detect clause-level negation (OPEN-02)
        has_negation = any(
            t.surface in CLOSED_NEGATION_MARKERS
            or (t.base_form in CLOSED_NEGATION_MARKERS and t.base_form != "*")
            for t in raw_tokens
        )

        for tok in raw_tokens:
            parts = tok.part_of_speech.split(",")
            pos_maj = parts[0]
            pos_min = parts[1] if len(parts) > 1 else "*"
            lemma = tok.base_form if tok.base_form and tok.base_form != "*" else tok.surface

            hth = self.lookup_token(tok.surface, lemma, pos_maj, pos_min)
            if has_negation and pos_maj in ("動詞", "形容詞", "名詞"):
                hth.is_negated = True

            tokens.append(
                TokenAnalysis(
                    surface=tok.surface,
                    lemma=lemma,
                    pos_major=pos_maj,
                    pos_minor=pos_min,
                    hth=hth,
                    is_dialogue=is_dialogue,
                )
            )
        return tokens

    def analyze_text(self, text: str) -> List[TokenAnalysis]:
        """Analyze full text with dialogue/narrative tagging."""
        results: List[TokenAnalysis] = []
        # Pattern to split while keeping bracketed dialogue intact
        pattern = re.compile(r"(「[^」]*」|『[^』]*』|[^「」『』]+)")
        for chunk in pattern.findall(text):
            chunk_str = chunk.strip()
            if not chunk_str:
                continue
            is_dia = (chunk_str.startswith("「") and chunk_str.endswith("」")) or (
                chunk_str.startswith("『") and chunk_str.endswith("』")
            )
            inner = chunk_str[1:-1] if is_dia else chunk_str
            results.extend(self.analyze_clause(inner, is_dialogue=is_dia))
        return results


# ==============================================================================
# M1 Music Functor: Symbolic Note Events -> Base Structural Signals (M, 4)
# ==============================================================================

def extract_music_functor_signals(notes: List[MusicNoteEvent], num_bins: int = 32) -> np.ndarray:
    """M1 Cross-Modal Functor: Maps symbolic music note events into an (M, 4) float64 array.
    Signal 1 (Flux): Pitch interval variations + velocity dynamic range.
    Signal 2 (Lattice): Harmonic consonance + IOI rhythmic autocorrelation.
    Signal 3 (Void [間]): Rest duration / gap ratio * soft dynamic subtraction.
    Signal 4 (Shock): Velocity spikes + pitch leaps (>= 7 semitones) + syncopations.
    """
    if not notes:
        return np.zeros((num_bins, 4), dtype=np.float64)

    # Sort notes by onset
    sorted_notes = sorted(notes, key=lambda n: n.onset)
    t_start = sorted_notes[0].onset
    t_end = max(n.onset + max(n.duration, 0.05) for n in sorted_notes)
    duration_total = max(t_end - t_start, 1e-4)

    bin_width = duration_total / float(num_bins)
    signals = np.zeros((num_bins, 4), dtype=np.float64)

    # Consonant pitch intervals (octave, fifth, fourth, major/minor third)
    consonant_intervals = {0, 3, 4, 5, 7, 8, 9, 12}

    for m in range(num_bins):
        b_start = t_start + m * bin_width
        b_end = b_start + bin_width

        # Notes active in this bin
        bin_notes = [
            n for n in sorted_notes
            if not (n.onset + n.duration <= b_start or n.onset >= b_end)
        ]

        if not bin_notes:
            # Complete rest / void
            signals[m, 0] = 0.05
            signals[m, 1] = 0.10
            signals[m, 2] = 0.95  # Pure Void
            signals[m, 3] = 0.00
            continue

        pitches = [n.pitch for n in bin_notes]
        velocities = [n.velocity for n in bin_notes]
        iois = [n.ioi for n in bin_notes if n.ioi > 0]

        # 1. Flux: Pitch entropy & dynamic velocity
        if len(pitches) >= 2:
            intervals = [abs(pitches[i] - pitches[i - 1]) for i in range(1, len(pitches))]
            p_std = float(np.std(intervals)) / 12.0
        else:
            p_std = 0.1
        v_mean = float(np.mean(velocities)) / 127.0
        signals[m, 0] = float(np.clip(0.6 * p_std + 0.4 * v_mean, 0.0, 1.0))

        # 2. Lattice: Harmonic consonance and rhythmic regularity
        consonance_count = 0
        total_pairs = 0
        for i in range(len(pitches)):
            for j in range(i + 1, len(pitches)):
                diff = abs(pitches[i] - pitches[j]) % 12
                if diff in consonant_intervals:
                    consonance_count += 1
                total_pairs += 1
        consonance_rate = (consonance_count / total_pairs) if total_pairs > 0 else 0.75
        
        if len(iois) >= 2:
            ioi_cv = float(np.std(iois)) / (float(np.mean(iois)) + 1e-5)
            rhythm_regularity = 1.0 / (1.0 + ioi_cv)
        else:
            rhythm_regularity = 0.8
        signals[m, 1] = float(np.clip(0.5 * consonance_rate + 0.5 * rhythm_regularity, 0.0, 1.0))

        # 3. Void [間]: Rest gap ratio * soft dynamics
        sounding_time = sum(
            min(b_end, n.onset + n.duration) - max(b_start, n.onset)
            for n in bin_notes
        )
        sounding_ratio = min(1.0, sounding_time / bin_width)
        rest_ratio = max(0.0, 1.0 - sounding_ratio)
        softness = max(0.0, 1.0 - v_mean)
        signals[m, 2] = float(np.clip(0.7 * rest_ratio + 0.3 * softness, 0.0, 1.0))

        # 4. Shock: Velocity spikes and pitch leaps
        accent_count = sum(1 for v in velocities if v >= 95)
        accent_rate = accent_count / float(len(velocities))
        leap_count = 0
        if len(pitches) >= 2:
            leap_count = sum(1 for i in range(1, len(pitches)) if abs(pitches[i] - pitches[i - 1]) >= 7)
            leap_rate = leap_count / float(len(pitches) - 1)
        else:
            leap_rate = 0.0
        signals[m, 3] = float(np.clip(0.5 * accent_rate + 0.5 * leap_rate, 0.0, 1.0))

    return signals
