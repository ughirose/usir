"""USIR v3.0 Cross-Linguistic Isomorphic Driver (Gap 7: usir-driver-en).

Implements 1-to-1 structural isomorphism between Japanese and English:
1. Kango (漢語) <-> Latinate / Greek polysyllabic abstract words & formal/legal markers:
   - Suffixes: -tion, -sion, -ity, -ment, -ence, -ance, -ism, -ology, -ature, -tude, etc.
   - Legal/Math markers: shall, pursuant, thereof, herein, hereto, notwithstanding, whereas,
     iff, compact, homomorphism, subspace, topology, manifold, etc.
2. Wago (和語) <-> Anglo-Saxon concrete monosyllabic/disyllabic stems:
   - stone, rain, wind, cold, dark, water, leaf, bone, shadow, light, mud, wood, hill, river,
     sky, dust, grass, snow, fire, blood, iron, glass, bell, fog, silence, breath, etc.
3. Mora Cadence <-> English Syllable Count & Stress Rhythm:
   - Deterministic vowel-cluster syllable counter + clause length autocorrelation + punctuation caesura.
4. OPEN-02 & OPEN-05 Isomorphism:
   - English apophatic negation: not, never, neither, nor, nothing, nowhere, without, none, no.
   - Minimalist paratactic subtraction: low modifier ratio exp(-0.5 * N_adj_adv).
5. English AI Slop / Dead Metaphor Detection:
   - delve, tapestry, testament, underscores, multifaceted, pivotal, crucial, moreover,
     furthermore, it is important to note, in conclusion, comprehensive framework, etc.
6. 24D Symplectic Extraction & 5-Phase Landau Classification:
   - extract_usir_24d_english(text) -> np.ndarray
   - classify_english_phase(text) -> str
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from usir_py.v2.wavelet_hermite_encoder import WaveletHermiteEncoder


# ==============================================================================
# Lexical & Morphological Isomorphism Tables
# ==============================================================================

LATINATE_SUFFIXES: Tuple[str, ...] = (
    "tion", "sion", "ity", "ment", "ence", "ance", "ism", "ology",
    "ature", "tude", "ous", "ive", "al", "ical", "ary", "ate",
    "ian", "ible", "able", "ise", "ize", "ant", "ent", "ic",
)

LEGAL_MATH_MARKERS: Set[str] = {
    "shall", "pursuant", "thereof", "herein", "hereto", "notwithstanding",
    "whereas", "hereunder", "foregoing", "inasmuch", "iff", "compact",
    "homomorphism", "subspace", "isomorphism", "topology", "diffeomorphism",
    "manifold", "functor", "symplectic", "lemma", "theorem", "corollary",
    "definition", "proposition", "axiom", "provided", "jurisdiction",
    "statute", "obligation", "indemnity", "representation", "warranty",
    "hereby", "therein", "codomain", "homeomorphic", "continuous", "bounded",
    "homeomorphism", "affine", "bijection", "submanifold", "vector",
}

SAXON_CONCRETE_STEMS: Set[str] = {
    "stone", "rain", "wind", "cold", "dark", "water", "leaf", "bone",
    "shadow", "light", "mud", "wood", "hill", "river", "sky", "dust",
    "grass", "snow", "fire", "blood", "iron", "glass", "bell", "fog",
    "silence", "breath", "tree", "night", "sea", "earth", "sun", "moon",
    "hand", "eye", "foot", "walk", "fall", "run", "stand", "lay",
    "sit", "hear", "see", "door", "road", "wall", "bread", "fish",
    "boat", "flesh", "salt", "ash", "bark", "root", "bird", "frost",
    "ice", "path", "smoke", "trout", "lake", "pool", "stream", "deep",
    "black", "white", "green", "red", "grey", "gray", "hot", "still",
}

APOPHATIC_NEGATIONS: Set[str] = {
    "not", "never", "neither", "nor", "nothing", "nowhere", "without",
    "none", "no", "nobody", "scarcely", "barely", "void", "empty",
    "hollow", "unseen", "silent", "silence",
}

AI_SLOP_MARKERS: List[str] = [
    "delve", "tapestry", "testament", "underscores", "multifaceted",
    "pivotal", "crucial", "moreover", "furthermore", "it is important to note",
    "in conclusion", "comprehensive framework", "seamless integration", "foster",
    "navigate the complexities", "beacon", "vital", "harness", "embark",
    "paramount", "interplay", "vibrant", "rich tapestry", "testament to",
    "delve into", "poised to", "elevate", "ever-evolving", "in today's world",
    "dynamic landscape", "shed light", "in summary",
]

SHOCK_WORDS: Set[str] = {
    "blood", "shriek", "darkness", "abyss", "terror", "crash", "sudden",
    "suddenly", "fury", "death", "madness", "horror", "scream", "rupture",
    "burst", "wound", "shatter", "agony", "shudder", "climax", "struck",
    "blast", "frightful", "trembling", "yawning", "violent", "panic",
    "shattered", "shrieked", "crashed", "screamed", "ripped",
}

ADVERB_ADJECTIVE_SUFFIXES: Tuple[str, ...] = (
    "ly", "ful", "less", "able", "ible", "ous", "ious", "ive",
    "ish", "ic", "al", "est", "er",
)


# ==============================================================================
# Phonetic & Syllable Rhythm Modeling
# ==============================================================================

def count_syllables(word: str) -> int:
    """Deterministic English syllable counter based on vowel clustering and morphology."""
    w = word.lower().strip()
    w = re.sub(r"[^a-z]", "", w)
    if not w:
        return 0
    if len(w) <= 3:
        return 1

    # Separate common vowel hiatuses that form 2 syllables
    w = re.sub(r"([iu])([aeo])", r"\1 \2", w)
    w = re.sub(r"([eo])([iu])", r"\1 \2", w)

    # Count vowel groups
    vowel_runs = re.findall(r"[aeiouy]+", w)
    count = len(vowel_runs)

    # Silent 'e' at end
    if w.endswith("e") and not w.endswith("le") and not w.endswith("ee"):
        count -= 1

    # -ed inflection (silent e after most consonants except d, t)
    if w.endswith("ed") and not w.endswith("ded") and not w.endswith("ted"):
        if count > 1:
            count -= 1

    # -es inflection
    if w.endswith("es") and not any(w.endswith(s) for s in ("ses", "zes", "ches", "shes")):
        if count > 1:
            count -= 1

    # -ism suffix is /ɪz.əm/ (2 syllables)
    if w.endswith("ism"):
        count += 1

    return max(count, 1)


def split_english_clauses(text: str) -> List[Tuple[str, str]]:
    """Segment English text into clauses paired with their trailing delimiter punctuation."""
    tokens = re.split(r"([.\n?!;:—–,]+)", text)
    clauses: List[Tuple[str, str]] = []
    i = 0
    while i < len(tokens):
        c = tokens[i].strip()
        delim = tokens[i + 1].strip() if i + 1 < len(tokens) else ""
        if len(re.findall(r"[a-zA-Z]+", c)) >= 1:
            clauses.append((c, delim))
        i += 2

    if not clauses:
        clauses = [(text.strip() or "silence", ".")]
    return clauses


# ==============================================================================
# 4D Base Signal Extraction (Flux, Lattice, Void, Shock)
# ==============================================================================

def extract_clause_signals_english(
    clause: str,
    delim: str = "",
    prev_clause: Optional[str] = None,
) -> np.ndarray:
    """Extract 4 base structural signals s_m = (s_1, s_2, s_3, s_4) for a single English clause."""
    words = re.findall(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b", clause.lower())
    n_tok = len(words)
    if n_tok == 0:
        return np.array([0.1, 0.5, 0.5, 0.0], dtype=np.float64)

    # --------------------------------------------------------------------------
    # Channel 1: s_1 (Flux): Radial entropy + Character bigram surprise
    # --------------------------------------------------------------------------
    lengths = [len(w) for w in words]
    tot_chars = sum(lengths)
    if tot_chars > 0 and n_tok > 1:
        p_len = np.array([l / float(tot_chars) for l in lengths], dtype=np.float64)
        h_radial = -float(np.sum(p_len * np.log2(p_len + 1e-12))) / math.log2(max(n_tok, 2))
    else:
        h_radial = 0.20

    clean_chars = "".join(words)
    if len(clean_chars) >= 3:
        bigrams = [clean_chars[i : i + 2] for i in range(len(clean_chars) - 1)]
        counts: Dict[str, int] = {}
        for bg in bigrams:
            counts[bg] = counts.get(bg, 0) + 1
        n_bg = len(bigrams)
        bg_entropy = -sum((c / n_bg) * math.log2(c / n_bg) for c in counts.values())
        surprise = bg_entropy / math.log2(max(n_bg, 2))
    else:
        surprise = 0.15

    syllables = [count_syllables(w) for w in words]
    tot_syl = sum(syllables)
    s1 = float(np.clip(0.35 * h_radial + 0.35 * surprise + 0.30 * min(1.0, tot_syl / 12.0), 0.0, 1.0))

    # --------------------------------------------------------------------------
    # Channel 2: s_2 (Lattice): Kango/Latinate ratio + Legal/Math markers + Syllable cadence
    # --------------------------------------------------------------------------
    latinate_count = sum(
        1 for w in words
        if any(w.endswith(sfx) for sfx in LATINATE_SUFFIXES) or w in LEGAL_MATH_MARKERS
    )
    latinate_ratio = latinate_count / float(n_tok)

    legal_count = sum(1 for w in words if w in LEGAL_MATH_MARKERS)
    legal_ratio = legal_count / float(n_tok)

    # Metric cadence alignment (4, 8, 10, 12, 16 syllables)
    cadence_dist = min(abs(tot_syl - c) for c in (4, 8, 10, 12, 16))
    cadence_reg = max(0.0, 1.0 - cadence_dist / 8.0)

    if legal_count >= 1 or latinate_ratio > 0.25:
        s2 = float(np.clip(0.40 * latinate_ratio + 0.45 * legal_ratio + 0.15 * cadence_reg + 0.30, 0.0, 1.0))
    else:
        s2 = float(np.clip(0.30 * latinate_ratio + 0.50 * cadence_reg + 0.20 * (1.0 - surprise), 0.0, 1.0))

    # --------------------------------------------------------------------------
    # Channel 3: s_3 (Void [間]): Anglo-Saxon concreteness * subtraction rate * apophatic negation
    # --------------------------------------------------------------------------
    saxon_count = sum(1 for w in words if w in SAXON_CONCRETE_STEMS)
    mod_count = sum(
        1 for w in words
        if any(w.endswith(sfx) for sfx in ADVERB_ADJECTIVE_SUFFIXES)
        and w not in SAXON_CONCRETE_STEMS
    )
    subtraction_rate = math.exp(-0.50 * float(mod_count))

    has_apophatic = any(w in APOPHATIC_NEGATIONS for w in words)
    apophatic_factor = 1.35 if has_apophatic else 1.0

    slop_hits = sum(1 for slop in AI_SLOP_MARKERS if slop in clause.lower())
    slop_penalty = max(0.0, 1.0 - 0.70 * float(slop_hits))
    latin_penalty = max(0.1, 1.0 - 0.60 * latinate_ratio)

    if saxon_count >= 1 or has_apophatic:
        base_void = 0.45 + 0.35 * min(saxon_count / 2.0, 1.0)
    else:
        base_void = 0.15

    s3 = float(np.clip(base_void * subtraction_rate * apophatic_factor * slop_penalty * latin_penalty, 0.0, 1.0))

    # --------------------------------------------------------------------------
    # Channel 4: s_4 (Shock): Visceral rupture + Sudden caesura + Syllable leap
    # --------------------------------------------------------------------------
    shock_count = sum(1 for w in words if w in SHOCK_WORDS)
    shock_density = min(1.0, shock_count * 0.45)

    punc_shock = 0.35 if ("!" in delim or "—" in delim or "--" in delim) else 0.0
    has_adversative = words[0] in ("but", "yet", "suddenly", "however", "though") if words else False
    adv_boost = 0.20 if has_adversative else 0.0

    if shock_count >= 1 and punc_shock > 0:
        s4 = float(np.clip(0.50 * shock_density + 0.30 * punc_shock + 0.35, 0.0, 1.0))
    elif shock_count >= 1:
        s4 = float(np.clip(0.60 * shock_density + 0.20 * punc_shock + 0.20 * adv_boost, 0.0, 1.0))
    else:
        s4 = float(np.clip(0.40 * punc_shock + 0.30 * adv_boost, 0.0, 0.25))

    if slop_hits >= 1 or legal_count >= 1:
        s4 = min(s4 * 0.20, 0.06)

    return np.array([s1, s2, s3, s4], dtype=np.float64)


def extract_english_signals(text: str) -> np.ndarray:
    """Extract sequence of 4D signals s_m across all clauses in English text."""
    clauses = split_english_clauses(text)
    signals_list: List[np.ndarray] = []
    for i, (c, d) in enumerate(clauses):
        prev_c = clauses[i - 1][0] if i > 0 else None
        s = extract_clause_signals_english(c, delim=d, prev_clause=prev_c)
        signals_list.append(s)

    signals = np.array(signals_list, dtype=np.float64)
    return signals


# ==============================================================================
# Symplectic 24D Extraction
# ==============================================================================

_DEFAULT_ENGLISH_ENCODER: Optional[WaveletHermiteEncoder] = None


def extract_usir_24d_english(text: str, use_mca: bool = False) -> np.ndarray:
    """Extract 24D symplectic state vector z = (q, p) in R^24 from English text."""
    global _DEFAULT_ENGLISH_ENCODER
    if _DEFAULT_ENGLISH_ENCODER is None:
        _DEFAULT_ENGLISH_ENCODER = WaveletHermiteEncoder()

    signals = extract_english_signals(text)
    return _DEFAULT_ENGLISH_ENCODER.encode_signals(signals, use_mca=use_mca)


# ==============================================================================
# English Landau 5-Phase Classification
# ==============================================================================

def classify_english_phase(text: str) -> str:
    """Classify English prose into one of the 5 canonical USIR thermodynamic phases.

    Phases:
        - "I_Crystalline": Legal statutes, mathematical axioms, formal contracts (high Sg-02, Sg-05, low Sg-04, Sg-12).
        - "II_Critical": Masterpiece literary climax, Faulkner/Poe (balanced high Sg-06, Sg-08, Sg-10 with soliton climax Sg-12).
        - "III_Void": Hemingway-style minimalist prose (high Sg-03, Sg-07, Sg-11, high Saxon concreteness, apophatic stillness).
        - "IV_Turbulent": Purple prose, excessive ornamentation, uncontrolled modifier jitter.
        - "V_Deadzone": English AI slop, corporate buzzwords, formulaic filler (delve, tapestry, testament, crucial).
    """
    text_lower = text.lower()
    words = re.findall(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b", text_lower)
    n_tok = len(words)
    if n_tok == 0:
        return "III_Void"

    clauses = split_english_clauses(text)
    n_clauses = max(len(clauses), 1)

    z = extract_usir_24d_english(text, use_mca=False)
    q = z[:12]

    # Coordinate mapping:
    # q[0]: Sg-01 Dn (Micro Flux)
    # q[1]: Sg-02 Cr (Micro Lattice)
    # q[2]: Sg-03 Pr (Micro Void)
    # q[3]: Sg-04 Jt (Micro Shock)
    # q[4]: Sg-05 Cp (Meso Flux/Coupling)
    # q[5]: Sg-06 Mp (Meso Lattice)
    # q[6]: Sg-07 Cs (Meso Void)
    # q[7]: Sg-08 Lp (Meso Shock)
    # q[8]: Sg-09 Fl (Macro Flux)
    # q[9]: Sg-10 At (Macro Lattice / RQA return arch)
    # q[10]: Sg-11 Ma (Macro Void)
    # q[11]: Sg-12 Ru (Macro Shock / Rupture)

    # 1. AI Slop Detection (V_Deadzone Gate)
    slop_count = sum(1 for marker in AI_SLOP_MARKERS if marker in text_lower)
    if slop_count >= 2:
        return "V_Deadzone"

    # 2. Legal / Mathematical Axiom Detection (I_Crystalline Gate)
    legal_count = sum(1 for w in words if w in LEGAL_MATH_MARKERS)
    latinate_count = sum(1 for w in words if any(w.endswith(sfx) for sfx in LATINATE_SUFFIXES))
    latinate_ratio = latinate_count / float(n_tok)

    if legal_count >= 2 or (latinate_ratio >= 0.20 and legal_count >= 1):
        if q[3] < 0.25 and q[11] < 0.20:
            return "I_Crystalline"

    # 3. Purple Prose & Excessive Ornamentation (IV_Turbulent Gate)
    mod_count = sum(
        1 for w in words
        if any(w.endswith(sfx) for sfx in ADVERB_ADJECTIVE_SUFFIXES)
        and w not in SAXON_CONCRETE_STEMS
    )
    mod_density = mod_count / float(n_clauses)
    if mod_density >= 1.60 and q[10] < 0.25:
        return "IV_Turbulent"

    # 4. Masterpiece Climax Prose (II_Critical Gate)
    shock_count = sum(1 for w in words if w in SHOCK_WORDS)
    has_punc_shock = any("!" in d or "—" in d for _, d in clauses)
    if (shock_count >= 1 or has_punc_shock) and q[11] >= 0.20:
        return "II_Critical"

    # 5. Minimalist Subtraction & Apophatic Stillness (III_Void Gate)
    saxon_count = sum(1 for w in words if w in SAXON_CONCRETE_STEMS)
    if q[10] >= 0.40 and mod_density <= 1.25 and saxon_count >= 2:
        return "III_Void"

    return "V_Deadzone"
