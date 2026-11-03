"""USIR Linguistic Gates (OPEN-01, OPEN-02, OPEN-03, OPEN-04).

Eliminates literary and stylistic blind spots via deterministic gates:
- OPEN-01: Concreteness attenuation gate C(u) and polysemy tri-state resolution
- OPEN-02: Dual sensory tensor (s+, s-) for negation, speculative, and modality scope
- OPEN-03: Onomatopoeia sound symbolism synthesis Psi_mimetic and Mimetic Immunity
- OPEN-04: Bakhtinian 2-channel orthogonal separation (narration vs dialogue) and Dialogic Collapse detection

References:
- 10_SPEC_PHASE_A_HEXAGONAL_AND_HTH_LEXICON.md Section 3
- 10_SPEC_PHASE_A_PARSER_FREE_AND_MULTIRESOLUTION.md Section 2
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# ==============================================================================
# OPEN-01: Concreteness C(u) & Polysemy Damping Gate
# ==============================================================================

ABSTRACT_STEM_MARKERS: Set[str] = {
    "性", "化", "的", "論", "主義", "構造", "概念", "制度",
    "単位", "課題", "側面", "関係", "問題", "状況", "機能",
    "傾向", "状態", "基準", "定義", "方針", "視線", "態度",
    "思惑", "意識", "立場", "印象", "理論", "思想", "原則",
}


def compute_concreteness(
    surface: str,
    lemma: str = "",
    sensory_vec: Optional[np.ndarray] = None,
) -> float:
    """Computes deterministic concept concreteness C(u) in [0.0, 1.0].

    Formula:
    If is_abstract_stem:
        C(u) = 0.15 * ||s_u||_1 (or 0.10 if ||s_u||_1 == 0)
    Else:
        C(u) = clip(0.35 + 0.65 * ||s_u||_inf, 0.0, 1.0)
    """
    word = lemma if lemma else surface
    is_abstract = any(m in word for m in ABSTRACT_STEM_MARKERS)

    if hasattr(sensory_vec, "sensory_vec"):
        sensory_vec = sensory_vec.sensory_vec

    if sensory_vec is not None and len(sensory_vec) > 0:
        vec = np.asarray(sensory_vec, dtype=float)
        norm_inf = float(np.max(np.abs(vec)))
        norm_l1 = float(np.sum(np.abs(vec)))
    else:
        norm_inf = 0.0
        norm_l1 = 0.0

    if is_abstract:
        if norm_l1 > 0:
            c = 0.15 * (norm_l1 / 8.0)
        else:
            c = 0.10
        return float(np.clip(c, 0.02, 0.28))
    else:
        c = 0.35 + 0.65 * norm_inf
        return float(np.clip(c, 0.35, 1.0))


@dataclass(frozen=True)
class PolysemyGateResult:
    case: str                     # "case1_physical" | "case2_synesthetic_metaphor" | "case3_abstract_transfer" | "case_transition" | "mimetic_immunity"
    slot: str                     # "primary_physical" | "secondary_abstract"
    effective_sensory: np.ndarray # Damped or raw sensory vector
    damping_factor: float         # Attenuation factor applied
    metaphor_bonus: float         # Bonus added to metaphor hop (Sg-08)
    concreteness_u: float         # Concreteness C(u)


def apply_open01_polysemy_damping(
    u_concreteness: float,
    v_primary_vec: np.ndarray,
    v_secondary_vec: Optional[np.ndarray] = None,
    u_sensory_vec: Optional[np.ndarray] = None,
    v_is_mimetic: bool = False,
) -> PolysemyGateResult:
    """Tri-state resolution on dependency edge (u -> v) (OPEN-01).

    - Mimetic Immunity: If v is mimetic, never damped (eff = primary, factor = 1.0).
    - Case 1 (Physical: C(u) >= 0.45): primary sense, no damping (factor = 1.0).
    - Case 2 (Synesthetic Metaphor: C(u) < 0.45 but <s_u, s_v> > 0.25): primary sense, bonus +0.35.
    - Case 3 (Abstract Transfer: C(u) < 0.30 and not Case 2): secondary sense, eff = C(u) * s_sec.
    - Transition band (0.30 <= C(u) < 0.45): soft blend.
    """
    v_pri = np.asarray(v_primary_vec, dtype=float)
    v_sec = np.asarray(v_secondary_vec, dtype=float) if v_secondary_vec is not None else v_pri * 0.3
    c_u = float(np.clip(u_concreteness, 0.0, 1.0))

    # 1. Mimetic Immunity
    if v_is_mimetic:
        return PolysemyGateResult(
            case="mimetic_immunity",
            slot="primary_physical",
            effective_sensory=v_pri.copy(),
            damping_factor=1.0,
            metaphor_bonus=0.0,
            concreteness_u=c_u,
        )

    # Check for Synesthetic Metaphor (Case 2)
    inner_product = 0.0
    if u_sensory_vec is not None and len(u_sensory_vec) > 0:
        u_vec = np.asarray(u_sensory_vec, dtype=float)
        inner_product = float(np.dot(u_vec, v_pri) / (np.linalg.norm(u_vec) * np.linalg.norm(v_pri) + 1e-9))

    if c_u < 0.45 and inner_product > 0.25:
        return PolysemyGateResult(
            case="case2_synesthetic_metaphor",
            slot="primary_physical",
            effective_sensory=v_pri.copy(),
            damping_factor=1.0,
            metaphor_bonus=0.35,
            concreteness_u=c_u,
        )

    # Case 1: Physical Description
    if c_u >= 0.45:
        return PolysemyGateResult(
            case="case1_physical",
            slot="primary_physical",
            effective_sensory=v_pri.copy(),
            damping_factor=1.0,
            metaphor_bonus=0.0,
            concreteness_u=c_u,
        )

    # Case 3: Abstract Transfer / Dead Metaphor
    if c_u < 0.30:
        eff = v_sec * c_u
        return PolysemyGateResult(
            case="case3_abstract_transfer",
            slot="secondary_abstract",
            effective_sensory=eff,
            damping_factor=c_u,
            metaphor_bonus=0.0,
            concreteness_u=c_u,
        )

    # Transition Band [0.30, 0.45)
    alpha = (c_u - 0.30) / 0.15
    damped_sec = v_sec * c_u
    blended = alpha * v_pri + (1.0 - alpha) * damped_sec
    chosen_slot = "primary_physical" if alpha >= 0.5 else "secondary_abstract"
    return PolysemyGateResult(
        case="case_transition",
        slot=chosen_slot,
        effective_sensory=blended,
        damping_factor=alpha + (1.0 - alpha) * c_u,
        metaphor_bonus=0.0,
        concreteness_u=c_u,
    )


# ==============================================================================
# OPEN-02: Negation & Modality Dual Sensory Tensor (s+, s-)
# ==============================================================================

NEGATION_PATTERNS: Tuple[str, ...] = (
    "ないで", "なかった", "なければ", "ない", "ぬ", "ず", "まい",
    "なし", "消え", "失わ", "などなかった", "わけではない", "ではない",
    "じゃなく", "なく", "無かった", "ゼロ",
)

SPECULATIVE_PATTERNS: Tuple[str, ...] = (
    "だろう", "でしょう", "かもしれない", "かのよう", "みたい",
    "らしい", "思われ", "と思われる", "ようだ", "ように", "気配",
    "彷彿", "ごとし", "ごとく",
)


def evaluate_modality_alpha(
    tail: str,
    is_negation_scope: bool = False,
    is_speculative_scope: bool = False,
) -> float:
    """Determines modality coefficient alpha_mod in [-1.0, 1.0].

    - Negation / absence: -0.5
    - Speculative / evidential / simile: +0.6
    - Affirmative: +1.0
    """
    if is_negation_scope or any(neg in tail for neg in NEGATION_PATTERNS):
        return -0.5
    if is_speculative_scope or any(spec in tail for spec in SPECULATIVE_PATTERNS):
        return 0.6
    return 1.0


@dataclass(frozen=True)
class DualSensoryTensorResult:
    alpha_mod: float
    s_plus: np.ndarray       # Perceptual reality tensor (mu in Sg-01, Sg-04)
    s_minus: float           # Cognitive afterimage contrast scalar (sigma in Sg-08, Sg-12)
    s_minus_vec: np.ndarray   # Cognitive afterimage contrast vector
    is_negated: bool
    is_speculative: bool


def compute_dual_sensory_tensor(
    effective_sensory: np.ndarray,
    alpha_mod: float,
) -> DualSensoryTensorResult:
    """Computes Dual Sensory Tensor (s+, s-) under modality scope (OPEN-02).

    - Perceptual Reality: s+ = max(0.0, alpha_mod) * s_eff
      (Negated items have s+ strictly zero!)
    - Cognitive Afterimage: s- = |alpha_mod| * ||s_eff||_2 * 1_{alpha_mod < 0}
      (Negation triggers a contrast spike in sigma momentum)
    """
    s_eff = np.asarray(effective_sensory, dtype=float)
    s_plus = max(0.0, alpha_mod) * s_eff

    norm_2 = float(np.linalg.norm(s_eff))
    is_neg = alpha_mod < 0.0
    s_minus = abs(alpha_mod) * norm_2 if is_neg else 0.0
    s_minus_vec = abs(alpha_mod) * s_eff if is_neg else np.zeros_like(s_eff)

    return DualSensoryTensorResult(
        alpha_mod=alpha_mod,
        s_plus=s_plus,
        s_minus=s_minus,
        s_minus_vec=s_minus_vec,
        is_negated=is_neg,
        is_speculative=(0.0 < alpha_mod < 1.0),
    )


# ==============================================================================
# OPEN-03: Onomatopoeia Sound Symbolism Synthesis & Mimetic Immunity
# ==============================================================================

VOICED_ROW = set("がぎぐげござじずぜぞだぢづでどばびぶべぼガギグゲゴザジズゼゾダヂヅデドバビブベボ")
VOICELESS_FRICATIVE_ROW = set("さしすせそしゃしゅしょはひふへほサシスセソシャシュショハヒフヘホ")
VOICELESS_STOPS_ROW = set("かきくけこカキクケコたちつてとタチツテト")
SEMI_VOICED_ROW = set("ぱぴぷぺぽパピプペポ")
NASAL_LIQUID_ROW = set("まみむめもなにぬねのらりるれろわやゆよマミムメモナニヌネノラリルレロワヤユヨ")

SOKUON_CHARS = set("っッ")
HATSUON_CHARS = set("んンー")


# Non-mimetic common words containing kana inflections
MIMETIC_EXCLUSIONS: Set[str] = {
    "なかった", "あった", "行った", "言った", "知った", "持った", "待った",
    "取った", "切った", "入った", "立った", "思った", "鳴った", "買った",
    "本当", "人間", "先生", "日本", "自分", "全部", "全然", "簡単", "困難",
    "関係", "問題", "そんな", "どんな", "あんな", "こんな", "みんな",
    "あたり", "とおり", "あまり", "ばかり", "かわり", "おわり", "つまり", "やはり",
    "たり", "だり",
}


def is_mimetic_word(word: str) -> bool:
    """Checks if word is Japanese onomatopoeia/mimetic based on phonology."""
    if not word or len(word) < 2:
        return False
    w = word.strip()
    if w in MIMETIC_EXCLUSIONS:
        return False

    # Check if word is kana-based (excluding kanji words like 先生, 簡単)
    if not re.fullmatch(r'^[ぁ-んァ-ヶー・]+$', w):
        return False

    # Check adverbial particle suffix (e.g. ぷつりと, ぎらぎらと, ふわりに)
    if len(w) >= 3 and w.endswith(("と", "に")) and not w.endswith(("っと", "んと")):
        base = w[:-1]
        if is_mimetic_word(base):
            return True

    # 1. 2-mora exact reduplication ABAB (e.g. きらきら, ぎらぎら, ざわざわ, ギラギラ)
    if len(w) == 4 and w[0:2] == w[2:4]:
        return True
    if len(w) >= 5 and w[0:2] == w[2:4]:
        return True

    # 2. Sokuon / ri pattern: AっBり, ABっと, Aり, Aりと (ぷつり, ぎゅっと, はっと, ぴたり, どっしり, ふわり)
    if re.fullmatch(r'^[ぁ-んァ-ヶ]っ[ぁ-んァ-ヶ]り$', w):
        return True
    if re.fullmatch(r'^[ぁ-んァ-ヶ]{1,3}っと?$', w):
        return True
    if re.fullmatch(r'^[ぁ-んァ-ヶ]{1,4}りと?$', w):
        return True

    # 3. Hatsuon pattern: Aん(と/わり) (しんと, じんわり, ぼんやり, すーっと)
    if re.fullmatch(r'^[ぁ-んァ-ヶ]ん(?:と|わり)?$', w):
        return True
    if re.fullmatch(r'^[ぁ-んァ-ヶ]ーっ?と$', w):
        return True

    return False


@dataclass(frozen=True)
class MimeticSynthesisResult:
    word: str
    sensory_vec: np.ndarray      # 8D sensory vector in [0.0, 1.0]
    concreteness: float          # High default concreteness (0.85)
    is_mimetic: bool
    mimetic_immunity: bool
    impulse_bonus: float
    rhythm_bonus: float


def synthesize_mimetic_8d(word: str) -> MimeticSynthesisResult:
    """Synthesizes 8D sensory vector Psi_mimetic(w) via phonological symbolism.

    Dimensions:
    0: Visual (luminance / chromaticity)
    1: Auditory (silence / friction / resonance)
    2: Tactile (roughness / smoothness)
    3: Taste/Smell
    4: Kinematic (weight / abruptness / speed)
    5: Temperature (cool / heat)
    6: Arousal (tension / energy)
    7: Reverberation (visceral / afterglow)
    """
    w = word.strip()
    if not w:
        return MimeticSynthesisResult(
            word="",
            sensory_vec=np.zeros(8, dtype=float),
            concreteness=0.5,
            is_mimetic=False,
            mimetic_immunity=False,
            impulse_bonus=0.0,
            rhythm_bonus=0.0,
        )

    vec = np.zeros(8, dtype=float)
    impulse_bonus = 0.0
    rhythm_bonus = 0.0

    # Strip adverbial suffix if not sokuon/hatsuon compound (e.g. ぷつりと -> ぷつり)
    core_w = w
    if len(w) >= 3 and w.endswith(("と", "に")) and not w.endswith(("っと", "んと")):
        core_w = w[:-1]

    c0 = core_w[0]
    # 1. Initial character class
    if c0 in VOICED_ROW:
        vec[4] += 0.45  # Kinematic weight
        vec[2] += 0.35  # Tactile roughness
        vec[6] += 0.35  # Arousal
    elif c0 in VOICELESS_FRICATIVE_ROW:
        vec[1] += 0.40  # Auditory silence/friction
        vec[2] += 0.30  # Tactile smoothness
        vec[5] += 0.30  # Temperature (cool)
    elif c0 in VOICELESS_STOPS_ROW:
        vec[0] += 0.35  # Visual lightness / sparkle
        vec[1] += 0.35  # Auditory crispness / click
        vec[4] += 0.30  # Kinematic sharpness / agility
    elif c0 in SEMI_VOICED_ROW:
        vec[0] += 0.40  # Visual brightness
        vec[4] += 0.45  # Kinematic abruptness
    elif c0 in NASAL_LIQUID_ROW:
        vec[2] += 0.30  # Tactile softness
        vec[7] += 0.30  # Visceral resonance

    # 2. Sokuon / Ri ending
    if any(c in SOKUON_CHARS for c in core_w) or core_w.endswith("り"):
        vec[4] += 0.35  # Kinematic abruptness
        impulse_bonus += 0.30

    # 3. Hatsuon / Prolonged vowel
    if any(c in HATSUON_CHARS for c in core_w):
        vec[7] += 0.45  # Visceral afterglow / reverberation
        vec[1] += 0.25  # Auditory resonance

    # 4. Reduplication ABAB
    if len(core_w) >= 4 and core_w[0:2] == core_w[2:4]:
        rhythm_bonus += 0.35
        vec[6] += 0.15  # Arousal

    # Clip to [0.0, 1.0]
    vec = np.clip(vec, 0.0, 1.0)
    concreteness = 0.85

    return MimeticSynthesisResult(
        word=w,
        sensory_vec=vec,
        concreteness=concreteness,
        is_mimetic=True,
        mimetic_immunity=True,
        impulse_bonus=impulse_bonus,
        rhythm_bonus=rhythm_bonus,
    )


# ==============================================================================
# OPEN-04: Bakhtinian 2-Channel Separation & Dialogic Collapse
# ==============================================================================

DIALOGUE_RE = re.compile(r'「([^」]+)」|『([^』]+)』')
SPEAKER_BEFORE_RE = re.compile(
    r'(?:^|[。！？!?\s、でにはからのとて]|しかし|そして|だが|また)([一-龯]{1,4}|[ぁ-んァ-ヶ]{2,6}|[a-zA-Z0-9]{2,10})(?:さん|君|くん|様|氏)?(?:は|が|も)\s*(?:笑って|怒って|叫んで|頷いて|呟いて|言って|振り返って|立ち止まって|慌てて|静かに)?\s*$'
)
SPEAKER_BEFORE_COLLOQUIAL_RE = re.compile(
    r'(?:^|[。！？!?\s、])([一-龯]{2,4}|[ぁ-んァ-ヶ]{2,6}|[a-zA-Z0-9]{2,10})って\s*$'
)
SPEAKER_AFTER_RE = re.compile(
    r'^と(?:、)?([一-龯ぁ-んァ-ヶa-zA-Z0-9]{1,8})(?:さん|君|くん|様|氏)?(?:は|が|も)?(?:怒鳴|叫|言|語|呟|問|答|述|告|呻|応|声|笑|促|命|尋|返)'
)
SPEAKER_VERB_AFTER_RE = re.compile(
    r'([一-龯ぁ-んァ-ヶa-zA-Z0-9]{1,8})(?:さん|君|くん|様|氏)?(?:は|が|も|と)(?:怒鳴|叫|言|語|呟|問|答|述|告|呻|応|声|笑|促|命|尋|返)'
)

SPEAKER_STOPWORDS: Set[str] = {
    "誰", "声", "言", "思", "笑", "怒", "泣", "叫", "頷", "呟", "走", "立", "しかし", "そして", "だが", "また", "その",
}


@dataclass(frozen=True)
class BakhtinianSeparationResult:
    narration_text: str
    speaker_utterances: Dict[str, List[str]]
    speaker_vectors: Dict[str, np.ndarray]
    min_voice_distance: float
    dialogic_collapse_detected: bool
    pairwise_distances: Dict[Tuple[str, str], float]


def extract_speaker(window_before: str, window_after: str) -> Optional[str]:
    """Extracts speaker from before or after the dialogue quote."""
    # Priority 1: Direct subject before quote (e.g. 太郎は「... or 太郎は笑って「...)
    m_before = SPEAKER_BEFORE_RE.search(window_before)
    if m_before:
        name = m_before.group(1).strip()
        if name not in SPEAKER_STOPWORDS and len(name) >= 1:
            return name

    m_colloq = SPEAKER_BEFORE_COLLOQUIAL_RE.search(window_before)
    if m_colloq:
        name = m_colloq.group(1).strip()
        if name not in SPEAKER_STOPWORDS and len(name) >= 2:
            return name

    # Priority 2: Direct speech attribution after quote (e.g. と太郎は言った / と次郎が叫んだ)
    m_after = SPEAKER_AFTER_RE.search(window_after)
    if m_after:
        name = m_after.group(1).strip()
        if name not in SPEAKER_STOPWORDS and len(name) >= 1:
            return name

    # Priority 3: General speech verb in following window
    m_verb = SPEAKER_VERB_AFTER_RE.search(window_after)
    if m_verb:
        name = m_verb.group(1).strip()
        if name not in SPEAKER_STOPWORDS and len(name) >= 1:
            return name

    return None


def compute_speech_style_vector(utterances: List[str]) -> np.ndarray:
    """Extracts a normalized style feature vector for a character's utterances."""
    if not utterances:
        return np.zeros(12, dtype=float)

    full_text = " ".join(utterances)
    chars = len(full_text)
    if chars == 0:
        return np.zeros(12, dtype=float)

    # 12-dimensional stylometric vector
    # 0: Mean utterance length
    # 1: Exclamation/question rate
    # 2: Kanji ratio
    # 3: Hiragana ratio
    # 4: Katakana ratio
    # 5: Keigo / formal markers (です/ます/ございます)
    # 6: Informal / rough markers (だ/ぞ/ぜ/じゃん/だろ)
    # 7: Self-referential pronouns (私 vs 俺 vs 僕)
    # 8: Question particle (か/かい/だい)
    # 9: Interjection/hesitation (あ、えっと、うーん)
    # 10: Ending particle (よ/ね/わ/な)
    # 11: Lexical entropy
    mean_len = float(chars) / max(len(utterances), 1)

    exclam = (full_text.count("！") + full_text.count("？") + full_text.count("!") + full_text.count("?")) / max(chars, 1)

    kanji = sum(1 for c in full_text if "\u4e00" <= c <= "\u9fff") / chars
    hira = sum(1 for c in full_text if "\u3040" <= c <= "\u309f") / chars
    kata = sum(1 for c in full_text if "\u30a0" <= c <= "\u30ff") / chars

    keigo = sum(full_text.count(w) for w in ("です", "ます", "ございます", "でしょうか", "いたします")) / max(len(utterances), 1)
    rough = sum(full_text.count(w) for w in ("だろ", "ぜ", "ぞ", "じゃん", "てめえ", "野郎", "くそ", "だな")) / max(len(utterances), 1)

    ore_count = full_text.count("俺") + full_text.count("おれ")
    watashi_count = full_text.count("私") + full_text.count("わたし")
    boku_count = full_text.count("僕") + full_text.count("ぼく")
    pronoun_balance = (ore_count * 1.0 - watashi_count * 1.0 + boku_count * 0.5) / max(len(utterances), 1)

    question_p = sum(full_text.count(w) for w in ("のか", "かい", "だい", "かしら")) / max(len(utterances), 1)
    interj = sum(full_text.count(w) for w in ("あっ", "えっ", "うーん", "おい", "やれやれ")) / max(len(utterances), 1)
    end_p = sum(full_text.count(w) for w in ("よね", "だよ", "わね", "わよ", "だなあ")) / max(len(utterances), 1)

    vec = np.array([
        min(mean_len / 50.0, 2.0),
        min(exclam * 20.0, 2.0),
        kanji,
        hira,
        kata,
        min(keigo, 2.0),
        min(rough, 2.0),
        np.clip(pronoun_balance, -2.0, 2.0),
        min(question_p, 2.0),
        min(interj, 2.0),
        min(end_p, 2.0),
        min(chars / 200.0, 1.0),
    ], dtype=float)

    return vec


def extract_dual_channel_bakhtinian(
    text: str,
    collapse_threshold: float = 0.15,
) -> BakhtinianSeparationResult:
    """Extracts narration and dialogue channels, detecting Dialogic Collapse (OPEN-04).

    - Extracts 「...」 and 『...』 spans.
    - Strips dialogue from narration channel to prevent CV_L contamination.
    - Assigns speaker labels and computes pairwise voice Euclidean distance Delta_voice.
    - Flags dialogic_collapse_detected = True if min(Delta_voice) < collapse_threshold.
    """
    if not text:
        return BakhtinianSeparationResult(
            narration_text="",
            speaker_utterances={},
            speaker_vectors={},
            min_voice_distance=1.0,
            dialogic_collapse_detected=False,
            pairwise_distances={},
        )

    # Find all dialogue spans
    matches = list(DIALOGUE_RE.finditer(text))
    speaker_utterances: Dict[str, List[str]] = {}
    default_speakers = ["Speaker_A", "Speaker_B"]
    dialogue_idx = 0

    for m in matches:
        utt = m.group(1) if m.group(1) is not None else m.group(2)
        if not utt:
            continue

        # Look around the match for speaker
        start, end = m.span()
        window_before = text[max(0, start - 30) : start]
        window_after = text[end : min(len(text), end + 30)]

        speaker = extract_speaker(window_before, window_after)
        if not speaker:
            speaker = default_speakers[dialogue_idx % 2]
            dialogue_idx += 1

        speaker_utterances.setdefault(speaker, []).append(utt)

    # Remove dialogue spans to construct pure narration
    narration = DIALOGUE_RE.sub(" ", text)
    narration = re.sub(r'\s+', ' ', narration).strip()

    # Compute speaker vectors
    speaker_vectors: Dict[str, np.ndarray] = {
        spk: compute_speech_style_vector(utts)
        for spk, utts in speaker_utterances.items()
    }

    # Pairwise distance
    speakers = list(speaker_utterances.keys())
    pairwise_distances: Dict[Tuple[str, str], float] = {}
    min_dist = 1.0
    collapse = False

    if len(speakers) >= 2:
        for i in range(len(speakers)):
            for j in range(i + 1, len(speakers)):
                s1, s2 = speakers[i], speakers[j]
                v1, v2 = speaker_vectors[s1], speaker_vectors[s2]
                dist = float(np.linalg.norm(v1 - v2))
                pairwise_distances[(s1, s2)] = dist
                if dist < min_dist:
                    min_dist = dist
        if min_dist < collapse_threshold:
            collapse = True

    return BakhtinianSeparationResult(
        narration_text=narration,
        speaker_utterances=speaker_utterances,
        speaker_vectors=speaker_vectors,
        min_voice_distance=min_dist,
        dialogic_collapse_detected=collapse,
        pairwise_distances=pairwise_distances,
    )
