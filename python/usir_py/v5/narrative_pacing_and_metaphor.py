"""Prototype implementation of HW-22, HW-24, and HW-25 (Spec 14.2 - 14.4).

Notice: All constants (e.g. metaphor compression kappa=1.35, closure penalty gamma=1.5)
are uncalibrated initial values. This is an experimental Python prototype for signal
confirmation on Aozora corpus, NOT for immediate Rust migration.

Components:
  - HW-22: Metaphorical Leap Distance (D_leap)
  - HW-24: Cliffhanger Intensity (C_cliff)
  - HW-25: Setting Disclosure UID Coupling Ratio (Gamma_UID)
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Uncalibrated Initial Constants (Spec 14.2 - 14.4)
# -----------------------------------------------------------------------------
KAPPA_METAPHOR_COMPRESSION = 1.35  # kappa_mode (simile=1.0, metaphor=1.35)
GAMMA_CLOSURE_PENALTY = 1.50       # gamma for exp(-gamma * D_closure)
EPSILON_CARRIER_FLOOR = 0.10       # epsilon_0 for carrier denominator

# Category lexicons for Selectional Preference Violation (SPV)
# Concrete / Sensory / Inanimate category
CONCRETE_SENSORY_CHARS = set(
    "光影水火石風土雨雪氷冷熱赤青白黒金銀木枝葉花鳥獣魚虫月星海波岩砂霧煙"
    "音声響鳴匂香味舌眼目耳手足肌膚骨肉血唇歯爪"
)

# Abstract / Psychological / Mental category
ABSTRACT_MENTAL_CHARS = set(
    "心想情愛哀怒悲憂喜苦悩痛魂念志気意夢幻死生滅虚幽寂霊幽魔運命誠真理"
    "時永遠過去未来世界記憶孤独沈黙希望絶望怖畏"
)

# Dead metaphor cliches (highly conventionalized similes/metaphors)
DEAD_METAPHOR_PATTERNS = [
    re.compile(p) for p in [
        r"雪のよう", r"火のよう", r"夢のよう", r"鬼のよう", r"死んだよう",
        r"胸が躍る", r"息をのむ", r"目頭が熱", r"血が通う", r"身を粉",
        r"手を焼く", r"足を引っぱる", r"腹を抱える", r"耳を傾ける",
    ]
]

# Simile markers
SIMILE_PATTERN = re.compile(r"(?:のよう[なだに]|のごと[きくだ]|みたい[なだに])")

# Closure markers (tidy resolution at end of window)
CLOSURE_PATTERN = re.compile(
    r"(?:であった|であったとさ|こうして|すべては|すべてが|終わった|眠りについた|帰っていった|静けさが|平穏が|幕を閉じた)"
)

# Open question / suspense markers
SUSPENSE_PATTERN = re.compile(
    r"(?:[？\?]|だろうか|かしら|のではなかろうか|のか|のではないか|のであろうか|ないか|ないだろうか)"
)

# Dialogue regex
DIALOGUE_PATTERN = re.compile(r"「[^「」]*」")

# Expository explanation markers
EXPOSITORY_PATTERN = re.compile(
    r"(?:とは|というのは|の構造|の歴史|の意味|の起源|と呼ばれる|に由来する|について説明すると)"
)

# Dynamic sensory verbs (kinetic carriers)
KINETIC_VERBS = re.compile(
    r"(?:走|歩|駆|飛|跳|叫|叫|喚|見|睨|凝視|聞|聴|殴|打|突|引|押|倒|崩|壊|破|裂|泣|笑|震|慄)"
)


# -----------------------------------------------------------------------------
# HW-22: Metaphorical Leap Distance (D_leap)
# -----------------------------------------------------------------------------
def extract_hw22_metaphor_leap(window_text: str) -> float:
    """HW-22: Metaphorical Leap Distance D_leap.

    Evaluates:
      D_leap = SPV(w_src, w_tgt) * phi_novelty * kappa_mode
    Where:
      - SPV is selectional preference violation between sensory concrete and abstract mental domains.
      - phi_novelty discounts conventionalized dead metaphors.
      - kappa_mode applies metaphor compression (1.0 for similes, 1.35 for unheralded metaphors).
    """
    body = re.sub(r"\s", "", window_text)
    if not body:
        return 0.0

    leaps: List[float] = []

    # 1. Similes
    for m in SIMILE_PATTERN.finditer(body):
        start, end = m.span()
        # Look at 8 characters before and after the simile marker
        pre_span = body[max(0, start - 8):start]
        post_span = body[end:min(len(body), end + 8)]

        # Check domain intersection
        has_concrete = any(c in CONCRETE_SENSORY_CHARS for c in pre_span + post_span)
        has_abstract = any(c in ABSTRACT_MENTAL_CHARS for c in pre_span + post_span)

        if has_concrete and has_abstract:
            spv = 1.0
        elif has_concrete or has_abstract:
            spv = 0.5
        else:
            spv = 0.2

        # Dead metaphor discounting
        context = body[max(0, start - 6):min(len(body), end + 6)]
        is_dead = any(dp.search(context) for dp in DEAD_METAPHOR_PATTERNS)
        phi_novelty = 0.25 if is_dead else 1.0

        leaps.append(spv * phi_novelty * 1.0)

    # 2. Unheralded Metaphors (sensory + abstract adjacent collocations without simile marker)
    # Scan bigrams / trigrams across sentences
    sents = re.split(r"[。！？!?]", body)
    for s in sents:
        if len(s) < 3:
            continue
        for i in range(len(s) - 2):
            c1, c2 = s[i], s[i + 1]
            if (c1 in CONCRETE_SENSORY_CHARS and c2 in ABSTRACT_MENTAL_CHARS) or \
               (c1 in ABSTRACT_MENTAL_CHARS and c2 in CONCRETE_SENSORY_CHARS):
                context = s[max(0, i - 3):min(len(s), i + 4)]
                is_dead = any(dp.search(context) for dp in DEAD_METAPHOR_PATTERNS)
                phi = 0.20 if is_dead else 1.0
                leaps.append(1.0 * phi * KAPPA_METAPHOR_COMPRESSION)

    if not leaps:
        return 0.0
    # Mean of top-3 leaps in the window (density-scaled)
    leaps.sort(reverse=True)
    return float(np.mean(leaps[:3]))


# -----------------------------------------------------------------------------
# HW-24: Cliffhanger Intensity (C_cliff)
# -----------------------------------------------------------------------------
def extract_hw24_cliffhanger(window_text: str) -> float:
    """HW-24: Episode/Scene Cliffhanger Intensity C_cliff.

    Evaluates:
      C_cliff = (V_Ely + alpha * ||a_term||_2) * exp(-gamma * D_closure)
    Where:
      - V_Ely: Bayesian suspense proxy (open questions, unresolved terminal interrogatives).
      - ||a_term||_2: Terminal staccato acceleration (shortening of final clauses).
      - D_closure: Tidy resolution penalty (dampens cliffhanger if ending is wrapped up).
    """
    body = re.sub(r"\s", "", window_text)
    if len(body) < 10:
        return 0.0

    # Tail portion (last 120 chars, or whole body if shorter)
    tail = body[-120:] if len(body) >= 120 else body

    # 1. Open questions / Bayesian suspense proxy
    suspense_matches = len(SUSPENSE_PATTERN.findall(tail))
    v_ely = min(2.5, suspense_matches * 1.2)

    # 2. Terminal acceleration (staccato shortening of sentences near tail vs window average)
    all_sents = [s for s in re.split(r"[。！？!?]", body) if len(s) >= 2]
    if all_sents:
        avg_len = np.mean([len(s) for s in all_sents])
        tail_sents = [s for s in re.split(r"[。！？!?]", tail) if len(s) >= 2]
        if tail_sents:
            tail_avg_len = np.mean([len(s) for s in tail_sents])
            # Acceleration if tail sentences are distinctly shorter
            accel = max(0.0, (avg_len - tail_avg_len) / max(avg_len, 1.0))
        else:
            accel = 0.0
    else:
        accel = 0.0

    # 3. Tidy resolution penalty (closure)
    closure_matches = len(CLOSURE_PATTERN.findall(tail))
    d_closure = min(2.0, closure_matches * 0.8)
    closure_damping = math.exp(-GAMMA_CLOSURE_PENALTY * d_closure)

    c_cliff = (v_ely * 1.0 + accel * 1.5) * closure_damping
    return float(c_cliff)


# -----------------------------------------------------------------------------
# HW-25: Setting Disclosure UID Coupling Ratio (Gamma_UID)
# -----------------------------------------------------------------------------
def extract_hw25_uid_ratio(window_text: str) -> float:
    """HW-25: Setting Disclosure Pace Allocation Coupling Ratio Gamma_UID.

    Evaluates:
      Gamma_UID = L_world / K_carrier
    Where:
      - L_world = r_new_entity * (1 + r_expository): Setting/world-building cognitive load.
      - K_carrier = epsilon_0 + alpha_turn * d_turn + alpha_sens * g_sensory: Kinetic momentum carrier.
    """
    body = re.sub(r"\s", "", window_text)
    n = max(len(body), 1)

    # 1. World-building load (kanji compound density >= 3 chars, technical/proper noun proxy)
    kanji_compounds = re.findall(r"[\u4e00-\u9fff]{3,}", body)
    compound_chars = sum(len(c) for c in kanji_compounds)
    r_entity = compound_chars / n

    # Expository explanations
    expository_matches = len(EXPOSITORY_PATTERN.findall(body))
    r_expository = min(2.0, expository_matches * 0.5)

    l_world = r_entity * (1.0 + r_expository)

    # 2. Kinetic momentum carrier (dialogue turns + action verbs)
    dialogues = DIALOGUE_PATTERN.findall(body)
    dia_chars = sum(len(d) for d in dialogues)
    d_turn = dia_chars / n

    kinetic_matches = len(KINETIC_VERBS.findall(body))
    g_sensory = min(1.0, kinetic_matches * 3.0 / n)

    k_carrier = EPSILON_CARRIER_FLOOR + 0.55 * d_turn + 0.45 * g_sensory

    gamma_uid = l_world / k_carrier
    return float(gamma_uid)


# -----------------------------------------------------------------------------
# Combined Feature Extractor
# -----------------------------------------------------------------------------
def extract_all_narrative_pacing(window_text: str) -> Dict[str, float]:
    """Extract all three prototype metrics for a given window."""
    return {
        "hw22_metaphor_leap": extract_hw22_metaphor_leap(window_text),
        "hw24_cliffhanger": extract_hw24_cliffhanger(window_text),
        "hw25_uid_ratio": extract_hw25_uid_ratio(window_text),
    }
