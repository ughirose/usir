"""GRPO Style Reward Engine using USIR 24D Symplectic State and Anti-Slop Penalty."""

import numpy as np
from .extractor_24d import extract_usir_24d
from .corpus_40_benchmark import CORPUS

# Precompute Phase Centroids from the Golden Corpus
_PROCESSED_CORPUS = [{**item, **extract_usir_24d(item["text"])} for item in CORPUS]

PHASE_CENTROIDS_24D = {}
for ph in ["I_Crystalline", "II_Critical", "III_Void", "IV_Turbulent", "V_Deadzone"]:
    sub = [r for r in _PROCESSED_CORPUS if r["phase"] == ph]
    if sub:
        PHASE_CENTROIDS_24D[ph] = np.mean([r["z_24d"] for r in sub], axis=0)

# Precompute empirical diagonal variances across all 40 passages for Mahalanobis scaling
_ALL_Z = np.array([r["z_24d"] for r in _PROCESSED_CORPUS])
_FEATURE_STD = np.std(_ALL_Z, axis=0) + 1e-4

# Cleanroom constraint dictionaries aligned with scripts/scan_cleanroom_copyright.py
BANNED_TARGET_KEYWORDS = [
    "メロス", "李徴", "島村", "下人", "芥川", "太宰", "中島", "梶井",
    "志賀", "川端", "芭蕉", "漱石", "羅生門", "山月記", "雪国", "檸檬"
]

HARD_AI_CLICHE_FRAMES = [
    "テーマ", "ように思われ", "ように感じられ", "させられたのだった",
    "様々な思いが", "特別な瞬間となった", "改めて深く", "強い後悔と悲しみ",
    "心の均衡を失った", "言いようのない不安を感じていた", "のではないだろうか",
    "と言えるだろう", "のだった。", "思われた。", "感じられた。",
    "傾向が見受けられ", "重要な要素", "多様な側面"
]

META_GOODHART_TERMS = [
    "シンプレクティック", "ハミルトニアン", "マハラノビス",
    "秩序変数", "臨界相", "ヴォイド相", "デッドゾーン"
]

MODERN_LOANWORDS = [
    "猛スピード", "スピード", "システム", "レベル", "ストレス", "タイミング",
    "イメージ", "メリット", "デメリット", "プロセス", "エネルギー", "パーセント",
    "アプローチ", "コミット", "モチベーション", "プロジェクト", "ターゲット",
    "シチュエーション", "プライド", "リアル", "クオリティ", "テンション",
    "リセット", "スペック", "ルーティン", "センス", "モチベ", "コスパ",
    "アピール", "リアリティ"
]

BRACKET_PAIRS = {
    "「": "」", "『": "』", "（": "）", "【": "】", "《": "》", "〔": "〕",
    "(": ")", "[": "]", "{": "}"
}


def compute_syntax_multiplier(text: str) -> float:
    """Computes M_syntax in [0.1, 1.0] evaluating bracket closure, punctuation, and repetition."""
    if not text:
        return 0.1

    # 1. Bracket matching
    open_brackets = set(BRACKET_PAIRS.keys())
    close_brackets = {v: k for k, v in BRACKET_PAIRS.items()}
    stack: list[str] = []
    matched = 0
    unmatched = 0

    for ch in text:
        if ch in open_brackets:
            stack.append(ch)
        elif ch in close_brackets:
            expected = close_brackets[ch]
            if stack and stack[-1] == expected:
                stack.pop()
                matched += 1
            else:
                unmatched += 1
    unmatched += len(stack)
    total_brk = (2 * matched) + unmatched
    bracket_score = 1.0 if total_brk == 0 else (2.0 * matched) / float(total_brk)

    # 2. Sentence ending integrity
    valid_terminators = {"。", "！", "？", "!", "?", "」", "』", "）", "…", "—"}
    stripped = text.strip()
    ending_score = 1.0 if stripped and stripped[-1] in valid_terminators else 0.70

    # 3. Degenerate repetition penalty
    rep_penalty = 0.0
    if len(text) >= 12:
        for i in range(len(text) - 8):
            quad = text[i:i+4]
            # If the 4-gram repeats 3+ times
            if text.count(quad) >= 4:
                rep_penalty = max(rep_penalty, 0.40)
                break

    m_syntax = float(np.clip(bracket_score * ending_score * (1.0 - rep_penalty), 0.10, 1.0))
    return m_syntax


def compute_semantic_multiplier(extracted: dict, text: str) -> float:
    """Computes M_semantic in [0.05, 1.0] from cadence_alive, abstract bloat, and AI clichés."""
    cadence = float(extracted.get("cadence_alive", 0.5))
    abst_bloat = float(extracted.get("abstract_b_mean", 0.0))

    # Cliché penalties
    cliche_count = sum(1 for f in HARD_AI_CLICHE_FRAMES if f in text)

    # Baseline semantic vitality: alive cadence + low abstract bloat
    vitality = (cadence ** 1.1) * max(0.05, (1.0 - 2.0 * abst_bloat))
    penalty_cliche = 0.60 ** min(cliche_count, 4)

    m_semantic = float(np.clip(vitality * penalty_cliche, 0.05, 1.0))
    return m_semantic


def compute_cleanroom_multiplier(
    text: str,
    is_authentic_masterwork: bool = False,
    masterwork_texts: list[str] | None = None,
) -> float:
    """Computes M_cleanroom in [0.05, 1.0] checking copyright and cleanroom restrictions."""
    if is_authentic_masterwork:
        # Authentic public domain masterwork: character names permitted, check only meta terms
        meta_count = sum(1 for m in META_GOODHART_TERMS if m in text)
        return float(0.05 if meta_count > 0 else 1.0)

    # For synthetic rewrites: strict zero-tolerance cleanroom
    banned_hits = sum(1 for w in BANNED_TARGET_KEYWORDS if w in text)
    meta_hits = sum(1 for m in META_GOODHART_TERMS if m in text)
    loan_hits = sum(1 for lw in MODERN_LOANWORDS if lw in text)
    cliche_hits = sum(1 for f in HARD_AI_CLICHE_FRAMES if f in text)

    # 7-gram leakage check if reference masterworks provided
    leak_hits = 0
    if masterwork_texts and len(text) >= 7:
        text_7grams = {text[i:i+7] for i in range(len(text) - 6)}
        for mw in masterwork_texts:
            if len(mw) >= 7:
                for i in range(len(mw) - 6):
                    if mw[i:i+7] in text_7grams:
                        leak_hits += 1
                        break
            if leak_hits > 0:
                break

    total_violations = banned_hits * 3 + meta_hits * 3 + leak_hits * 4 + loan_hits + cliche_hits
    if total_violations == 0:
        return 1.0
    return float(max(0.05, 0.20 ** min(total_violations, 3)))


def compute_v3_multiplicative_grpo_reward(
    text: str,
    target: str = "OBSIDIAN_VOID",
    is_authentic_masterwork: bool = False,
    masterwork_texts: list[str] | None = None,
    alpha_metric: float = 1.00,
) -> dict:
    r"""Computes v3.0 4-factor multiplicative penalty GRPO reward:

    R_GRPO(y) = M_syntax(y) * M_semantic(y) * M_cleanroom(y) * exp(-1/2 * ||z(y) - z*||_G^2)
    where ||z(y) - z*||_G^2 = (1 / (24 * alpha)) * \sum_{i=1}^{24} ((z_i - z*_i) / sigma_i)^2.

    Returns:
        dict containing 'reward', 'm_syntax', 'm_semantic', 'm_cleanroom', 'exp_dist', 'dist_g'.
    """
    extracted = extract_usir_24d(text)
    z = extracted["z_24d"]

    if target in USIR_CANONICAL_TARGETS:
        target_mu = USIR_CANONICAL_TARGETS[target]["centroid"]
    elif target in PHASE_CENTROIDS_24D:
        target_mu = PHASE_CENTROIDS_24D[target]
    else:
        raise ValueError(f"Unknown target '{target}'.")

    # Precision-weighted Mahalanobis distance squared: ||z - z*||_G^2
    diff_norm = (z - target_mu) / _FEATURE_STD
    # Scaled by dimension 24 and calibration factor alpha
    dist_g_sq = float(np.sum(diff_norm**2) / (24.0 * alpha_metric))
    dist_g = float(np.sqrt(max(0.0, dist_g_sq)))

    # Factor 1: Gaussian style manifold attractor
    exp_dist = float(np.exp(-0.5 * dist_g_sq))

    # Factor 2: Syntax multiplier
    m_syntax = compute_syntax_multiplier(text)

    # Factor 3: Semantic vitality / anti-slop multiplier
    m_semantic = compute_semantic_multiplier(extracted, text)

    # Factor 4: Cleanroom compliance multiplier
    m_cleanroom = compute_cleanroom_multiplier(
        text,
        is_authentic_masterwork=is_authentic_masterwork,
        masterwork_texts=masterwork_texts,
    )

    # 4-Factor Multiplicative Reward in [0, 1]
    r_grpo = float(np.clip(m_syntax * m_semantic * m_cleanroom * exp_dist, 0.0, 1.0))

    return {
        "reward": r_grpo,
        "m_syntax": m_syntax,
        "m_semantic": m_semantic,
        "m_cleanroom": m_cleanroom,
        "exp_dist": exp_dist,
        "dist_g": dist_g,
        "target": target,
        "cadence_alive": float(extracted["cadence_alive"]),
        "len_cv": float(extracted["len_cv"]),
        "extracted": extracted,
    }

USIR_CANONICAL_TARGETS = {
    "AMBER_CRITICAL": {
        "phase": "II_Critical",
        "centroid": PHASE_CENTROIDS_24D.get("II_Critical"),
        "description": "Phase II Self-Organized Criticality (1/f rhythm volatility, dynamic leap)",
    },
    "OBSIDIAN_VOID": {
        "phase": "III_Void",
        "centroid": PHASE_CENTROIDS_24D.get("III_Void"),
        "description": "Phase III Superfluid Void (Wago brevity, stillness operators, clean ending cut)",
    },
    "PORCELAIN_CRYSTAL": {
        "phase": "I_Crystalline",
        "centroid": PHASE_CENTROIDS_24D.get("I_Crystalline"),
        "description": "Phase I Crystalline Order (High formal structure, axiomatic symmetry)",
    },
}


def compute_grpo_reward(
    text: str,
    target: str = "OBSIDIAN_VOID",
    beta: float = 0.08,
    slop_penalty_weight: float = 1.20,
) -> dict:
    """Computes GRPO style reinforcement learning reward R_USIR(y).

    Args:
        text: Input generated text candidate y.
        target: Target particle or phase name ('OBSIDIAN_VOID', 'AMBER_CRITICAL', or a phase name).
        beta: Distance scaling factor.
        slop_penalty_weight: Weight of the narrative AI slop penalty.

    Returns:
        dict containing 'reward', 'distance', 'slop_deadness', 'phase_detected', and 'extracted'.
    """
    extracted = extract_usir_24d(text)
    z = extracted["z_24d"]

    if target in USIR_CANONICAL_TARGETS:
        target_mu = USIR_CANONICAL_TARGETS[target]["centroid"]
    elif target in PHASE_CENTROIDS_24D:
        target_mu = PHASE_CENTROIDS_24D[target]
    else:
        raise ValueError(f"Unknown target '{target}'. Valid targets: {list(USIR_CANONICAL_TARGETS.keys())} or {list(PHASE_CENTROIDS_24D.keys())}")

    # Normalized precision-weighted Euclidean distance
    diff_norm = (z - target_mu) / _FEATURE_STD
    dist_sq = float(np.sum(diff_norm**2))
    dist = float(np.sqrt(dist_sq))

    # Base similarity reward in [0, 1]
    r_sim = float(np.exp(-beta * dist))

    # Slop deadness penalty derived from cadence alive
    # cadence_alive is in [0.08, 1.0], 1.0 being alive and <= 0.32 being dead
    slop_deadness = float(1.0 - extracted["cadence_alive"])
    abst_penalty = float(extracted["abstract_b_mean"] * 4.0)
    total_penalty = float(slop_penalty_weight * (slop_deadness * 0.70 + abst_penalty * 0.30))

    total_reward = float(r_sim - total_penalty)

    return {
        "reward": total_reward,
        "similarity_score": r_sim,
        "distance": dist,
        "slop_penalty": total_penalty,
        "cadence_alive": extracted["cadence_alive"],
        "len_cv": extracted["len_cv"],
        "extracted": extracted,
    }
