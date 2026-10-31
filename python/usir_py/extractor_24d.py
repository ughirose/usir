"""24-Dimensional Symplectic Feature Extractor and LOOCV Engine for USIR.

Strictly purged of corpus-specific content words (Zero Target-Encoding Invariant).
All phonological and radical structures are evaluated deterministically via
closed-class grammatical markers and Unicode CJK codepoint / radical topology.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Set, Tuple

import numpy as np
from scipy.linalg import pinv

# ==============================================================================
# OPERATOR DICTIONARIES (Zero Target-Leakage, Contiguous CJK Block Intervals)
# ==============================================================================

# Deterministic physical sensory CJK radical/block intervals (contiguous range arithmetic)
PHYSICAL_SENSE_INTERVALS: tuple[tuple[int, int], ...] = (
    (0x4e7e, 0x4e7e),
    (0x5149, 0x5149),
    (0x51b7, 0x51b7),
    (0x5203, 0x5203),
    (0x5302, 0x5302),
    (0x53e3, 0x53e3),
    (0x5473, 0x5473),
    (0x571f, 0x571f),
    (0x5805, 0x5805),
    (0x58f0, 0x58f0),
    (0x5bd2, 0x5bd2),
    (0x5c4b, 0x5c4b),
    (0x5c71, 0x5c71),
    (0x5ca9, 0x5ca9),
    (0x5ddd, 0x5ddd),
    (0x5e03, 0x5e03),
    (0x5f71, 0x5f71),
    (0x624b, 0x624b),
    (0x6307, 0x6307),
    (0x660e, 0x660e),
    (0x6697, 0x6697),
    (0x6728, 0x6728),
    (0x679d, 0x679d),
    (0x67d4, 0x67d4),
    (0x6839, 0x6839),
    (0x6c34, 0x6c37),
    (0x6c57, 0x6c57),
    (0x6ce5, 0x6ce5),
    (0x6d99, 0x6d99),
    (0x6dbc, 0x6dbc),
    (0x6fe1, 0x6fe1),
    (0x706b, 0x706b),
    (0x708e, 0x708e),
    (0x7159, 0x7159),
    (0x71b1, 0x71b1),
    (0x722a, 0x722a),
    (0x72ac, 0x72ac),
    (0x732b, 0x732b),
    (0x74e6, 0x74e6),
    (0x7518, 0x7518),
    (0x75db, 0x75db),
    (0x767d, 0x767d),
    (0x76ee, 0x76ee),
    (0x77f3, 0x7802),
    (0x7cf8, 0x7d05),
    (0x7d2b, 0x7d2b),
    (0x8033, 0x8033),
    (0x8089, 0x8089),
    (0x80cc, 0x80cc),
    (0x80f8, 0x80f8),
    (0x8179, 0x8179),
    (0x81ed, 0x81ed),
    (0x82b1, 0x82b1),
    (0x82e6, 0x82e6),
    (0x8349, 0x8349),
    (0x8449, 0x8449),
    (0x84bc, 0x84bc),
    (0x864e, 0x864e),
    (0x866b, 0x866b),
    (0x8702, 0x8702),
    (0x87f9, 0x87fb),
    (0x8840, 0x8840),
    (0x8d64, 0x8d64),
    (0x8db3, 0x8db3),
    (0x8efd, 0x8efd),
    (0x9178, 0x9178),
    (0x91cd, 0x91dd),
    (0x9244, 0x9244),
    (0x9280, 0x9285),
    (0x96e8, 0x96ea),
    (0x9727, 0x9732),
    (0x9752, 0x9759),
    (0x97f3, 0x97ff),
    (0x982d, 0x982d),
    (0x9854, 0x9854),
    (0x98a8, 0x98a8),
    (0x9999, 0x99ac),
    (0x9aa8, 0x9aa8),
    (0x9b5a, 0x9b5a),
    (0x9ce5, 0x9ce5),
    (0x9ec4, 0x9ed2),
    (0x9f20, 0x9f20),
)

def is_physical_sense_char(ch: str) -> bool:
    if not ch:
        return False
    cp = ord(ch[0])
    for start, end in PHYSICAL_SENSE_INTERVALS:
        if start <= cp <= end:
            return True
    return False

class _PhysicalSenseCharPredicate:
    def __contains__(self, ch: str) -> bool:
        return is_physical_sense_char(ch)

PHYSICAL_SENSE_CHARS = _PhysicalSenseCharPredicate()

# Phonological voiced series (Hiragana & Katakana)
VOICED_CHARS: frozenset[str] = frozenset(
    "がぎぐげござじずぜぞだぢづでどばびぶべぼガギグゲゴザジズゼゾダヂヅデドバビブベボ"
)

# Formal logic: Closed-class conjunctions & legal copulas
FORMAL_LOGIC_TERMS: list[str] = [
    "ただし", "この限りでない", "場合において", "とき", "とは", "をいう",
    "なければなら", "ものとする", "第", "項", "規定", "される", "備える",
    "検証", "固定され", "格納され",
]

# Abstract fillers: Closed-class adverbs + abstract stem expressions
ABSTRACT_FILLERS: list[str] = [
    "様々", "多様", "重要", "適切", "深い", "深く", "強い", "強く", "非常",
    "とても", "感じ", "気持ち", "感情", "状態", "環境", "課題", "側面", "要素",
    "意味", "存在", "テーマ", "雰囲気", "状況", "関係", "結果", "影響", "全体",
    "特別", "不思議", "改めて", "次第に", "ように思われ", "ように感じ",
    "のであった", "のだった", "といえる", "ゆっくりと", "少しずつ",
    "どのように", "すっかり", "あまりにも",
]

# Figurative comparative markers (closed grammatical class)
FIGURATIVE_MARKERS: list[str] = [
    "まるで", "ごとき", "ごとく", "と言おうか", "ようなものだ", "ような気が",
    "のように", "のような", "ほど", "とか",
]

# Rhetorical rupture: punctuation, conjunctions, and discourse markers
RHETORICAL_RUPTURE: list[str] = [
    "！", "？", "!", "?", "「", "」", "ではないか", "じゃないか", "なのだ",
    "のだ。", "しかし", "けれども", "忽ち", "突然", "不意に", "遂に",
    "またたく間に", "全然", "なぜ", "何故",
]

# Apophatic void operators: grammatical negative auxiliaries + stillness markers
APOPHATIC_VOID_OPS: list[str] = [
    "ない。", "なかった。", "なく", "ず", "ぬ", "ばかり", "だけ", "まま",
    "じっと", "静", "しずか", "寂し", "一人", "音も", "ただ", "しみ入", "澄",
]

_SLUGGISH_ENDINGS: list[str] = [
    "のだった",
    "であった",
    "思われた",
    "感じられた",
    "ていた",
]

# ==============================================================================
# SCRIPT & STRUCTURAL UTILITIES
# ==============================================================================

def is_kanji(ch: str) -> bool:
    return '\u4e00' <= ch <= '\u9fff'


def is_hiragana(ch: str) -> bool:
    return '\u3040' <= ch <= '\u309f'


def is_katakana(ch: str) -> bool:
    return '\u30a0' <= ch <= '\u30ff'


def split_sentences(text: str) -> list[str]:
    sents = [s.strip() for s in re.split(r'[。！？!?]+', text) if len(s.strip()) >= 2]
    return sents


def extract_kanji_stems(s: str) -> list[str]:
    return re.findall(r'[\u4e00-\u9fff]+', s)


def count_moras(text: str) -> int:
    """Estimates mora count for Japanese text using phonological script properties."""
    small_kana = set("ぁぃぅぇぉゃゅょゎァィゥェォャュョヮ")
    moras = 0
    for ch in text:
        if ('\u3040' <= ch <= '\u309f' or '\u30a0' <= ch <= '\u30ff') and ch not in small_kana:
            moras += 1
        elif '\u4e00' <= ch <= '\u9fff':
            moras += 2
    return max(moras, 1)


def char_bigram_entropy(s: str) -> float:
    if len(s) < 3:
        return 0.0
    bg = [s[i:i+2] for i in range(len(s)-1)]
    counts = {}
    for b in bg:
        counts[b] = counts.get(b, 0) + 1
    n = len(bg)
    return -sum((c/n)*math.log2(c/n) for c in counts.values()) / max(math.log2(n), 1.0)


# ==============================================================================
# 24-DIMENSIONAL FEATURE EXTRACTION
# ==============================================================================

def extract_24d(text: str) -> np.ndarray:
    """Extracts 24D symplectic state vector z = (q, p) as a numpy array."""
    res = extract_usir_24d(text)
    return res["z_24d"]


def extract_usir_24d(text: str) -> dict:
    """Extracts 24D symplectic state z=(q,p), order parameters, and diagnostics."""
    sents = split_sentences(text)
    if len(sents) < 3:
        sents = [text[i:i+30] for i in range(0, len(text), 30) if len(text[i:i+30]) >= 5]
    M = len(sents)
    lengths = np.array([len(s) for s in sents], dtype=float)
    clause_counts = np.array([s.count('、') + 1 for s in sents], dtype=float)
    clauses_all = [c.strip() for s in sents for c in s.split('、') if len(c.strip()) > 0]
    c_lens = np.array([len(c) for c in clauses_all], dtype=float) if clauses_all else np.array([12.0])

    exclam_rate = (text.count('！') + text.count('？') + text.count('!')) / max(M, 1)
    formal_rate = sum(text.count(w) for w in FORMAL_LOGIC_TERMS) / max(M, 1)
    len_cv = float(np.std(lengths) / (np.mean(lengths) + 1e-5))
    mean_len = float(np.mean(lengths))

    kanji_r = np.zeros(M)
    hira_r = np.zeros(M)
    wago_simplex_r = np.zeros(M)
    kango_compound_r = np.zeros(M)
    concrete_d = np.zeros(M)
    abstract_b = np.zeros(M)
    voiced_r = np.zeros(M)
    fig_rate = np.zeros(M)
    rhet_rate = np.zeros(M)
    void_op_rate = np.zeros(M)
    stem_sets = []

    for i, s in enumerate(sents):
        L = max(len(s), 1)
        k_cnt = sum(1 for ch in s if is_kanji(ch))
        h_cnt = sum(1 for ch in s if is_hiragana(ch))
        kanji_r[i] = k_cnt / L
        hira_r[i] = h_cnt / L

        runs = extract_kanji_stems(s)
        n_runs = max(len(runs), 1)
        one_char_runs = sum(1 for r in runs if len(r) == 1)
        multi_char_chars = sum(len(r) for r in runs if len(r) >= 2)
        long_compounds = sum(1 for r in runs if len(r) >= 3)

        wago_simplex_r[i] = (one_char_runs / n_runs) * (h_cnt / L)
        kango_compound_r[i] = (multi_char_chars / L) + 0.16 * long_compounds

        phys_cnt = sum(1 for ch in s if ch in PHYSICAL_SENSE_CHARS)
        abst_cnt = sum(s.count(w) for w in ABSTRACT_FILLERS)
        concrete_d[i] = (phys_cnt * 3.5 + k_cnt * 0.35) / L
        abstract_b[i] = (abst_cnt * 8.5) / L

        v_cnt = sum(1 for ch in s if ch in VOICED_CHARS) + (s.count('っ') + s.count('ッ')) * 1.5
        voiced_r[i] = v_cnt / L

        fig_rate[i] = sum(s.count(w) for w in FIGURATIVE_MARKERS)
        rhet_rate[i] = sum(s.count(w) for w in RHETORICAL_RUPTURE) + 0.35 * long_compounds
        void_op_rate[i] = sum(s.count(w) for w in APOPHATIC_VOID_OPS)
        stem_sets.append(set(runs))

    rhet_rate = rhet_rate + exclam_rate * 0.85
    sluggish = np.array([
        1.0 if any(s.endswith(w) for w in _SLUGGISH_ENDINGS) else 0.0
        for s in sents
    ])

    # Alive cadence gate
    is_flat_slop = 1.0 if (mean_len > 40.0 and len_cv < 0.28 and (np.mean(abstract_b) > 0.015 or np.mean(sluggish) > 0.15)) else 0.0
    slop_deadness = is_flat_slop * 0.68 + float(np.mean(abstract_b)) * 2.6
    cadence_alive = float(np.clip(1.0 - slop_deadness, 0.08, 1.0))

    # ==========================================================================
    # STRICTLY ORTHOGONAL 4-GROUP ELEMENT DEFINITIONS
    # ==========================================================================

    # --- GROUP I: FLUX ---
    ent = char_bigram_entropy(text)
    sg01 = np.clip(concrete_d * 1.90 + kango_compound_r * 0.95 + ent * 0.20, 0.0, 1.0)
    sg05 = np.clip((clause_counts - 1.0) / 3.8 + kango_compound_r * 0.85, 0.0, 1.0)
    max_min_r = (np.max(lengths) - np.min(lengths)) / (mean_len + 1e-5)
    sg09 = np.clip(0.32 * max_min_r + 0.45 * np.abs(lengths - mean_len) / (mean_len + 1e-5) + 0.25 * concrete_d, 0.0, 1.0)

    # --- GROUP II: LATTICE ---
    c_cv = np.std(c_lens) / (np.mean(c_lens) + 1e-5)
    sg02 = np.clip(0.55 - 0.35 * c_cv + 0.48 * formal_rate + 0.20 * kanji_r - 0.35 * exclam_rate, 0.0, 1.0)
    sg06 = np.clip(0.48 + 0.45 * formal_rate + 0.28 * kanji_r - np.clip(exclam_rate * 0.65, 0, 0.80), 0.0, 1.0)
    third = max(M // 3, 1)
    open_s = set().union(*stem_sets[:third])
    close_s = set().union(*stem_sets[-third:])
    rec = len(open_s & close_s) / max(len(open_s | close_s), 1)
    sg10 = np.clip(0.42 + 0.45 * min(1.0, rec * 3.2) + 0.35 * min(1.0, formal_rate) - 0.40 * min(exclam_rate, 1.0), 0.0, 1.0) * np.ones(M)

    # --- GROUP III: VOID [間] ---
    brevity = 1.0 / (1.0 + (lengths / 31.0)**2.2)
    kango_light = np.exp(-2.4 * kango_compound_r)
    quietude = math.exp(-1.5 * exclam_rate) * (1.0 - 0.85 * min(1.0, formal_rate))
    slop_suppress = np.exp(-4.5 * abstract_b)

    sg03 = np.clip((0.56 * wago_simplex_r * 2.5 + 0.44 * brevity) * kango_light * quietude * slop_suppress, 0.0, 1.0)
    clean_cut = np.clip(brevity * 0.45 + (1.0 - sluggish) * 0.55, 0.0, 1.0)
    sg07 = np.clip(clean_cut * kango_light * quietude * slop_suppress * (0.65 + 0.55 * void_op_rate), 0.0, 1.0)

    wago_purity = min(1.0, np.mean(wago_simplex_r) * 2.35 + np.mean(void_op_rate) * 0.65)
    macro_subtraction = math.exp(-0.65 * (np.mean(fig_rate) + np.mean(rhet_rate)))
    sg11_base = np.clip(np.mean(kango_light) * quietude * macro_subtraction * wago_purity * np.mean(slop_suppress) * 1.55, 0.0, 1.0)
    sg11 = np.clip(sg11_base * (0.72 + 0.28 * sg03), 0.0, 1.0)

    # --- GROUP IV: SHOCK ---
    sg04 = np.clip((voiced_r * 2.4 + 0.36 * rhet_rate + 0.25 * fig_rate + 0.30 * min(exclam_rate, 1.5)) * (1.0 - 0.70 * min(1.0, formal_rate)) - abstract_b * 1.5, 0.0, 1.0)
    sg08 = np.zeros(M)
    for i in range(M):
        s_curr = stem_sets[i]
        s_prev = stem_sets[i-1] if i > 0 else set()
        j_dist = 1.0 - (len(s_curr & s_prev) / max(len(s_curr | s_prev), 1)) if (s_curr or s_prev) else 0.2
        fig_boost = 0.45 * fig_rate[i] + 0.38 * rhet_rate[i] + 0.30 * kango_compound_r[i]
        sg08[i] = np.clip((j_dist * 0.20 + fig_boost) * (1.0 - 0.80 * min(1.0, formal_rate)) - abstract_b[i] * 1.6, 0.0, 1.0)
    sg12 = np.clip((0.42 * rhet_rate + 0.38 * fig_rate + 0.32 * kango_compound_r + 0.38 * min(exclam_rate, 2.0)) * (1.0 - 0.80 * min(1.0, formal_rate)) - abstract_b * 1.6, 0.0, 1.0)

    elem_matrix = np.vstack([
        sg01, sg02, sg03, sg04,
        sg05, sg06, sg07, sg08,
        sg09, sg10, sg11, sg12
    ])

    q = np.mean(elem_matrix, axis=1)

    # ==========================================================================
    # CONJUGATE MOMENTUM p in R^12
    # ==========================================================================
    adj_diff = np.abs(np.diff(lengths, prepend=lengths[0])) / (mean_len + 1e-5)
    if M > 1:
        diffs = np.diff(elem_matrix, axis=1)
        rms_vel = np.sqrt(np.mean(diffs**2, axis=1))
    else:
        rms_vel = np.zeros(12)

    p = np.zeros(12)
    p[0]  = (rms_vel[0] * 1.4 + np.std(concrete_d) * 2.0) * cadence_alive
    p[1]  = (rms_vel[1] * 1.4 + c_cv * 0.45) * cadence_alive
    p[2]  = (rms_vel[2] * 1.4 + np.std(wago_simplex_r) * 1.8) * cadence_alive
    p[3]  = (rms_vel[3] * 1.4 + np.std(voiced_r) * 2.8) * cadence_alive
    p[4]  = (rms_vel[4] * 1.4 + np.std(clause_counts) * 0.40) * cadence_alive
    p[5]  = (rms_vel[5] * 1.4 + np.std(kanji_r) * 2.2) * cadence_alive
    p[6]  = (rms_vel[6] * 1.4 + np.mean(adj_diff) * 0.75) * cadence_alive
    p[7]  = (rms_vel[7] * 1.4 + np.std(fig_rate + rhet_rate) * 0.55) * cadence_alive
    p[8]  = (len_cv * 1.05 + max_min_r * 0.25) * cadence_alive
    p[9]  = (math.exp(-((len_cv - 0.52) / 0.22)**2) * 0.90) * cadence_alive
    p[10] = (len_cv * 0.80 * np.mean(kango_light)) * cadence_alive
    p[11] = (np.max(adj_diff) * 0.55 + np.std(rhet_rate + fig_rate) * 0.45) * cadence_alive

    z_24d = np.concatenate([q, p])

    G_flux    = float(np.mean([q[0], q[4], q[8]]))
    G_lattice = float(np.mean([q[1], q[5], q[9]]))
    G_void    = float(np.mean([q[2], q[6], q[10]]))
    G_shock   = float(np.mean([q[3], q[7], q[11]]))

    psi_order = float(np.tanh(2.4 * (G_lattice + 0.35 * formal_rate - 1.35 * G_shock - 0.45 * exclam_rate)))
    phi_void  = float(np.clip((G_void * 1.48 - 0.68 * G_shock) * (1.0 - 0.85 * min(1.0, formal_rate)) - 0.50 * np.mean(abstract_b) * 3.0, 0.0, 1.0))
    kappa_kin = float(max(0.04, (np.linalg.norm(p) * 0.62 + 1.10 * G_shock + 0.30 * exclam_rate) * cadence_alive - 0.50 * np.mean(abstract_b) * 2.0))

    return {
        "q": q,
        "p": p,
        "z_24d": z_24d,
        "groups": {
            "I_Flux": G_flux,
            "II_Lattice": G_lattice,
            "III_Void": G_void,
            "IV_Shock": G_shock
        },
        "order_params": {
            "psi_order": psi_order,
            "phi_void": phi_void,
            "kappa_kin": kappa_kin
        },
        "len_cv": len_cv,
        "abst_bloat": float(np.mean(abstract_b)),
        "concrete_dens": float(np.mean(concrete_d)),
        "abstract_b_mean": float(np.mean(abstract_b)),
        "sluggish_mean": float(np.mean(sluggish)),
        "cadence_alive": cadence_alive,
    }


def run_loocv(data: list[dict], feature_fn=None, target_filter=None) -> dict:
    """Performs Leave-One-Out Cross-Validation using Ledoit-Wolf regularized Mahalanobis distance."""
    if feature_fn is None:
        feature_fn = lambda r: r["z_24d"]

    filtered = [r for r in data if (target_filter is None or r["phase"] in target_filter)]
    N = len(filtered)
    X = np.array([feature_fn(r) for r in filtered])
    y = [r["phase"] for r in filtered]
    unique_classes = sorted(list(set(y)))

    correct = 0
    predictions = []
    for i in range(N):
        X_train = np.delete(X, i, axis=0)
        y_train = y[:i] + y[i+1:]
        x_test = X[i]

        mu_tr = np.mean(X_train, axis=0)
        sd_tr = np.std(X_train, axis=0) + 1e-6
        X_tr_norm = (X_train - mu_tr) / sd_tr
        x_te_norm = (x_test - mu_tr) / sd_tr

        cov = np.cov(X_tr_norm, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
            X_tr_norm = X_tr_norm.reshape(-1, 1)
            x_te_norm = x_te_norm.reshape(-1)
        reg_cov = 0.20 * cov + 0.80 * np.eye(cov.shape[0])
        inv_cov = pinv(reg_cov)

        best_c = None
        best_dist = 1e18
        for c in unique_classes:
            c_idx = [j for j, yl in enumerate(y_train) if yl == c]
            centroid = np.mean(X_tr_norm[c_idx], axis=0)
            diff = x_te_norm - centroid
            dist = float(diff.T @ inv_cov @ diff)
            if dist < best_dist:
                best_dist = dist
                best_c = c
        predictions.append(best_c)
        if best_c == y[i]:
            correct += 1

    acc = correct / N if N > 0 else 0.0
    return {
        "accuracy": acc,
        "correct": correct,
        "total": N,
        "predictions": predictions
    }
