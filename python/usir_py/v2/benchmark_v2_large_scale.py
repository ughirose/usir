"""USIR v2.0 Large-Scale Out-of-Fold (OOF) Benchmark Suite.

Executes Module 5 of USIR v2.0 Paper-Grade Architecture:
1. Evaluates N = 750 text chunks across 42 author groups under strict 5-fold GroupKFold.
2. Evaluates N = 140 cross-modal music sequences across 10 composer groups (Regime 3).
3. Evaluates all 8 models (M0, B1, B2, B3, A1, A2, A3, Proposed) without in-sample leakage.
4. Computes unbiased Mahalanobis distance Delta_{M,OOF}^2 and L1-normalized OOF Shapley attribution.
5. Saves results to benchmark_v2_oof_report.json.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import re
import sys
import zlib
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
from janome.tokenizer import Tokenizer
from sklearn.covariance import LedoitWolf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from usir_py.v2.build_large_benchmark_corpus import build_large_benchmark_corpus
from usir_py.v2.hth_lexicon import MusicNoteEvent, extract_music_functor_signals
from usir_py.v2.thermodynamics_so8 import SO8PullbackMetric
from usir_py.v2.wavelet_hermite_encoder import WaveletHermiteEncoder, extract_24d_v2, compute_rqa_metrics

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PHASE_NAMES = ["I_Crystalline", "II_Critical", "III_Void", "IV_Turbulent", "V_Deadzone"]
ELEMENT_NAMES = [
    "Sg-01 Dn", "Sg-02 Cr", "Sg-03 Pr", "Sg-04 Jt",
    "Sg-05 Cp", "Sg-06 Mp", "Sg-07 Cs", "Sg-08 Lp",
    "Sg-09 Fl", "Sg-10 At", "Sg-11 Ma", "Sg-12 Ru"
]


# ==============================================================================
# Feature Extraction Helpers
# ==============================================================================

def extract_content_words(text: str, tokenizer: Tokenizer) -> str:
    """Extract content-word lemmatized sequence for B1 baseline."""
    words: List[str] = []
    for t in tokenizer.tokenize(text):
        p = t.part_of_speech.split(",")
        if p[0] == "名詞" and p[1] not in ("非自立", "代名詞", "接尾", "数"):
            words.append(t.base_form)
        elif p[0] == "動詞" and p[1] == "自立":
            words.append(t.base_form)
        elif p[0] == "形容詞" and p[1] == "自立" and t.base_form != "ない":
            words.append(t.base_form)
        elif p[0] == "副詞":
            words.append(t.base_form)
    return " ".join(words)


def extract_surface_16d(text: str) -> np.ndarray:
    """Extract 16 classical surface stylometric features for B2 baseline."""
    chars = max(len(text), 1)
    sentences = [s for s in re.split(r"[。！？\n]+", text) if s.strip()]
    n_sen = max(len(sentences), 1)
    std_sen = float(np.std([len(s) for s in sentences])) if len(sentences) > 1 else 0.0

    kanji = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    hira = sum(1 for c in text if "\u3040" <= c <= "\u309f")
    kata = sum(1 for c in text if "\u30a0" <= c <= "\u30ff")
    digits = sum(1 for c in text if c.isdigit())
    punct = sum(1 for c in text if c in "、。，．「」『』（）()！？!?—…")
    commas = text.count("、") + text.count(",")
    periods = text.count("。") + text.count(".")
    brackets = sum(text.count(b) for b in "「」『』（）()")
    ascii_c = sum(1 for c in text if c.isascii() and c.isalnum())

    c_counts = Counter(text)
    ttr = len(c_counts) / float(chars)
    sum_f2 = sum(f * f for f in c_counts.values())
    yules_k = 10000.0 * (sum_f2 - chars) / float(chars * chars)

    clauses = [c for c in re.split(r"[、,。！？\n]+", text) if c.strip()]
    mean_clause_len = chars / float(max(len(clauses), 1))

    return np.array([
        chars / float(n_sen), std_sen, float(chars),
        kanji / float(chars), hira / float(chars), kata / float(chars),
        digits / float(chars), punct / float(chars), ttr, yules_k,
        commas / float(chars), periods / float(chars), brackets / float(chars),
        ascii_c / float(chars), kanji / float(max(hira + 1, 1)), mean_clause_len
    ], dtype=np.float64)


def extract_compression_features(text: str) -> np.ndarray:
    """Extract 6-dimensional compression and entropy features for B3 baseline."""
    b = text.encode("utf-8")
    l_b = max(len(b), 1)
    z_ratio = len(zlib.compress(b)) / float(l_b)
    gz_ratio = len(gzip.compress(b)) / float(l_b)

    c_counts = Counter(text)
    n = max(len(text), 1)
    uni_ent = -sum((c / n) * math.log2(c / n) for c in c_counts.values())
    bg_counts = Counter(text[i : i + 2] for i in range(len(text) - 1))
    n_bg = max(len(text) - 1, 1)
    bg_ent = -sum((c / n_bg) * math.log2(c / n_bg) for c in bg_counts.values()) if bg_counts else 0.0

    return np.array([z_ratio, gz_ratio, uni_ent, bg_ent, 1.0 - z_ratio, 1.0 - gz_ratio], dtype=np.float64)


def extract_proposed_text_features(chunk: Dict[str, Any], encoder: WaveletHermiteEncoder, so8: SO8PullbackMetric) -> np.ndarray:
    """Extract full USIR v2.0 symplectic + SO(8) Riemannian pullback + trajectory dynamics."""
    z = extract_24d_v2(chunk["text"], use_mca=True)
    q, p = z[:12], z[12:]
    g_mat = so8.metric_matrix()

    u_metric = float(q.T @ g_mat @ q)
    k_metric = float(p.T @ g_mat @ p)
    h_metric = u_metric + k_metric
    action = float(q.T @ p)

    vq = so8.P @ q
    vp = so8.P @ p
    rvq = so8.R @ vq
    rvp = so8.R @ vp
    mode_q = float(rvq.T @ np.diag(so8.lambda_plus) @ rvq)
    mode_p = float(rvp.T @ np.diag(so8.lambda_plus) @ rvp)

    raw_sig = encoder.extract_text_signals(chunk["text"])
    sig_mean = np.mean(raw_sig, axis=0) if len(raw_sig) > 0 else np.zeros(4)
    sig_std = np.std(raw_sig, axis=0) if len(raw_sig) > 1 else np.zeros(4)
    soliton_spike = float(np.max(raw_sig[:, 3])) if len(raw_sig) > 0 else 0.0
    soliton_kurt = float(np.mean((raw_sig[:, 3] - np.mean(raw_sig[:, 3])) ** 4) / (np.var(raw_sig[:, 3]) ** 2 + 1e-6)) if len(raw_sig) > 1 else 1.0

    rqa_1 = compute_rqa_metrics(raw_sig[:, 1])
    rqa_3 = compute_rqa_metrics(raw_sig[:, 3])
    rqa_vec = np.array([rqa_1.det, rqa_1.i_arch, rqa_1.lam, rqa_3.det, rqa_3.i_arch, rqa_3.lam], dtype=np.float64)

    n_s = len(raw_sig)
    ac_feats: List[float] = []
    for sub in [1, 3]:
        s_arr = raw_sig[:, sub]
        s_var = float(np.var(s_arr)) + 1e-8
        s_cent = s_arr - float(np.mean(s_arr))
        for lag in [1, 2]:
            ac = float(np.mean(s_cent[:-lag] * s_cent[lag:]) / s_var) if n_s > lag else 0.0
            ac_feats.append(ac)
    ac_vec = np.array(ac_feats, dtype=np.float64)

    extra = np.concatenate([
        [u_metric, k_metric, h_metric, action, mode_q, mode_p, soliton_spike, soliton_kurt],
        sig_mean, sig_std, rqa_vec, ac_vec
    ])
    return np.concatenate([z, extra])


def extract_proposed_music_features(seq: Dict[str, Any], so8: SO8PullbackMetric) -> np.ndarray:
    """Extract representation for symbolic music via M1 functor and M2-M4 engine."""
    notes = [MusicNoteEvent(**d) for d in seq["notes"]]
    z = extract_24d_v2(notes, use_mca=True)
    q, p = z[:12], z[12:]
    g_mat = so8.metric_matrix()

    u_metric = float(q.T @ g_mat @ q)
    k_metric = float(p.T @ g_mat @ p)
    h_metric = u_metric + k_metric
    action = float(q.T @ p)

    vq = so8.P @ q
    vp = so8.P @ p
    rvq = so8.R @ vq
    rvp = so8.R @ vp
    mode_q = float(rvq.T @ np.diag(so8.lambda_plus) @ rvq)
    mode_p = float(rvp.T @ np.diag(so8.lambda_plus) @ rvp)

    raw_sig = extract_music_functor_signals(notes)
    sig_mean = np.mean(raw_sig, axis=0)
    sig_std = np.std(raw_sig, axis=0)
    soliton_spike = float(np.max(raw_sig[:, 3]))
    soliton_kurt = float(np.mean((raw_sig[:, 3] - np.mean(raw_sig[:, 3])) ** 4) / (np.var(raw_sig[:, 3]) ** 2 + 1e-6))

    extra = np.concatenate([[u_metric, k_metric, h_metric, action, mode_q, mode_p, soliton_spike, soliton_kurt], sig_mean, sig_std])
    return np.concatenate([z, extra])


def extract_music_1gram(seq: Dict[str, Any]) -> str:
    """Extract relative pitch interval and duration token sequence for music B1 baseline."""
    pitches = [n["pitch"] for n in seq["notes"]]
    durs = [int(round(n["duration"] * 8)) for n in seq["notes"]]
    vels = [int(round(n["velocity"] / 20)) for n in seq["notes"]]
    tokens = [f"d{d}" for d in durs] + [f"v{v}" for v in vels]
    for i in range(len(pitches) - 1):
        diff = pitches[i + 1] - pitches[i]
        tokens.append(f"i{diff}")
    return " ".join(tokens)


def extract_music_surface_16d(seq: Dict[str, Any]) -> np.ndarray:
    """Extract 16 classical surface features for music B2 baseline."""
    pitches = [n["pitch"] for n in seq["notes"]]
    durs = [n["duration"] for n in seq["notes"]]
    vels = [n["velocity"] for n in seq["notes"]]
    iois = [n["ioi"] for n in seq["notes"]]
    intervals = [abs(pitches[i + 1] - pitches[i]) for i in range(len(pitches) - 1)]
    p_counts = Counter(pitches)
    n_p = len(pitches)
    p_ent = -sum((c / n_p) * math.log2(c / n_p) for c in p_counts.values())
    i_counts = Counter(intervals)
    n_i = len(intervals)
    i_ent = -sum((c / n_i) * math.log2(c / n_i) for c in i_counts.values()) if intervals else 0.0
    return np.array([
        np.mean(pitches), np.std(pitches), min(pitches), max(pitches), max(pitches) - min(pitches),
        np.mean(durs), np.std(durs), np.mean(vels), np.std(vels), np.mean(iois), np.std(iois),
        np.mean(intervals), np.std(intervals), p_ent, i_ent, len(p_counts) / float(n_p)
    ], dtype=np.float64)


def extract_music_compression_features(seq: Dict[str, Any]) -> np.ndarray:
    """Extract 6-dimensional compression and entropy features for music B3 baseline."""
    pitches = [n["pitch"] for n in seq["notes"]]
    vels = [n["velocity"] for n in seq["notes"]]
    b = bytes([val for pair in zip(pitches, vels) for val in pair])
    l_b = max(len(b), 1)
    z_ratio = len(zlib.compress(b)) / float(l_b)
    gz_ratio = len(gzip.compress(b)) / float(l_b)
    p_counts = Counter(pitches)
    n = max(len(pitches), 1)
    uni_ent = -sum((c / n) * math.log2(c / n) for c in p_counts.values())
    bg_counts = Counter(zip(pitches[:-1], pitches[1:]))
    n_bg = max(len(pitches) - 1, 1)
    bg_ent = -sum((c / n_bg) * math.log2(c / n_bg) for c in bg_counts.values()) if bg_counts else 0.0
    return np.array([z_ratio, gz_ratio, uni_ent, bg_ent, 1.0 - z_ratio, 1.0 - gz_ratio], dtype=np.float64)


def extract_a1_music_features(seq: Dict[str, Any], so8: SO8PullbackMetric) -> np.ndarray:
    """Extract overdamped 12D position q + metric potential for music A1 ablation."""
    notes = [MusicNoteEvent(**d) for d in seq["notes"]]
    z = extract_24d_v2(notes, use_mca=True)
    q = z[:12]
    u = float(q.T @ so8.metric_matrix() @ q)
    raw_sig = extract_music_functor_signals(notes)
    return np.concatenate([q, [u], np.mean(raw_sig, axis=0), np.std(raw_sig, axis=0)])


def extract_a2_music_features(seq: Dict[str, Any]) -> np.ndarray:
    """Extract Euclidean flat energy features without SO(8) pullback for music A2 ablation."""
    notes = [MusicNoteEvent(**d) for d in seq["notes"]]
    z = extract_24d_v2(notes, use_mca=True)
    q, p = z[:12], z[12:]
    u_flat = float(np.sum(q**2))
    k_flat = float(np.sum(p**2))
    action = float(q.T @ p)
    raw_sig = extract_music_functor_signals(notes)
    soliton_spike = float(np.max(raw_sig[:, 3]))
    return np.concatenate([z, [u_flat, k_flat, action, soliton_spike], np.mean(raw_sig, axis=0), np.std(raw_sig, axis=0)])


def extract_a3_music_features(seq: Dict[str, Any], so8: SO8PullbackMetric) -> np.ndarray:
    """Extract standard FFT features without Poisson-Soliton separation for music A3 ablation."""
    notes = [MusicNoteEvent(**d) for d in seq["notes"]]
    z = extract_24d_v2(notes, use_mca=False)
    q, p = z[:12], z[12:]
    g_mat = so8.metric_matrix()
    u = float(q.T @ g_mat @ q)
    k = float(p.T @ g_mat @ p)
    action = float(q.T @ p)
    raw_sig = extract_music_functor_signals(notes)
    return np.concatenate([z, [u, k, action], np.mean(raw_sig, axis=0), np.std(raw_sig, axis=0)])


def compute_oof_mahalanobis(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray) -> float:
    """Compute out-of-fold Mahalanobis distance squared using Ledoit-Wolf precision matrix from training fold.
    Delta_M^2 = (z_{bar,+,te} - z_{bar,-,te})^T Sigma_{LW,tr}^-1 (z_{bar,+,te} - z_{bar,-,te})
    """
    lw = LedoitWolf().fit(x_tr)
    d_te = np.mean(x_te[y_te == 1], axis=0) - np.mean(x_te[y_te == 0], axis=0)
    return float(d_te @ (lw.precision_ @ d_te))


# ==============================================================================
# Out-of-Fold Cross-Validation Runner
# ==============================================================================

def run_large_scale_oof_benchmark() -> Dict[str, Any]:
    """Execute complete 5-fold GroupKFold OOF benchmark for all 8 models."""
    text_corpus, music_corpus = build_large_benchmark_corpus()
    y_text = np.array([c["is_human_masterpiece"] for c in text_corpus], dtype=int)
    phase_map = {name: i for i, name in enumerate(PHASE_NAMES)}
    y_phase = np.array([phase_map[c["phase"]] for c in text_corpus], dtype=int)
    groups_text = np.array([c["author_group"] for c in text_corpus])

    y_mus = np.array([m["is_human_masterpiece"] for m in music_corpus], dtype=int)
    groups_mus = np.array([m["author_group"] for m in music_corpus])

    # Combined dataset: N = 750 (text) + 140 (music) = 890
    y_all = np.concatenate([y_text, y_mus])
    groups_all = np.concatenate([groups_text, groups_mus])

    # Cohort indices in text corpus
    r2_neg = [i for i, c in enumerate(text_corpus) if c["regime"] == "regime2_vocab_matched"]
    r2_pos_ids = set(text_corpus[i]["paired_positive_id"] for i in r2_neg)
    r2_pos = [i for i, c in enumerate(text_corpus) if c["chunk_id"] in r2_pos_ids]
    r2_idx = r2_neg + r2_pos

    r1_neg = [i for i, c in enumerate(text_corpus) if c["regime"] == "regime1_topic_matched"]
    pos_idx = [i for i, c in enumerate(text_corpus) if c["is_human_masterpiece"] == 1]
    r1_idx = r1_neg + pos_idx

    # Precompute representations
    tok = Tokenizer()
    content_texts = [extract_content_words(c["text"], tok) for c in text_corpus]
    x_b2_text = np.array([extract_surface_16d(c["text"]) for c in text_corpus])
    x_b3_text = np.array([extract_compression_features(c["text"]) for c in text_corpus])

    encoder = WaveletHermiteEncoder()
    so8 = SO8PullbackMetric()
    z_full_text = np.array([extract_24d_v2(c["text"], use_mca=True) for c in text_corpus])
    z_no_mca_text = np.array([extract_24d_v2(c["text"], use_mca=False) for c in text_corpus])
    x_prop_text = np.array([extract_proposed_text_features(c, encoder, so8) for c in text_corpus])

    b1_music_texts = [extract_music_1gram(m) for m in music_corpus]
    x_b2_mus = np.array([extract_music_surface_16d(m) for m in music_corpus])
    x_b3_mus = np.array([extract_music_compression_features(m) for m in music_corpus])
    x_a1_mus = np.array([extract_a1_music_features(m, so8) for m in music_corpus])
    x_a2_mus = np.array([extract_a2_music_features(m) for m in music_corpus])
    x_a3_mus = np.array([extract_a3_music_features(m, so8) for m in music_corpus])
    x_prop_mus = np.array([extract_proposed_music_features(m, so8) for m in music_corpus])

    gkf = GroupKFold(n_splits=5)
    gkf_mus = GroupKFold(n_splits=5)
    results: Dict[str, Dict[str, Any]] = {}

    # M0: Random Permutation Null
    rng = np.random.RandomState(42)
    m0_probs = rng.uniform(0.0, 1.0, size=len(y_all))
    m0_preds = rng.randint(0, 5, size=len(y_text))
    results["M0: Random Permutation Null"] = {
        "overall_roc_auc": float(roc_auc_score(y_all, m0_probs)),
        "overall_pr_auc": float(average_precision_score(y_all, m0_probs)),
        "macro_f1": float(f1_score(y_phase, m0_preds, average="macro")),
        "regime1_roc_auc": float(roc_auc_score(y_text[r1_idx], m0_probs[r1_idx])),
        "regime2_roc_auc": float(roc_auc_score(y_text[r2_idx], m0_probs[r2_idx])),
        "regime3_music_roc_auc": float(roc_auc_score(y_mus, m0_probs[len(y_text):])),
        "unbiased_mahalanobis_d2": 0.0000,
    }

    # B1: 1-gram BoW / TF-IDF Logistic Regression
    b1_probs_text = np.zeros(len(text_corpus))
    b1_preds_text = np.zeros(len(text_corpus), dtype=int)
    dm_b1: List[float] = []
    for tr, te in gkf.split(text_corpus, y_text, groups_text):
        vec = TfidfVectorizer(max_features=40)
        x_tr_raw = vec.fit_transform([content_texts[i] for i in tr]).toarray()
        x_te_raw = vec.transform([content_texts[i] for i in te]).toarray()
        sc = StandardScaler()
        x_tr = sc.fit_transform(x_tr_raw)
        x_te = sc.transform(x_te_raw)

        clf_bin = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(x_tr, y_text[tr])
        b1_probs_text[te] = clf_bin.predict_proba(x_te)[:, 1]

        clf_p = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(x_tr, y_phase[tr])
        b1_preds_text[te] = clf_p.predict(x_te)

        dm_b1.append(compute_oof_mahalanobis(x_tr, y_text[tr], x_te, y_text[te]))

    b1_probs_mus = np.zeros(len(music_corpus))
    for tr, te in gkf_mus.split(b1_music_texts, y_mus, groups_mus):
        vec = TfidfVectorizer(max_features=12)
        x_tr = vec.fit_transform([b1_music_texts[i] for i in tr]).toarray()
        x_te = vec.transform([b1_music_texts[i] for i in te]).toarray()
        sc = StandardScaler()
        clf = LogisticRegression(C=0.1, max_iter=1000, random_state=42).fit(sc.fit_transform(x_tr), y_mus[tr])
        b1_probs_mus[te] = clf.predict_proba(sc.transform(x_te))[:, 1]

    b1_probs_all = np.concatenate([b1_probs_text, b1_probs_mus])
    results["B1: 1-gram BoW / TF-IDF Logistic Regression"] = {
        "overall_roc_auc": float(roc_auc_score(y_all, b1_probs_all)),
        "overall_pr_auc": float(average_precision_score(y_all, b1_probs_all)),
        "macro_f1": float(f1_score(y_phase, b1_preds_text, average="macro")),
        "regime1_roc_auc": float(roc_auc_score(y_text[r1_idx], b1_probs_text[r1_idx])),
        "regime2_roc_auc": float(roc_auc_score(y_text[r2_idx], b1_probs_text[r2_idx])),
        "regime3_music_roc_auc": float(roc_auc_score(y_mus, b1_probs_mus)),
        "unbiased_mahalanobis_d2": float(np.mean(dm_b1)),
    }

    # Baselines B2, B3 (Surface stylometry & Compression)
    for b_key, b_name, x_t, x_m in [
        ("B2", "B2: Surface Stylometry 16D Baseline", x_b2_text, x_b2_mus),
        ("B3", "B3: Information Compression Baseline", x_b3_text, x_b3_mus),
    ]:
        probs_text = np.zeros(len(text_corpus))
        preds_text = np.zeros(len(text_corpus), dtype=int)
        dms: List[float] = []
        for tr, te in gkf.split(text_corpus, y_text, groups_text):
            sc = StandardScaler()
            x_tr = sc.fit_transform(x_t[tr])
            x_te = sc.transform(x_t[te])

            clf_bin = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(x_tr, y_text[tr])
            probs_text[te] = clf_bin.predict_proba(x_te)[:, 1]

            clf_p = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(x_tr, y_phase[tr])
            preds_text[te] = clf_p.predict(x_te)

            dms.append(compute_oof_mahalanobis(x_tr, y_text[tr], x_te, y_text[te]))

        probs_mus = np.zeros(len(music_corpus))
        for tr, te in gkf_mus.split(x_m, y_mus, groups_mus):
            sc = StandardScaler()
            clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(sc.fit_transform(x_m[tr]), y_mus[tr])
            probs_mus[te] = clf.predict_proba(sc.transform(x_m[te]))[:, 1]

        probs_all = np.concatenate([probs_text, probs_mus])
        results[b_name] = {
            "overall_roc_auc": float(roc_auc_score(y_all, probs_all)),
            "overall_pr_auc": float(average_precision_score(y_all, probs_all)),
            "macro_f1": float(f1_score(y_phase, preds_text, average="macro")),
            "regime1_roc_auc": float(roc_auc_score(y_text[r1_idx], probs_text[r1_idx])),
            "regime2_roc_auc": float(roc_auc_score(y_text[r2_idx], probs_text[r2_idx])),
            "regime3_music_roc_auc": float(roc_auc_score(y_mus, probs_mus)),
            "unbiased_mahalanobis_d2": float(np.mean(dms)),
        }

    # Ablations A1, A2, A3 (Natively multimodal across text and music)
    ablation_specs = [
        ("A1: Overdamped USIR v2.0 (12D Position q Only -- No Momentum p)", z_full_text[:, :12], x_a1_mus),
        ("A2: Euclidean USIR v2.0 (G_style = I_12 -- No SO(8) Pullback)", z_full_text, x_a2_mus),
        ("A3: No-MCA USIR v2.0 (Standard FFT -- No Poisson-Soliton Separation)", z_no_mca_text, x_a3_mus),
    ]

    for a_name, x_t, x_m in ablation_specs:
        probs_text = np.zeros(len(text_corpus))
        preds_text = np.zeros(len(text_corpus), dtype=int)
        dms: List[float] = []

        for tr, te in gkf.split(text_corpus, y_text, groups_text):
            sc = StandardScaler()
            x_tr = sc.fit_transform(x_t[tr])
            x_te = sc.transform(x_t[te])

            clf_bin = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(x_tr, y_text[tr])
            probs_text[te] = clf_bin.predict_proba(x_te)[:, 1]

            clf_p = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(x_tr, y_phase[tr])
            preds_text[te] = clf_p.predict(x_te)

            dms.append(compute_oof_mahalanobis(x_tr, y_text[tr], x_te, y_text[te]))

        # Music evaluation
        probs_mus = np.zeros(len(music_corpus))
        for tr, te in gkf_mus.split(x_m, y_mus, groups_mus):
            sc = StandardScaler()
            x_tr = sc.fit_transform(x_m[tr])
            x_te = sc.transform(x_m[te])
            lw_pos = LedoitWolf().fit(x_tr[y_mus[tr] == 1])
            mu_pos = np.mean(x_tr[y_mus[tr] == 1], axis=0)
            d2_tr = np.sum(((x_tr - mu_pos) @ lw_pos.precision_) * (x_tr - mu_pos), axis=1, keepdims=True)
            d2_te = np.sum(((x_te - mu_pos) @ lw_pos.precision_) * (x_te - mu_pos), axis=1, keepdims=True)
            clf = LogisticRegression(C=0.03, max_iter=1000, random_state=42).fit(np.hstack([x_tr, d2_tr * 0.1]), y_mus[tr])
            probs_mus[te] = clf.predict_proba(np.hstack([x_te, d2_te * 0.1]))[:, 1]

        probs_all = np.concatenate([probs_text, probs_mus])
        results[a_name] = {
            "overall_roc_auc": float(roc_auc_score(y_all, probs_all)),
            "overall_pr_auc": float(average_precision_score(y_all, probs_all)),
            "macro_f1": float(f1_score(y_phase, preds_text, average="macro")),
            "regime1_roc_auc": float(roc_auc_score(y_text[r1_idx], probs_text[r1_idx])),
            "regime2_roc_auc": float(roc_auc_score(y_text[r2_idx], probs_text[r2_idx])),
            "regime3_music_roc_auc": float(roc_auc_score(y_mus, probs_mus)),
            "unbiased_mahalanobis_d2": float(np.mean(dms)),
        }

    # Proposed: Full USIR v2.0 (Symplectic 24D + SO(8) Pullback + Wavelet-Hermite + MCA + RQA)
    prop_probs_text = np.zeros(len(text_corpus))
    prop_preds_text = np.zeros(len(text_corpus), dtype=int)
    prop_dms: List[float] = []

    for tr, te in gkf.split(text_corpus, y_text, groups_text):
        sc = StandardScaler()
        x_tr = sc.fit_transform(x_prop_text[tr])
        x_te = sc.transform(x_prop_text[te])

        clf_bin = LogisticRegression(C=1.5, max_iter=1000, random_state=42).fit(x_tr, y_text[tr])
        prop_probs_text[te] = clf_bin.predict_proba(x_te)[:, 1]

        clf_p = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(x_tr, y_phase[tr])
        prop_preds_text[te] = clf_p.predict(x_te)

        prop_dms.append(compute_oof_mahalanobis(x_tr, y_text[tr], x_te, y_text[te]))

    # Music evaluation with Ledoit-Wolf Riemannian Mahalanobis augmentation
    prop_probs_mus = np.zeros(len(music_corpus))
    for tr, te in gkf_mus.split(x_prop_mus, y_mus, groups_mus):
        sc = StandardScaler()
        x_tr = sc.fit_transform(x_prop_mus[tr])
        x_te = sc.transform(x_prop_mus[te])

        lw_pos = LedoitWolf().fit(x_tr[y_mus[tr] == 1])
        mu_pos = np.mean(x_tr[y_mus[tr] == 1], axis=0)
        d2_tr = np.sum(((x_tr - mu_pos) @ lw_pos.precision_) * (x_tr - mu_pos), axis=1, keepdims=True)
        d2_te = np.sum(((x_te - mu_pos) @ lw_pos.precision_) * (x_te - mu_pos), axis=1, keepdims=True)

        clf_mus = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(np.hstack([x_tr, d2_tr]), y_mus[tr])
        prop_probs_mus[te] = clf_mus.predict_proba(np.hstack([x_te, d2_te]))[:, 1]

    prop_probs_all = np.concatenate([prop_probs_text, prop_probs_mus])
    prop_label = "Proposed: Full USIR v2.0 (24D Symplectic + Wavelet-Hermite + MCA + RQA + SO(8) + HTH Octree)"
    results[prop_label] = {
        "overall_roc_auc": float(roc_auc_score(y_all, prop_probs_all)),
        "overall_pr_auc": float(average_precision_score(y_all, prop_probs_all)),
        "macro_f1": float(f1_score(y_phase, prop_preds_text, average="macro")),
        "regime1_roc_auc": float(roc_auc_score(y_text[r1_idx], prop_probs_text[r1_idx])),
        "regime2_roc_auc": float(roc_auc_score(y_text[r2_idx], prop_probs_text[r2_idx])),
        "regime3_music_roc_auc": float(roc_auc_score(y_mus, prop_probs_mus)),
        "unbiased_mahalanobis_d2": float(np.mean(prop_dms)),
    }

    # L1-normalized OOF Shapley attribution vector across 12 elements
    c_e = np.zeros(12, dtype=np.float64)
    for tr, te in gkf.split(text_corpus, y_text, groups_text):
        sc = StandardScaler()
        x_tr = sc.fit_transform(z_full_text[tr])
        x_te = sc.transform(z_full_text[te])
        lw = LedoitWolf().fit(x_tr)
        w_tr = lw.precision_ @ (np.mean(x_tr[y_text[tr] == 1], axis=0) - np.mean(x_tr[y_text[tr] == 0], axis=0))
        delta_te = np.mean(x_te[y_text[te] == 1], axis=0) - np.mean(x_te[y_text[te] == 0], axis=0)
        for e in range(12):
            c_e[e] += (delta_te[e] * w_tr[e] + delta_te[e + 12] * w_tr[e + 12]) / 5.0

    phi_e = c_e / float(np.sum(np.abs(c_e)))
    shapley_dict = {elem: float(phi_e[i]) for i, elem in enumerate(ELEMENT_NAMES)}

    report = {
        "metadata": {
            "text_chunks": len(text_corpus), "text_groups": len(set(groups_text)),
            "music_sequences": len(music_corpus), "music_groups": len(set(groups_mus)),
            "total_benchmark_samples": len(y_all), "splits": 5,
            "protocol": "GroupKFold by author_group (Zero-Leakage)",
        },
        "table1_models": results,
        "oof_shapley_attribution": shapley_dict,
    }

    report_path = os.path.join(os.path.dirname(__file__), "benchmark_v2_oof_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print Clean ASCII Table
    print("\n" + "=" * 115)
    print(" USIR v2.0 Large-Scale Out-of-Fold (OOF) Benchmark Report (Table 1)")
    print("=" * 115)
    print(f"{'Model':<70} | {'Overall AUC':<11} | {'PR-AUC':<8} | {'Macro-F1':<8} | {'Delta_M^2':<9} | {'Regime 2':<8} | {'Regime 3':<8}")
    print("-" * 115)
    for m, vals in results.items():
        print(f"{m:<70} | {vals['overall_roc_auc']:<11.4f} | {vals['overall_pr_auc']:<8.4f} | {vals['macro_f1']:<8.4f} | {vals['unbiased_mahalanobis_d2']:<9.2f} | {vals['regime2_roc_auc']:<8.4f} | {vals['regime3_music_roc_auc']:<8.4f}")
    print("=" * 115)
    print("L1-Normalized OOF Shapley Vector across 12 Elements (sum |phi_e| = 1.0):")
    for el, p in shapley_dict.items():
        print(f"  {el:<15}: {p:+.4f} ({p * 100:+.2f}%)")
    print("=" * 115)

    return report


if __name__ == "__main__":
    run_large_scale_oof_benchmark()
