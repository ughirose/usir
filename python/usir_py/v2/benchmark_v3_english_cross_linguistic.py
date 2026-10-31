"""USIR v3.0 Cross-Linguistic Universality Benchmark Suite (Table 4).

Evaluates the English Isomorphic Driver (usir-driver-en) on a large-scale
English public-domain benchmark (N = 400 chunks across 21 author/source groups)
under strict 5-Fold GroupKFold(n_splits=5, groups=author_group) out-of-fold (OOF)
cross-validation without in-sample leakage.

Evaluates all 7 Table 4 models:
1. M0_Null_Prior: Random permutation null prior.
2. B1_Content_1gram_BoW: Content 1-gram bag-of-words (TF-IDF).
3. B2_Surface_Stylometry_16D: 16 classical lexical and character surface stylometrics.
4. B3_Compression_Entropy: 6D compression and information entropy features.
5. A1_USIR_Position_Only_12D: Overdamped USIR 12D position coordinates (no momentum p).
6. A2_USIR_Euclidean_24D: Flat Euclidean 24D symplectic state (no SO(8) pullback / trajectory).
7. Proposed_Full_USIR_v3_English: Full 24D symplectic + SO(8) pullback + RQA + trajectory dynamics.
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
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from usir_py.v2.build_english_benchmark_corpus import build_english_benchmark_corpus
from usir_py.v2.english_driver import extract_english_signals, extract_usir_24d_english
from usir_py.v2.thermodynamics_so8 import SO8PullbackMetric
from usir_py.v2.wavelet_hermite_encoder import compute_rqa_metrics

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PHASE_NAMES = ["I_Crystalline", "II_Critical", "III_Void", "IV_Turbulent", "V_Deadzone"]
ELEMENT_NAMES = [
    "Sg-01 Dn", "Sg-02 Cr", "Sg-03 Pr", "Sg-04 Jt",
    "Sg-05 Cp", "Sg-06 Mp", "Sg-07 Cs", "Sg-08 Lp",
    "Sg-09 Fl", "Sg-10 At", "Sg-11 Ma", "Sg-12 Ru",
]


# ==============================================================================
# Feature Extraction Functions
# ==============================================================================

def extract_b1_tokens(text: str) -> str:
    """Extract lowercased word token sequence for B1 1-gram baseline."""
    words = re.findall(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b", text.lower())
    return " ".join(words)


def extract_b2_classical_surface_16d(text: str) -> np.ndarray:
    """Extract 16 classical lexical and character surface stylometric features.

    These surface features measure bag-of-character and bag-of-word statistics
    without sequential trajectory or clause-order dynamics, properly collapsing
    to AUC ~0.50 on Theiler surrogates where multiset frequencies are preserved.
    """
    chars = max(len(text), 1)
    words = re.findall(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b", text)
    n_words = max(len(words), 1)

    uppercase = sum(1 for c in text if c.isupper())
    lowercase = sum(1 for c in text if c.islower())
    digits = sum(1 for c in text if c.isdigit())
    whitespace = sum(1 for c in text if c.isspace())
    punct = sum(1 for c in text if not c.isalnum() and not c.isspace())

    commas = text.count(",")
    periods = text.count(".")
    apostrophes = text.count("'")
    hyphens = text.count("-") + text.count("—")

    text_l = text.lower()
    e_ratio = text_l.count("e") / float(chars)
    t_ratio = text_l.count("t") / float(chars)
    a_ratio = text_l.count("a") / float(chars)

    word_counts = Counter(w.lower() for w in words)
    ttr = len(word_counts) / float(n_words)
    sum_f2 = sum(f * f for f in word_counts.values())
    yules_k = 10000.0 * (sum_f2 - n_words) / float(n_words * n_words) if n_words > 1 else 0.0

    mean_word_len = sum(len(w) for w in words) / float(n_words)

    return np.array([
        float(chars), float(n_words), mean_word_len,
        uppercase / float(chars), lowercase / float(chars), digits / float(chars),
        whitespace / float(chars), punct / float(chars), commas / float(chars),
        periods / float(chars), apostrophes / float(chars), hyphens / float(chars),
        e_ratio, t_ratio, ttr, yules_k,
    ], dtype=np.float64)


def extract_b3_compression(text: str) -> np.ndarray:
    """Extract 6-dimensional compression ratio and character entropy features."""
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


def extract_a1_features(text: str) -> np.ndarray:
    """Ablation 1: Overdamped 12D position q + potential energy without momentum p."""
    z = extract_usir_24d_english(text, use_mca=False)
    q = z[:12]
    raw_sig = extract_english_signals(text)
    return np.concatenate([q, [float(q.T @ q)], np.mean(raw_sig, axis=0), np.std(raw_sig, axis=0)])


def extract_a2_features(text: str) -> np.ndarray:
    """Ablation 2: Flat Euclidean 24D symplectic state without SO(8) pullback or RQA."""
    z = extract_usir_24d_english(text, use_mca=False)
    q, p = z[:12], z[12:]
    u_flat = float(q.T @ q)
    k_flat = float(p.T @ p)
    action = float(q.T @ p)
    raw_sig = extract_english_signals(text)
    return np.concatenate([z, [u_flat, k_flat, action], np.mean(raw_sig, axis=0), np.std(raw_sig, axis=0)])


def extract_proposed_features(text: str, so8: SO8PullbackMetric) -> np.ndarray:
    """Full Proposed USIR v3.0: 24D Symplectic + SO(8) Pullback + Wavelet-Hermite + RQA + CV_L."""
    z = extract_usir_24d_english(text, use_mca=True)
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

    raw_sig = extract_english_signals(text)
    sig_mean = np.mean(raw_sig, axis=0) if len(raw_sig) > 0 else np.zeros(4)
    sig_std = np.std(raw_sig, axis=0) if len(raw_sig) > 1 else np.zeros(4)
    soliton_spike = float(np.max(raw_sig[:, 3])) if len(raw_sig) > 0 else 0.0
    soliton_kurt = (
        float(np.mean((raw_sig[:, 3] - np.mean(raw_sig[:, 3])) ** 4) / (np.var(raw_sig[:, 3]) ** 2 + 1e-6))
        if len(raw_sig) > 1
        else 1.0
    )

    rqa_1 = compute_rqa_metrics(raw_sig[:, 1])
    rqa_3 = compute_rqa_metrics(raw_sig[:, 3])
    rqa_vec = np.array([rqa_1.det, rqa_1.i_arch, rqa_1.lam, rqa_3.det, rqa_3.i_arch, rqa_3.lam], dtype=np.float64)

    sentences = [s for s in re.split(r"[.!?\n]+", text) if s.strip()]
    sen_lens = [len(re.findall(r"\b[a-zA-Z]+\b", s)) for s in sentences]
    cv_l = float(np.std(sen_lens) / (np.mean(sen_lens) + 1e-6)) if len(sen_lens) > 1 else 0.0

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
        [u_metric, k_metric, h_metric, action, mode_q, mode_p, soliton_spike, soliton_kurt, cv_l],
        sig_mean, sig_std, rqa_vec, ac_vec,
    ])
    return np.concatenate([z, extra])


def compute_oof_mahalanobis(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray) -> float:
    """Compute out-of-fold Mahalanobis distance squared using Ledoit-Wolf precision matrix from training fold."""
    lw = LedoitWolf().fit(x_tr)
    pos_te = x_te[y_te == 1]
    neg_te = x_te[y_te == 0]
    if len(pos_te) == 0 or len(neg_te) == 0:
        return 0.0
    d_te = np.mean(pos_te, axis=0) - np.mean(neg_te, axis=0)
    return float(d_te @ (lw.precision_ @ d_te))


# ==============================================================================
# Master Out-of-Fold Cross-Validation Runner
# ==============================================================================

def run_english_cross_linguistic_benchmark(save_reports: bool = True) -> Dict[str, Any]:
    """Execute complete 5-fold GroupKFold OOF benchmark across all 7 Table 4 models."""
    corpus = build_english_benchmark_corpus()
    y_bin = np.array([c["is_human_masterpiece"] for c in corpus], dtype=int)
    phase_map = {name: i for i, name in enumerate(PHASE_NAMES)}
    y_phase = np.array([phase_map[c["phase"]] for c in corpus], dtype=int)
    groups = np.array([c["author_group"] for c in corpus])

    # Sub-cohort indices for Regimes 1 and 2
    r2_neg = [i for i, c in enumerate(corpus) if c["regime"] == "regime2_theiler_surrogate"]
    r2_pos_ids = set(corpus[i]["paired_positive_id"] for i in r2_neg)
    r2_pos = [i for i, c in enumerate(corpus) if c["chunk_id"] in r2_pos_ids]
    r2_idx = r2_neg + r2_pos

    r1_neg = [i for i, c in enumerate(corpus) if c["regime"] == "regime1_topic_matched"]
    pos_idx = [i for i, c in enumerate(corpus) if c["is_human_masterpiece"] == 1]
    r1_idx = r1_neg + pos_idx

    # Precompute representations for all chunks
    b1_texts = [extract_b1_tokens(c["text"]) for c in corpus]
    x_b2 = np.array([extract_b2_classical_surface_16d(c["text"]) for c in corpus])
    x_b3 = np.array([extract_b3_compression(c["text"]) for c in corpus])
    x_a1 = np.array([extract_a1_features(c["text"]) for c in corpus])
    x_a2 = np.array([extract_a2_features(c["text"]) for c in corpus])

    so8 = SO8PullbackMetric()
    x_prop = np.array([extract_proposed_features(c["text"], so8) for c in corpus])
    z_full = np.array([extract_usir_24d_english(c["text"], use_mca=True) for c in corpus])

    gkf = GroupKFold(n_splits=5)
    results: Dict[str, Dict[str, Any]] = {}

    # --------------------------------------------------------------------------
    # M0: Random Permutation Null Prior
    # --------------------------------------------------------------------------
    rng = np.random.RandomState(42)
    m0_probs = rng.uniform(0.0, 1.0, size=len(y_bin))
    m0_preds = rng.randint(0, 5, size=len(y_bin))
    results["M0_Null_Prior"] = {
        "overall_roc_auc": float(roc_auc_score(y_bin, m0_probs)),
        "overall_pr_auc": float(average_precision_score(y_bin, m0_probs)),
        "macro_f1": float(f1_score(y_phase, m0_preds, average="macro")),
        "regime1_roc_auc": float(roc_auc_score(y_bin[r1_idx], m0_probs[r1_idx])),
        "regime2_roc_auc": float(roc_auc_score(y_bin[r2_idx], m0_probs[r2_idx])),
        "unbiased_mahalanobis_d2": 0.0000,
    }

    # --------------------------------------------------------------------------
    # B1: Content 1-gram BoW / TF-IDF
    # --------------------------------------------------------------------------
    b1_probs = np.zeros(len(corpus))
    b1_preds = np.zeros(len(corpus), dtype=int)
    dm_b1: List[float] = []
    for tr, te in gkf.split(corpus, y_bin, groups):
        vec = TfidfVectorizer(max_features=50)
        x_tr = vec.fit_transform([b1_texts[i] for i in tr]).toarray()
        x_te = vec.transform([b1_texts[i] for i in te]).toarray()

        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42).fit(x_tr, y_bin[tr])
        b1_probs[te] = clf.predict_proba(x_te)[:, 1]

        clf_p = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42).fit(x_tr, y_phase[tr])
        b1_preds[te] = clf_p.predict(x_te)

        dm_b1.append(compute_oof_mahalanobis(x_tr, y_bin[tr], x_te, y_bin[te]))

    results["B1_Content_1gram_BoW"] = {
        "overall_roc_auc": float(roc_auc_score(y_bin, b1_probs)),
        "overall_pr_auc": float(average_precision_score(y_bin, b1_probs)),
        "macro_f1": float(f1_score(y_phase, b1_preds, average="macro")),
        "regime1_roc_auc": float(roc_auc_score(y_bin[r1_idx], b1_probs[r1_idx])),
        "regime2_roc_auc": float(roc_auc_score(y_bin[r2_idx], b1_probs[r2_idx])),
        "unbiased_mahalanobis_d2": float(np.mean(dm_b1)),
    }

    # --------------------------------------------------------------------------
    # B2, B3, A1, A2, Proposed
    # --------------------------------------------------------------------------
    model_specs = [
        ("B2_Surface_Stylometry_16D", x_b2),
        ("B3_Compression_Entropy", x_b3),
        ("A1_USIR_Position_Only_12D", x_a1),
        ("A2_USIR_Euclidean_24D", x_a2),
        ("Proposed_Full_USIR_v3_English", x_prop),
    ]

    for name, feat_mat in model_specs:
        probs = np.zeros(len(corpus))
        preds = np.zeros(len(corpus), dtype=int)
        dms: List[float] = []

        for tr, te in gkf.split(corpus, y_bin, groups):
            sc = StandardScaler()
            x_tr = sc.fit_transform(feat_mat[tr])
            x_te = sc.transform(feat_mat[te])

            clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42).fit(x_tr, y_bin[tr])
            probs[te] = clf.predict_proba(x_te)[:, 1]

            clf_p = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42).fit(x_tr, y_phase[tr])
            preds[te] = clf_p.predict(x_te)

            dms.append(compute_oof_mahalanobis(x_tr, y_bin[tr], x_te, y_bin[te]))

        results[name] = {
            "overall_roc_auc": float(roc_auc_score(y_bin, probs)),
            "overall_pr_auc": float(average_precision_score(y_bin, probs)),
            "macro_f1": float(f1_score(y_phase, preds, average="macro")),
            "regime1_roc_auc": float(roc_auc_score(y_bin[r1_idx], probs[r1_idx])),
            "regime2_roc_auc": float(roc_auc_score(y_bin[r2_idx], probs[r2_idx])),
            "unbiased_mahalanobis_d2": float(np.mean(dms)),
        }

    # --------------------------------------------------------------------------
    # L1-Normalized OOF Shapley Attribution Vector across 12 Elements
    # --------------------------------------------------------------------------
    c_e = np.zeros(12, dtype=np.float64)
    for tr, te in gkf.split(corpus, y_bin, groups):
        sc = StandardScaler()
        x_tr = sc.fit_transform(z_full[tr])
        x_te = sc.transform(z_full[te])
        lw = LedoitWolf().fit(x_tr)
        diff_tr = np.mean(x_tr[y_bin[tr] == 1], axis=0) - np.mean(x_tr[y_bin[tr] == 0], axis=0)
        w_tr = lw.precision_ @ diff_tr
        delta_te = np.mean(x_te[y_bin[te] == 1], axis=0) - np.mean(x_te[y_bin[te] == 0], axis=0)
        for e in range(12):
            c_e[e] += (delta_te[e] * w_tr[e] + delta_te[e + 12] * w_tr[e + 12]) / 5.0

    phi_e = c_e / float(np.sum(np.abs(c_e)))
    shapley_dict = {elem: float(phi_e[i]) for i, elem in enumerate(ELEMENT_NAMES)}

    report = {
        "metadata": {
            "title": "USIR v3.0 Cross-Linguistic Universality Benchmark (Table 4)",
            "benchmark_language": "English (Public Domain)",
            "text_chunks": len(corpus),
            "author_groups": len(set(groups)),
            "splits": 5,
            "protocol": "GroupKFold by author_group (Zero-Leakage Out-of-Fold)",
            "phases": PHASE_NAMES,
        },
        "table4_models": results,
        "oof_shapley_attribution": shapley_dict,
    }

    if save_reports:
        # Save to python/usir_py/v2/benchmark_v3_english_oof_report.json
        py_report_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "benchmark_v3_english_oof_report.json")
        )
        with open(py_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Save to artifacts/benchmark_v3_english_oof_report.json
        artifacts_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "artifacts")
        )
        os.makedirs(artifacts_dir, exist_ok=True)
        art_report_path = os.path.join(artifacts_dir, "benchmark_v3_english_oof_report.json")
        with open(art_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    # Print Clean ASCII Table
    print("\n" + "=" * 115)
    print(" USIR v3.0 English Cross-Linguistic Universality Benchmark Report (Table 4)")
    print("=" * 115)
    print(f"{'Model':<35} | {'Overall AUC':<11} | {'PR-AUC':<8} | {'Macro-F1':<8} | {'Regime 1':<8} | {'Regime 2':<8} | {'Delta_M^2':<9}")
    print("-" * 115)
    for m, vals in results.items():
        print(f"{m:<35} | {vals['overall_roc_auc']:<11.4f} | {vals['overall_pr_auc']:<8.4f} | {vals['macro_f1']:<8.4f} | {vals['regime1_roc_auc']:<8.4f} | {vals['regime2_roc_auc']:<8.4f} | {vals['unbiased_mahalanobis_d2']:<9.2f}")
    print("=" * 115)
    print("L1-Normalized OOF Shapley Vector across 12 Elements (sum |phi_e| = 1.0):")
    for el, p in shapley_dict.items():
        print(f"  {el:<15}: {p:+.4f} ({p * 100:+.2f}%)")
    print("=" * 115)

    return report


if __name__ == "__main__":
    run_english_cross_linguistic_benchmark()
