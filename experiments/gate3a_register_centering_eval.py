#!/usr/bin/env python
"""USIR v5 Stage B-9 Task 3: Gate 3a Register-Specific Centering & Diachronic Offset Alignment.

Comparative Evaluation of 4 Schemes under LOO (Leave-One-Out) on:
- 32 samples (data/usir_v5_eval/tier_c_modern_benchmark.jsonl, 4 registers x 8 samples)
- 64 samples (data/usir_v5_eval/tier_c_modern_benchmark_expanded.jsonl, 4 registers x 16 samples)

Schemes:
- Scheme 0 (Raw_R4_Centering):
  Aozora Bunko 106 works (10 authors) baseline mean & covariance in 31D ortho_v2.
  Mahalanobis distance & outlier rates (3-sigma and chi^2_0.95).
- Scheme 1 (Continuous_Core_Subspace):
  Core continuous subspace (ortho24_cont: 24D) excluding sparse final particles (zero-rate > 80%)
  and comma-dependent dimensions.
- Scheme 2 (LOO_Era_Shift_Alignment):
  LOO Diachronic drift alignment z_tilde_i = z_i - delta_era_LOO relative to Aozora baseline.
  Testing if R1 modern colloquial fiction falls within normal distribution (outlier rate < 15%).
- Scheme 3 (Within_Register_LOO_Centering):
  Register-specific LOO centering and scaling for each r in {R1, R2, R3, R4}.
  Elimination of between-register variance ratio eta^2_register and stability of intra-register distance structure.

Outputs:
  data/usir_v5_eval/usir_v5_b9_register_centering_and_level3.json
"""
from __future__ import annotations

import functools
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments"))

from fable_v5_grand_shootout import load_works
from usir_py.orthographic_prior import (
    ORTHO_EXT_V2_NAMES,
    NARR_V2_NAMES,
    ortho_matrix_ext_v2,
    ortho_matrix_narr_v2,
)

BENCHMARK_32_PATH = ROOT / "data" / "usir_v5_eval" / "tier_c_modern_benchmark.jsonl"
BENCHMARK_64_PATH = ROOT / "data" / "usir_v5_eval" / "tier_c_modern_benchmark_expanded.jsonl"
TASK1_STAGE_JSON_PATH = ROOT / "data" / "usir_v5_eval" / "usir_v5_stage_b8_scale_separation_and_gate3a.json"
TASK2_U6_META_PATH = ROOT / "docs" / "handoff" / "v10" / "u6_guards_calibration_metadata.json"
OUTPUT_JSON_PATH = ROOT / "data" / "usir_v5_eval" / "usir_v5_b9_register_centering_and_level3.json"
OUTPUT_B10_JSON_PATH = ROOT / "data" / "usir_v5_eval" / "usir_v5_b10_gate3a_scene_scale_and_level3.json"
OUTPUT_B11_JSON_PATH = ROOT / "data" / "usir_v5_eval" / "usir_v5_b11_gate3a_calibrated_and_f14_parity.json"
OUTPUT_B12_JSON_PATH = ROOT / "data" / "usir_v5_eval" / "usir_v5_b12_canonical_eval.json"
OUTPUT_B12_SUMMARY_PATH = ROOT / "docs" / "handoff" / "v10" / "gate3a_register_centering_summary.json"

# Gate status constants (Strictly frozen & maintained according to repository rules)
GATE3A_STATUS = "SHIFT_QUANTIFIED_NOT_PASSED"
GATE3B_STATUS = "GATE3B_FROZEN"

# Stage B-12 Task 2: Explicit primary / auxiliary alarm threshold specification (threshold values unchanged)
PRIMARY_ALARM_THRESHOLD_KEY = "mc_indep_p95"
PRIMARY_ALARM_THRESHOLD_D = 6.9267
AUXILIARY_ALARM_THRESHOLD_KEY = "mc_r4corr_p95"
AUXILIARY_ALARM_THRESHOLD_D = 7.6574
ALARM_THRESHOLD_SPEC: Dict[str, Any] = {
    "primary_alarm": {
        "role": "primary",
        "label_ja": "一次警報",
        "threshold_key": PRIMARY_ALARM_THRESHOLD_KEY,
        "threshold_d": PRIMARY_ALARM_THRESHOLD_D,
        "mc_quantile": "95%",
        "null_model": "independent_gaussian_24d_loo",
        "description": "Primary alarm threshold (一次警報): mc_indep_p95 (d=6.9267).",
    },
    "auxiliary_alarm": {
        "role": "auxiliary",
        "label_ja": "補助警報",
        "threshold_key": AUXILIARY_ALARM_THRESHOLD_KEY,
        "threshold_d": AUXILIARY_ALARM_THRESHOLD_D,
        "mc_quantile": "95%",
        "null_model": "r4_correlated_gaussian_24d_loo",
        "description": "Auxiliary alarm threshold (補助警報): mc_r4corr_p95 (d=7.6574).",
    },
    "thresholds_unchanged_note": (
        "Explicitly documents primary (mc_indep_p95, d=6.9267) and auxiliary (mc_r4corr_p95, d=7.6574) roles; "
        "threshold values and Gate 3a/3b decisions remain unchanged."
    ),
}


def get_alarm_threshold_specification() -> Dict[str, Any]:
    """Return the canonical Gate 3a primary/auxiliary alarm threshold specification."""
    return json.loads(json.dumps(ALARM_THRESHOLD_SPEC, ensure_ascii=False))

# 31D ortho_v2 dimensions
ALL_ORTHO_NAMES = list(ORTHO_EXT_V2_NAMES) + list(NARR_V2_NAMES)
assert len(ALL_ORTHO_NAMES) == 31

# Exclusions for ortho24_cont (Task 1 continuous core parity: comma [2,3,29] + 4 sparse finals [21,22,23,24])
COMMA_DIMS = [2, 3, 29]
SPARSE_FINALS = [21, 22, 23, 24]
EXCLUDED_DIMS_7 = sorted(COMMA_DIMS + SPARSE_FINALS)
CONT24_DIMS = [i for i in range(31) if i not in EXCLUDED_DIMS_7]
assert len(CONT24_DIMS) == 24

# Exclusions for ortho24_no_sparse_finals (Gate 3a zero-rate > 80% sparse sentence-final/bracket dims: [19..25])
SPARSE_FINALS_ALL_7 = [19, 20, 21, 22, 23, 24, 25]  # だ, い, う, す, ぬ, か, 」
NO_SPARSE_FINALS_24_DIMS = [i for i in range(31) if i not in SPARSE_FINALS_ALL_7]
assert len(NO_SPARSE_FINALS_24_DIMS) == 24


def bootstrap_ci(
    data: np.ndarray,
    stat_func=np.mean,
    n_boot: int = 2000,
    ci: float = 0.95,
    rng_seed: int = 42,
) -> Tuple[float, float]:
    """Compute non-parametric bootstrap percentile confidence interval."""
    if len(data) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(rng_seed)
    n = len(data)
    boot_stats = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats[b] = stat_func(sample)
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_stats, 100 * alpha))
    high = float(np.percentile(boot_stats, 100 * (1.0 - alpha)))
    return low, high


def compute_eta_squared(matrix: np.ndarray, labels: Sequence[str]) -> float:
    """Compute multivariate eta^2_register = SS_between / SS_total."""
    unique_labels = sorted(list(set(labels)))
    overall_mean = np.mean(matrix, axis=0)
    ss_total = float(np.sum((matrix - overall_mean) ** 2))
    if ss_total < 1e-12:
        return 0.0

    ss_between = 0.0
    for lab in unique_labels:
        idx = [i for i, l in enumerate(labels) if l == lab]
        group = matrix[idx]
        group_mean = np.mean(group, axis=0)
        ss_between += len(group) * float(np.sum((group_mean - overall_mean) ** 2))

    return float(np.clip(ss_between / ss_total, 0.0, 1.0))


def regularized_mahalanobis(
    x: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
    shrinkage: float = 0.15,
) -> float:
    """Compute regularized Mahalanobis distance with diagonal shrinkage."""
    diff = x - mean
    d = len(diff)
    if cov.shape[0] < d:
        # High-dimensional: purely diagonal Mahalanobis with variance floor
        var_safe = np.maximum(np.diag(cov), 1e-6)
        return float(np.sqrt(np.sum((diff ** 2) / var_safe)))

    # Ledoit-Wolf style shrinkage
    target = np.diag(np.diag(cov) + 1e-5)
    reg_cov = (1.0 - shrinkage) * cov + shrinkage * target
    try:
        inv_cov = np.linalg.pinv(reg_cov)
        val = diff.dot(inv_cov).dot(diff)
        return float(np.sqrt(max(val, 0.0)))
    except Exception:
        var_safe = np.maximum(np.diag(cov), 1e-6)
        return float(np.sqrt(np.sum((diff ** 2) / var_safe)))


def diagonal_mahalanobis(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    """Compute diagonal Mahalanobis distance."""
    var_safe = np.maximum(var, 1e-6)
    diff = x - mean
    return float(np.sqrt(np.sum((diff ** 2) / var_safe)))


def load_benchmark_records(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@functools.lru_cache(maxsize=1)
def extract_aozora_106_atlas() -> Dict[str, Any]:
    """Extract ortho_v2 31D and cont24 for all 106 Aozora Bunko works."""
    print("Loading 106 Aozora Bunko works and extracting orthographic profiles...")
    works = load_works()
    assert len(works) == 106, f"Expected 106 works, got {len(works)}"
    texts = [w["text"] for w in works]

    ext = ortho_matrix_ext_v2(texts)
    narr = ortho_matrix_narr_v2(texts)
    full31 = np.concatenate([ext, narr], axis=1)  # (106, 31)
    cont24 = full31[:, CONT24_DIMS]               # (106, 24)

    mean_31 = np.mean(full31, axis=0)
    var_31 = np.var(full31, axis=0, ddof=1)
    cov_31 = np.cov(full31, rowvar=False)

    mean_24 = np.mean(cont24, axis=0)
    var_24 = np.var(cont24, axis=0, ddof=1)
    cov_24 = np.cov(cont24, rowvar=False)

    # Compute baseline self-Mahalanobis distances to establish 3-sigma thresholds
    d_m_31_self = np.array([regularized_mahalanobis(full31[i], mean_31, cov_31) for i in range(106)])
    thresh_3sigma_31 = float(np.mean(d_m_31_self) + 3.0 * np.std(d_m_31_self, ddof=1))
    thresh_chi2_31 = float(np.sqrt(stats.chi2.ppf(0.95, df=31)))

    d_m_24_self = np.array([regularized_mahalanobis(cont24[i], mean_24, cov_24) for i in range(106)])
    thresh_3sigma_24 = float(np.mean(d_m_24_self) + 3.0 * np.std(d_m_24_self, ddof=1))
    thresh_chi2_24 = float(np.sqrt(stats.chi2.ppf(0.95, df=24)))

    return {
        "full31": full31,
        "cont24": cont24,
        "mean_31": mean_31,
        "var_31": var_31,
        "cov_31": cov_31,
        "thresh_3sigma_31": thresh_3sigma_31,
        "thresh_chi2_31": thresh_chi2_31,
        "self_d_m_31_mean": float(np.mean(d_m_31_self)),
        "self_d_m_31_std": float(np.std(d_m_31_self, ddof=1)),
        "mean_24": mean_24,
        "var_24": var_24,
        "cov_24": cov_24,
        "thresh_3sigma_24": thresh_3sigma_24,
        "thresh_chi2_24": thresh_chi2_24,
        "self_d_m_24_mean": float(np.mean(d_m_24_self)),
        "self_d_m_24_std": float(np.std(d_m_24_self, ddof=1)),
    }


def evaluate_baseline_a_diagonal(
    X31: np.ndarray,
    labels: List[str],
    reg_indices: Dict[str, List[int]],
    modern_indices: List[int],
) -> Dict[str, Any]:
    """Evaluate Schemes 0, 1, and 2 using Baseline A: In-benchmark R4 diagonal Mahalanobis."""
    registers = ["R1", "R2", "R3", "R4"]
    r4_idx = reg_indices["R4"]
    r1_idx = reg_indices["R1"]

    def _run_space(X: np.ndarray, space_name: str, dropped_dims: List[int]) -> Dict[str, Any]:
        df = X.shape[1]
        X_r4 = X[r4_idx]
        mu_r4 = np.mean(X_r4, axis=0)
        var_r4 = np.maximum(np.var(X_r4, axis=0, ddof=1), 1e-6)
        d_r4_self = np.array([diagonal_mahalanobis(X_r4[i], mu_r4, var_r4) for i in range(len(X_r4))])
        thresh_3s = float(np.mean(d_r4_self) + 3.0 * np.std(d_r4_self, ddof=1))
        thresh_c2 = float(np.sqrt(stats.chi2.ppf(0.95, df=df)))

        d_all = np.array([diagonal_mahalanobis(X[i], mu_r4, var_r4) for i in range(len(X))])
        per_reg = {}
        for r in registers:
            idx = reg_indices[r]
            d_vals = d_all[idx]
            per_reg[r] = {
                "n": len(idx),
                "d_m_mean": round(float(np.mean(d_vals)), 4),
                "d_m_std": round(float(np.std(d_vals, ddof=1)), 4),
                "outlier_rate_3sigma_pct": round(float(np.mean(d_vals > thresh_3s) * 100.0), 2),
                "outlier_rate_chi2_p95_pct": round(float(np.mean(d_vals > thresh_c2) * 100.0), 2),
            }
        mod_d = d_all[modern_indices]
        mod_agg = {
            "n": len(modern_indices),
            "d_m_mean": round(float(np.mean(mod_d)), 4),
            "outlier_rate_3sigma_pct": round(float(np.mean(mod_d > thresh_3s) * 100.0), 2),
            "outlier_rate_chi2_p95_pct": round(float(np.mean(mod_d > thresh_c2) * 100.0), 2),
        }

        # Scheme 2: LOO Era Shift Alignment on R1 and Modern Pooled
        def _loo_align(target_idx: List[int]) -> Dict[str, Any]:
            d_un = []
            d_al = []
            for i in target_idx:
                others = [j for j in target_idx if j != i]
                delta = np.mean(X[others], axis=0) - mu_r4
                d_un.append(diagonal_mahalanobis(X[i], mu_r4, var_r4))
                d_al.append(diagonal_mahalanobis(X[i] - delta, mu_r4, var_r4))
            d_un_arr = np.array(d_un)
            d_al_arr = np.array(d_al)
            out_al_3s = float(np.mean(d_al_arr > thresh_3s) * 100.0)
            out_al_c2 = float(np.mean(d_al_arr > thresh_c2) * 100.0)
            return {
                "n_samples": len(target_idx),
                "unaligned": {
                    "d_m_mean": round(float(np.mean(d_un_arr)), 4),
                    "outlier_rate_3sigma_pct": round(float(np.mean(d_un_arr > thresh_3s) * 100.0), 2),
                    "outlier_rate_chi2_p95_pct": round(float(np.mean(d_un_arr > thresh_c2) * 100.0), 2),
                },
                "aligned": {
                    "d_m_mean": round(float(np.mean(d_al_arr)), 4),
                    "outlier_rate_3sigma_pct": round(out_al_3s, 2),
                    "outlier_rate_chi2_p95_pct": round(out_al_c2, 2),
                },
                "passes_15pct_safety_threshold": bool(out_al_3s < 15.0 and out_al_c2 < 15.0),
            }

        return {
            "space_name": space_name,
            "dimension": df,
            "dropped_dimensions": [ALL_ORTHO_NAMES[i] for i in dropped_dims],
            "thresholds": {
                "3sigma_threshold": round(thresh_3s, 4),
                "chi2_p95_threshold": round(thresh_c2, 4),
            },
            "unaligned_modern_aggregate": mod_agg,
            "unaligned_per_register": per_reg,
            "scheme_2_loo_era_shift_alignment": {
                "R1_fiction_alignment": _loo_align(r1_idx),
                "all_modern_pooled_alignment": _loo_align(modern_indices),
            },
        }

    return {
        "description": "Baseline A: In-benchmark R4 diagonal Mahalanobis (Gate 3a canonical metric parity)",
        "scheme_0_ortho_v2_31d": _run_space(X31, "ortho_v2_31d", []),
        "scheme_1_ortho24_cont": _run_space(X31[:, CONT24_DIMS], "ortho24_cont", EXCLUDED_DIMS_7),
        "scheme_1_ortho24_no_sparse_finals": _run_space(
            X31[:, NO_SPARSE_FINALS_24_DIMS], "ortho24_no_sparse_finals", SPARSE_FINALS_ALL_7
        ),
    }


def evaluate_within_register_outliers(
    X: np.ndarray,
    reg_indices: Dict[str, List[int]],
    dim_names: List[str],
) -> Dict[str, Any]:
    """Compute within-register LOO Mahalanobis outlier rates and single-nonzero collapse diagnostics."""
    registers = ["R1", "R2", "R3", "R4"]
    df = X.shape[1]
    r4_idx = reg_indices["R4"]
    X_r4 = X[r4_idx]
    mu_r4 = np.mean(X_r4, axis=0)
    var_r4 = np.maximum(np.var(X_r4, axis=0, ddof=1), 1e-6)
    d_r4_self = np.array([diagonal_mahalanobis(X_r4[i], mu_r4, var_r4) for i in range(len(X_r4))])
    thresh_3s = float(np.mean(d_r4_self) + 3.0 * np.std(d_r4_self, ddof=1))
    thresh_c2 = float(np.sqrt(stats.chi2.ppf(0.95, df=df)))

    var_pool = np.maximum(
        np.mean([np.var(X[reg_indices[r]], axis=0, ddof=1) for r in registers], axis=0),
        1e-6,
    )

    modes: Dict[str, Any] = {}
    collapse_events: List[Dict[str, Any]] = []

    for mode_name in ["loo_mean_r4_var", "loo_mean_shrinkage_loo_var", "loo_mean_unregularized_loo_var"]:
        per_reg: Dict[str, Any] = {}
        mod_dists: List[float] = []
        for r in registers:
            idx = reg_indices[r]
            sub = X[idx]
            n_r = len(sub)
            d_loo: List[float] = []
            for i in range(n_r):
                others = [j for j in range(n_r) if j != i]
                mu_loo = np.mean(sub[others], axis=0)
                if mode_name == "loo_mean_r4_var":
                    v_use = var_r4
                elif mode_name == "loo_mean_shrinkage_loo_var":
                    v_loo = np.var(sub[others], axis=0, ddof=1)
                    v_use = np.maximum(0.8 * v_loo + 0.2 * var_pool, 1e-6)
                else:
                    v_raw = np.var(sub[others], axis=0, ddof=1)
                    # Record single-nonzero variance collapse events where v_raw == 0 while sample i is nonzero
                    collapsed_dims = [
                        dim_names[d_i]
                        for d_i in range(df)
                        if v_raw[d_i] < 1e-12 and abs(sub[i, d_i] - mu_loo[d_i]) > 1e-4
                    ]
                    if collapsed_dims:
                        collapse_events.append({
                            "register": r,
                            "sample_local_idx": i,
                            "collapsed_dims": collapsed_dims,
                        })
                    v_use = np.maximum(v_raw, 1e-6)

                d_val = diagonal_mahalanobis(sub[i], mu_loo, v_use)
                d_loo.append(d_val)
                if r != "R4":
                    mod_dists.append(d_val)

            d_arr = np.array(d_loo)
            per_reg[r] = {
                "n": n_r,
                "d_m_mean": round(float(np.mean(d_arr)), 4),
                "d_m_max": round(float(np.max(d_arr)), 4),
                "outlier_rate_3sigma_pct": round(float(np.mean(d_arr > thresh_3s) * 100.0), 2),
                "outlier_rate_chi2_p95_pct": round(float(np.mean(d_arr > thresh_c2) * 100.0), 2),
            }
        mod_arr = np.array(mod_dists)
        modes[mode_name] = {
            "modern_aggregate": {
                "n": len(mod_arr),
                "d_m_mean": round(float(np.mean(mod_arr)), 4),
                "d_m_max": round(float(np.max(mod_arr)), 4),
                "outlier_rate_3sigma_pct": round(float(np.mean(mod_arr > thresh_3s) * 100.0), 2),
                "outlier_rate_chi2_p95_pct": round(float(np.mean(mod_arr > thresh_c2) * 100.0), 2),
            },
            "per_register": per_reg,
        }

    return {
        "dimension": df,
        "thresholds": {
            "r4_3sigma_threshold": round(thresh_3s, 4),
            "chi2_p95_threshold": round(thresh_c2, 4),
        },
        "modes": modes,
        "unregularized_loo_collapse_audit": {
            "n_collapse_events": len(collapse_events),
            "events": collapse_events,
            "explanation": (
                "Single-Nonzero LOO Variance Collapse: When only 1 sample in a small register cohort has a non-zero "
                "value on a sparse dimension, leaving that sample out yields LOO variance = 0.0, dividing by the 1e-6 "
                "variance floor and inflating Mahalanobis distance. Shrinkage (0.8*var_loo + 0.2*var_pool) or R4 baseline variance prevents this."
            ),
        },
    }


def evaluate_dataset(
    records: List[Dict[str, Any]],
    aozora_atlas: Dict[str, Any],
    dataset_name: str,
) -> Dict[str, Any]:
    """Run full 4-scheme evaluation on a benchmark dataset (32 or 64 samples)."""
    n_total = len(records)
    labels = [r["register"] for r in records]
    registers = ["R1", "R2", "R3", "R4"]
    reg_indices = {r: [i for i, lab in enumerate(labels) if lab == r] for r in registers}
    n_per_reg = len(reg_indices["R1"])

    # Extract 31D and 24D matrices
    X31 = np.array([r["profile"]["ortho_v2_31d"] for r in records])
    X24 = X31[:, CONT24_DIMS]

    modern_indices = [i for i, lab in enumerate(labels) if lab != "R4"]
    n_modern = len(modern_indices)

    # Atlas baselines
    mean_31_atlas = aozora_atlas["mean_31"]
    cov_31_atlas = aozora_atlas["cov_31"]
    thresh_3sigma_31 = aozora_atlas["thresh_3sigma_31"]
    thresh_chi2_31 = aozora_atlas["thresh_chi2_31"]

    mean_24_atlas = aozora_atlas["mean_24"]
    cov_24_atlas = aozora_atlas["cov_24"]
    thresh_3sigma_24 = aozora_atlas["thresh_3sigma_24"]
    thresh_chi2_24 = aozora_atlas["thresh_chi2_24"]

    # =========================================================================
    # Scheme 0: Raw_R4_Centering (Full 31D ortho_v2 against Aozora Atlas)
    # =========================================================================
    d_m_s0 = np.array([regularized_mahalanobis(X31[i], mean_31_atlas, cov_31_atlas) for i in range(n_total)])

    s0_per_reg = {}
    for r in registers:
        idx = reg_indices[r]
        d_vals = d_m_s0[idx]
        out_3s = float(np.mean(d_vals > thresh_3sigma_31) * 100.0)
        out_c2 = float(np.mean(d_vals > thresh_chi2_31) * 100.0)
        ci_3s = bootstrap_ci(d_vals, lambda arr: np.mean(arr > thresh_3sigma_31) * 100.0)
        ci_c2 = bootstrap_ci(d_vals, lambda arr: np.mean(arr > thresh_chi2_31) * 100.0)
        s0_per_reg[r] = {
            "n": len(idx),
            "d_m_mean": round(float(np.mean(d_vals)), 4),
            "d_m_std": round(float(np.std(d_vals, ddof=1)), 4),
            "outlier_rate_3sigma_pct": round(out_3s, 2),
            "outlier_rate_3sigma_ci95": [round(ci_3s[0], 2), round(ci_3s[1], 2)],
            "outlier_rate_chi2_p95_pct": round(out_c2, 2),
            "outlier_rate_chi2_p95_ci95": [round(ci_c2[0], 2), round(ci_c2[1], 2)],
        }

    modern_d_s0 = d_m_s0[modern_indices]
    s0_modern_3s = float(np.mean(modern_d_s0 > thresh_3sigma_31) * 100.0)
    s0_modern_c2 = float(np.mean(modern_d_s0 > thresh_chi2_31) * 100.0)
    ci_s0_m3s = bootstrap_ci(modern_d_s0, lambda arr: np.mean(arr > thresh_3sigma_31) * 100.0)
    ci_s0_mc2 = bootstrap_ci(modern_d_s0, lambda arr: np.mean(arr > thresh_chi2_31) * 100.0)

    scheme_0_results = {
        "dimension": 31,
        "feature_space": "ortho_v2_31d",
        "baseline_atlas": "Aozora_Bunko_106_works",
        "thresholds": {
            "3sigma_threshold": round(thresh_3sigma_31, 4),
            "chi2_p95_threshold": round(thresh_chi2_31, 4),
        },
        "modern_aggregate": {
            "n": n_modern,
            "d_m_mean": round(float(np.mean(modern_d_s0)), 4),
            "outlier_rate_3sigma_pct": round(s0_modern_3s, 2),
            "outlier_rate_3sigma_ci95": [round(ci_s0_m3s[0], 2), round(ci_s0_m3s[1], 2)],
            "outlier_rate_chi2_p95_pct": round(s0_modern_c2, 2),
            "outlier_rate_chi2_p95_ci95": [round(ci_s0_mc2[0], 2), round(ci_s0_mc2[1], 2)],
        },
        "per_register": s0_per_reg,
    }

    # =========================================================================
    # Scheme 1: Continuous_Core_Subspace (ortho24_cont: 24D)
    # =========================================================================
    d_m_s1 = np.array([regularized_mahalanobis(X24[i], mean_24_atlas, cov_24_atlas) for i in range(n_total)])

    s1_per_reg = {}
    for r in registers:
        idx = reg_indices[r]
        d_vals = d_m_s1[idx]
        out_3s = float(np.mean(d_vals > thresh_3sigma_24) * 100.0)
        out_c2 = float(np.mean(d_vals > thresh_chi2_24) * 100.0)
        ci_3s = bootstrap_ci(d_vals, lambda arr: np.mean(arr > thresh_3sigma_24) * 100.0)
        ci_c2 = bootstrap_ci(d_vals, lambda arr: np.mean(arr > thresh_chi2_24) * 100.0)
        s1_per_reg[r] = {
            "n": len(idx),
            "d_m_mean": round(float(np.mean(d_vals)), 4),
            "d_m_std": round(float(np.std(d_vals, ddof=1)), 4),
            "outlier_rate_3sigma_pct": round(out_3s, 2),
            "outlier_rate_3sigma_ci95": [round(ci_3s[0], 2), round(ci_3s[1], 2)],
            "outlier_rate_chi2_p95_pct": round(out_c2, 2),
            "outlier_rate_chi2_p95_ci95": [round(ci_c2[0], 2), round(ci_c2[1], 2)],
        }

    modern_d_s1 = d_m_s1[modern_indices]
    s1_modern_3s = float(np.mean(modern_d_s1 > thresh_3sigma_24) * 100.0)
    s1_modern_c2 = float(np.mean(modern_d_s1 > thresh_chi2_24) * 100.0)
    ci_s1_m3s = bootstrap_ci(modern_d_s1, lambda arr: np.mean(arr > thresh_3sigma_24) * 100.0)
    ci_s1_mc2 = bootstrap_ci(modern_d_s1, lambda arr: np.mean(arr > thresh_chi2_24) * 100.0)

    scheme_1_results = {
        "dimension": 24,
        "feature_space": "ortho24_cont",
        "dropped_dimensions_7": [ALL_ORTHO_NAMES[i] for i in EXCLUDED_DIMS_7],
        "baseline_atlas": "Aozora_Bunko_106_works",
        "thresholds": {
            "3sigma_threshold": round(thresh_3sigma_24, 4),
            "chi2_p95_threshold": round(thresh_chi2_24, 4),
        },
        "modern_aggregate": {
            "n": n_modern,
            "d_m_mean": round(float(np.mean(modern_d_s1)), 4),
            "outlier_rate_3sigma_pct": round(s1_modern_3s, 2),
            "outlier_rate_3sigma_ci95": [round(ci_s1_m3s[0], 2), round(ci_s1_m3s[1], 2)],
            "outlier_rate_chi2_p95_pct": round(s1_modern_c2, 2),
            "outlier_rate_chi2_p95_ci95": [round(ci_s1_mc2[0], 2), round(ci_s1_mc2[1], 2)],
        },
        "per_register": s1_per_reg,
    }

    # =========================================================================
    # Scheme 2: LOO_Era_Shift_Alignment (時代オフセット補正)
    # =========================================================================
    # Evaluated on R1 (Modern Colloquial Fiction) and All Modern Prose (R1-R3)
    # Testing both 31D and 24D spaces
    def run_loo_era_alignment(
        data_matrix: np.ndarray,
        target_indices: List[int],
        mean_atlas: np.ndarray,
        cov_atlas: np.ndarray,
        thresh_3s: float,
        thresh_c2: float,
    ) -> Dict[str, Any]:
        n_samples = len(target_indices)
        d_m_unaligned = []
        d_m_aligned = []
        drift_norms = []

        for i_idx, i in enumerate(target_indices):
            # LOO mean across remaining samples in target set
            loo_other_idx = [j for j in target_indices if j != i]
            loo_mean = np.mean(data_matrix[loo_other_idx], axis=0)
            delta_era_loo = loo_mean - mean_atlas
            drift_norms.append(float(np.linalg.norm(delta_era_loo)))

            # Original sample
            z_raw = data_matrix[i]
            d_raw = regularized_mahalanobis(z_raw, mean_atlas, cov_atlas)
            d_m_unaligned.append(d_raw)

            # Aligned sample: z_tilde = z - delta_era_loo
            z_tilde = z_raw - delta_era_loo
            d_aligned = regularized_mahalanobis(z_tilde, mean_atlas, cov_atlas)
            d_m_aligned.append(d_aligned)

        d_unaligned = np.array(d_m_unaligned)
        d_aligned = np.array(d_m_aligned)

        out_un_3s = float(np.mean(d_unaligned > thresh_3s) * 100.0)
        out_un_c2 = float(np.mean(d_unaligned > thresh_c2) * 100.0)
        out_al_3s = float(np.mean(d_aligned > thresh_3s) * 100.0)
        out_al_c2 = float(np.mean(d_aligned > thresh_c2) * 100.0)

        ci_al_3s = bootstrap_ci(d_aligned, lambda arr: np.mean(arr > thresh_3s) * 100.0)
        ci_al_c2 = bootstrap_ci(d_aligned, lambda arr: np.mean(arr > thresh_c2) * 100.0)

        # Safety criterion: outlier rate < 15.0%
        passes_15pct_safety = bool(out_al_3s < 15.0 and out_al_c2 < 15.0)

        return {
            "n_samples": n_samples,
            "mean_drift_vector_l2_norm": round(float(np.mean(drift_norms)), 4),
            "unaligned": {
                "d_m_mean": round(float(np.mean(d_unaligned)), 4),
                "outlier_rate_3sigma_pct": round(out_un_3s, 2),
                "outlier_rate_chi2_p95_pct": round(out_un_c2, 2),
            },
            "aligned": {
                "d_m_mean": round(float(np.mean(d_aligned)), 4),
                "outlier_rate_3sigma_pct": round(out_al_3s, 2),
                "outlier_rate_3sigma_ci95": [round(ci_al_3s[0], 2), round(ci_al_3s[1], 2)],
                "outlier_rate_chi2_p95_pct": round(out_al_c2, 2),
                "outlier_rate_chi2_p95_ci95": [round(ci_al_c2[0], 2), round(ci_al_c2[1], 2)],
            },
            "outlier_rate_reduction_3sigma_delta": round(out_un_3s - out_al_3s, 2),
            "outlier_rate_reduction_chi2_delta": round(out_un_c2 - out_al_c2, 2),
            "passes_15pct_safety_threshold": passes_15pct_safety,
        }

    r1_idx = reg_indices["R1"]
    scheme_2_results = {
        "description": "LOO Diachronic drift alignment z_tilde = z - delta_era_LOO vs Aozora Atlas",
        "R1_fiction_alignment": {
            "ortho_v2_31d": run_loo_era_alignment(
                X31, r1_idx, mean_31_atlas, cov_31_atlas, thresh_3sigma_31, thresh_chi2_31
            ),
            "ortho24_cont": run_loo_era_alignment(
                X24, r1_idx, mean_24_atlas, cov_24_atlas, thresh_3sigma_24, thresh_chi2_24
            ),
        },
        "all_modern_pooled_alignment": {
            "ortho_v2_31d": run_loo_era_alignment(
                X31, modern_indices, mean_31_atlas, cov_31_atlas, thresh_3sigma_31, thresh_chi2_31
            ),
            "ortho24_cont": run_loo_era_alignment(
                X24, modern_indices, mean_24_atlas, cov_24_atlas, thresh_3sigma_24, thresh_chi2_24
            ),
        },
    }

    # =========================================================================
    # Scheme 3: Within_Register_LOO_Centering (レジスター別中心化)
    # =========================================================================
    # 1. Variance Decomposition: eta^2_register across stages
    # - Raw
    # - Global Column Centered
    # - Global Standardized
    # - Within-Register LOO Centered
    # - Within-Register LOO Standardized

    # Global column centering
    X31_glob_c = X31 - np.mean(X31, axis=0)
    X31_glob_s = X31_glob_c / np.maximum(np.std(X31, axis=0, ddof=1), 1e-12)

    X24_glob_c = X24 - np.mean(X24, axis=0)
    X24_glob_s = X24_glob_c / np.maximum(np.std(X24, axis=0, ddof=1), 1e-12)

    # Within-register LOO centering & standardization
    X31_within_c = np.zeros_like(X31)
    X31_within_s = np.zeros_like(X31)
    X24_within_c = np.zeros_like(X24)
    X24_within_s = np.zeros_like(X24)

    # Also compute intra-register distance structure stability
    # Pairwise distance correlation before and after centering within each register
    intra_distance_correlations = {}

    for r in registers:
        idx = reg_indices[r]
        sub_X31 = X31[idx]
        sub_X24 = X24[idx]
        n_r = len(idx)

        # Pairwise euclidean distances before
        pdist_orig = []
        for i in range(n_r):
            for j in range(i + 1, n_r):
                pdist_orig.append(float(np.linalg.norm(sub_X24[i] - sub_X24[j])))

        sub_c_31 = np.zeros_like(sub_X31)
        sub_s_31 = np.zeros_like(sub_X31)
        sub_c_24 = np.zeros_like(sub_X24)
        sub_s_24 = np.zeros_like(sub_X24)

        for i in range(n_r):
            loo_idx = [j for j in range(n_r) if j != i]
            loo_mean_31 = np.mean(sub_X31[loo_idx], axis=0)
            loo_std_31 = np.maximum(np.std(sub_X31[loo_idx], axis=0, ddof=1), 1e-6)

            loo_mean_24 = np.mean(sub_X24[loo_idx], axis=0)
            loo_std_24 = np.maximum(np.std(sub_X24[loo_idx], axis=0, ddof=1), 1e-6)

            sub_c_31[i] = sub_X31[i] - loo_mean_31
            sub_s_31[i] = (sub_X31[i] - loo_mean_31) / loo_std_31

            sub_c_24[i] = sub_X24[i] - loo_mean_24
            sub_s_24[i] = (sub_X24[i] - loo_mean_24) / loo_std_24

        X31_within_c[idx] = sub_c_31
        X31_within_s[idx] = sub_s_31
        X24_within_c[idx] = sub_c_24
        X24_within_s[idx] = sub_s_24

        # Pairwise euclidean distances after LOO centering
        pdist_centered = []
        for i in range(n_r):
            for j in range(i + 1, n_r):
                pdist_centered.append(float(np.linalg.norm(sub_c_24[i] - sub_c_24[j])))

        # Spearman rank correlation between pairwise distances
        sp_corr, _ = stats.spearmanr(pdist_orig, pdist_centered)
        pe_corr, _ = stats.pearsonr(pdist_orig, pdist_centered)
        intra_distance_correlations[r] = {
            "spearman_rank_corr": round(float(sp_corr), 4),
            "pearson_corr": round(float(pe_corr), 4),
            "distance_structure_preserved": bool(sp_corr >= 0.90),
        }

    # Compute eta^2 across conditions
    eta2_31_raw = compute_eta_squared(X31, labels)
    eta2_31_glob_c = compute_eta_squared(X31_glob_c, labels)
    eta2_31_glob_s = compute_eta_squared(X31_glob_s, labels)
    eta2_31_within_c = compute_eta_squared(X31_within_c, labels)
    eta2_31_within_s = compute_eta_squared(X31_within_s, labels)

    eta2_24_raw = compute_eta_squared(X24, labels)
    eta2_24_glob_c = compute_eta_squared(X24_glob_c, labels)
    eta2_24_glob_s = compute_eta_squared(X24_glob_s, labels)
    eta2_24_within_c = compute_eta_squared(X24_within_c, labels)
    eta2_24_within_s = compute_eta_squared(X24_within_s, labels)

    baseline_a = evaluate_baseline_a_diagonal(X31, labels, reg_indices, modern_indices)
    within_outliers_cont24 = evaluate_within_register_outliers(
        X24, reg_indices, [ALL_ORTHO_NAMES[i] for i in CONT24_DIMS]
    )
    within_outliers_nosparse24 = evaluate_within_register_outliers(
        X31[:, NO_SPARSE_FINALS_24_DIMS],
        reg_indices,
        [ALL_ORTHO_NAMES[i] for i in NO_SPARSE_FINALS_24_DIMS],
    )

    scheme_3_results = {
        "description": "Within-register LOO centering & standardization: elimination of between-register variance",
        "eta_squared_register_elimination": {
            "ortho_v2_31d": {
                "eta2_raw": round(eta2_31_raw, 4),
                "eta2_global_centered": round(eta2_31_glob_c, 4),
                "eta2_global_standardized": round(eta2_31_glob_s, 4),
                "eta2_within_register_loo_centered": round(eta2_31_within_c, 6),
                "eta2_within_register_loo_standardized": round(eta2_31_within_s, 6),
                "register_variance_elimination_rate_pct": round((1.0 - eta2_31_within_c / max(eta2_31_raw, 1e-12)) * 100.0, 2),
            },
            "ortho24_cont": {
                "eta2_raw": round(eta2_24_raw, 4),
                "eta2_global_centered": round(eta2_24_glob_c, 4),
                "eta2_global_standardized": round(eta2_24_glob_s, 4),
                "eta2_within_register_loo_centered": round(eta2_24_within_c, 6),
                "eta2_within_register_loo_standardized": round(eta2_24_within_s, 6),
                "register_variance_elimination_rate_pct": round((1.0 - eta2_24_within_c / max(eta2_24_raw, 1e-12)) * 100.0, 2),
            },
        },
        "intra_register_distance_structure_fidelity": intra_distance_correlations,
        "within_register_outlier_rates": {
            "ortho24_cont": within_outliers_cont24,
            "ortho24_no_sparse_finals": within_outliers_nosparse24,
        },
    }

    return {
        "dataset_name": dataset_name,
        "n_samples": n_total,
        "n_per_register": n_per_reg,
        "baseline_a_r4_in_benchmark_diagonal": baseline_a,
        "scheme_0_raw_r4_centering_31d": scheme_0_results,
        "scheme_1_continuous_core_subspace_24d": scheme_1_results,
        "scheme_2_loo_era_shift_alignment": scheme_2_results,
        "scheme_3_within_register_loo_centering": scheme_3_results,
    }


def compute_b10_mc_quantiles(
    X24_r4: np.ndarray,
    seed: int = 42,
    n_trials: int = 5000,
) -> Dict[str, Any]:
    """Compute B-10 finite-sample Monte Carlo null quantiles (24D & 31D) under exact B-10 §4 setup.

    Setup: n_r=16, m=15, nu=14, LOO Student-t inflation factor = (16/15)*(14/12) = 1.244444.
    Pools all n_trials * 16 LOO D^2 evaluations to extract the finite-sample 95% and 99% quantiles.
    """
    std_scene_r4 = np.sqrt(np.var(X24_r4, axis=0, ddof=1))
    active_23_idx = [i for i, s in enumerate(std_scene_r4) if s > 1e-6]
    assert len(active_23_idx) == 23

    X23_r4 = X24_r4[:, active_23_idx]
    corr23 = np.corrcoef(X23_r4, rowvar=False)
    eigvals23 = np.linalg.eigvalsh(corr23)[::-1]
    min_eig = np.min(eigvals23)
    cov_safe = corr23.copy()
    if min_eig < 1e-6:
        cov_safe += np.eye(23) * (1e-6 - min_eig)
    L_chol = np.linalg.cholesky(cov_safe)

    m_val = 15
    nu_val = 14
    inflation_factor = (1.0 + 1.0 / m_val) * (nu_val / (nu_val - 2.0))

    rng = np.random.default_rng(seed)
    chi2_95_24 = float(stats.chi2.ppf(0.95, df=24))
    chi2_99_24 = float(stats.chi2.ppf(0.99, df=24))

    def _run_loo_mc(generator_fn) -> Tuple[List[float], List[float], np.ndarray]:
        exceed_trial_rates: List[float] = []
        mean_trial_d2: List[float] = []
        pooled_d2 = np.empty(n_trials * 16, dtype=float)
        offset = 0
        for _ in range(n_trials):
            X_sim = generator_fn()
            d2_list = []
            for i in range(16):
                others = np.delete(X_sim, i, axis=0)
                mu_l = np.mean(others, axis=0)
                var_l = np.maximum(np.var(others, axis=0, ddof=1), 1e-6)
                d2 = float(np.sum((X_sim[i] - mu_l) ** 2 / var_l))
                d2_list.append(d2)
            d2_arr = np.array(d2_list, dtype=float)
            pooled_d2[offset : offset + 16] = d2_arr
            offset += 16
            exceed_trial_rates.append(float(np.mean(d2_arr > chi2_95_24)))
            mean_trial_d2.append(float(np.mean(d2_arr)))
        return exceed_trial_rates, mean_trial_d2, pooled_d2

    mc_exceed_indep, mc_d2_indep, pooled_d2_indep = _run_loo_mc(lambda: rng.normal(size=(16, 24)))
    mc_exceed_r4, mc_d2_r4, pooled_d2_r4 = _run_loo_mc(
        lambda: np.hstack([rng.normal(size=(16, 23)) @ L_chol.T, np.zeros((16, 1))])
    )

    # Separate RNG for 31D independent baseline so 24D RNG stream remains bit-identical to B-10
    rng31 = np.random.default_rng(seed)
    chi2_95_31 = float(stats.chi2.ppf(0.95, df=31))
    pooled_d2_31 = np.empty(n_trials * 16, dtype=float)
    exceed_31 = []
    offset31 = 0
    for _ in range(n_trials):
        X_sim31 = rng31.normal(size=(16, 31))
        d2_list31 = []
        for i in range(16):
            others = np.delete(X_sim31, i, axis=0)
            mu_l = np.mean(others, axis=0)
            var_l = np.maximum(np.var(others, axis=0, ddof=1), 1e-6)
            d2 = float(np.sum((X_sim31[i] - mu_l) ** 2 / var_l))
            d2_list31.append(d2)
        d2_arr31 = np.array(d2_list31, dtype=float)
        pooled_d2_31[offset31 : offset31 + 16] = d2_arr31
        offset31 += 16
        exceed_31.append(float(np.mean(d2_arr31 > chi2_95_31)))

    def _quantile_block(pooled: np.ndarray) -> Dict[str, float]:
        q95_d2 = float(np.percentile(pooled, 95.0))
        q99_d2 = float(np.percentile(pooled, 99.0))
        return {
            "mc_p95_d2": round(q95_d2, 4),
            "mc_p95_d": round(float(np.sqrt(q95_d2)), 4),
            "mc_p99_d2": round(q99_d2, 4),
            "mc_p99_d": round(float(np.sqrt(q99_d2)), 4),
            "n_samples_total": int(len(pooled)),
            "exceedance_rate_mc_p95_pct": round(float(np.mean(pooled > q95_d2) * 100.0), 2),
            "exceedance_rate_mc_p99_pct": round(float(np.mean(pooled > q99_d2) * 100.0), 2),
        }

    return {
        "seed": seed,
        "n_trials": n_trials,
        "m_loo_observations": m_val,
        "nu_degrees_of_freedom": nu_val,
        "theoretical_d2_inflation_factor": round(inflation_factor, 6),
        "active_23_idx": active_23_idx,
        "corr23": corr23,
        "eigvals23": eigvals23,
        "chi2_ref_24d": {
            "chi2_p95_d2": round(chi2_95_24, 4),
            "chi2_p95_d": round(float(np.sqrt(chi2_95_24)), 4),
            "chi2_p99_d2": round(chi2_99_24, 4),
            "chi2_p99_d": round(float(np.sqrt(chi2_99_24)), 4),
        },
        "independent_gaussian_24d": {
            "mean_d2": round(float(np.mean(mc_d2_indep)), 4),
            "asymptotic_expected_d2": 24.0,
            "empirical_inflation_ratio": round(float(np.mean(mc_d2_indep)) / 24.0, 4),
            "exceedance_rate_chi2_p95_pct": round(float(np.mean(mc_exceed_indep) * 100.0), 2),
            **_quantile_block(pooled_d2_indep),
        },
        "r4_correlated_gaussian_24d": {
            "mean_d2": round(float(np.mean(mc_d2_r4)), 4),
            "exceedance_rate_chi2_p95_pct": round(float(np.mean(mc_exceed_r4) * 100.0), 2),
            **_quantile_block(pooled_d2_r4),
        },
        "independent_gaussian_31d": {
            "mean_d2": round(float(np.mean(pooled_d2_31)), 4),
            "asymptotic_expected_d2": 31.0,
            "empirical_inflation_ratio": round(float(np.mean(pooled_d2_31)) / 31.0, 4),
            "exceedance_rate_chi2_p95_pct": round(float(np.mean(exceed_31) * 100.0), 2),
            **_quantile_block(pooled_d2_31),
        },
    }


def get_or_compute_b10_mc_quantiles(
    b10_json_path: Optional[Path] = None,
    X24_r4: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Load MC 95%/99% quantiles from B-10 artifact if present; else recompute under B-10 §4."""
    target = Path(b10_json_path) if b10_json_path is not None else OUTPUT_B10_JSON_PATH
    if target.exists():
        data = json.loads(target.read_text(encoding="utf-8"))
        mc = (
            data.get("b10_scene_scale_redesign_evaluation", {})
            .get("r4_nominal_level_exceedance_quantification", {})
            .get("2_finite_sample_loo_student_t_inflation", {})
            .get("monte_carlo_5000_trials", {})
        )
        indep = mc.get("independent_gaussian_24d", {})
        r4corr = mc.get("r4_correlated_gaussian_24d", {})
        if "mc_p95_d" in indep and "mc_p99_d" in indep and "mc_p95_d" in r4corr and "mc_p99_d" in r4corr:
            return {
                "source": "loaded_from_b10_json",
                "b10_json_path": str(target),
                "independent_gaussian_24d": indep,
                "r4_correlated_gaussian_24d": r4corr,
                "independent_gaussian_31d": mc.get("independent_gaussian_31d", {}),
            }

    if X24_r4 is None:
        records = load_benchmark_records(BENCHMARK_64_PATH)
        r4_records = [r for r in records if r["register"] == "R4"]
        X24_r4 = np.array([r["profile"]["ortho_v2_31d"] for r in r4_records])[:, CONT24_DIMS]
    res = compute_b10_mc_quantiles(X24_r4, seed=42, n_trials=5000)
    res["source"] = "recomputed_b10_sec4_conditions"
    res["b10_json_path"] = str(target)
    return res


def evaluate_b10_scene_scale_redesign(
    records_64: List[Dict[str, Any]],
    aozora_atlas: Dict[str, Any],
    task1_stage_json_path: Path,
) -> Dict[str, Any]:
    """USIR v5 Stage B-10: Scene-level denominator redesign & finite-sample quantification."""
    labels = [r["register"] for r in records_64]
    registers = ["R1", "R2", "R3", "R4"]
    reg_indices = {r: [i for i, lab in enumerate(labels) if lab == r] for r in registers}
    modern_indices = [i for i, lab in enumerate(labels) if lab != "R4"]

    X31 = np.array([r["profile"]["ortho_v2_31d"] for r in records_64])
    X24 = X31[:, CONT24_DIMS]

    # Atlas baselines
    mu_aozora_31 = aozora_atlas["mean_31"]
    thresh_3s_atlas_31 = aozora_atlas["thresh_3sigma_31"]
    mu_aozora_24 = aozora_atlas["mean_24"]
    thresh_3s_atlas_24 = aozora_atlas["thresh_3sigma_24"]
    std_work_24 = np.sqrt(aozora_atlas["var_24"])

    # -------------------------------------------------------------------------
    # 1. Task 2 Fixed H1/H2 Ratings from B-8/B-9 Stage JSON
    # -------------------------------------------------------------------------
    task1_json = json.loads(task1_stage_json_path.read_text(encoding="utf-8"))
    l3b = task1_json["level3b_scale_separated_specificity"]
    h1_raw = l3b["h1_micro_scale"]
    h2_raw = l3b["h2_macro_scale"]

    task2_fixed_ratings = {
        "protocol": "Two-sided Wilcoxon signed-rank test with Bonferroni correction (k=4)",
        "h1_micro_scale": {
            "ortho28": {
                "dimension": 28,
                "n_pairs": h1_raw["ortho28"]["n_pairs"],
                "mean_cos_spec": h1_raw["ortho28"]["mean_cos_spec"],
                "vs_random": {
                    "w_stat": h1_raw["ortho28"]["vs_random"]["w_stat"],
                    "p_one_sided": h1_raw["ortho28"]["vs_random"]["p_one_sided"],
                    "p_two_sided": h1_raw["ortho28"]["vs_random"]["p_two_sided"],
                    "bonferroni_p_two": h1_raw["ortho28"]["vs_random"]["bonferroni_p_two"],
                    "significant_after_bonferroni": False,
                },
                "vs_other": {
                    "w_stat": h1_raw["ortho28"]["vs_other"]["w_stat"],
                    "p_one_sided": h1_raw["ortho28"]["vs_other"]["p_one_sided"],
                    "p_two_sided": h1_raw["ortho28"]["vs_other"]["p_two_sided"],
                    "bonferroni_p_two": h1_raw["ortho28"]["vs_other"]["bonferroni_p_two"],
                    "significant_after_bonferroni": True,
                },
            },
            "ortho28_plus_kasane11": {
                "dimension": 39,
                "n_pairs": h1_raw["ortho28_plus_kasane11"]["n_pairs"],
                "mean_cos_spec": h1_raw["ortho28_plus_kasane11"]["mean_cos_spec"],
                "vs_random": {
                    "w_stat": h1_raw["ortho28_plus_kasane11"]["vs_random"]["w_stat"],
                    "p_one_sided": h1_raw["ortho28_plus_kasane11"]["vs_random"]["p_one_sided"],
                    "p_two_sided": h1_raw["ortho28_plus_kasane11"]["vs_random"]["p_two_sided"],
                    "bonferroni_p_two": h1_raw["ortho28_plus_kasane11"]["vs_random"]["bonferroni_p_two"],
                    "significant_after_bonferroni": False,
                },
                "vs_other": {
                    "w_stat": h1_raw["ortho28_plus_kasane11"]["vs_other"]["w_stat"],
                    "p_one_sided": h1_raw["ortho28_plus_kasane11"]["vs_other"]["p_one_sided"],
                    "p_two_sided": h1_raw["ortho28_plus_kasane11"]["vs_other"]["p_two_sided"],
                    "bonferroni_p_two": h1_raw["ortho28_plus_kasane11"]["vs_other"]["bonferroni_p_two"],
                    "significant_after_bonferroni": True,
                },
            },
            "ortho24_cont_plus_kasane8_active": {
                "dimension": 32,
                "n_pairs": h1_raw["ortho24_cont_plus_kasane8_active"]["n_pairs"],
                "mean_cos_spec": h1_raw["ortho24_cont_plus_kasane8_active"]["mean_cos_spec"],
                "vs_random": {
                    "w_stat": h1_raw["ortho24_cont_plus_kasane8_active"]["vs_random"]["w_stat"],
                    "p_one_sided": h1_raw["ortho24_cont_plus_kasane8_active"]["vs_random"]["p_one_sided"],
                    "p_two_sided": h1_raw["ortho24_cont_plus_kasane8_active"]["vs_random"]["p_two_sided"],
                    "bonferroni_p_two": h1_raw["ortho24_cont_plus_kasane8_active"]["vs_random"]["bonferroni_p_two"],
                    "significant_after_bonferroni": False,
                },
                "vs_other": {
                    "w_stat": h1_raw["ortho24_cont_plus_kasane8_active"]["vs_other"]["w_stat"],
                    "p_one_sided": h1_raw["ortho24_cont_plus_kasane8_active"]["vs_other"]["p_one_sided"],
                    "p_two_sided": h1_raw["ortho24_cont_plus_kasane8_active"]["vs_other"]["p_two_sided"],
                    "bonferroni_p_two": h1_raw["ortho24_cont_plus_kasane8_active"]["vs_other"]["bonferroni_p_two"],
                    "significant_after_bonferroni": True,
                },
            },
        },
        "h2_macro_scale": {
            "6pole": {
                "dimension": 6,
                "n_pairs": h2_raw["6pole"]["n_pairs"],
                "mean_cos_spec": h2_raw["6pole"]["mean_cos_spec"],
                "vs_random": {
                    "w_stat": h2_raw["6pole"]["vs_random"]["w_stat"],
                    "p_one_sided": h2_raw["6pole"]["vs_random"]["p_one_sided"],
                    "p_two_sided": h2_raw["6pole"]["vs_random"]["p_two_sided"],
                    "bonferroni_p_two": h2_raw["6pole"]["vs_random"]["bonferroni_p_two"],
                    "significant_after_bonferroni": False,
                },
                "vs_other": {
                    "w_stat": h2_raw["6pole"]["vs_other"]["w_stat"],
                    "p_one_sided": h2_raw["6pole"]["vs_other"]["p_one_sided"],
                    "p_two_sided": h2_raw["6pole"]["vs_other"]["p_two_sided"],
                    "bonferroni_p_two": h2_raw["6pole"]["vs_other"]["bonferroni_p_two"],
                    "significant_after_bonferroni": True,
                },
            },
            "6pole_plus_hw25": {
                "dimension": 7,
                "n_pairs": h2_raw["6pole_plus_hw25"]["n_pairs"],
                "mean_cos_spec": h2_raw["6pole_plus_hw25"]["mean_cos_spec"],
                "vs_random": {
                    "w_stat": h2_raw["6pole_plus_hw25"]["vs_random"]["w_stat"],
                    "p_one_sided": h2_raw["6pole_plus_hw25"]["vs_random"]["p_one_sided"],
                    "p_two_sided": h2_raw["6pole_plus_hw25"]["vs_random"]["p_two_sided"],
                    "bonferroni_p_two": h2_raw["6pole_plus_hw25"]["vs_random"]["bonferroni_p_two"],
                    "significant_after_bonferroni": False,
                },
                "vs_other": {
                    "w_stat": h2_raw["6pole_plus_hw25"]["vs_other"]["w_stat"],
                    "p_one_sided": h2_raw["6pole_plus_hw25"]["vs_other"]["p_one_sided"],
                    "p_two_sided": h2_raw["6pole_plus_hw25"]["vs_other"]["p_two_sided"],
                    "bonferroni_p_two": h2_raw["6pole_plus_hw25"]["vs_other"]["bonferroni_p_two"],
                    "significant_after_bonferroni": True,
                },
            },
            "4pole_cont_plus_hw25": {
                "dimension": 5,
                "n_pairs": h2_raw["4pole_cont_plus_hw25"]["n_pairs"],
                "mean_cos_spec": h2_raw["4pole_cont_plus_hw25"]["mean_cos_spec"],
                "vs_random": {
                    "w_stat": h2_raw["4pole_cont_plus_hw25"]["vs_random"]["w_stat"],
                    "p_one_sided": h2_raw["4pole_cont_plus_hw25"]["vs_random"]["p_one_sided"],
                    "p_two_sided": h2_raw["4pole_cont_plus_hw25"]["vs_random"]["p_two_sided"],
                    "bonferroni_p_two": h2_raw["4pole_cont_plus_hw25"]["vs_random"]["bonferroni_p_two"],
                    "significant_after_bonferroni": False,
                },
                "vs_other": {
                    "w_stat": h2_raw["4pole_cont_plus_hw25"]["vs_other"]["w_stat"],
                    "p_one_sided": h2_raw["4pole_cont_plus_hw25"]["vs_other"]["p_one_sided"],
                    "p_two_sided": h2_raw["4pole_cont_plus_hw25"]["vs_other"]["p_two_sided"],
                    "bonferroni_p_two": h2_raw["4pole_cont_plus_hw25"]["vs_other"]["bonferroni_p_two"],
                    "significant_after_bonferroni": True,
                },
            },
        },
        "fixed_ratings_summary": {
            "specificity_vs_other": "Specificities vs other authors/styles remain statistically significant across all spaces after Bonferroni correction (k=4, two-sided p_bonf in [0.0129, 0.0420] < 0.05).",
            "specificity_vs_isotropic_random": "Specificities vs isotropic random perturbations are nominal p < 0.05 at best and do NOT survive two-sided Bonferroni correction (k=4); in H2 5D continuous core (4pole_cont_plus_hw25), vs-random effect completely disappears (p_two = 0.4060, bonferroni_p_two = 1.0000).",
            "prohibited_claims_guard": [
                "Do NOT claim 'approach toward target exceeding random' on the macro side.",
                "Do NOT claim 'discrete poles carried a spurious signal' for the 5D cos drop."
            ],
        },
    }

    # -------------------------------------------------------------------------
    # 2. Precompute Finite-Sample MC Null Quantiles (B-10 §4 exact setup: seed=42, 5000 trials)
    # -------------------------------------------------------------------------
    X24_r4 = X24[reg_indices["R4"]]
    mc_quant = compute_b10_mc_quantiles(X24_r4, seed=42, n_trials=5000)
    mc_24d_thresh = {
        "mc_indep_p95": mc_quant["independent_gaussian_24d"]["mc_p95_d"],
        "mc_indep_p99": mc_quant["independent_gaussian_24d"]["mc_p99_d"],
        "mc_r4corr_p95": mc_quant["r4_correlated_gaussian_24d"]["mc_p95_d"],
        "mc_r4corr_p99": mc_quant["r4_correlated_gaussian_24d"]["mc_p99_d"],
    }
    mc_31d_thresh = {
        "mc_indep_p95": mc_quant["independent_gaussian_31d"]["mc_p95_d"],
        "mc_indep_p99": mc_quant["independent_gaussian_31d"]["mc_p99_d"],
    }

    # -------------------------------------------------------------------------
    # 3. Scene-Scale Evaluation (Condition a & b across 24D and 31D)
    # -------------------------------------------------------------------------
    def _eval_space(
        X: np.ndarray,
        space_name: str,
        mu_aozora: np.ndarray,
        thresh_3s_atlas: float,
        mc_thresh: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        d = X.shape[1]
        thresh_c2_95 = float(np.sqrt(stats.chi2.ppf(0.95, df=d)))
        thresh_c2_99 = float(np.sqrt(stats.chi2.ppf(0.99, df=d)))
        thresh_c2_9973 = float(np.sqrt(stats.chi2.ppf(0.9973, df=d)))
        mc_thresh = mc_thresh or {}

        var_insample = {r: np.maximum(np.var(X[reg_indices[r]], axis=0, ddof=1), 1e-6) for r in registers}
        var_pool = np.maximum(np.mean([var_insample[r] for r in registers], axis=0), 1e-6)

        def _summarize_arr(arr: np.ndarray, thresh_3s_r4: float) -> Dict[str, Any]:
            rec: Dict[str, Any] = {
                "n": len(arr),
                "d_m_mean": round(float(np.mean(arr)), 4),
                "d_m_std": round(float(np.std(arr, ddof=1)), 4),
                "d_m_max": round(float(np.max(arr)), 4),
                "outlier_rate_atlas_3sigma_pct": round(float(np.mean(arr > thresh_3s_atlas) * 100.0), 2),
                "outlier_rate_r4_3sigma_pct": round(float(np.mean(arr > thresh_3s_r4) * 100.0), 2),
                "outlier_rate_chi2_p95_pct": round(float(np.mean(arr > thresh_c2_95) * 100.0), 2),
                "outlier_rate_chi2_p99_pct": round(float(np.mean(arr > thresh_c2_99) * 100.0), 2),
                "outlier_rate_chi2_p9973_pct": round(float(np.mean(arr > thresh_c2_9973) * 100.0), 2),
            }
            if "mc_indep_p95" in mc_thresh:
                rec["outlier_rate_mc_indep_p95_pct"] = round(float(np.mean(arr > mc_thresh["mc_indep_p95"]) * 100.0), 2)
                rec["outlier_rate_mc_indep_p99_pct"] = round(float(np.mean(arr > mc_thresh["mc_indep_p99"]) * 100.0), 2)
            if "mc_r4corr_p95" in mc_thresh:
                rec["outlier_rate_mc_r4corr_p95_pct"] = round(float(np.mean(arr > mc_thresh["mc_r4corr_p95"]) * 100.0), 2)
                rec["outlier_rate_mc_r4corr_p99_pct"] = round(float(np.mean(arr > mc_thresh["mc_r4corr_p99"]) * 100.0), 2)
            return rec

        def _run_cond(is_centered: bool) -> Dict[str, Any]:
            modes_res = {}
            for var_mode in ["in_sample", "shrinkage_loo", "unregularized_loo"]:
                d_per_reg = {r: [] for r in registers}
                for r in registers:
                    idx = reg_indices[r]
                    sub = X[idx]
                    n_r = len(idx)
                    for i in range(n_r):
                        others = [j for j in range(n_r) if j != i]
                        if is_centered:
                            mu_ref = np.mean(sub[others], axis=0)
                        else:
                            mu_ref = mu_aozora

                        if var_mode == "in_sample":
                            v_use = var_insample[r]
                        elif var_mode == "shrinkage_loo":
                            v_loo = np.var(sub[others], axis=0, ddof=1)
                            v_use = np.maximum(0.8 * v_loo + 0.2 * var_pool, 1e-6)
                        else:
                            v_loo = np.var(sub[others], axis=0, ddof=1)
                            v_use = np.maximum(v_loo, 1e-6)

                        d_val = diagonal_mahalanobis(sub[i], mu_ref, v_use)
                        d_per_reg[r].append(d_val)

                d_r4_arr = np.array(d_per_reg["R4"])
                thresh_3s_r4 = float(np.mean(d_r4_arr) + 3.0 * np.std(d_r4_arr, ddof=1))

                reg_summary = {r: _summarize_arr(np.array(d_per_reg[r]), thresh_3s_r4) for r in registers}

                mod_d_list = []
                for r in ["R1", "R2", "R3"]:
                    mod_d_list.extend(d_per_reg[r])
                mod_summary = _summarize_arr(np.array(mod_d_list), thresh_3s_r4)

                modes_res[var_mode] = {
                    "r4_empirical_3sigma_threshold": round(thresh_3s_r4, 4),
                    "per_register": reg_summary,
                    "modern_aggregate": mod_summary,
                }
            return modes_res

        # Regularized Covariance LOO on R4 (Diagnostic on rank deficiency)
        d_cov_r4 = []
        X_r4 = X[reg_indices["R4"]]
        for i in range(len(X_r4)):
            others = np.delete(X_r4, i, axis=0)
            mu_loo = np.mean(others, axis=0)
            cov_loo = np.cov(others, rowvar=False)
            d_val = regularized_mahalanobis(X_r4[i], mu_loo, cov_loo, shrinkage=0.15)
            d_cov_r4.append(d_val)
        d_cov_r4_arr = np.array(d_cov_r4)
        cov_outlier_pct = float(np.mean(d_cov_r4_arr > thresh_c2_95) * 100.0)

        # eta^2 decomposition
        X_glob_c = X - np.mean(X, axis=0)
        X_glob_s = X_glob_c / np.maximum(np.std(X, axis=0, ddof=1), 1e-12)
        X_within_c = np.zeros_like(X)
        X_within_s = np.zeros_like(X)
        intra_fidelity = {}

        for r in registers:
            idx = reg_indices[r]
            sub = X[idx]
            n_r = len(idx)
            sub_c = np.zeros_like(sub)
            sub_s = np.zeros_like(sub)
            d_mahal_in = []
            d_mahal_loo = []

            mu_in = np.mean(sub, axis=0)
            var_in = np.maximum(np.var(sub, axis=0, ddof=1), 1e-6)

            for i in range(n_r):
                others = [j for j in range(n_r) if j != i]
                mu_loo = np.mean(sub[others], axis=0)
                var_loo = np.maximum(np.var(sub[others], axis=0, ddof=1), 1e-6)

                sub_c[i] = sub[i] - mu_loo
                sub_s[i] = (sub[i] - mu_loo) / np.sqrt(var_loo)

                d_mahal_in.append(diagonal_mahalanobis(sub[i], mu_in, var_in))
                d_mahal_loo.append(diagonal_mahalanobis(sub[i], mu_loo, var_loo))

            X_within_c[idx] = sub_c
            X_within_s[idx] = sub_s

            pdist_raw = []
            pdist_loo_c = []
            pdist_loo_s = []
            for i in range(n_r):
                for j in range(i + 1, n_r):
                    pdist_raw.append(float(np.linalg.norm(sub[i] - sub[j])))
                    pdist_loo_c.append(float(np.linalg.norm(sub_c[i] - sub_c[j])))
                    pdist_loo_s.append(float(np.linalg.norm(sub_s[i] - sub_s[j])))

            sp_eucl, _ = stats.spearmanr(pdist_raw, pdist_loo_c)
            sp_mahal, _ = stats.spearmanr(d_mahal_in, d_mahal_loo)
            sp_std, _ = stats.spearmanr(pdist_raw, pdist_loo_s)

            intra_fidelity[r] = {
                "spearman_euclidean_raw_vs_loo_centered": round(float(sp_eucl), 4),
                "spearman_mahalanobis_insample_vs_loo": round(float(sp_mahal), 4),
                "spearman_pairwise_raw_vs_loo_standardized": round(float(sp_std), 4),
                "distance_structure_preserved": bool(sp_eucl >= 0.90),
            }

        eta2_raw = compute_eta_squared(X, labels)
        eta2_glob_c = compute_eta_squared(X_glob_c, labels)
        eta2_glob_s = compute_eta_squared(X_glob_s, labels)
        eta2_within_c = compute_eta_squared(X_within_c, labels)
        eta2_within_s = compute_eta_squared(X_within_s, labels)

        thresholds_dict: Dict[str, Any] = {
            "atlas_3sigma": round(thresh_3s_atlas, 4),
            "chi2_p95": round(thresh_c2_95, 4),
            "chi2_p99": round(thresh_c2_99, 4),
            "chi2_p9973": round(thresh_c2_9973, 4),
            "canonical_denominator": "sigma2_scene_r_diagonal",
            "canonical_threshold_family": "finite_sample_mc_loo_student_t_null",
            "primary_alarm_threshold_key": PRIMARY_ALARM_THRESHOLD_KEY,
            "primary_alarm_threshold_d": PRIMARY_ALARM_THRESHOLD_D,
            "auxiliary_alarm_threshold_key": AUXILIARY_ALARM_THRESHOLD_KEY,
            "auxiliary_alarm_threshold_d": AUXILIARY_ALARM_THRESHOLD_D,
            **{k: round(float(v), 4) for k, v in mc_thresh.items()},
        }

        return {
            "space_name": space_name,
            "dimension": d,
            "thresholds": thresholds_dict,
            "condition_a_uncentered_scene_scaled": _run_cond(is_centered=False),
            "condition_b_within_register_loo_centered_scene_scaled": _run_cond(is_centered=True),
            "covariance_rank_deficiency_diagnostic": {
                "dimension_d": d,
                "degrees_of_freedom_nu": 14,
                "n_samples_per_register": 16,
                "rank_deficient": bool(d > 14),
                "r4_loo_outlier_rate_chi2_p95_pct": round(cov_outlier_pct, 2),
                "explanation": (
                    f"With n_r=16 samples per register, LOO sample covariance has degrees of freedom nu=14. "
                    f"Since dimension d={d} > nu=14, the sample covariance matrix is mathematically rank-deficient. "
                    f"Diagonal shrinkage (0.15) inverts near-nullspace noise, producing {cov_outlier_pct:.2f}% false outliers "
                    f"on the R4 self-control cohort. This proves why univariate scene-level variance is necessary."
                ),
            },
            "eta_squared_register_elimination": {
                "eta2_raw": round(eta2_raw, 4),
                "eta2_global_centered": round(eta2_glob_c, 4),
                "eta2_global_standardized": round(eta2_glob_s, 4),
                "eta2_within_register_loo_centered": round(eta2_within_c, 6),
                "eta2_within_register_loo_standardized": round(eta2_within_s, 6),
                "register_variance_elimination_rate_pct": round((1.0 - eta2_within_c / max(eta2_raw, 1e-12)) * 100.0, 2),
            },
            "intra_register_distance_structure_fidelity": intra_fidelity,
        }

    scene_scale_24d = _eval_space(X24, "ortho24_cont", mu_aozora_24, thresh_3s_atlas_24, mc_24d_thresh)
    scene_scale_31d = _eval_space(X31, "ortho_v2_31d", mu_aozora_31, thresh_3s_atlas_31, mc_31d_thresh)

    # -------------------------------------------------------------------------
    # 4. R4 Nominal Level Exceedance Quantification
    # -------------------------------------------------------------------------
    # 1) Scale ratio sigma_scene / sigma_work on 23 active continuous dims
    var_scene_r4 = np.var(X24_r4, axis=0, ddof=1)
    std_scene_r4 = np.sqrt(var_scene_r4)
    active_23_idx = mc_quant["active_23_idx"]

    per_dim_scale = {}
    ratios = []
    for idx_i in active_23_idx:
        dim_name = ALL_ORTHO_NAMES[CONT24_DIMS[idx_i]]
        s_scene = float(std_scene_r4[idx_i])
        s_work = float(std_work_24[idx_i])
        r_val = s_scene / max(s_work, 1e-6)
        ratios.append(r_val)
        per_dim_scale[dim_name] = {
            "dim_idx_24d": idx_i,
            "std_scene_r4": round(s_scene, 4),
            "std_work_aozora": round(s_work, 4),
            "ratio_scene_to_work": round(r_val, 4),
        }
    mean_scale_ratio = float(np.mean(ratios))
    median_scale_ratio = float(np.median(ratios))
    mean_var_ratio = float(np.mean(np.array(ratios) ** 2))

    # 2) Finite-sample LOO Student-t inflation & MC quantiles (reusing precomputed mc_quant)
    corr23 = mc_quant["corr23"]
    eigvals23 = mc_quant["eigvals23"]

    # 3) R4 correlation matrix geometry
    off_diag_mask = ~np.eye(23, dtype=bool)
    sum_off_diag_rho2 = float(np.sum(corr23[off_diag_mask] ** 2))
    eff_rank = float((np.sum(eigvals23) ** 2) / np.sum(eigvals23 ** 2))
    top_5_eigs = [round(float(e), 4) for e in eigvals23[:5]]

    # 4) Single-nonzero LOO variance collapse in 31D (R4_16 on v2_final_ぬ)
    r4_sub31 = X31[reg_indices["R4"]]
    others_15 = np.delete(r4_sub31, 15, axis=0)
    mu_l31 = np.mean(others_15, axis=0)
    var_l31_unreg = np.maximum(np.var(others_15, axis=0, ddof=1), 1e-6)
    var_pool31 = np.maximum(np.mean([np.var(X31[reg_indices[r]], axis=0, ddof=1) for r in registers], axis=0), 1e-6)
    var_l31_shrink = np.maximum(0.8 * np.var(others_15, axis=0, ddof=1) + 0.2 * var_pool31, 1e-6)
    var_in31 = np.maximum(np.var(r4_sub31, axis=0, ddof=1), 1e-6)
    mu_in31 = np.mean(r4_sub31, axis=0)

    d_r4_16_unreg = diagonal_mahalanobis(r4_sub31[15], mu_l31, var_l31_unreg)
    d_r4_16_shrink = diagonal_mahalanobis(r4_sub31[15], mu_l31, var_l31_shrink)
    d_r4_16_insample = diagonal_mahalanobis(r4_sub31[15], mu_in31, var_in31)

    r4_nominal_quant = {
        "1_scale_ratio_scene_vs_work_23_active_dims": {
            "active_dims_count": 23,
            "mean_scale_ratio": round(mean_scale_ratio, 4),
            "median_scale_ratio": round(median_scale_ratio, 4),
            "mean_variance_ratio": round(mean_var_ratio, 4),
            "per_dimension_details": per_dim_scale,
            "explanation": (
                "Scene-level variance sigma_scene^2 is on average 2.2645x larger than work-level variance sigma_work^2 "
                "(mean scale ratio 1.4233x, (mean)^2 = 2.0258x) due to short-scene sample variance vs full-book averaging. "
                "B-9 evaluated scenes against work-level denominators, creating an artificial 81.25% (31D) / 62.50% (24D) outlier rate on R4. "
                "Using sigma_scene resolves this scale mismatch."
            ),
        },
        "2_finite_sample_loo_student_t_inflation": {
            "seed": mc_quant["seed"],
            "n_trials": mc_quant["n_trials"],
            "m_loo_observations": mc_quant["m_loo_observations"],
            "nu_degrees_of_freedom": mc_quant["nu_degrees_of_freedom"],
            "theoretical_d2_inflation_factor": mc_quant["theoretical_d2_inflation_factor"],
            "chi2_ref_24d": mc_quant["chi2_ref_24d"],
            "monte_carlo_5000_trials": {
                "independent_gaussian_24d": mc_quant["independent_gaussian_24d"],
                "r4_correlated_gaussian_24d": mc_quant["r4_correlated_gaussian_24d"],
                "independent_gaussian_31d": mc_quant["independent_gaussian_31d"],
            },
            "explanation": (
                "Under LOO at n_r=16 (m=15, nu=14), standardized distance squares follow a scaled Student-t distribution with E[D^2] inflated by 1.2444x. "
                "Consequently, the nominal 5% asymptotic chi2_0.95 threshold (d=6.0345) is exceeded by 22.29% (independent) to 24.51% (R4-correlated) of true null samples. "
                "Finite-sample Monte Carlo quantiles (24D indep: 95%=6.9267, 99%=7.7032; 24D R4-corr: 95%=7.6574, 99%=9.0381) calibrate this threshold."
            ),
        },
        "3_r4_correlation_matrix_geometry_23_active_dims": {
            "n_active_dims": 23,
            "sum_off_diagonal_rho_squared": round(sum_off_diag_rho2, 4),
            "effective_rank_participation_ratio": round(eff_rank, 4),
            "top_5_eigenvalues": top_5_eigs,
            "all_eigenvalues": [round(float(e), 4) for e in eigvals23],
        },
        "4_single_nonzero_loo_variance_collapse_31d": {
            "affected_register": "R4",
            "affected_sample": "R4_16",
            "dimension_name": "v2_final_ぬ",
            "dimension_index_31d": 23,
            "sample_value": round(float(X31[reg_indices["R4"][15], 23]), 6),
            "unregularized_loo_variance": 0.0,
            "unregularized_loo_distance": round(d_r4_16_unreg, 4),
            "shrinkage_loo_distance": round(d_r4_16_shrink, 4),
            "insample_distance": round(d_r4_16_insample, 4),
            "explanation": (
                "When only a single sample in a cohort has a non-zero value on a sparse particle dimension (v2_final_ぬ in R4_16), "
                "omitting it produces zero LOO variance, hitting the 1e-6 floor and exploding distance to 52.79. "
                "Shrinkage LOO (18.38) and in-sample variance (5.32) prevent this collapse."
            ),
        },
    }

    # -------------------------------------------------------------------------
    # 4. Reference Only: Scheme 2 LOO Era Shift Alignment
    # -------------------------------------------------------------------------
    # Explicit disclaimer as required by repository rules
    reference_only_scheme_2 = {
        "disclaimer": (
            "本値は自己レジスター平均を減算した差分を他分布（青空文庫アトラス）の分散で除算した同義反復的補正"
            "（self-register mean subtraction divided by another distribution's sigma）であり、独立した収束証拠とはみなさない参考値である。"
        ),
        "r1_colloquial_fiction_alignment": {
            "baseline_a_r4_diagonal_31d": {
                "unaligned_outlier_rate_3sigma_pct": 0.0,
                "unaligned_outlier_rate_chi2_p95_pct": 62.5,
                "aligned_outlier_rate_3sigma_pct": 0.0,
                "aligned_outlier_rate_chi2_p95_pct": 0.0,
            },
            "baseline_b_aozora_atlas_24d": {
                "unaligned_outlier_rate_3sigma_pct": 100.0,
                "aligned_outlier_rate_3sigma_pct": 0.0,
                "aligned_outlier_rate_chi2_p95_pct": 25.0,
            },
        },
        "pooled_modern_prose_alignment": {
            "baseline_a_r4_diagonal_31d": {
                "unaligned_outlier_rate_3sigma_pct": 58.33,
                "unaligned_outlier_rate_chi2_p95_pct": 87.5,
                "aligned_outlier_rate_3sigma_pct": 0.0,
                "aligned_outlier_rate_chi2_p95_pct": 35.42,
            },
            "baseline_b_aozora_atlas_24d": {
                "unaligned_outlier_rate_3sigma_pct": 100.0,
                "aligned_outlier_rate_3sigma_pct": 72.92,
                "aligned_outlier_rate_chi2_p95_pct": 87.5,
            },
        },
    }

    # -------------------------------------------------------------------------
    # 5. Gate 3a / 3b Decision & Proposal to Orion
    # -------------------------------------------------------------------------
    gate_3a_decision = {
        "status": GATE3A_STATUS,
        "official_reason": (
            "Modern prose across registers (R1-R3) exhibits significant diachronic and genre-specific distributional shift "
            "relative to the pre-war Aozora Bunko baseline. While scene-level within-register scaling stabilizes metrics, "
            "uncentered modern prose continues to fail pre-war acceptance thresholds. Gate 3a remains officially SHIFT_QUANTIFIED_NOT_PASSED."
        ),
        "proposal_to_orion": {
            "summary": "Formal proposal to Orion regarding future Gate 3a criteria update and transition to register-conditioned manifold modeling.",
            "items": [
                "1. Denominator Redesign: Replace work-level denominator sigma_work with scene-level denominator sigma_scene to match sample resolution (400-800 chars) and eliminate false outlier inflation (scale ratio 1.42x / variance 2.03x).",
                "2. Finite-Sample LOO Adjustment: Account for the 1.2444x Student-t inflation at n_r=16 in thresholding; asymptotic chi2_0.95 naturally yields ~22-25% exceedance under finite-sample null.",
                "3. Register-Conditioned Intake: Mandate Scheme 3 (Within-Register LOO Centering with Shrinkage Variance) for text ingestion, which achieves 100% elimination of between-register variance (eta^2 -> 0.000) while preserving internal distance geometry (rho=1.000).",
                "4. Gate 3b Prerequisite: Maintain Gate 3b as GATE3B_FROZEN until automated register classification and conditioned standardization pipelines are validated."
            ],
        },
    }

    return {
        "gate_3a_official_decision": gate_3a_decision,
        "gate_3b_official_decision": {
            "status": GATE3B_STATUS,
            "reason": "Gate 3b remains strictly frozen pending Orion consensus and deployment of register-conditioned normalization pipeline.",
        },
        "task2_fixed_h1_h2_ratings": task2_fixed_ratings,
        "scene_scale_evaluation": {
            "ortho24_cont": scene_scale_24d,
            "ortho_v2_31d": scene_scale_31d,
            "r4_nominal_level_exceedance_quantification": r4_nominal_quant,
        },
        "reference_only_scheme_2": reference_only_scheme_2,
    }


def build_b11_threshold_calibration_comparison(b10_eval: Dict[str, Any]) -> Dict[str, Any]:
    """Build Stage B-11 Task 1 comparison between old chi2 and finite-sample MC thresholds."""
    s24 = b10_eval["scene_scale_evaluation"]["ortho24_cont"]
    mc = b10_eval["scene_scale_evaluation"]["r4_nominal_level_exceedance_quantification"][
        "2_finite_sample_loo_student_t_inflation"
    ]["monte_carlo_5000_trials"]

    def _extract_mode_comparison(cond_dict: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for var_mode in ["in_sample", "shrinkage_loo", "unregularized_loo"]:
            m = cond_dict[var_mode]
            pr = m["per_register"]
            agg = m["modern_aggregate"]

            def _pick(entry: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    "n": entry["n"],
                    "d_m_mean": entry["d_m_mean"],
                    "d_m_max": entry["d_m_max"],
                    "old_chi2_p95_pct": entry["outlier_rate_chi2_p95_pct"],
                    "old_chi2_p99_pct": entry["outlier_rate_chi2_p99_pct"],
                    "old_chi2_p9973_pct": entry["outlier_rate_chi2_p9973_pct"],
                    "new_mc_indep_p95_pct": entry["outlier_rate_mc_indep_p95_pct"],
                    "new_mc_indep_p99_pct": entry["outlier_rate_mc_indep_p99_pct"],
                    "new_mc_r4corr_p95_pct": entry["outlier_rate_mc_r4corr_p95_pct"],
                    "new_mc_r4corr_p99_pct": entry["outlier_rate_mc_r4corr_p99_pct"],
                }

            out[var_mode] = {
                "R4": _pick(pr["R4"]),
                "R1": _pick(pr["R1"]),
                "R2": _pick(pr["R2"]),
                "R3": _pick(pr["R3"]),
                "R1_to_R3_pooled": _pick(agg),
            }
        return out

    return {
        "canonical_space": "ortho24_cont",
        "dimension": 24,
        "canonical_denominator": "diagonal_sigma2_scene_r",
        "gate_3a_status": GATE3A_STATUS,
        "gate_3b_status": GATE3B_STATUS,
        "threshold_calibration_note": (
            "Threshold update from asymptotic chi2_0.95 to finite-sample Monte Carlo null quantiles (95%/99%) "
            "is strictly a calibration of the judgment criterion for finite-sample LOO inflation (1.2444x at n_r=16) "
            "and R4 correlation; it is NOT a change to Gate 3a's official decision (SHIFT_QUANTIFIED_NOT_PASSED)."
        ),
        "scheme2_governance_note": b10_eval["reference_only_scheme_2"]["disclaimer"],
        "scheme3_governance_note": (
            "Condition (b) Scheme 3 within-register LOO convergence is an intake-normalization diagnostic only "
            "and is NOT counted as Gate 3a pass evidence."
        ),
        "primary_alarm_threshold_key": PRIMARY_ALARM_THRESHOLD_KEY,
        "primary_alarm_threshold_d": PRIMARY_ALARM_THRESHOLD_D,
        "auxiliary_alarm_threshold_key": AUXILIARY_ALARM_THRESHOLD_KEY,
        "auxiliary_alarm_threshold_d": AUXILIARY_ALARM_THRESHOLD_D,
        "alarm_threshold_specification": get_alarm_threshold_specification(),
        "thresholds": s24["thresholds"],
        "monte_carlo_quantiles": {
            "seed": 42,
            "n_trials": 5000,
            "n_loo_evaluations_per_null": 80000,
            "independent_gaussian_24d": mc["independent_gaussian_24d"],
            "r4_correlated_gaussian_24d": mc["r4_correlated_gaussian_24d"],
        },
        "condition_a_uncentered_scene_scaled": _extract_mode_comparison(
            s24["condition_a_uncentered_scene_scaled"]
        ),
        "condition_b_scheme3_loo_centered_diagnostic_only": _extract_mode_comparison(
            s24["condition_b_within_register_loo_centered_scene_scaled"]
        ),
    }


def build_b12_gate3a_summary(b10_eval: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build Stage B-12 Gate 3a output JSON with explicit primary/auxiliary alarm thresholds."""
    if b10_eval is None:
        records_64 = load_benchmark_records(BENCHMARK_64_PATH)
        atlas = extract_aozora_106_atlas()
        b10_eval = evaluate_b10_scene_scale_redesign(records_64, atlas, TASK1_STAGE_JSON_PATH)
    calib = build_b11_threshold_calibration_comparison(b10_eval)
    return {
        "metadata": {
            "title": "USIR v5 Stage B-12: Gate 3a Summary with Explicit Primary & Auxiliary Alarm Thresholds",
            "version": "1.2",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gate_3a_status": GATE3A_STATUS,
            "gate_3b_status": GATE3B_STATUS,
            "primary_alarm_threshold_key": PRIMARY_ALARM_THRESHOLD_KEY,
            "primary_alarm_threshold_d": PRIMARY_ALARM_THRESHOLD_D,
            "auxiliary_alarm_threshold_key": AUXILIARY_ALARM_THRESHOLD_KEY,
            "auxiliary_alarm_threshold_d": AUXILIARY_ALARM_THRESHOLD_D,
        },
        "gate_3a_status": GATE3A_STATUS,
        "gate_3b_status": GATE3B_STATUS,
        "primary_alarm_threshold_key": PRIMARY_ALARM_THRESHOLD_KEY,
        "primary_alarm_threshold_d": PRIMARY_ALARM_THRESHOLD_D,
        "auxiliary_alarm_threshold_key": AUXILIARY_ALARM_THRESHOLD_KEY,
        "auxiliary_alarm_threshold_d": AUXILIARY_ALARM_THRESHOLD_D,
        "alarm_threshold_specification": get_alarm_threshold_specification(),
        "gate3a_calibrated_evaluation": calib,
        "reference_only_scheme_2": b10_eval["reference_only_scheme_2"],
    }


def main(overwrite_frozen: bool = False) -> None:
    t0 = time.time()
    print("=== USIR v5 Stage B-12: Gate 3a Register Centering & Canonical Evaluation ===")

    aozora_atlas = extract_aozora_106_atlas()
    assert BENCHMARK_32_PATH.exists(), f"Missing {BENCHMARK_32_PATH}"
    assert BENCHMARK_64_PATH.exists(), f"Missing {BENCHMARK_64_PATH}"
    records_64 = load_benchmark_records(BENCHMARK_64_PATH)

    b10_eval = evaluate_b10_scene_scale_redesign(records_64, aozora_atlas, TASK1_STAGE_JSON_PATH)
    b12_summary = build_b12_gate3a_summary(b10_eval)

    OUTPUT_B12_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_B12_SUMMARY_PATH.write_text(json.dumps(b12_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote Stage B-12 Gate 3a summary JSON: {OUTPUT_B12_SUMMARY_PATH}")

    from scale_separated_level3 import (
        evaluate_f14_hybrid_alignment_and_parity,
        verify_kasane8_index_unification_and_recalc_h1_h2,
    )

    f14_parity = evaluate_f14_hybrid_alignment_and_parity()
    kasane8_recalc = verify_kasane8_index_unification_and_recalc_h1_h2()

    b12_canonical = {
        "metadata": {
            "title": "USIR v5 Stage B-12: Canonical Dice Gate, Explicit Alarm Thresholds, & KASANE8 H1/H2 Re-calculation",
            "version": "1.2",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "base_commit": "ae992e5",
            "gate_3a_status": GATE3A_STATUS,
            "gate_3b_status": GATE3B_STATUS,
            "frozen_artifacts_untouched": not overwrite_frozen,
        },
        "task1_dice_gate_canonical_parity": f14_parity,
        "task2_gate3a_explicit_alarm_thresholds": b12_summary,
        "task3_kasane8_unification_and_h1_h2_recalc": kasane8_recalc,
    }
    OUTPUT_B12_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_B12_JSON_PATH.write_text(json.dumps(b12_canonical, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote Stage B-12 Canonical JSON: {OUTPUT_B12_JSON_PATH} ({round(time.time() - t0, 2)}s)")

    if overwrite_frozen:
        records_32 = load_benchmark_records(BENCHMARK_32_PATH)
        eval_32 = evaluate_dataset(records_32, aozora_atlas, "tier_c_modern_benchmark_32")
        eval_64 = evaluate_dataset(records_64, aozora_atlas, "tier_c_modern_benchmark_expanded_64")
        print(f"Frozen artifacts overwrite skipped unless needed ({len(eval_32['dataset_name'])}, {len(eval_64['dataset_name'])})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gate 3a Register Centering & Stage B-12 Canonical Evaluation")
    parser.add_argument("--overwrite-frozen", action="store_true", help="Do not set; frozen B-9..B-11 artifacts must stay unchanged")
    args = parser.parse_args()
    main(overwrite_frozen=args.overwrite_frozen)
