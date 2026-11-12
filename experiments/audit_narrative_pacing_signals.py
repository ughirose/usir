"""Signal Confirmation Audit for HW-22, HW-24, and HW-25 Prototype Metrics on Aozora 106 Works (11,723 windows).

Evaluates:
  1. Distribution (mean, std, median, Q1, Q3, IQR, min, max) across 11,723 windows.
  2. One-Way ANOVA: Between-author variance vs within-author window noise (F-statistic, p-value, eta^2).
  3. Objective ruthlessly honest signal strength classification (Weak / Moderate / Strong).
  4. Note that all constants are uncalibrated initial values.

Output: docs/handoff/v10/narrative_pacing_prototype_results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy import stats

USIR = Path("c:/dev/usir")
for p in ("python", "scripts", "experiments"):
    sys.path.insert(0, str(USIR / p))

from fable_v5_grand_shootout import load_works  # noqa: E402
from usir_py import style_atlas as sa  # noqa: E402
from usir_py.v5.narrative_pacing_and_metaphor import (  # noqa: E402
    extract_all_narrative_pacing,
)

OUT_DIR = USIR / "docs" / "handoff" / "v10"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "narrative_pacing_prototype_results.json"


def main() -> None:
    t0 = time.time()
    works = load_works()
    print(f"Loaded {len(works)} works")

    # Group windows by author
    author_windows: Dict[str, List[str]] = {}
    all_windows = []
    window_author_ids = []

    for w in works:
        aid = w["author_id"]
        if aid not in author_windows:
            author_windows[aid] = []
        wins = sa.make_windows(w["text"])
        author_windows[aid].extend(wins)
        all_windows.extend(wins)
        window_author_ids.extend([aid] * len(wins))

    n_windows = len(all_windows)
    aids = sorted(list(author_windows.keys()))
    print(f"Total windows: {n_windows}, Authors: {len(aids)}")
    assert n_windows == 11723

    # Extract metrics for all windows
    t_ext0 = time.time()
    print("Extracting HW-22, HW-24, HW-25 across all 11,723 windows...")
    hw22_vals = []
    hw24_vals = []
    hw25_vals = []

    for i, win in enumerate(all_windows):
        if (i + 1) % 3000 == 0:
            print(f"  Processed {i + 1}/{n_windows} windows...")
        m = extract_all_narrative_pacing(win)
        hw22_vals.append(m["hw22_metaphor_leap"])
        hw24_vals.append(m["hw24_cliffhanger"])
        hw25_vals.append(m["hw25_uid_ratio"])

    hw22_arr = np.array(hw22_vals, dtype=float)
    hw24_arr = np.array(hw24_vals, dtype=float)
    hw25_arr = np.array(hw25_vals, dtype=float)
    t_ext = time.time() - t_ext0
    print(f"Extracted all metrics in {t_ext:.2f}s")

    metrics_dict = {
        "HW-22 (D_leap, 比喩跳躍距離)": hw22_arr,
        "HW-24 (C_cliff, 章末引き強度)": hw24_arr,
        "HW-25 (Gamma_UID, 設定開示結合比)": hw25_arr,
    }

    results = {}
    print("\n--- Statistical & Signal Analysis ---")
    for name, arr in metrics_dict.items():
        q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        iqr = q3 - q1

        # Per-author values for ANOVA
        author_groups = [arr[np.array(window_author_ids) == a] for a in aids]
        f_stat, p_val = stats.f_oneway(*author_groups)

        # Eta-squared effect size
        grand_mean = np.mean(arr)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in author_groups)
        ss_total = np.sum((arr - grand_mean) ** 2)
        eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0

        # Signal classification
        if eta_sq < 0.06:
            signal_verdict = "WEAK_SIGNAL (窓内ノイズに埋没。作家識別への寄与は極めて微弱)"
        elif eta_sq < 0.14:
            signal_verdict = "MODERATE_SIGNAL (中程度の作家間差が存在)"
        else:
            signal_verdict = "STRONG_SIGNAL (顕著な作家間差が存在)"

        per_author_stats = {}
        for a, g in zip(aids, author_groups):
            ag_q1, ag_q3 = float(np.percentile(g, 25)), float(np.percentile(g, 75))
            per_author_stats[a] = {
                "n_windows": len(g),
                "mean": round(float(np.mean(g)), 4),
                "median": round(float(np.median(g)), 4),
                "iqr": round(float(ag_q3 - ag_q1), 4),
                "std": round(float(np.std(g)), 4),
            }

        results[name] = {
            "uncalibrated_constants_note": "Constants (kappa=1.35, gamma=1.5, epsilon=0.1) are uncalibrated initial values.",
            "overall_distribution": {
                "mean": round(float(np.mean(arr)), 4),
                "std": round(float(np.std(arr)), 4),
                "median": round(float(np.median(arr)), 4),
                "q1": round(q1, 4),
                "q3": round(q3, 4),
                "iqr": round(float(iqr), 4),
                "min": round(float(np.min(arr)), 4),
                "max": round(float(np.max(arr)), 4),
            },
            "anova": {
                "f_statistic": round(float(f_stat), 4),
                "p_value": float(p_val),
                "eta_squared": round(eta_sq, 4),
                "df_between": len(aids) - 1,
                "df_within": n_windows - len(aids),
            },
            "signal_assessment": signal_verdict,
            "per_author": per_author_stats,
        }

        print(f"{name}:")
        print(f"  Distribution: Median={np.median(arr):.4f}, IQR={iqr:.4f}, Mean={np.mean(arr):.4f} +/- {np.std(arr):.4f}")
        print(f"  ANOVA: F={f_stat:.2f}, p={p_val:.4e}, eta^2={eta_sq:.4f} -> {signal_verdict}")

    summary = {
        "status": "PASS",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_works": len(works),
        "n_windows": n_windows,
        "n_authors": len(aids),
        "metrics": results,
        "runtime_sec": round(time.time() - t0, 2),
    }

    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote narrative pacing results to {OUT_JSON}")


if __name__ == "__main__":
    main()
