"""Stage B-21: Tier 2 Length Drift Experiment (Truncation Analysis).

Evaluates whether the positive Spearman correlations with log(char_count)
observed in F6/B-19 (n=105: hvg_mean_degree rho=0.1467, rqa_det rho=0.1698)
are driven by mechanical length dependence (within-work drift) or intrinsic
between-work stylistic differences.
"""

from __future__ import annotations

import json
import math
import pathlib
import statistics
import sys
from collections import defaultdict

import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import tier2_rg as t2  # noqa: E402

REPO_ROOT = SCRIPT_DIR.parents[2]
SCENES_PATH = REPO_ROOT / "data" / "v5" / "aozora_106works_scenes.jsonl"
CACHE_DIR = REPO_ROOT / "data" / "aozora_cache"
OUT_JSON = SCRIPT_DIR / "b21_results.json"

METRICS = [
    "hvg_kl",
    "hvg_assort",
    "hvg_mean_degree",
    "rqa_det",
    "rqa_lam",
    "rqa_tt",
]

TRUNC_STEPS = [128, 256, 512, 1024]


def load_corpus() -> list[dict]:
    scenes = [json.loads(line) for line in SCENES_PATH.open(encoding="utf-8")]
    by_work: dict[tuple[str, str], list[dict]] = defaultdict(list)
    order: list[tuple[str, str]] = []
    for s in scenes:
        key = (s["author_id"], s["work_id"])
        if key not in by_work:
            order.append(key)
        by_work[key].append(s)

    cache_files = list(CACHE_DIR.glob("*.txt"))
    works: list[dict] = []
    for aid, wid in order:
        suf = f"_{aid}_{wid}.txt"
        matches = [p for p in cache_files if p.name.endswith(suf)]
        if not matches:
            continue
        text = matches[0].read_text(encoding="utf-8")
        feats, sents = t2.sentences(text)
        if len(feats) < t2.BLOCK_LEN:
            continue
        chars = t2.char_count_non_ws(text)
        works.append(
            {
                "author_id": aid,
                "work_id": wid,
                "file": matches[0].name,
                "n_scenes": len(by_work[(aid, wid)]),
                "n_sentences": len(feats),
                "chars": chars,
                "log_chars": math.log(chars),
                "feats": feats,
                "sents": sents,
            }
        )
    return works


def ols_slope(x: list[float], y: list[float]) -> float:
    xm = statistics.mean(x)
    ym = statistics.mean(y)
    num = sum((xi - xm) * (yi - ym) for xi, yi in zip(x, y))
    den = sum((xi - xm) ** 2 for xi in x)
    return num / den if den > 1e-12 else 0.0


def main() -> None:
    works = load_corpus()
    print(f"Loaded {len(works)} eligible works (>= 64 sentences).")

    # 1. Full corpus (n=105) baseline Tier 2 metrics
    full_corpus_metrics: list[dict] = []
    for w in works:
        m = t2.compute_tier2_all(w["feats"])
        m["author_id"] = w["author_id"]
        m["work_id"] = w["work_id"]
        m["n_scenes"] = w["n_scenes"]
        m["chars"] = w["chars"]
        m["log_chars"] = w["log_chars"]
        full_corpus_metrics.append(m)

    log_chars_all = [m["log_chars"] for m in full_corpus_metrics]
    baseline_rho = {
        k: t2.spearman([m[k] for m in full_corpus_metrics], log_chars_all)
        for k in METRICS
    }
    print("Baseline Spearman vs log(chars) (n=105):", {k: round(v, 4) for k, v in baseline_rho.items()})

    # Between-work stats on full corpus (n=105)
    between_full_105 = {}
    for k in METRICS:
        vals = [m[k] for m in full_corpus_metrics]
        between_full_105[k] = {
            "mean": statistics.mean(vals),
            "sd": statistics.stdev(vals),
            "var": statistics.variance(vals),
            "min": min(vals),
            "max": max(vals),
        }

    # 2. Target selection: Top 10 works by scene count (all have n_scenes >= 126, n_sentences >= 2557)
    sorted_by_scenes = sorted(works, key=lambda w: (w["n_scenes"], w["n_sentences"]), reverse=True)
    targets = sorted_by_scenes[:10]
    print("\nSelected 10 target works:")
    for w in targets:
        print(f"  {w['work_id']} (author {w['author_id']}): scenes={w['n_scenes']}, sents={w['n_sentences']}, chars={w['chars']}")

    # 3. Truncation experiment on the 10 target works
    trunc_results: list[dict] = []
    for w in targets:
        steps: list[tuple[str, int]] = [(str(k), k) for k in TRUNC_STEPS if k < w["n_sentences"]]
        steps.append(("full", w["n_sentences"]))
        rows = []
        for label, k in steps:
            sub_feats = w["feats"][:k]
            sub_sents = w["sents"][:k]
            sub_chars = sum(len(s) for s in sub_sents)
            m = t2.compute_tier2_all(sub_feats)
            m["label"] = label
            m["k_sents"] = k
            m["chars"] = sub_chars
            m["log_chars"] = math.log(sub_chars)
            rows.append(m)
        trunc_results.append(
            {
                "author_id": w["author_id"],
                "work_id": w["work_id"],
                "file": w["file"],
                "n_scenes": w["n_scenes"],
                "n_sentences": w["n_sentences"],
                "chars": w["chars"],
                "rows": rows,
            }
        )

    # 4. Within-work drift vs between-work spread (on the 10 targets)
    drift_summary = {}
    for k in METRICS:
        deltas_full_128 = []
        deltas_512_128 = []
        slopes_log_chars = []
        within_vars = []
        full_vals_10 = []
        val128_10 = []
        for tr in trunc_results:
            r_map = {r["label"]: r for r in tr["rows"]}
            v128 = r_map["128"][k]
            v512 = r_map["512"][k]
            vfull = r_map["full"][k]
            deltas_full_128.append(vfull - v128)
            deltas_512_128.append(v512 - v128)
            full_vals_10.append(vfull)
            val128_10.append(v128)
            xs = [r["log_chars"] for r in tr["rows"]]
            ys = [r[k] for r in tr["rows"]]
            slopes_log_chars.append(ols_slope(xs, ys))
            within_vars.append(statistics.variance(ys))

        between_var_10 = statistics.variance(full_vals_10)
        between_sd_10 = statistics.stdev(full_vals_10)
        mean_within_var = statistics.mean(within_vars)
        mean_within_sd = math.sqrt(mean_within_var)
        # Expected shift over corpus log(chars) IQR / SD
        logc_sd_105 = statistics.stdev(log_chars_all)
        mean_slope = statistics.mean(slopes_log_chars)
        expected_corr_from_slope = mean_slope * logc_sd_105 / between_full_105[k]["sd"]

        drift_summary[k] = {
            "delta_full_minus_128_mean": statistics.mean(deltas_full_128),
            "delta_full_minus_128_sd": statistics.stdev(deltas_full_128),
            "delta_full_minus_128_values": deltas_full_128,
            "delta_512_minus_128_mean": statistics.mean(deltas_512_128),
            "slope_per_log_char_mean": mean_slope,
            "slope_per_log_char_sd": statistics.stdev(slopes_log_chars),
            "within_work_sd_rms": mean_within_sd,
            "within_work_var_mean": mean_within_var,
            "between_work_sd_10": between_sd_10,
            "between_work_var_10": between_var_10,
            "between_work_sd_105": between_full_105[k]["sd"],
            "between_work_var_105": between_full_105[k]["var"],
            "variance_ratio_within_over_between105": mean_within_var / between_full_105[k]["var"],
            "predicted_pearson_from_within_slope": expected_corr_from_slope,
        }

    # 5. Cross-work correlation re-evaluation under truncation (n=105)
    # (A) Cap all 105 works at K sentences: min(n_sents, cap)
    cap_correlations = {}
    for cap in [128, 256, 512]:
        vals_by_m: dict[str, list[float]] = {k: [] for k in METRICS}
        trunc_logc = []
        for w in works:
            k_cut = min(w["n_sentences"], cap)
            m = t2.compute_tier2_all(w["feats"][:k_cut])
            sub_chars = sum(len(s) for s in w["sents"][:k_cut])
            trunc_logc.append(math.log(sub_chars))
            for k in METRICS:
                vals_by_m[k].append(m[k])
        cap_correlations[str(cap)] = {
            "n": len(works),
            "vs_full_log_chars": {k: t2.spearman(vals_by_m[k], log_chars_all) for k in METRICS},
            "vs_trunc_log_chars": {k: t2.spearman(vals_by_m[k], trunc_logc) for k in METRICS},
        }

    # (B) Uniform-length subset: works with n_sentences >= K, all truncated to exact K sentences
    uniform_correlations = {}
    for k_unif in [128, 256, 512]:
        sub_works = [w for w in works if w["n_sentences"] >= k_unif]
        sub_full_logc = [w["log_chars"] for w in sub_works]
        vals_by_m = {k: [] for k in METRICS}
        vals_full_by_m = {k: [] for k in METRICS}
        # Map work_id to full metric
        full_map = {(m["author_id"], m["work_id"]): m for m in full_corpus_metrics}
        for w in sub_works:
            m = t2.compute_tier2_all(w["feats"][:k_unif])
            mf = full_map[(w["author_id"], w["work_id"])]
            for k in METRICS:
                vals_by_m[k].append(m[k])
                vals_full_by_m[k].append(mf[k])
        uniform_correlations[str(k_unif)] = {
            "n": len(sub_works),
            "truncated_vs_full_log_chars": {k: t2.spearman(vals_by_m[k], sub_full_logc) for k in METRICS},
            "full_vs_full_log_chars_same_subset": {k: t2.spearman(vals_full_by_m[k], sub_full_logc) for k in METRICS},
        }

    out = {
        "n_corpus": len(works),
        "baseline_spearman_vs_log_chars_n105": baseline_rho,
        "between_work_stats_n105": between_full_105,
        "target_works": trunc_results,
        "drift_summary": drift_summary,
        "cap_correlations": cap_correlations,
        "uniform_length_correlations": uniform_correlations,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved results to {OUT_JSON}")

    print("\n--- Drift Summary (10 targets) ---")
    for k in ["hvg_mean_degree", "rqa_det", "hvg_kl", "hvg_assort", "rqa_lam", "rqa_tt"]:
        ds = drift_summary[k]
        print(
            f"{k:16s} | baseline_rho={baseline_rho[k]:+.4f} | "
            f"delta(full-128)={ds['delta_full_minus_128_mean']:+.4f}±{ds['delta_full_minus_128_sd']:.4f} | "
            f"slope/logC={ds['slope_per_log_char_mean']:+.4f} | "
            f"within_SD={ds['within_work_sd_rms']:.4f} vs between_SD(105)={ds['between_work_sd_105']:.4f} | "
            f"pred_r={ds['predicted_pearson_from_within_slope']:+.4f}"
        )

    print("\n--- Cap Correlations (n=105) ---")
    for cap, obj in cap_correlations.items():
        print(f"cap={cap} vs full_log_chars:  hvg_mean_degree={obj['vs_full_log_chars']['hvg_mean_degree']:+.4f}, rqa_det={obj['vs_full_log_chars']['rqa_det']:+.4f}")
        print(f"cap={cap} vs trunc_log_chars: hvg_mean_degree={obj['vs_trunc_log_chars']['hvg_mean_degree']:+.4f}, rqa_det={obj['vs_trunc_log_chars']['rqa_det']:+.4f}")

    print("\n--- Uniform-Length Subset Correlations ---")
    for k_unif, obj in uniform_correlations.items():
        print(
            f"exact K={k_unif} (n={obj['n']}): "
            f"trunc vs full_logC: hvg_deg={obj['truncated_vs_full_log_chars']['hvg_mean_degree']:+.4f}, rqa_det={obj['truncated_vs_full_log_chars']['rqa_det']:+.4f} | "
            f"full vs full_logC: hvg_deg={obj['full_vs_full_log_chars_same_subset']['hvg_mean_degree']:+.4f}, rqa_det={obj['full_vs_full_log_chars_same_subset']['rqa_det']:+.4f}"
        )


if __name__ == "__main__":
    main()
