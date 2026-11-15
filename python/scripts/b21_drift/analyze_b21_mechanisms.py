"""Deep-dive mechanism analysis for Stage B-21 Tier 2 length drift."""

from __future__ import annotations

import json
import math
import pathlib
import statistics
import sys

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import tier2_rg as t2  # noqa: E402
from run_b21_drift import METRICS, load_corpus  # noqa: E402


def main() -> None:
    d = json.loads((SCRIPT_DIR / "b21_results.json").read_text(encoding="utf-8"))

    print("=== 1. PER-WORK TRUNCATION TABLE (10 TARGETS) ===")
    for w in d["target_works"]:
        print(f"\n[{w['work_id']}] author={w['author_id']} scenes={w['n_scenes']} sents={w['n_sentences']} chars={w['chars']}")
        for r in w["rows"]:
            print(
                f"  {r['label']:>5s} (sents={r['n_sentences']:4d}, blk={r['n_blocks']:3d}): "
                f"hvg_deg={r['hvg_mean_degree']:.4f} | rqa_det={r['rqa_det']:.4f} | "
                f"hvg_kl={r['hvg_kl']:.4f} | hvg_ass={r['hvg_assort']:+.4f} | "
                f"rqa_lam={r['rqa_lam']:.4f} | rqa_tt={r['rqa_tt']:.4f}"
            )

    print("\n=== 2. WITHIN-WORK MONOTONICITY (DIRECTION OF FULL - 128) ===")
    for m in ["hvg_mean_degree", "rqa_det", "hvg_kl", "hvg_assort", "rqa_lam", "rqa_tt"]:
        vals = d["drift_summary"][m]["delta_full_minus_128_values"]
        pos = sum(1 for v in vals if v > 0)
        neg = sum(1 for v in vals if v < 0)
        print(f"  {m:16s}: pos={pos}/10, neg={neg}/10, min={min(vals):+.4f}, median={statistics.median(vals):+.4f}, max={max(vals):+.4f}")

    print("\n=== 3. DECOMPOSING log(chars) = log(n_sentences) + log(mean_sent_len) (n=105) ===")
    works = load_corpus()
    full_m = [t2.compute_tier2_all(w["feats"]) for w in works]
    log_chars = [w["log_chars"] for w in works]
    log_sents = [math.log(w["n_sentences"]) for w in works]
    log_msl = [math.log(w["chars"] / w["n_sentences"]) for w in works]
    print(f"  Spearman(log_sents, log_msl) = {t2.spearman(log_sents, log_msl):+.4f}")
    for m in METRICS:
        col = [r[m] for r in full_m]
        print(
            f"  {m:16s}: vs log_chars={t2.spearman(col, log_chars):+.4f} | "
            f"vs log_sents={t2.spearman(col, log_sents):+.4f} | "
            f"vs log_mean_sent_len={t2.spearman(col, log_msl):+.4f}"
        )

    print("\n=== 4. SENSITIVITY: RQA DET WITH FULL-WORK Z-SCORE VS LOCAL SLICE Z-SCORE (10 TARGETS) ===")
    targets = sorted(works, key=lambda w: (w["n_scenes"], w["n_sentences"]), reverse=True)[:10]
    deltas_local = []
    deltas_global_z = []
    for w in targets:
        full_cloud = t2.zscore_cloud(w["feats"])
        # local zscore at 128 vs full
        m_128_local = t2.compute_tier2_all(w["feats"][:128])["rqa_det"]
        m_full = t2.compute_tier2_all(w["feats"])["rqa_det"]
        # global zscore at 128
        b_128 = [
            t2.recurrence(full_cloud[start : start + 64])["det"]
            for start in range(0, 128 - 64 + 1, 32)
        ]
        m_128_global = t2.quantile_sorted(sorted(b_128), 0.5)
        deltas_local.append(m_full - m_128_local)
        deltas_global_z.append(m_full - m_128_global)
    print(f"  Local z-score  delta(full-128): mean={statistics.mean(deltas_local):+.4f}, sd={statistics.stdev(deltas_local):.4f}")
    print(f"  Global z-score delta(full-128): mean={statistics.mean(deltas_global_z):+.4f}, sd={statistics.stdev(deltas_global_z):.4f}")

    print("\n=== 5. MIDDLE WINDOW (512 SENTS) VS FULL WORK (10 TARGETS) ===")
    # Check if opening 512 differs from middle 512
    for m in ["hvg_mean_degree", "rqa_det"]:
        diff_prefix = []
        diff_middle = []
        for w in targets:
            n = w["n_sentences"]
            mid_start = (n - 512) // 2
            full_v = t2.compute_tier2_all(w["feats"])[m]
            pref_v = t2.compute_tier2_all(w["feats"][:512])[m]
            mid_v = t2.compute_tier2_all(w["feats"][mid_start : mid_start + 512])[m]
            diff_prefix.append(full_v - pref_v)
            diff_middle.append(full_v - mid_v)
        print(
            f"  {m:16s}: full-prefix512={statistics.mean(diff_prefix):+.4f}±{statistics.stdev(diff_prefix):.4f} | "
            f"full-middle512={statistics.mean(diff_middle):+.4f}±{statistics.stdev(diff_middle):.4f}"
        )


if __name__ == "__main__":
    main()
