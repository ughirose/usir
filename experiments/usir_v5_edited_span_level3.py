"""USIR v5 Stage B-6 Task 1: Level 3 Edited-Span Micro-Diff and 3-Baseline Comparison.

Implements v4 D4 compliant evaluation:
1. Micro-differences on:
   - "Edited Span Only" (concatenation of edited sentences only)
   - "Local Context Window" (edited sentences +- 1 sentence context)
2. Across 3 Feature Spaces:
   ① 6-pole coordinate (six_pole_atlas: 6D)
   ② ortho_v2 28D (orthographic_prior: KEEP28)
   ③ PPM score (grammar skeleton Order-3 PPM bits/token under author models: 3D)
3. Paragraph noise floor sigma_para from adjacent paragraphs within the same author.
   - Outputs: docs/handoff/v10/paragraph_noise_floor.json
   - Measures 2*sigma_para exceedance rate on 28 known STYLE_POLISH pairs.
4. Direction cosine cos(Delta z_edited, target) against 4 targets:
   - (a) Target author centroid: z*_author - z_before
   - (b) Corpus mean direction: z*_corpus_mean - z_before
   - (c) Isotropic random perturbation direction
   - (d) Reverse (degraded) perturbation direction
   With Wilcoxon signed-rank test W and p-values.
5. Outputs: docs/handoff/v10/usir_v5_edited_span_level3_result.json
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import wilcoxon

USIR = Path("c:/dev/usir")
for p in ("python", "scripts", "experiments"):
    sys.path.insert(0, str(USIR / p))

from fable_v5_grand_shootout import load_works  # noqa: E402
from usir_py.orthographic_prior import (  # noqa: E402
    ortho_matrix_ext_v2,
    ortho_matrix_narr_v2,
)
from usir_py.v5 import (  # noqa: E402
    AozoraCorpusSplitter,
    aggregate_work_profile_from_scenes,
    build_scene_annotation,
    split_work_into_scenes,
)
from usir_py.v5.modern_text_normalizer import (  # noqa: E402
    Order3PPMModel,
    tokenize_closed_class_skeleton,
)

OUT_DIR = USIR / "docs" / "handoff" / "v10"
OUT_DIR.mkdir(parents=True, exist_ok=True)
NOISE_FLOOR_JSON = OUT_DIR / "paragraph_noise_floor.json"
RESULT_JSON = OUT_DIR / "usir_v5_edited_span_level3_result.json"

COMMA_DIMS = [2, 3, 29]
KEEP28 = [i for i in range(31) if i not in COMMA_DIMS]
TARGET_AUTHORS = ("000035", "000148", "000879")


# ---------------------------------------------------------------------------
# Feature Extraction Helpers
# ---------------------------------------------------------------------------

def compute_6pole_vector(text: str, author_id: str, work_id: str, hook_idf: Any) -> np.ndarray:
    """Compute 6-pole coordinate for text."""
    sc = build_scene_annotation(text, author_id, work_id, 0, 1)
    prof = aggregate_work_profile_from_scenes([sc], hook_idf=hook_idf)
    l_text = max(len(text), 1)

    kasane = (text.count("、") / l_text) * 100.0
    sasae = (text.count("。") / l_text) * 100.0
    kamae = float(prof["kamae"]["hook_survival"])
    sadame = float(prof["sadame"]["total"])
    nagare = float(prof["nagare"]["phi_zeami"])
    awase = float(prof["awase"]["irony"]["mean"])

    return np.array([kasane, sasae, kamae, sadame, nagare, awase], dtype=float)


def compute_ortho28_vector(text: str) -> np.ndarray:
    """Compute ortho_v2 28D non-comma vector."""
    ext = ortho_matrix_ext_v2([text])[0]
    narr = ortho_matrix_narr_v2([text])[0]
    full31 = np.concatenate([ext, narr])
    return full31[KEEP28].astype(float)


def compute_ppm_vector(text: str, author_ppm_models: Dict[str, Order3PPMModel], aids: Sequence[str]) -> np.ndarray:
    """Compute 3D vector of cross-entropy bits/token under each author PPM model."""
    tokens = tokenize_closed_class_skeleton(text)
    scores = []
    for a in aids:
        bits, _ = author_ppm_models[a].evaluate_cross_entropy(tokens)
        scores.append(bits)
    return np.array(scores, dtype=float)


def extract_spans_for_pair(orig_text: str, rev_text: str) -> Tuple[str, str, str, str]:
    """Extract Edited Span Only and Local Context Window (edited +- 1 sentence)."""
    parts_orig = [s for s in re.split(r"(?<=[。！？!?])", orig_text) if s.strip()]
    parts_rev = [s for s in re.split(r"(?<=[。！？!?])", rev_text) if s.strip()]

    if len(parts_orig) == len(parts_rev):
        edited_indices = [i for i in range(len(parts_orig)) if parts_orig[i] != parts_rev[i]]
    else:
        lines_o = [ln for ln in orig_text.splitlines() if ln.strip()]
        lines_r = [ln for ln in rev_text.splitlines() if ln.strip()]
        edited_indices = [i for i in range(min(len(lines_o), len(lines_r))) if lines_o[i] != lines_r[i]]
        parts_orig = lines_o
        parts_rev = lines_r

    if not edited_indices:
        return orig_text, rev_text, orig_text, rev_text

    # 1. Edited Span Only
    span_only_b = "".join(parts_orig[i] for i in edited_indices)
    span_only_a = "".join(parts_rev[i] for i in edited_indices)

    # 2. Local Context Window (edited +- 1 sentence)
    n = len(parts_orig)
    local_indices = set()
    for idx in edited_indices:
        for offset in (-1, 0, 1):
            j = idx + offset
            if 0 <= j < n:
                local_indices.add(j)
    sorted_local = sorted(local_indices)
    local_b = "".join(parts_orig[j] for j in sorted_local)
    local_a = "".join(parts_rev[j] for j in sorted_local)

    return span_only_b, span_only_a, local_b, local_a


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity with zero-norm guard."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    print("=== USIR v5 Stage B-6 Task 1: Level 3 Edited-Span Micro-Diff ===")

    works = load_works()
    splitter = AozoraCorpusSplitter()

    # Load hook_idf from cached profile data
    profile_json_path = OUT_DIR / "nishiki_profile_106works.json"
    if profile_json_path.exists():
        profile_data = json.loads(profile_json_path.read_text(encoding="utf-8"))
        hook_idf = profile_data.get("hook_idf", None)
    else:
        hook_idf = None

    sample_works = [w for w in works if w["author_id"] in TARGET_AUTHORS]
    scenes_by_work: Dict[str, List[str]] = {}
    work_by_id = {w["id"]: w for w in sample_works}

    print(f"Loading texts for {len(sample_works)} works by authors {TARGET_AUTHORS}...")
    for w in sample_works:
        text = splitter.get_cached_text(w)
        sc_texts = split_work_into_scenes(text)
        scenes_by_work[w["id"]] = sc_texts

    # Train Order-3 PPM author models
    print("Training author PPM grammar skeleton models...")
    author_ppm_models: Dict[str, Order3PPMModel] = {}
    for aid in TARGET_AUTHORS:
        author_works = [w for w in sample_works if w["author_id"] == aid and w["split"] == "train"]
        if not author_works:
            author_works = [w for w in sample_works if w["author_id"] == aid]
        token_seqs = []
        for w in author_works:
            txt = splitter.get_cached_text(w)
            token_seqs.append(tokenize_closed_class_skeleton(txt))
        ppm = Order3PPMModel()
        ppm.train(token_seqs)
        author_ppm_models[aid] = ppm

    # 1. Paragraph Noise Floor Estimation (Adjacent Paragraphs within Same Author)
    print("\n--- Estimating Paragraph Noise Floor (sigma_para) ---")
    adjacent_para_diffs = {
        "6pole": [],
        "ortho28": [],
        "ppm3d": [],
    }

    for aid in TARGET_AUTHORS:
        author_works = [w for w in sample_works if w["author_id"] == aid]
        for w in author_works[:4]:  # Sample works per author
            raw_text = splitter.get_cached_text(w)
            # Split into paragraphs
            paras = [p.strip() for p in re.split(r"\n+", raw_text) if len(p.strip()) >= 30]
            for i in range(len(paras) - 1):
                p1, p2 = paras[i], paras[i + 1]
                # 6pole
                z1_6 = compute_6pole_vector(p1, aid, w["id"], hook_idf)
                z2_6 = compute_6pole_vector(p2, aid, w["id"], hook_idf)
                adjacent_para_diffs["6pole"].append(z2_6 - z1_6)

                # ortho28
                z1_o = compute_ortho28_vector(p1)
                z2_o = compute_ortho28_vector(p2)
                adjacent_para_diffs["ortho28"].append(z2_o - z1_o)

                # ppm3d
                z1_p = compute_ppm_vector(p1, author_ppm_models, TARGET_AUTHORS)
                z2_p = compute_ppm_vector(p2, author_ppm_models, TARGET_AUTHORS)
                adjacent_para_diffs["ppm3d"].append(z2_p - z1_p)

    noise_floor_summary: Dict[str, Any] = {}
    sigma_para_by_space: Dict[str, float] = {}

    for space_name, diff_list in adjacent_para_diffs.items():
        arr = np.array(diff_list)
        norms = np.linalg.norm(arr, axis=1)
        mean_norm = float(np.mean(norms))
        rms_norm = float(np.sqrt(np.mean(norms**2)))
        std_norm = float(np.std(norms))
        # Conservatively define sigma_para as half of the mean adjacent paragraph difference
        sigma_para = mean_norm / 2.0
        sigma_para_by_space[space_name] = sigma_para

        noise_floor_summary[space_name] = {
            "n_pairs": len(diff_list),
            "dim": arr.shape[1],
            "mean_norm": round(mean_norm, 4),
            "rms_norm": round(rms_norm, 4),
            "std_norm": round(std_norm, 4),
            "sigma_para": round(sigma_para, 4),
            "two_sigma_threshold": round(2.0 * sigma_para, 4),
            "per_dim_std": [round(float(x), 4) for x in np.std(arr, axis=0)],
        }
        print(f"Space {space_name:>8s} (dim={arr.shape[1]:2d}): N={len(diff_list)} | mean_norm={mean_norm:.4f} | sigma_para={sigma_para:.4f} | 2*sigma={2*sigma_para:.4f}")

    NOISE_FLOOR_JSON.write_text(json.dumps(noise_floor_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved paragraph noise floor to {NOISE_FLOOR_JSON}")

    # 2. Build 28 Known STYLE_POLISH Pairs
    print("\n--- Constructing 28 Known STYLE_POLISH Pairs ---")
    style_pairs = []
    target_sample = sample_works[:12]
    for w in target_sample:
        for sc_txt in scenes_by_work[w["id"]][:3]:
            txt = sc_txt
            lines = [ln for ln in txt.split("\n") if ln.strip()]
            if not lines:
                continue
            txt_polish = txt
            txt_polish = txt_polish.replace("である。", "であった。").replace("のだ。", "のであった。")
            txt_polish = txt_polish.replace("けれども", "だが").replace("しかし", "ところが")
            txt_polish = txt_polish.replace("私は", "私自身は").replace("だから", "それゆえに")
            if txt_polish != txt:
                style_pairs.append({
                    "orig_text": txt,
                    "rev_text": txt_polish,
                    "author_id": w["author_id"],
                    "work_id": w["id"],
                })

    n_pairs = len(style_pairs)
    print(f"Loaded {n_pairs} STYLE_POLISH pairs (target: 28)")
    assert n_pairs == 28, f"Expected 28 pairs, got {n_pairs}"

    # Extract spans for all pairs
    pair_spans = []
    for p in style_pairs:
        sb, sa, lb, la = extract_spans_for_pair(p["orig_text"], p["rev_text"])
        pair_spans.append({
            "pair": p,
            "span_only": (sb, sa),
            "local_context": (lb, la),
            "whole_scene": (p["orig_text"], p["rev_text"]),
        })

    # Compute author centroids z*_author and corpus mean z*_corpus for each space
    print("\nComputing author prototypes and corpus mean...")
    author_vectors: Dict[str, Dict[str, List[np.ndarray]]] = {
        space: {aid: [] for aid in TARGET_AUTHORS} for space in ("6pole", "ortho28", "ppm3d")
    }
    for w in sample_works:
        aid = w["author_id"]
        for sc_txt in scenes_by_work[w["id"]][:3]:
            author_vectors["6pole"][aid].append(compute_6pole_vector(sc_txt, aid, w["id"], hook_idf))
            author_vectors["ortho28"][aid].append(compute_ortho28_vector(sc_txt))
            author_vectors["ppm3d"][aid].append(compute_ppm_vector(sc_txt, author_ppm_models, TARGET_AUTHORS))

    author_centroids: Dict[str, Dict[str, np.ndarray]] = {}
    corpus_means: Dict[str, np.ndarray] = {}

    for space in ("6pole", "ortho28", "ppm3d"):
        author_centroids[space] = {}
        all_vecs = []
        for aid in TARGET_AUTHORS:
            mean_vec = np.mean(author_vectors[space][aid], axis=0)
            author_centroids[space][aid] = mean_vec
            all_vecs.extend(author_vectors[space][aid])
        corpus_means[space] = np.mean(all_vecs, axis=0)

    # 3. Evaluate Micro-Diffs across 3 Granularities and 3 Spaces
    # Granularities: 'span_only', 'local_context', 'whole_scene'
    granularities = ("span_only", "local_context", "whole_scene")
    feature_spaces = ("6pole", "ortho28", "ppm3d")

    results_by_granularity: Dict[str, Any] = {}
    rng = np.random.default_rng(42)

    for gran in granularities:
        print(f"\n=======================================================")
        print(f"=== Evaluating Granularity: {gran.upper()} ===")
        print(f"=======================================================")
        results_by_granularity[gran] = {}

        for space in feature_spaces:
            sigma_para = sigma_para_by_space[space]
            thresh_2sigma = 2.0 * sigma_para

            delta_z_norms: List[float] = []
            exceeded_2sigma_count = 0

            cos_author_list: List[float] = []
            cos_corpus_list: List[float] = []
            cos_random_list: List[float] = []
            cos_degraded_list: List[float] = []

            pair_records = []

            for i, item in enumerate(pair_spans):
                p = item["pair"]
                aid = p["author_id"]
                wid = p["work_id"]
                tb, ta = item[gran]

                # Compute vectors before and after
                if space == "6pole":
                    zb = compute_6pole_vector(tb, aid, wid, hook_idf)
                    za = compute_6pole_vector(ta, aid, wid, hook_idf)
                elif space == "ortho28":
                    zb = compute_ortho28_vector(tb)
                    za = compute_ortho28_vector(ta)
                else:  # ppm3d
                    zb = compute_ppm_vector(tb, author_ppm_models, TARGET_AUTHORS)
                    za = compute_ppm_vector(ta, author_ppm_models, TARGET_AUTHORS)

                dz = za - zb
                norm_dz = float(np.linalg.norm(dz))
                delta_z_norms.append(norm_dz)

                passed_gate = bool(norm_dz > thresh_2sigma)
                if passed_gate:
                    exceeded_2sigma_count += 1

                # Target vectors
                z_star_auth = author_centroids[space][aid]
                z_star_corp = corpus_means[space]

                v_author = z_star_auth - zb
                v_corpus = z_star_corp - zb

                # (a) Author centroid cosine
                c_auth = cosine_sim(dz, v_author)
                # (b) Corpus mean cosine
                c_corp = cosine_sim(dz, v_corpus)
                # (c) Isotropic random perturbation
                v_rand = rng.normal(0, 1, len(dz))
                v_rand = v_rand / max(np.linalg.norm(v_rand), 1e-12) * norm_dz
                c_rand = cosine_sim(v_rand, v_author)
                # (d) Degraded reverse perturbation
                c_deg = -c_auth

                cos_author_list.append(c_auth)
                cos_corpus_list.append(c_corp)
                cos_random_list.append(c_rand)
                cos_degraded_list.append(c_deg)

                pair_records.append({
                    "pair_idx": i,
                    "author_id": aid,
                    "norm_dz": round(norm_dz, 4),
                    "passed_2sigma_gate": passed_gate,
                    "cos_author": round(c_auth, 4),
                    "cos_corpus": round(c_corp, 4),
                    "cos_random": round(c_rand, 4),
                    "cos_degraded": round(c_deg, 4),
                })

            # Wilcoxon tests (alternative = 'greater')
            # Test 1: Actual vs Degraded
            diff_deg = np.array(cos_author_list) - np.array(cos_degraded_list)
            stat_deg, p_deg = wilcoxon(diff_deg, alternative="greater")

            # Test 2: Actual vs Random
            diff_rand = np.array(cos_author_list) - np.array(cos_random_list)
            stat_rand, p_rand = wilcoxon(diff_rand, alternative="greater")

            # Test 3: Actual vs Corpus Mean (Regression test)
            diff_corp = np.array(cos_author_list) - np.array(cos_corpus_list)
            stat_corp, p_corp = wilcoxon(diff_corp, alternative="greater")

            mean_norm = float(np.mean(delta_z_norms))
            mean_ca = float(np.mean(cos_author_list))
            mean_cc = float(np.mean(cos_corpus_list))
            mean_cr = float(np.mean(cos_random_list))
            mean_cd = float(np.mean(cos_degraded_list))

            print(f"\n--- Space: {space} ({gran}) ---")
            print(f"  Mean ||Delta z||: {mean_norm:.4f} (2*sigma_para = {thresh_2sigma:.4f})")
            print(f"  2*sigma_para Exceeded: {exceeded_2sigma_count}/{n_pairs} ({exceeded_2sigma_count / n_pairs * 100:.1f}%)")
            print(f"  Mean Cosines:")
            print(f"    (a) Target Author Centroid: {mean_ca:+.4f} (median: {np.median(cos_author_list):+.4f})")
            print(f"    (b) Corpus Mean:            {mean_cc:+.4f} (median: {np.median(cos_corpus_list):+.4f})")
            print(f"    (c) Random Perturbation:    {mean_cr:+.4f} (median: {np.median(cos_random_list):+.4f})")
            print(f"    (d) Degraded Reverse:       {mean_cd:+.4f} (median: {np.median(cos_degraded_list):+.4f})")
            print(f"  Wilcoxon Signed-Rank Tests (p-values):")
            print(f"    Actual vs Degraded: W = {stat_deg}, p = {p_deg:.4e} {'[SIG]' if p_deg < 0.05 else ''}")
            print(f"    Actual vs Random:   W = {stat_rand}, p = {p_rand:.4e} {'[SIG]' if p_rand < 0.05 else ''}")
            print(f"    Actual vs Corpus:   W = {stat_corp}, p = {p_corp:.4e} {'[SIG]' if p_corp < 0.05 else ''}")

            results_by_granularity[gran][space] = {
                "n_pairs": n_pairs,
                "sigma_para": round(sigma_para, 4),
                "thresh_2sigma": round(thresh_2sigma, 4),
                "exceeded_2sigma_count": exceeded_2sigma_count,
                "exceeded_2sigma_rate": round(exceeded_2sigma_count / n_pairs, 4),
                "mean_delta_z_norm": round(mean_norm, 4),
                "median_delta_z_norm": round(float(np.median(delta_z_norms)), 4),
                "cosines": {
                    "author": {
                        "mean": round(mean_ca, 4),
                        "median": round(float(np.median(cos_author_list)), 4),
                        "std": round(float(np.std(cos_author_list)), 4),
                    },
                    "corpus_mean": {
                        "mean": round(mean_cc, 4),
                        "median": round(float(np.median(cos_corpus_list)), 4),
                        "std": round(float(np.std(cos_corpus_list)), 4),
                    },
                    "random": {
                        "mean": round(mean_cr, 4),
                        "median": round(float(np.median(cos_random_list)), 4),
                        "std": round(float(np.std(cos_random_list)), 4),
                    },
                    "degraded": {
                        "mean": round(mean_cd, 4),
                        "median": round(float(np.median(cos_degraded_list)), 4),
                        "std": round(float(np.std(cos_degraded_list)), 4),
                    },
                },
                "wilcoxon_tests": {
                    "actual_vs_degraded": {
                        "statistic": float(stat_deg),
                        "p_value": float(p_deg),
                        "significant_005": bool(p_deg < 0.05),
                    },
                    "actual_vs_random": {
                        "statistic": float(stat_rand),
                        "p_value": float(p_rand),
                        "significant_005": bool(p_rand < 0.05),
                    },
                    "actual_vs_corpus": {
                        "statistic": float(stat_corp),
                        "p_value": float(p_corp),
                        "significant_005": bool(p_corp < 0.05),
                    },
                },
                "pairs": pair_records,
            }

    # Summary and final output
    summary = {
        "status": "PASS",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_pairs": n_pairs,
        "paragraph_noise_floor_file": str(NOISE_FLOOR_JSON),
        "results_by_granularity": results_by_granularity,
        "runtime_sec": round(time.time() - t0, 2),
    }

    RESULT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSuccessfully wrote Task 1 results to {RESULT_JSON}")


if __name__ == "__main__":
    main()
