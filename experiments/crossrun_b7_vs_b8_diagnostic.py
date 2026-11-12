"""USIR v5 Stage B8-DIAG Task 1: 2x2 Cross-Run Diagnostic between B-7 and B-8.

Separates whether the collapse of residual specificity signals in Stage B-8
was caused by 'Data Effect (Pair Expansion)' or 'Implementation/Definition Effect'.

2x2 Matrix:
- Cell (1,1): B7 Implementation x B7 Original Data (Micro 28 pairs + D10 Macro 12 pairs)
- Cell (1,2): B8 Implementation x B7 Original Data (Micro 28 pairs + D10 Macro 12 pairs)
- Cell (2,1): B7 Implementation x B8 Expanded Data (Class A 42 pairs + Class B 22 pairs)
- Cell (2,2): B8 Implementation x B8 Expanded Data (Class A 42 pairs + Class B 22 pairs)

Spaces evaluated:
- ortho28 (28D)
- ortho28_plus_kasane11 (39D)
- 6pole (6D)
- 6pole_plus_hw25 (7D)
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from scipy.stats import wilcoxon

REPO_ROOT = Path("c:/dev/usir")
for p in ("python", "scripts", "experiments"):
    sys.path.insert(0, str(REPO_ROOT / p))

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
from usir_py.v5.narrative_pacing_and_metaphor import extract_hw25_uid_ratio  # noqa: E402

COMMA_DIMS = [2, 3, 29]
KEEP28 = [i for i in range(31) if i not in COMMA_DIMS]
TARGET_AUTHORS = ("000035", "000148", "000879")

# ---------------------------------------------------------------------------
# Feature Extractors
# ---------------------------------------------------------------------------

def extract_ortho28(text: str) -> np.ndarray:
    ext = ortho_matrix_ext_v2([text])[0]
    narr = ortho_matrix_narr_v2([text])[0]
    full31 = np.concatenate([ext, narr])
    return full31[KEEP28].astype(float)


def extract_kasane11(text: str) -> np.ndarray:
    n = max(len(text), 1)
    c_comma = text.count("、") / n
    c_period = text.count("。") / n
    c_te = text.count("て") / n
    c_de = text.count("で") / n
    c_ni = text.count("に") / n
    c_wa = text.count("は") / n
    c_ga = text.count("が") / n
    c_no = text.count("の") / n
    c_tada = (text.count("た") + text.count("だ")) / n
    conj_matches = sum(text.count(w) for w in ["だが", "しかし", "ところが", "それゆえ", "けれども", "だから"])
    c_conj = conj_matches / n
    clauses = [cl for cl in text.replace("。", "、").split("、") if cl.strip()]
    mean_cl = float(np.mean([len(cl) for cl in clauses])) if clauses else float(n)
    return np.array([
        c_comma, c_period, c_te, c_de, c_ni, c_wa, c_ga, c_no, c_tada, c_conj, mean_cl / 100.0
    ], dtype=float)


def extract_ortho28_kasane11(text: str) -> np.ndarray:
    return np.concatenate([extract_ortho28(text), extract_kasane11(text)])


def extract_6pole(text: str, author_id: str, work_id: str, hook_idf: Any) -> np.ndarray:
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


def extract_6pole_hw25(text: str, author_id: str, work_id: str, hook_idf: Any) -> np.ndarray:
    p6 = extract_6pole(text, author_id, work_id, hook_idf)
    hw25 = extract_hw25_uid_ratio(text)
    return np.concatenate([p6, [hw25]])


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def safe_wilcoxon_tests(x: Sequence[float], y: Sequence[float]) -> Dict[str, Any]:
    """Perform both one-sided (greater) and two-sided Wilcoxon signed-rank tests."""
    diff = np.array(x, dtype=float) - np.array(y, dtype=float)
    nonzero = diff[np.abs(diff) > 1e-12]
    if len(nonzero) < 3:
        return {
            "w_stat": 0.0,
            "p_one_sided": 1.0,
            "p_two_sided": 1.0,
            "sig_005_one": False,
            "sig_005_two": False,
            "n_nonzero": len(nonzero),
        }
    try:
        res_one = wilcoxon(diff, alternative="greater")
        w_stat = float(res_one.statistic)
        p_one = float(res_one.pvalue)
    except Exception:
        w_stat, p_one = 0.0, 1.0

    try:
        res_two = wilcoxon(diff, alternative="two-sided")
        p_two = float(res_two.pvalue)
    except Exception:
        p_two = 1.0

    return {
        "w_stat": w_stat,
        "p_one_sided": p_one,
        "p_two_sided": p_two,
        "sig_005_one": bool(p_one < 0.05),
        "sig_005_two": bool(p_two < 0.05),
        "n_nonzero": len(nonzero),
    }


# ---------------------------------------------------------------------------
# B-7 Implementation Engine
# ---------------------------------------------------------------------------

def run_b7_implementation(
    pairs: List[Dict[str, Any]],
    space_name: str,
    extract_fn: Any,
    corpus_mean: np.ndarray,
    author_centroids: Dict[str, np.ndarray],
    style_centroids: Optional[Dict[str, np.ndarray]] = None,
    all_aids: Optional[List[str]] = None,
    rng_seed: int = 42,
) -> Dict[str, Any]:
    """Run B-7 residual orthogonalization implementation on a given set of pairs."""
    rng = np.random.default_rng(rng_seed)
    dim = len(corpus_mean)

    v_perp_norms = []
    rho_perps = []
    theta_degs = []
    quasi_flags = []

    cos_specs = []
    cos_rands = []
    cos_others = []

    for p in pairs:
        tb = p["text_before"]
        ta = p["text_after"]
        zb = extract_fn(tb)
        za = extract_fn(ta)
        dz = za - zb
        norm_dz = float(np.linalg.norm(dz))

        # Determine target centroid z_star_target
        if "target_style" in p and style_centroids and p["target_style"] in style_centroids:
            z_star_target = style_centroids[p["target_style"]]
            is_style = True
        else:
            aid = p.get("author_id", "000035")
            if aid in author_centroids:
                z_star_target = author_centroids[aid]
            else:
                # fallback to corpus mean
                z_star_target = corpus_mean
            is_style = False

        z_star_corp = corpus_mean

        # B-7 Gram-Schmidt
        v_corp = z_star_corp - zb
        norm_vc = np.linalg.norm(v_corp)
        u_corp = v_corp / norm_vc if norm_vc > 1e-12 else np.zeros_like(v_corp)

        v_target = z_star_target - zb
        norm_vt = np.linalg.norm(v_target)

        if norm_vt < 1e-12:
            rho_perp = 0.0
            theta_deg = 0.0
            v_perp = np.zeros_like(v_target)
            u_spec = np.zeros_like(v_target)
            quasi = True
        else:
            proj = float(np.dot(v_target, u_corp))
            v_perp = v_target - proj * u_corp
            norm_vp = float(np.linalg.norm(v_perp))
            rho_perp = float(norm_vp / norm_vt)
            cos_th = float(np.clip(proj / norm_vt, -1.0, 1.0))
            theta_deg = float(math.degrees(math.acos(cos_th)))
            u_spec = v_perp / norm_vp if norm_vp > 1e-12 else np.zeros_like(v_perp)
            quasi = bool(rho_perp < 0.10)

        v_perp_norms.append(float(np.linalg.norm(v_perp)))
        rho_perps.append(rho_perp)
        theta_degs.append(theta_deg)
        quasi_flags.append(quasi)

        # 1. cos_spec
        c_spec = cosine_sim(dz, u_spec)
        cos_specs.append(c_spec)

        # 2. vs isotropic random
        v_rand = rng.normal(0, 1, dim)
        norm_vr = np.linalg.norm(v_rand)
        if norm_vr > 1e-12:
            v_rand = v_rand / norm_vr * norm_dz
        cos_rands.append(cosine_sim(v_rand, u_spec))

        # 3. vs other authors or other styles
        if is_style and style_centroids:
            target_st = p["target_style"]
            other_styles = [s for s in style_centroids.keys() if s != target_st]
            other_cos = []
            for ost in other_styles:
                z_ost = style_centroids[ost]
                v_t_ost = z_ost - zb
                norm_vt_ost = np.linalg.norm(v_t_ost)
                if norm_vt_ost > 1e-12:
                    pr = float(np.dot(v_t_ost, u_corp))
                    vp = v_t_ost - pr * u_corp
                    nvp = np.linalg.norm(vp)
                    us = vp / nvp if nvp > 1e-12 else np.zeros_like(vp)
                    other_cos.append(cosine_sim(dz, us))
            cos_others.append(float(np.mean(other_cos)) if other_cos else 0.0)
        else:
            aid = p.get("author_id", "000035")
            other_aids = [a for a in (all_aids or list(author_centroids.keys())) if a != aid and a in author_centroids]
            other_cos = []
            for oaid in other_aids:
                z_oa = author_centroids[oaid]
                v_t_oa = z_oa - zb
                norm_vt_oa = np.linalg.norm(v_t_oa)
                if norm_vt_oa > 1e-12:
                    pr = float(np.dot(v_t_oa, u_corp))
                    vp = v_t_oa - pr * u_corp
                    nvp = np.linalg.norm(vp)
                    us = vp / nvp if nvp > 1e-12 else np.zeros_like(vp)
                    other_cos.append(cosine_sim(dz, us))
            cos_others.append(float(np.mean(other_cos)) if other_cos else 0.0)

    # Statistical tests
    test_rand = safe_wilcoxon_tests(cos_specs, cos_rands)
    test_other = safe_wilcoxon_tests(cos_specs, cos_others)

    return {
        "n_pairs": len(pairs),
        "n_quasi": sum(quasi_flags),
        "median_rho_perp": round(float(np.median(rho_perps)), 4),
        "median_theta_deg": round(float(np.median(theta_degs)), 2),
        "mean_cos_spec": round(float(np.mean(cos_specs)), 4),
        "mean_cos_rand": round(float(np.mean(cos_rands)), 4),
        "mean_cos_other": round(float(np.mean(cos_others)), 4),
        "vs_random": test_rand,
        "vs_other": test_other,
    }


# ---------------------------------------------------------------------------
# B-8 Implementation Engine
# ---------------------------------------------------------------------------

def run_b8_implementation(
    pairs: List[Dict[str, Any]],
    space_name: str,
    extract_fn: Any,
    corp_vec: np.ndarray,
    proto_map: Dict[str, np.ndarray],
    rng_seed: int = 123,
) -> Dict[str, Any]:
    """Run B-8 implementation as coded in scale_separated_level3.py."""
    rng = np.random.default_rng(rng_seed)
    dim = len(corp_vec)

    rhos = []
    thetas = []
    quasis = []
    cos_rands = []
    cos_others = []

    norm_uc = float(np.linalg.norm(corp_vec))
    u_corp_unit = corp_vec / norm_uc if norm_uc > 1e-12 else np.zeros_like(corp_vec)

    for p in pairs:
        zb = extract_fn(p["text_before"])
        za = extract_fn(p["text_after"])
        v_t = za - zb
        norm_vt = float(np.linalg.norm(v_t))

        if norm_vt < 1e-12 or norm_uc < 1e-12:
            v_perp = np.zeros_like(v_t)
            rho_perp = 0.0
            theta_deg = 0.0
            u_s = np.zeros_like(v_t)
            q = True
        else:
            proj = float(np.dot(v_t, u_corp_unit))
            v_perp = v_t - proj * u_corp_unit
            norm_vp = float(np.linalg.norm(v_perp))
            rho_perp = float(norm_vp / norm_vt)
            cos_theta = float(np.clip(proj / norm_vt, -1.0, 1.0))
            theta_deg = float(math.degrees(math.acos(cos_theta)))
            u_s = v_perp / norm_vp if norm_vp > 1e-12 else np.zeros_like(v_perp)
            q = bool(rho_perp < 0.10)

        rhos.append(rho_perp)
        thetas.append(theta_deg)
        quasis.append(q)

        # Random isotropic comparison: dot(u_s, rand_v)
        rand_v = rng.standard_normal(dim)
        n_rv = np.linalg.norm(rand_v)
        if n_rv > 1e-12:
            rand_v /= n_rv
        cos_rands.append(float(np.dot(u_s, rand_v)))

        # Other author comparison
        aid = p.get("author_id", "")
        other_aids = [a for a in proto_map.keys() if a != aid]
        if other_aids:
            other_proto_mean = np.mean([proto_map[a] for a in other_aids], axis=0)
            v_other = other_proto_mean - corp_vec
            norm_vo = np.linalg.norm(v_other)
            if norm_vo > 1e-12:
                cos_others.append(float(np.dot(u_s, v_other / norm_vo)))
            else:
                cos_others.append(0.0)
        else:
            cos_others.append(0.0)

    # B-8 Wilcoxon tests against 0 (both one-sided and two-sided)
    test_rand = safe_wilcoxon_tests(cos_rands, [0.0] * len(cos_rands))
    test_other = safe_wilcoxon_tests(cos_others, [0.0] * len(cos_others))

    return {
        "n_pairs": len(pairs),
        "n_quasi": sum(quasis),
        "median_rho_perp": round(float(np.median(rhos)), 4),
        "median_theta_deg": round(float(np.median(thetas)), 2),
        "mean_cos_spec": round(float(np.mean(cos_rands)), 4),  # B-8 effect size was mean of cos_rands / cos_others
        "mean_cos_rand": round(float(np.mean(cos_rands)), 4),
        "mean_cos_other": round(float(np.mean(cos_others)), 4),
        "vs_random": test_rand,
        "vs_other": test_other,
    }


# ---------------------------------------------------------------------------
# Data Builders
# ---------------------------------------------------------------------------

def build_b7_data(works: List[Dict[str, Any]], splitter: AozoraCorpusSplitter) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Construct B-7 28 micro-polish pairs and 12 D10 macro pairs."""
    from usir_v5_residual_specificity_level3 import extract_spans_for_pair, CROSSED_SOURCES, TRANSLATOR_STYLES

    # 1. 28 micro pairs
    sample_target_works = [w for w in works if w["author_id"] in TARGET_AUTHORS][:12]
    style_pairs = []
    for w in sample_target_works:
        txt = splitter.get_cached_text(w)
        scs = split_work_into_scenes(txt)
        for sc_txt in scs[:3]:
            lines = [ln for ln in sc_txt.split("\n") if ln.strip()]
            if not lines:
                continue
            txt_polish = sc_txt
            txt_polish = txt_polish.replace("である。", "であった。").replace("のだ。", "のであった。")
            txt_polish = txt_polish.replace("けれども", "だが").replace("しかし", "ところが")
            txt_polish = txt_polish.replace("私は", "私自身は").replace("だから", "それゆえに")
            if txt_polish != sc_txt:
                sb, sa, _, _ = extract_spans_for_pair(sc_txt, txt_polish)
                style_pairs.append({
                    "text_before": sb,
                    "text_after": sa,
                    "author_id": w["author_id"],
                    "work_id": w["id"],
                    "pair_type": "micro_polish",
                })
    style_pairs = style_pairs[:28]

    # 2. 12 D10 macro pairs (TR_STANDARD -> 3 styles)
    d10_cells = {}
    for src in CROSSED_SOURCES:
        for trans in TRANSLATOR_STYLES:
            paras = []
            for ev_text, ev_title in src["events"]:
                styled_text = trans["transform"](ev_text)
                expanded = f"【{src['title']} - {ev_title}】\n{styled_text}\n" * 2
                paras.append(expanded)
            full_text = "\n".join(paras)
            d10_cells[f"{src['source_id']}_{trans['trans_id']}"] = full_text

    d10_pairs = []
    for src in CROSSED_SOURCES:
        sid = src["source_id"]
        z_std_txt = d10_cells[f"{sid}_TR_STANDARD"]
        for tid in ("TR_RIGID", "TR_EMOTIONAL", "TR_CONVERSATIONAL"):
            z_target_txt = d10_cells[f"{sid}_{tid}"]
            d10_pairs.append({
                "text_before": z_std_txt,
                "text_after": z_target_txt,
                "source_id": sid,
                "author_id": "D10_TRANSLATOR",
                "base_style": "TR_STANDARD",
                "target_style": tid,
                "pair_type": "macro_revision",
            })

    return style_pairs, d10_pairs


def load_b8_data() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load B-8 Class A (42 pairs) and Class B (22 pairs)."""
    dataset_path = REPO_ROOT / "data" / "usir_v5_eval" / "u6_calibration_and_level3_dataset.jsonl"
    class_a = []
    class_b = []
    styles = [("TR_RIGID", "蓋し"), ("TR_EMOTIONAL", "ああ、"), ("TR_CONVERSATIONAL", "ねえ、"), ("TR_STANDARD", "こうして")]

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item["class"] == "Class_A_micro_polish":
                class_a.append({
                    "text_before": item["text_before"],
                    "text_after": item["text_after"],
                    "author_id": item.get("author_id", "000035"),
                    "work_id": item.get("work_id", "work"),
                    "pair_type": "micro_polish",
                })
            elif item["class"] == "Class_B_macro_revision":
                tb = item["text_before"]
                ta = item["text_after"]
                # Detect style for D10 pairs
                sb = [s[0] for s in styles if s[1] in tb]
                sa = [s[0] for s in styles if s[1] in ta]
                t_style = sa[0] if sa else None
                b_style = sb[0] if sb else None

                class_b.append({
                    "text_before": tb,
                    "text_after": ta,
                    "author_id": item.get("author_id", "D10_TRANSLATOR"),
                    "work_id": item.get("work_id", "macro"),
                    "base_style": b_style,
                    "target_style": t_style,
                    "pair_type": "macro_revision",
                })

    return class_a, class_b


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    print("=== USIR v5 Stage B8-DIAG Task 1: 2x2 Cross-Run Diagnostic ===")

    works = load_works()
    splitter = AozoraCorpusSplitter()
    all_aids = sorted({w["author_id"] for w in works})

    profile_json = REPO_ROOT / "docs" / "handoff" / "v10" / "nishiki_profile_106works.json"
    hook_idf = json.loads(profile_json.read_text(encoding="utf-8")).get("hook_idf") if profile_json.exists() else None

    # Load 106-works scene vectors for B-7 reference
    print("Extracting reference centroids across all 106 works...")
    scenes_by_aid: Dict[str, List[str]] = {a: [] for a in all_aids}
    for w in works:
        txt = splitter.get_cached_text(w)
        scs = split_work_into_scenes(txt)
        scenes_by_aid[w["author_id"]].extend(scs[:3])

    spaces = {
        "ortho28": extract_ortho28,
        "ortho28_plus_kasane11": extract_ortho28_kasane11,
        "6pole": lambda t: extract_6pole(t, "author", "work", hook_idf),
        "6pole_plus_hw25": lambda t: extract_6pole_hw25(t, "author", "work", hook_idf),
    }

    # Author centroids and corpus means across 4 spaces
    author_centroids_all: Dict[str, Dict[str, np.ndarray]] = {s: {} for s in spaces}
    corpus_means_all: Dict[str, np.ndarray] = {}

    for s_name, fn in spaces.items():
        all_vecs = []
        for aid, scs in scenes_by_aid.items():
            if scs:
                vecs = [fn(sc) for sc in scs]
                author_centroids_all[s_name][aid] = np.mean(vecs, axis=0)
                all_vecs.extend(vecs)
        corpus_means_all[s_name] = np.mean(all_vecs, axis=0)

    # D10 Style centroids across 4 spaces
    from usir_v5_residual_specificity_level3 import CROSSED_SOURCES, TRANSLATOR_STYLES
    d10_style_centroids: Dict[str, Dict[str, np.ndarray]] = {s: {} for s in spaces}
    d10_corpus_means: Dict[str, np.ndarray] = {}

    for s_name, fn in spaces.items():
        all_d10_vecs = []
        for trans in TRANSLATOR_STYLES:
            tid = trans["trans_id"]
            trans_vecs = []
            for src in CROSSED_SOURCES:
                paras = [trans["transform"](ev[0]) for ev in src["events"]]
                full_t = "\n".join(paras)
                v = fn(full_t)
                trans_vecs.append(v)
                all_d10_vecs.append(v)
            d10_style_centroids[s_name][tid] = np.mean(trans_vecs, axis=0)
        d10_corpus_means[s_name] = np.mean(all_d10_vecs, axis=0)

    # Build datasets
    print("Building datasets...")
    b7_micro_28, b7_macro_12 = build_b7_data(works, splitter)
    b8_micro_42, b8_macro_22 = load_b8_data()

    print(f"B7 data: Micro={len(b7_micro_28)}, Macro={len(b7_macro_12)}")
    print(f"B8 data: Micro={len(b8_micro_42)}, Macro={len(b8_macro_22)}")

    matrix_results: Dict[str, Dict[str, Any]] = {
        "Cell_1_1_B7impl_x_B7data": {},
        "Cell_1_2_B8impl_x_B7data": {},
        "Cell_2_1_B7impl_x_B8data": {},
        "Cell_2_2_B8impl_x_B8data": {},
    }

    # Helper for running all 4 spaces on a given cell
    def run_all_spaces_b7_impl(micro_pairs, macro_pairs) -> Dict[str, Any]:
        cell_out = {}
        for s_name, fn in spaces.items():
            # For micro spaces (ortho28, ortho28+kasane11), test on micro_pairs
            # For macro spaces (6pole, 6pole+hw25), test on macro_pairs
            target_pairs = micro_pairs if "ortho" in s_name else macro_pairs
            c_mean = d10_corpus_means[s_name] if "6pole" in s_name else corpus_means_all[s_name]
            st_cent = d10_style_centroids[s_name] if "6pole" in s_name else None

            res = run_b7_implementation(
                target_pairs,
                s_name,
                fn,
                corpus_mean=c_mean,
                author_centroids=author_centroids_all[s_name],
                style_centroids=st_cent,
                all_aids=all_aids,
            )
            cell_out[s_name] = res
        return cell_out

    def run_all_spaces_b8_impl(micro_pairs, macro_pairs) -> Dict[str, Any]:
        cell_out = {}
        for s_name, fn in spaces.items():
            target_pairs = micro_pairs if "ortho" in s_name else macro_pairs
            res = run_b8_implementation(
                target_pairs,
                s_name,
                fn,
                corp_vec=corpus_means_all[s_name],
                proto_map=author_centroids_all[s_name],
            )
            cell_out[s_name] = res
        return cell_out

    # Execute 2x2
    print("\nRunning Cell (1,1): B7 Implementation x B7 Data...")
    matrix_results["Cell_1_1_B7impl_x_B7data"] = run_all_spaces_b7_impl(b7_micro_28, b7_macro_12)

    print("Running Cell (1,2): B8 Implementation x B7 Data...")
    matrix_results["Cell_1_2_B8impl_x_B7data"] = run_all_spaces_b8_impl(b7_micro_28, b7_macro_12)

    print("Running Cell (2,1): B7 Implementation x B8 Data...")
    matrix_results["Cell_2_1_B7impl_x_B8data"] = run_all_spaces_b7_impl(b8_micro_42, b8_macro_22)

    print("Running Cell (2,2): B8 Implementation x B8 Data...")
    matrix_results["Cell_2_2_B8impl_x_B8data"] = run_all_spaces_b8_impl(b8_micro_42, b8_macro_22)

    # Print Comparison Table
    print("\n" + "=" * 100)
    print("=== 2x2 CROSS-RUN DIAGNOSTIC COMPARISON TABLE ===")
    print("=" * 100)
    header = f"{'Cell':<26} | {'Space':<20} | {'N':>3} | {'Quasi':>5} | {'rho_perp':>8} | {'theta':>7} | {'cos_spec':>8} | {'vs Rand (p_1, p_2)':<20} | {'vs Other (p_1, p_2)':<20}"
    print(header)
    print("-" * 100)

    for cell_key, cell_data in matrix_results.items():
        for s_name, res in cell_data.items():
            r_one = f"{res['vs_random']['p_one_sided']:.4f}"
            r_two = f"{res['vs_random']['p_two_sided']:.4f}"
            o_one = f"{res['vs_other']['p_one_sided']:.4f}"
            o_two = f"{res['vs_other']['p_two_sided']:.4f}"
            row = (
                f"{cell_key:<26} | {s_name:<20} | {res['n_pairs']:>3} | {res['n_quasi']:>5} | "
                f"{res['median_rho_perp']:>8.4f} | {res['median_theta_deg']:>6.1f}° | {res['mean_cos_spec']:>+8.4f} | "
                f"({r_one:>6}, {r_two:>6}) | ({o_one:>6}, {o_two:>6})"
            )
            print(row)
        print("-" * 100)

    # Save output json
    out_file = REPO_ROOT / "data" / "usir_v5_eval" / "crossrun_b7_vs_b8_diagnostic_table.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(matrix_results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved cross-run diagnostic results to {out_file} ({round(time.time() - t0, 2)}s)")


if __name__ == "__main__":
    main()
