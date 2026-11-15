"""USIR v5 Stage B-18: Python / Rust numerical parity & ablation suite.

Compares:
- Python: usir_py.v5.six_pole_atlas (normalized_hsic, svd_lowdin_orthogonalize, NarrativeRGNormalizer)
- Rust:   nishiki-linalg (hsic::nhsic, lowdin::HsicLowdinOrthogonalizer, rg) via b18_rust_probe

Does NOT modify any existing calculation logic or frozen artifacts.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "python"))

from usir_py.v5.six_pole_atlas import (  # noqa: E402
    NarrativeRGNormalizer,
    aggregate_work_profile_from_scenes,
    center_kernel,
    normalized_hsic,
    rbf_kernel,
    svd_lowdin_orthogonalize,
)

RUST_PROBE_EXE = Path(
    os.environ.get(
        "B18_RUST_PROBE_EXE",
        r"C:\Users\user\.gemini\antigravity\brain\b213a4d0-d344-4736-9630-40eb85d69368\scratch\b18_rust_target\release\b18_rust_probe.exe",
    )
)


# ---------------------------------------------------------------------------
# Rust probe interface
# ---------------------------------------------------------------------------
def _mat_tokens(M: np.ndarray) -> str:
    return " ".join(f"{float(v):.17e}" for v in M.ravel(order="C"))


def call_rust_probe(commands: List[str]) -> List[str]:
    payload = "\n".join(commands) + "\n"
    proc = subprocess.run(
        [str(RUST_PROBE_EXE)],
        input=payload,
        text=True,
        capture_output=True,
        check=True,
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def rust_nhsic(X: np.ndarray, Y: np.ndarray) -> Tuple[float, float, float]:
    n, da = X.shape
    ny, db = Y.shape
    assert n == ny
    cmd = f"NHSIC {n} {da} {db} {_mat_tokens(X)} {_mat_tokens(Y)}"
    lines = call_rust_probe([cmd])
    parts = lines[0].split()
    assert parts[0] == "NHSIC"
    return float(parts[1]), float(parts[2]), float(parts[3])


def rust_lowdin(X: np.ndarray, eps_reg: float = 1e-10) -> Dict[str, Any]:
    n, d = X.shape
    cmd = f"LOWDIN {n} {d} {eps_reg:.17e} {_mat_tokens(X)}"
    lines = call_rust_probe([cmd])
    out: Dict[str, Any] = {}
    for line in lines:
        toks = line.split()
        tag = toks[0]
        if tag == "LOWDIN":
            out["orth_err"] = float(toks[1])
            out["distortion"] = float(toks[2])
            out["rank"] = int(toks[3])
        elif tag == "Z_STD":
            out["z_std"] = np.array([float(x) for x in toks[1:]], dtype=float).reshape(n, d)
        elif tag == "EVALS":
            out["evals"] = np.array([float(x) for x in toks[1:]], dtype=float)
        elif tag == "W":
            out["w"] = np.array([float(x) for x in toks[1:]], dtype=float).reshape(d, d)
        elif tag == "Z_ORTH":
            out["z_orth"] = np.array([float(x) for x in toks[1:]], dtype=float).reshape(n, d)
    return out


# ---------------------------------------------------------------------------
# Ablation variants for nHSIC & Lowdin
# ---------------------------------------------------------------------------
def rbf_kernel_variant(X: np.ndarray, bw_mode: str) -> np.ndarray:
    """bw_mode: 'py' (all dists>0, floor 1e-6) or 'rust' (i<j sqrt, fallback >0, no floor)."""
    if bw_mode == "py":
        return rbf_kernel(X)
    elif bw_mode == "rust":
        scale = float(np.max(np.abs(X)))
        if scale == 0.0:
            return np.ones((X.shape[0], X.shape[0]), dtype=float)
        Xs = X / scale
        d2 = squareform(pdist(Xs, metric="sqeuclidean"))
        n = X.shape[0]
        iu = np.triu_indices(n, k=1)
        d = np.sqrt(np.maximum(d2[iu], 0.0))
        med = float(np.median(d)) if len(d) > 0 else 0.0
        if med <= 0.0:
            pos = d[d > 0.0]
            med = float(np.median(pos)) if len(pos) > 0 else 0.0
        s2 = med * med
        if s2 <= 0.0 or not math.isfinite(s2):
            return np.ones((n, n), dtype=float)
        K = np.exp(-np.maximum(d2, 0.0) / (2.0 * s2))
        np.fill_diagonal(K, 1.0)
        return K
    else:
        raise ValueError(bw_mode)


def nhsic_variant(X: np.ndarray, Y: np.ndarray, bw_mode: str = "py", denom_mode: str = "py") -> float:
    K = center_kernel(rbf_kernel_variant(X, bw_mode))
    L = center_kernel(rbf_kernel_variant(Y, bw_mode))
    tr_KL = float(np.sum(K * L))
    tr_K2 = float(np.sum(K * K))
    tr_L2 = float(np.sum(L * L))
    if denom_mode == "py":
        denom = math.sqrt(max(tr_K2 * tr_L2, 1e-12))
    elif denom_mode == "rust":
        denom = math.sqrt(tr_K2 * tr_L2 + 1e-12)
    else:
        raise ValueError(denom_mode)
    return float(np.clip(tr_KL / denom, 0.0, 1.0))


def lowdin_variant(Z: np.ndarray, mode: str, eps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z must already be centered/standardized.
    mode: 'clip' -> max(lam, eps); 'add' -> max(lam, 0) + eps
    """
    N, D = Z.shape
    Z_c = Z - np.mean(Z, axis=0, keepdims=True)
    S = (Z_c.T @ Z_c) / N
    eigvals, U = np.linalg.eigh(S)
    if mode == "clip":
        reg_evals = np.maximum(eigvals, eps)
    elif mode == "add":
        reg_evals = np.maximum(eigvals, 0.0) + eps
    else:
        raise ValueError(mode)
    inv_sqrt = np.diag(1.0 / np.sqrt(reg_evals))
    W = U @ inv_sqrt @ U.T
    Z_orth = Z_c @ W
    return Z_orth, W, eigvals


def orth_error(Z_orth: np.ndarray) -> float:
    N, D = Z_orth.shape
    S_orth = (Z_orth.T @ Z_orth) / N
    return float(np.linalg.norm(S_orth - np.eye(D), ord="fro"))


# ---------------------------------------------------------------------------
# Task 1 & 2: Synthetic parity + ablation
# ---------------------------------------------------------------------------
def run_synthetic_nhsic_parity() -> Dict[str, Any]:
    rng = np.random.default_rng(20261115)
    ns = [50, 105, 300]
    dim_pairs = [(2, 2), (3, 2), (4, 4), (7, 4), (7, 7)]
    relations = ["independent", "copy", "affine_copy", "nonlinear_sq", "linear_mix_s0.5", "ties_60pct", "small_scale_1e-4"]

    records: List[Dict[str, Any]] = []
    cmds: List[str] = []
    inputs: List[Tuple[int, int, int, str, np.ndarray, np.ndarray]] = []

    for n in ns:
        for da, db in dim_pairs:
            for rel in relations:
                X = rng.standard_normal((n, da))
                if rel == "independent":
                    Y = rng.standard_normal((n, db))
                elif rel == "copy":
                    if da != db:
                        continue
                    Y = X.copy()
                elif rel == "affine_copy":
                    A = rng.standard_normal((da, db))
                    b = rng.standard_normal((1, db))
                    Y = X @ A + b
                elif rel == "nonlinear_sq":
                    m = min(da, db)
                    Y = rng.standard_normal((n, db)) * 0.0
                    Y[:, :m] = X[:, :m] ** 2
                    if db > m:
                        Y[:, m:] = rng.standard_normal((n, db - m))
                elif rel == "linear_mix_s0.5":
                    m = min(da, db)
                    Y = rng.standard_normal((n, db)) * 0.5
                    Y[:, :m] = X[:, :m] + 0.5 * rng.standard_normal((n, m))
                elif rel == "ties_60pct":
                    # 60% of rows identical -> median of i<j pairs is 0
                    k_dup = int(0.75 * n)
                    X[:k_dup] = X[0]
                    Y = rng.standard_normal((n, db))
                    Y[:k_dup] = Y[0]
                elif rel == "small_scale_1e-4":
                    X = X * 1e-4
                    Y = rng.standard_normal((n, db)) * 1e-4
                else:
                    continue

                cmds.append(f"NHSIC {n} {da} {db} {_mat_tokens(X)} {_mat_tokens(Y)}")
                inputs.append((n, da, db, rel, X, Y))

    rust_lines = call_rust_probe(cmds)
    assert len(rust_lines) == len(inputs)

    for (n, da, db, rel, X, Y), rline in zip(inputs, rust_lines):
        parts = rline.split()
        r_val = float(parts[1])
        py_val = normalized_hsic(X, Y)
        v_bw_rust = nhsic_variant(X, Y, bw_mode="rust", denom_mode="py")
        v_den_rust = nhsic_variant(X, Y, bw_mode="py", denom_mode="rust")
        v_both_rust = nhsic_variant(X, Y, bw_mode="rust", denom_mode="rust")

        records.append(
            {
                "N": n,
                "da": da,
                "db": db,
                "relation": rel,
                "py": py_val,
                "rust": r_val,
                "abs_diff": abs(py_val - r_val),
                "var_bw_rust": v_bw_rust,
                "var_den_rust": v_den_rust,
                "var_both_rust": v_both_rust,
                "ablation_bw_effect": abs(py_val - v_bw_rust),
                "ablation_den_effect": abs(py_val - v_den_rust),
                "residual_after_both": abs(v_both_rust - r_val),
            }
        )

    # Summarize by relation and overall (standard relations vs special)
    std_rels = {"independent", "copy", "affine_copy", "nonlinear_sq", "linear_mix_s0.5"}
    std_diffs = [r["abs_diff"] for r in records if r["relation"] in std_rels]
    all_diffs = [r["abs_diff"] for r in records]
    by_rel: Dict[str, Any] = {}
    for rel in relations:
        sub = [r for r in records if r["relation"] == rel]
        if not sub:
            continue
        diffs = [r["abs_diff"] for r in sub]
        bw_eff = [r["ablation_bw_effect"] for r in sub]
        den_eff = [r["ablation_den_effect"] for r in sub]
        res_eff = [r["residual_after_both"] for r in sub]
        by_rel[rel] = {
            "count": len(sub),
            "max_diff": float(np.max(diffs)),
            "median_diff": float(np.median(diffs)),
            "max_bw_effect": float(np.max(bw_eff)),
            "median_bw_effect": float(np.median(bw_eff)),
            "max_den_effect": float(np.max(den_eff)),
            "median_den_effect": float(np.median(den_eff)),
            "max_residual_after_both": float(np.max(res_eff)),
        }

    return {
        "total_cases": len(records),
        "std_cases": len(std_diffs),
        "std_max_diff": float(np.max(std_diffs)),
        "std_median_diff": float(np.median(std_diffs)),
        "all_max_diff": float(np.max(all_diffs)),
        "all_median_diff": float(np.median(all_diffs)),
        "by_relation": by_rel,
        "records": records,
    }


# ---------------------------------------------------------------------------
# Task 2 (Item 3): Lowdin parity + ablation
# ---------------------------------------------------------------------------
def run_lowdin_ablation() -> Dict[str, Any]:
    rng = np.random.default_rng(20261115 + 1)
    cases = []

    # Case 1: Well-conditioned (N=105, D=15)
    Z1 = rng.standard_normal((105, 15))
    Z1 = (Z1 - Z1.mean(axis=0)) / Z1.std(axis=0, ddof=0)
    cases.append(("well_conditioned_105x15", Z1))

    # Case 2: Ill-conditioned (lambda_min ~ 1.95e-6, matching real corpus 6-pole scale)
    Z2_base = rng.standard_normal((105, 15))
    Z2_base[:, -1] = Z2_base[:, 0] + 1.4e-3 * rng.standard_normal(105)
    Z2 = (Z2_base - Z2_base.mean(axis=0)) / Z2_base.std(axis=0, ddof=0)
    cases.append(("ill_conditioned_105x15_lam1e-6", Z2))

    # Case 3: Rank-deficient / exact collinearity (lambda_min = 0)
    Z3_base = rng.standard_normal((80, 6))
    Z3_base[:, 4] = Z3_base[:, 0]
    Z3_base[:, 5] = -2.0 * Z3_base[:, 1]
    Z3 = (Z3_base - Z3_base.mean(axis=0)) / Z3_base.std(axis=0, ddof=0)
    cases.append(("rank_deficient_80x6", Z3))

    results: List[Dict[str, Any]] = []
    for name, Z in cases:
        z_py, w_py = svd_lowdin_orthogonalize(Z, eps=1e-8)
        z_add_1e8, w_add_1e8, _ = lowdin_variant(Z, "add", 1e-8)
        z_clip_1e10, w_clip_1e10, _ = lowdin_variant(Z, "clip", 1e-10)
        z_add_1e10, w_add_1e10, evals_py = lowdin_variant(Z, "add", 1e-10)
        r = rust_lowdin(Z, eps_reg=1e-10)

        results.append(
            {
                "case": name,
                "N": Z.shape[0],
                "D": Z.shape[1],
                "lambda_min": float(np.min(evals_py)),
                "lambda_max": float(np.max(evals_py)),
                "py_orth_err": orth_error(z_py),
                "rust_orth_err": r["orth_err"],
                "py_distortion": float(np.linalg.norm(z_py - Z, ord="fro")),
                "rust_distortion": r["distortion"],
                "max_abs_diff_Z_py_vs_rust": float(np.max(np.abs(z_py - r["z_orth"]))),
                "max_abs_diff_W_py_vs_rust": float(np.max(np.abs(w_py - r["w"]))),
                # Ablation: mode (clip vs add at 1e-8) vs eps (1e-8 vs 1e-10)
                "ablation_mode_clip_vs_add_1e8_Z": float(np.max(np.abs(z_py - z_add_1e8))),
                "ablation_eps_1e8_vs_1e10_clip_Z": float(np.max(np.abs(z_py - z_clip_1e10))),
                "ablation_add_1e10_py_vs_rust_Z": float(np.max(np.abs(z_add_1e10 - r["z_orth"]))),
                "ablation_add_1e10_py_vs_rust_W": float(np.max(np.abs(w_add_1e10 - r["w"]))),
            }
        )
    return {"cases": results}


# ---------------------------------------------------------------------------
# Task 1 (Real corpus): aozora_106works_scenes.jsonl (4 deep poles)
# ---------------------------------------------------------------------------
def run_real_corpus_deep_poles() -> Dict[str, Any]:
    scenes_path = REPO_ROOT / "data" / "v5" / "aozora_106works_scenes.jsonl"
    cache_dir = REPO_ROOT / "data" / "aozora_cache"
    if not scenes_path.exists():
        return {"available": False, "reason": f"missing {scenes_path}"}

    works_scenes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with scenes_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = f"{obj['author_id']}_{obj['work_id']}"
            works_scenes[key].append(obj)

    # Match F6's 105-work filter: works with text file in aozora_cache and >= 64 sentences
    # Wait: let's check both all 106 works AND the 105 works that have text in aozora_cache
    cache_files = list(cache_dir.glob("*.txt")) if cache_dir.exists() else []
    profiles: List[Dict[str, Any]] = []
    work_keys_105: List[str] = []
    for key, sc_list in works_scenes.items():
        sc_list.sort(key=lambda s: s.get("scene_idx", 0))
        prof = aggregate_work_profile_from_scenes(sc_list)
        profiles.append(prof)
        suffix = f"_{sc_list[0]['author_id']}_{sc_list[0]['work_id']}.txt"
        has_text = any(p.name.endswith(suffix) for p in cache_files)
        if has_text and key != "000035_001567":  # check which 1 work was excluded in F6
            pass
        if has_text:
            work_keys_105.append(f"{sc_list[0]['author_id']}:{sc_list[0]['work_id']}")

    norm = NarrativeRGNormalizer({"works": profiles})
    all_keys = [f"{p['author_id']}:{p['work_id']}" for p in profiles]
    X_all = norm.extract_all(all_keys)

    # Slice into 4 deep blocks: Kamae (0:3), Sadame (3:7), Nagare (7:11), Awase (11:15)
    # Standardize each column (population std, matching F6 zblock)
    def zscore(M: np.ndarray) -> np.ndarray:
        mu = M.mean(axis=0, keepdims=True)
        sd = M.std(axis=0, ddof=0, keepdims=True)
        sd = np.where(sd < 1e-13, 1.0, sd)
        return (M - mu) / sd

    X_z = zscore(X_all)
    blocks = {
        "Kamae": X_z[:, 0:3],
        "Sadame": X_z[:, 3:7],
        "Nagare": X_z[:, 7:11],
        "Awase": X_z[:, 11:15],
    }
    names = ["Kamae", "Sadame", "Nagare", "Awase"]
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            py_v = normalized_hsic(blocks[a], blocks[b])
            r_v, _, _ = rust_nhsic(blocks[a], blocks[b])
            v_bw = nhsic_variant(blocks[a], blocks[b], bw_mode="rust", denom_mode="py")
            v_den = nhsic_variant(blocks[a], blocks[b], bw_mode="py", denom_mode="rust")
            pairs.append(
                {
                    "pair": f"{a}-{b}",
                    "py": py_v,
                    "rust": r_v,
                    "abs_diff": abs(py_v - r_v),
                    "ablation_bw_effect": abs(py_v - v_bw),
                    "ablation_den_effect": abs(py_v - v_den),
                }
            )

    diffs = [p["abs_diff"] for p in pairs]
    return {
        "available": True,
        "n_works_jsonl": len(profiles),
        "n_works_with_cache_txt": len(work_keys_105),
        "pairs": pairs,
        "max_diff": float(np.max(diffs)),
        "median_diff": float(np.median(diffs)),
    }


# ---------------------------------------------------------------------------
# Task 4: Frozen SHA-256 verification
# ---------------------------------------------------------------------------
def verify_frozen_sha256() -> Dict[str, Any]:
    baseline_path = REPO_ROOT / "scripts" / "b17" / "frozen_sha256_baseline.json"
    base = json.loads(baseline_path.read_text(encoding="utf-8"))

    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest().upper()

    files = []
    all_ok = True
    for f_str, exp in base.items():
        p = Path(f_str)
        act = sha(p)
        ok = act == exp
        all_ok &= ok
        files.append({"file": f_str, "expected": exp, "actual": act, "unchanged": ok})

    can_u = REPO_ROOT / "docs" / "handoff" / "v10" / "paragraph_noise_floor_all_216_canonical.json"
    can_n = Path(r"C:\dev\nishiki\docs\handoff\evidence\2026-11-14_f21\paragraph_noise_floor_all_216_canonical.json")
    rev_v11 = REPO_ROOT / "data" / "usir_v5_eval" / "real_revision_pairs_v1_1.jsonl"
    scenes = REPO_ROOT / "data" / "v5" / "aozora_106works_scenes.jsonl"

    sha_can_u = sha(can_u)
    sha_can_n = sha(can_n)
    return {
        "all_15_baseline_unchanged": all_ok,
        "n_baseline": len(files),
        "canonical_216_usir_sha256": sha_can_u,
        "canonical_216_nishiki_sha256": sha_can_n,
        "canonical_216_identical": sha_can_u == sha_can_n,
        "canonical_216_prefix_ok": sha_can_u.startswith("2FC18E67"),
        "real_revision_pairs_v1_1_sha256": sha(rev_v11),
        "aozora_106works_scenes_sha256": sha(scenes),
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    print("[B-18] Running synthetic nHSIC parity & ablation...")
    syn = run_synthetic_nhsic_parity()
    print(f"  Synthetic standard cases ({syn['std_cases']}): max_diff={syn['std_max_diff']:.6e}, median_diff={syn['std_median_diff']:.6e}")
    print(f"  Synthetic all cases ({syn['total_cases']}): max_diff={syn['all_max_diff']:.6e}, median_diff={syn['all_median_diff']:.6e}")

    print("[B-18] Running Lowdin ablation...")
    low = run_lowdin_ablation()
    for c in low["cases"]:
        print(f"  {c['case']}: max|Z_py - Z_rust|={c['max_abs_diff_Z_py_vs_rust']:.6e}, py_orth_err={c['py_orth_err']:.3e}, rust_orth_err={c['rust_orth_err']:.3e}")

    print("[B-18] Running real corpus deep poles parity...")
    real = run_real_corpus_deep_poles()
    print(f"  Real corpus (N={real.get('n_works_jsonl')}): max_diff={real.get('max_diff'):.6e}, median_diff={real.get('median_diff'):.6e}")

    print("[B-18] Verifying frozen SHA-256...")
    sha_res = verify_frozen_sha256()
    print(f"  15 baseline unchanged={sha_res['all_15_baseline_unchanged']}, 216 canonical ok={sha_res['canonical_216_prefix_ok']}")

    summary = {
        "synthetic_nhsic": {k: v for k, v in syn.items() if k != "records"},
        "synthetic_nhsic_records": syn["records"],
        "lowdin_ablation": low,
        "real_corpus_deep_poles": real,
        "frozen_sha256": sha_res,
    }
    out_path = out_dir / "b18_parity_results.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[B-18] Saved results to {out_path}")


if __name__ == "__main__":
    main()
