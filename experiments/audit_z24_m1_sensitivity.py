"""Audit M1 (comma removal) sensitivity of z0-z23 across all 106 Aozora works (11,723 windows).

Evaluates:
  1. Dimensional breakdown of M1 sensitivity for z0-z23.
  2. Performance (LOWO 46 / Held-out 60 / M1 flip count) for:
     - Canonical A55: Z24 + v1_31
     - Z24 + v2_31
     - Z24 + v2_28
     - Pure invariant subset: z_inv + v2_28 (and v2_28 alone if z_inv is empty)
     - Standalone v2_31, v1_31, v2_28, v1_28, Z24
  3. Experimental export: artifacts/style_atlas/fisher_v2.json
Output: docs/handoff/v10/z24_m1_sensitivity_and_fisher_v2.json
"""
from __future__ import annotations

import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

USIR = Path("c:/dev/usir")
for p in ("python", "scripts", "experiments"):
    sys.path.insert(0, str(USIR / p))

from fable_v5_grand_shootout import REG, load_works, pack  # noqa: E402
from fetch_aozora import CACHE_DIR  # noqa: E402
from usir_py import style_atlas as sa  # noqa: E402
from usir_py.extractor_24d import extract_usir_24d  # noqa: E402
from usir_py.fisher_atlas import fit_fisher  # noqa: E402
from usir_py.orthographic_prior import (  # noqa: E402
    ORTHO_EXT_NAMES,
    ORTHO_EXT_V2_NAMES,
    NARR_NAMES,
    NARR_V2_NAMES,
    ortho_matrix_ext,
    ortho_matrix_ext_v2,
    ortho_matrix_narr,
    ortho_matrix_narr_v2,
)

COMMA_DIMS = [2, 3, 29]
KEEP28 = [i for i in range(31) if i not in COMMA_DIMS]
OUT_DIR = USIR / "docs" / "handoff" / "v10"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR = USIR / "artifacts" / "style_atlas"
ART_DIR.mkdir(parents=True, exist_ok=True)


def wilson_interval(n_correct: int, n: int, confidence: float = 0.95) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    z = 1.95996  # 95%
    p = n_correct / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return [max(0.0, float(center - half)), min(1.0, float(center + half))]


def main() -> None:
    t0 = time.time()
    works = load_works()
    train = [w for w in works if w["split"] == "train"]
    held = [w for w in works if w["split"] != "train"]
    aids = sorted({w["author_id"] for w in train})
    print(f"Works: {len(works)} (train: {len(train)}, held-out: {len(held)}, authors: {len(aids)})")

    # Load cache or make windows
    cache = pickle.loads((CACHE_DIR / "window_features_v1.pkl").read_bytes())
    wins = []
    wins_m1 = []
    span = {}
    Z_orig_list = []

    for w in works:
        key = (w["sha256"], sa.WINDOW_CHARS, sa.STRIDE_CHARS)
        assert key in cache, f"Missing cache for {w['id']}"
        w_wins = sa.make_windows(w["text"])
        span[w["id"]] = (len(wins), len(wins) + len(w_wins))
        wins.extend(w_wins)
        wins_m1.extend([w_txt.replace("、", "") for w_txt in w_wins])
        Z_orig_list.append(cache[key]["Z"])

    Z_orig = np.vstack(Z_orig_list)
    assert len(wins) == 11723
    assert Z_orig.shape == (11723, 24)
    print(f"Total windows: {len(wins)}")

    # 1. Compute Z_m1 on all windows
    t_z0 = time.time()
    print("Computing z0-z23 under M1 (comma removal) on 11,723 windows...")
    Z_m1_list = []
    for i, w_m1 in enumerate(wins_m1):
        if (i + 1) % 2000 == 0:
            print(f"  Processed {i + 1}/{len(wins_m1)} windows...")
        Z_m1_list.append(extract_usir_24d(w_m1)["z_24d"])
    Z_m1 = np.array(Z_m1_list, dtype=float)
    t_zm1 = time.time() - t_z0
    print(f"Computed Z_m1 in {t_zm1:.2f}s, shape: {Z_m1.shape}")

    # Check Z sensitivity per dimension
    z_diff = np.abs(Z_orig - Z_m1)
    max_z_diff = np.max(z_diff, axis=0)
    mean_z_diff = np.mean(z_diff, axis=0)
    moved_windows_z = np.sum(z_diff > 1e-12, axis=0)

    z_inv_dims = [i for i in range(24) if max_z_diff[i] < 1e-12]
    print(f"Z invariant dimensions (< 1e-12): {z_inv_dims} ({len(z_inv_dims)}/24)")
    for i in range(24):
        print(f"  z{i:02d}: max_diff={max_z_diff[i]:.4e}, mean_diff={mean_z_diff[i]:.4e}, moved={moved_windows_z[i]}/{len(wins)}")

    # 2. Compute v1 and v2 representations
    print("Computing v1 and v2 ortho representations...")
    V1 = np.hstack([ortho_matrix_ext(wins), ortho_matrix_narr(wins)])
    V1_m1 = np.hstack([ortho_matrix_ext(wins_m1), ortho_matrix_narr(wins_m1)])
    V2 = np.hstack([ortho_matrix_ext_v2(wins), ortho_matrix_narr_v2(wins)])
    V2_m1 = np.hstack([ortho_matrix_ext_v2(wins_m1), ortho_matrix_narr_v2(wins_m1)])

    # Verify v2 invariance on 28 non-comma dimensions
    v2_diff = np.abs(V2 - V2_m1)
    v2_28_moved = np.sum(v2_diff[:, KEEP28] > 1e-12)
    print(f"V2 non-comma 28 dims moved windows: {v2_28_moved}")
    assert v2_28_moved == 0, f"Expected 0 moved windows on V2 non-comma dims, got {v2_28_moved}"

    # 3. Evaluate LOWO / Held-out / M1 prediction flips
    def per_work(M):
        return {w["id"]: M[span[w["id"]][0]:span[w["id"]][1]] for w in works}

    pw_Z = per_work(Z_orig)
    pw_Zm1 = per_work(Z_m1)
    pw_V1 = per_work(V1)
    pw_V1m1 = per_work(V1_m1)
    pw_V2 = per_work(V2)
    pw_V2m1 = per_work(V2_m1)

    systems = {
        "Z24+v1_31 (=A55, 正本)": {
            "build": lambda m1: {w["id"]: np.hstack([pw_Zm1[w["id"]] if m1 else pw_Z[w["id"]], pw_V1m1[w["id"]] if m1 else pw_V1[w["id"]]]) for w in works},
            "desc": "Canonical 55-D: z24 + ortho_v1 31D",
        },
        "Z24+v2_31": {
            "build": lambda m1: {w["id"]: np.hstack([pw_Zm1[w["id"]] if m1 else pw_Z[w["id"]], pw_V2m1[w["id"]] if m1 else pw_V2[w["id"]]]) for w in works},
            "desc": "55-D: z24 + ortho_v2 31D",
        },
        "Z24+v2_28": {
            "build": lambda m1: {w["id"]: np.hstack([pw_Zm1[w["id"]] if m1 else pw_Z[w["id"]], (pw_V2m1[w["id"]] if m1 else pw_V2[w["id"]])[:, KEEP28]]) for w in works},
            "desc": "52-D: z24 + ortho_v2 28D (excluding 3 comma dims)",
        },
        "v2_28 (読点非依存28D)": {
            "build": lambda m1: {w["id"]: (pw_V2m1[w["id"]] if m1 else pw_V2[w["id"]])[:, KEEP28] for w in works},
            "desc": "28-D: ortho_v2 without comma dims (strictly M1-invariant)",
        },
        "v2_31": {
            "build": lambda m1: {w["id"]: pw_V2m1[w["id"]] if m1 else pw_V2[w["id"]] for w in works},
            "desc": "31-D: ortho_v2 full",
        },
        "v1_31": {
            "build": lambda m1: {w["id"]: pw_V1m1[w["id"]] if m1 else pw_V1[w["id"]] for w in works},
            "desc": "31-D: ortho_v1 full",
        },
        "v1_28 (読点3次元なし)": {
            "build": lambda m1: {w["id"]: (pw_V1m1[w["id"]] if m1 else pw_V1[w["id"]])[:, KEEP28] for w in works},
            "desc": "28-D: ortho_v1 without comma dims",
        },
        "Z24": {
            "build": lambda m1: {w["id"]: pw_Zm1[w["id"]] if m1 else pw_Z[w["id"]] for w in works},
            "desc": "24-D: z0-z23 alone",
        },
    }

    if z_inv_dims:
        systems[f"z_inv({len(z_inv_dims)}D)+v2_28"] = {
            "build": lambda m1: {w["id"]: np.hstack([(pw_Zm1[w["id"]] if m1 else pw_Z[w["id"]])[:, z_inv_dims], (pw_V2m1[w["id"]] if m1 else pw_V2[w["id"]])[:, KEEP28]]) for w in works},
            "desc": f"{len(z_inv_dims) + 28}-D: strictly invariant Z dims + ortho_v2 28D",
        }

    results = {}
    print("\n--- Evaluating Systems ---")
    for name, spec in systems.items():
        X_base = spec["build"](False)
        X_pert = spec["build"](True)

        lowo_base, lowo_pert = [], []
        for t in train:
            rest = [w for w in train if w["id"] != t["id"]]
            m = fit_fisher({a: np.concatenate([X_base[w["id"]] for w in rest if w["author_id"] == a]) for a in aids}, reg=REG)
            lowo_base.append({"id": t["id"], "author": t["author_id"], "scores": m.scores(X_base[t["id"]])})
            lowo_pert.append({"id": t["id"], "author": t["author_id"], "scores": m.scores(X_pert[t["id"]])})

        m_all = fit_fisher({a: np.concatenate([X_base[w["id"]] for w in train if w["author_id"] == a]) for a in aids}, reg=REG)
        ho_base = [{"id": w["id"], "author": w["author_id"], "scores": m_all.scores(X_base[w["id"]])} for w in held]
        ho_pert = [{"id": w["id"], "author": w["author_id"], "scores": m_all.scores(X_pert[w["id"]])} for w in held]

        base_pack = pack(lowo_base, ho_base, works, aids)
        pert_pack = pack(lowo_pert, ho_pert, works, aids)

        base_pred_lowo = {r["id"]: r["predicted"] for r in base_pack["lowo"]["rows"]}
        pert_pred_lowo = {r["id"]: r["predicted"] for r in pert_pack["lowo"]["rows"]}
        flips_lowo = sum(1 for wid in base_pred_lowo if base_pred_lowo[wid] != pert_pred_lowo[wid])

        base_pred_ho = {r["id"]: r["predicted"] for r in base_pack["heldout"]["rows"]}
        pert_pred_ho = {r["id"]: r["predicted"] for r in pert_pack["heldout"]["rows"]}
        flips_ho = sum(1 for wid in base_pred_ho if base_pred_ho[wid] != pert_pred_ho[wid])

        dims = next(iter(X_base.values())).shape[1]
        lo_c = base_pack["lowo"]["n_correct"]
        ho_c = base_pack["heldout"]["n_correct"]
        mq_lo_c = pert_pack["lowo"]["n_correct"]
        mq_ho_c = pert_pack["heldout"]["n_correct"]

        results[name] = {
            "dims": dims,
            "description": spec["desc"],
            "lowo": {
                "n_correct": lo_c,
                "n": 46,
                "accuracy": round(lo_c / 46.0, 4),
                "wilson95": [round(x, 4) for x in wilson_interval(lo_c, 46)],
            },
            "heldout": {
                "n_correct": ho_c,
                "n": 60,
                "accuracy": round(ho_c / 60.0, 4),
                "wilson95": [round(x, 4) for x in wilson_interval(ho_c, 60)],
            },
            "m1_query": {
                "lowo_correct": mq_lo_c,
                "lowo_flips": flips_lowo,
                "heldout_correct": mq_ho_c,
                "heldout_flips": flips_ho,
            },
        }

        print(f"{name:<28} dims={dims:2d} | LOWO {lo_c:2d}/46 ({results[name]['lowo']['wilson95']}) | "
              f"Held {ho_c:2d}/60 ({results[name]['heldout']['wilson95']}) | "
              f"M1 query: LOWO {mq_lo_c:2d}/46 (flips {flips_lowo:2d}), Held {mq_ho_c:2d}/60 (flips {flips_ho:2d})")

    # 4. Generate experimental artifacts/style_atlas/fisher_v2.json
    # Fit Fisher on train works using v2_28 (or z_inv+v2_28)
    # Target feature names: 28 dims without commas
    v2_names = list(ORTHO_EXT_V2_NAMES) + list(NARR_V2_NAMES)
    v2_28_names = [v2_names[i] for i in KEEP28]
    X_train_v2_28 = {a: np.concatenate([pw_V2[w["id"]][:, KEEP28] for w in train if w["author_id"] == a]) for a in aids}
    fisher_v2_model = fit_fisher(X_train_v2_28, reg=REG)

    # Structure identical to FisherProjection serializable format
    fisher_v2_data = {
        "status": "EXPERIMENTAL_PARALLEL_LAYER",
        "notice": "This artifact is for experimental ortho_v2 evaluation only. Canonical path remains fisher_v1.json.",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "schema": "usir_fisher_atlas_v2",
        "feature": "v2_28 (ortho_v2 non-comma)",
        "authors": aids,
        "feature_names": v2_28_names,
        "n_features": len(v2_28_names),
        "center": fisher_v2_model.center.tolist(),
        "scale": fisher_v2_model.scale.tolist(),
        "W": fisher_v2_model.W.tolist(),
        "eigvals": fisher_v2_model.eigvals.tolist(),
        "class_means_proj": {str(k): v.tolist() for k, v in fisher_v2_model.means.items()},
        "mu_bar_proj": fisher_v2_model.mu_bar.tolist(),
        "reg": REG,
    }
    fisher_v2_path = ART_DIR / "fisher_v2.json"
    fisher_v2_path.write_text(json.dumps(fisher_v2_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote experimental Fisher v2 artifact to {fisher_v2_path}")

    # Output full json
    summary = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_works": len(works),
        "n_windows": len(wins),
        "z_m1_sensitivity": {
            "z_inv_dims": z_inv_dims,
            "n_z_inv": len(z_inv_dims),
            "max_z_diff": {f"z{i:02d}": float(max_z_diff[i]) for i in range(24)},
            "mean_z_diff": {f"z{i:02d}": float(mean_z_diff[i]) for i in range(24)},
            "moved_windows": {f"z{i:02d}": int(moved_windows_z[i]) for i in range(24)},
        },
        "systems": results,
        "runtime_sec": round(time.time() - t0, 2),
    }

    out_json_path = OUT_DIR / "z24_m1_sensitivity_and_fisher_v2.json"
    out_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote summary to {out_json_path}")


if __name__ == "__main__":
    main()
