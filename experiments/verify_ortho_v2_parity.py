"""Verify parity between Rust ortho_v2 (crates/usir-metrics/src/ortho_v2.rs)
and Python ortho_v2 (usir_py/orthographic_prior.py) across all 106 Aozora works (11,723 windows).

Target: max absolute difference < 1e-12 across all 31 dimensions.
Output: docs/handoff/v10/ortho_v2_parity_result.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

USIR = Path("c:/dev/usir")
for p in ("python", "scripts", "experiments"):
    sys.path.insert(0, str(USIR / p))

from fable_v5_grand_shootout import load_works  # noqa: E402
from usir_py import style_atlas as sa  # noqa: E402
from usir_py.orthographic_prior import (  # noqa: E402
    ORTHO_EXT_V2_NAMES,
    NARR_V2_NAMES,
    ortho_matrix_ext_v2,
    ortho_matrix_narr_v2,
)

EXE = Path("C:/dev/nishiki/target/release/nishiki-hardcore.exe")
OUT_DIR = USIR / "docs" / "handoff" / "v10"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "ortho_v2_parity_result.json"


def main() -> None:
    t0 = time.time()
    works = load_works()
    print(f"Loaded {len(works)} works")

    wins = []
    work_spans = {}
    for w in works:
        w_wins = sa.make_windows(w["text"])
        work_spans[w["id"]] = (len(wins), len(wins) + len(w_wins))
        wins.extend(w_wins)

    n_windows = len(wins)
    print(f"Total windows: {n_windows}")
    assert n_windows == 11723, f"Expected 11723 windows, got {n_windows}"

    # 1. Python calculation
    t_py0 = time.time()
    print("Computing Python ortho_v2...")
    py_ext = ortho_matrix_ext_v2(wins)
    py_narr = ortho_matrix_narr_v2(wins)
    py_v2 = np.hstack([py_ext, py_narr])
    t_py = time.time() - t_py0
    print(f"Python ortho_v2 computed in {t_py:.2f}s, shape: {py_v2.shape}")

    # 2. Rust calculation via nishiki-hardcore ortho
    t_rs0 = time.time()
    print("Running Rust nishiki-hardcore ortho...")
    job_file = USIR / "_parity_job.json"
    out_file = USIR / "_parity_out.json"
    job_file.write_text(json.dumps({"texts": wins, "m1": False}, ensure_ascii=False), encoding="utf-8")

    r = subprocess.run([str(EXE), "ortho", str(job_file), str(out_file)], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, f"Rust execution failed: {r.stderr}"
    ro = json.loads(out_file.read_text(encoding="utf-8"))
    job_file.unlink(missing_ok=True)
    out_file.unlink(missing_ok=True)

    rust_v2 = np.array([x["v2"] for x in ro["rows"]], dtype=float)
    t_rs = time.time() - t_rs0
    print(f"Rust ortho computed in {t_rs:.2f}s, shape: {rust_v2.shape}")

    # 3. Parity verification
    all_names = list(ORTHO_EXT_V2_NAMES) + list(NARR_V2_NAMES)
    assert len(all_names) == 31
    assert py_v2.shape == rust_v2.shape == (n_windows, 31)

    abs_diff = np.abs(py_v2 - rust_v2)
    max_abs_diff_overall = float(np.max(abs_diff))
    max_diff_per_dim = {name: float(np.max(abs_diff[:, i])) for i, name in enumerate(all_names)}
    mean_diff_per_dim = {name: float(np.mean(abs_diff[:, i])) for i, name in enumerate(all_names)}

    print(f"Overall max absolute difference: {max_abs_diff_overall:.4e}")
    assert max_abs_diff_overall < 1e-12, f"Parity failure: {max_abs_diff_overall} >= 1e-12"

    result = {
        "status": "PASS",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_works": len(works),
        "n_windows": n_windows,
        "n_dims": 31,
        "threshold": 1e-12,
        "max_abs_diff_overall": max_abs_diff_overall,
        "is_within_threshold": bool(max_abs_diff_overall < 1e-12),
        "max_diff_per_dim": max_diff_per_dim,
        "mean_diff_per_dim": mean_diff_per_dim,
        "runtime_sec": {
            "python": round(t_py, 2),
            "rust": round(t_rs, 2),
            "total": round(time.time() - t0, 2),
        },
    }

    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote parity result to {OUT_JSON}")


if __name__ == "__main__":
    main()
