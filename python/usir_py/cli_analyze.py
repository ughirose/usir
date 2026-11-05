"""CLI: analyze a Japanese text against the 24-D style atlas.

    python -m usir_py.cli_analyze <file.txt> [--atlas artifacts/style_atlas] [--out x.usir] [--json]

Prints the 24-D (q, p) profile, the 5-phase occupancy diagnosis and a ranking of the atlas authors by
GMM log-likelihood (sum and mean over 600-char / stride-300 windows).  With --out, a Tier-M genome
distilled from the input text is written after it passes the clean-room gate (no >= 7-char
n-gram of the input, no banned keywords).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import style_atlas as sa

Q_NAMES = ["Sg-01 Gr", "Sg-02 Cr", "Sg-03 Pr", "Sg-04 Jt", "Sg-05 Cp", "Sg-06 Sy",
           "Sg-07 Cs", "Sg-08 Lp", "Sg-09 Dy", "Sg-10 At", "Sg-11 Ma", "Sg-12 Ru"]
P_NAMES = ["p_" + n for n in Q_NAMES]


def _resolve_atlas_dir(arg: str) -> Path:
    p = Path(arg)
    if p.is_dir():
        return p
    alt = Path(__file__).resolve().parents[2] / arg
    return alt if alt.is_dir() else p


def _labels(atlas_dir: Path) -> Dict[str, str]:
    meta = atlas_dir / "atlas.json"
    if not meta.exists():
        return {}
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
        return {a: v.get("label_ja", a) for a, v in data.get("authors", {}).items()}
    except (OSError, ValueError):
        return {}


def analyze(text: str, atlas_dir: Optional[Path] = None, out_path: Optional[Path] = None) -> Dict[str, Any]:
    F = sa.featurize_text(text)
    if len(F["Z"]) == 0:
        raise ValueError("text is too short to analyze (need >= 20 characters)")
    prof = sa.profile_from_features(F)
    groups = {
        "I_Flux": float(np.mean([prof["mean_q"][i] for i in (0, 4, 8)])),
        "II_Lattice": float(np.mean([prof["mean_q"][i] for i in (1, 5, 9)])),
        "III_Void": float(np.mean([prof["mean_q"][i] for i in (2, 6, 10)])),
        "IV_Shock": float(np.mean([prof["mean_q"][i] for i in (3, 7, 11)])),
    }
    occ = prof["phase_occupancy"]
    result: Dict[str, Any] = {
        "n_chars": len(text),
        "n_windows": prof["n_windows"],
        "window_chars": sa.WINDOW_CHARS,
        "stride_chars": sa.STRIDE_CHARS,
        "q": dict(zip(Q_NAMES, prof["mean_q"])),
        "p": dict(zip(P_NAMES, prof["mean_p"])),
        "groups": groups,
        "p_norm_mean": prof["p_norm_mean"],
        "cv_l": {"mean": prof["cv_l_mean"], "std": prof["cv_l_std"]},
        "phase_occupancy": occ,
        "dominant_phase": max(occ, key=occ.get),
        "ranking": [],
    }

    if atlas_dir is not None and atlas_dir.is_dir():
        gmms = sa.load_atlas_gmms(atlas_dir)
        labels = _labels(atlas_dir)
        rows = []
        for aid, g in gmms.items():
            ll = g.loglik(F["Z"])
            rows.append({"author_id": aid, "label": labels.get(aid, aid),
                         "sum_loglik": float(ll.sum()), "mean_loglik": float(ll.mean())})
        rows.sort(key=lambda r: r["sum_loglik"], reverse=True)
        result["ranking"] = rows

    if out_path is not None:
        works = [("input", F["Z"])]
        blob = sa.distill_genome(works, shrinkage="scaled")
        audit = sa.audit_cleanroom_binary(blob, text)
        result["out"] = {"path": str(out_path), "cleanroom_passed": audit["passed"], "byte_length": audit["byte_length"]}
        if audit["passed"]:
            Path(out_path).write_bytes(blob)
        else:
            result["out"]["written"] = False
    return result


def format_text(r: Dict[str, Any]) -> str:
    lines = [f"文字数 {r['n_chars']} / 窓数 {r['n_windows']} (窓 {r['window_chars']} 字, 刻み {r['stride_chars']} 字)", "", "[24次元 平均]"]
    for k, v in r["q"].items():
        lines.append(f"  q {k}: {v:.4f}")
    for k, v in r["p"].items():
        lines.append(f"  p {k[2:]}: {v:.4f}")
    g = r["groups"]
    lines += ["", "[4族] " + "  ".join(f"{k}={v:.3f}" for k, v in g.items()),
              f"||p|| 平均={r['p_norm_mean']:.3f}  CV_L 平均={r['cv_l']['mean']:.3f} (窓間 std {r['cv_l']['std']:.3f})",
              "", "[5相 占有率] (24次元重心への最近傍; 重心は約280字の黄金コーパス由来で、窓長600字とは尺度が異なる)"]
    for ph, v in r["phase_occupancy"].items():
        lines.append(f"  {ph}: {v:.3f}")
    lines.append(f"  最多: {r['dominant_phase']}")
    if r["ranking"]:
        lines += ["", "[近傍作家 (GMM 対数尤度, 窓合計の降順)]"]
        for i, row in enumerate(r["ranking"], 1):
            lines.append(f"  {i:2d}. {row['label']} ({row['author_id']})  sum={row['sum_loglik']:.1f}  mean/window={row['mean_loglik']:.2f}")
    if "out" in r:
        o = r["out"]
        lines += ["", f"[.usir 出力] {o['path']} cleanroom={'通過' if o['cleanroom_passed'] else '不通過(未書込)'} ({o['byte_length']} bytes)"]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m usir_py.cli_analyze", description=__doc__.split("\n")[0])
    ap.add_argument("file", help="UTF-8 text file")
    ap.add_argument("--atlas", default="artifacts/style_atlas", help="directory of <author_id>.usir (+ atlas.json)")
    ap.add_argument("--out", default=None, help="write a Tier-M .usir distilled from this text")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = ap.parse_args(argv)

    text = Path(args.file).read_text(encoding="utf-8")
    atlas_dir = _resolve_atlas_dir(args.atlas)
    if not atlas_dir.is_dir():
        print(f"warning: atlas directory not found: {args.atlas}; ranking skipped", file=sys.stderr)
    res = analyze(text, atlas_dir if atlas_dir.is_dir() else None, Path(args.out) if args.out else None)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1) if args.json else format_text(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
