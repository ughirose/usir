"""Offline recompute of the Phase 16 Climax direction metrics with the A1 `_direction_metrics` (Climax scored without discharge).

Inputs : docs/handoff/v6/p16*.json, p175_all.json (paragraph_texts, wells and discharge norms are stored).
Output : docs/handoff/v6/phase16_n5_recomputed.json
Run    : cd python; uv run python ../scripts/recompute_phase16_n5.py

Only the discharge NORMS are stored, not the vectors. They decide the steered set (norm > 1e-9) that defines q_base, so the
text-based metrics (dS, dF, delta_sf, d_realised, d_realised_reg, cos_ref) are exactly reproducible. cos_p_dq / d_target need the
vector and are NOT recomputed (the stored value is carried over where it exists).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
V6 = ROOT / "docs" / "handoff" / "v6"
sys.path.insert(0, str(ROOT / "python"))
spec = importlib.util.spec_from_file_location("run_v4_recompute", ROOT / "python" / "colab" / "run_v4_bidirectional_experiment.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

FILES = ["p16_3b_A", "p16_3b_B", "p16b_A", "p16b_B", "p16b_C", "p16b_D", "p16_15b_s3", "p175_all"]
COND_NAME = {"new_c": "Steered", "ctrl": "Ctrl"}
# Source of each (model, cond) cell used for the n=5 stats: 3B Ctrl comes from the earlier p16_3b pass, everything else from p16b
STAT_SRC = {
    ("3B", "Steered"): ["p16b_A", "p16b_B"], ("3B", "Ctrl"): ["p16_3b_A", "p16_3b_B"],
    ("1.5B", "Steered"): ["p16b_A", "p16b_B"], ("1.5B", "Ctrl"): ["p16b_A", "p16b_B"],
    ("7B", "Steered"): ["p16b_C", "p16b_D"], ("7B", "Ctrl"): ["p16b_C", "p16b_D"],
}
SEEDS = [3, 42, 137, 256, 999]


def model_tag(name: str) -> str:
    return "1.5B" if "1.5B" in name else ("3B" if "3B" in name else ("7B" if "7B" in name else name))


def recompute(run: dict) -> dict:
    texts = run["paragraph_texts"]
    wells = run.get("wells") or []
    names = {0: "Narrative", 1: "Dialogue", 2: "Climax"}
    norms = run["discharged"]
    paras = []
    for i, t in enumerate(texts):
        dn = float(norms[i])
        paras.append({"index": i, "well": names[wells[i]], "text": t, "discharged_momentum": ([dn] + [0.0] * 11) if dn > 0 else [0.0] * 12})
    new = mod._direction_metrics(paras)
    old = {d["index"]: d for d in run["direction"]}
    out = []
    for d in new:
        if d["well"] != "Climax":
            continue
        o = old.get(d["index"], {})
        out.append({
            "index": d["index"], "discharge_norm": float(norms[d["index"]]), "was_scored_before": "delta_sf" in o,
            "delta_sf": d.get("delta_sf"), "d_realised": d.get("d_realised"), "d_realised_reg": d.get("d_realised_reg"),
            "dS": d.get("dS"), "dF": d.get("dF"), "cos_ref": d.get("cos_ref"),
            "cos_p_dq_stored": o.get("cos_p_dq"),
            "stored": {k: o.get(k) for k in ("delta_sf", "d_realised", "d_realised_reg", "dS", "dF")},
            "mean_sentence_len": d.get("mean_sentence_len"),
        })
    return {"climax": out}


def main() -> None:
    runs = []
    for f in FILES:
        data = json.loads((V6 / f"{f}.json").read_text(encoding="utf-8"))
        for key, job in data.items():
            for cond_key in ("new_c", "ctrl"):
                r = job.get(cond_key) if isinstance(job, dict) else None
                if not isinstance(r, dict) or "paragraph_texts" not in r:
                    continue
                rc = recompute(dict(r, wells=job["wells"]))
                runs.append({"file": f, "job": key, "model": model_tag(job["model"]), "seed": job["seed"], "cond": COND_NAME[cond_key], **rc})

    # (1) consistency with the stored values for Climax paragraphs that were scored before (discharge > 0)
    diffs = []
    for r in runs:
        for c in r["climax"]:
            if c["was_scored_before"]:
                for k, sv in c["stored"].items():
                    if sv is not None and c[k] is not None:
                        diffs.append(abs(sv - c[k]))
    consistency = {
        "n_scored_before_values": len(diffs), "max_abs_diff": max(diffs) if diffs else None,
        "n_climax_total": sum(len(r["climax"]) for r in runs),
        "n_climax_newly_scored": sum(1 for r in runs for c in r["climax"] if not c["was_scored_before"]),
    }

    # (2) per-run aggregate (mean over Climax paragraphs) and paired Steered - Ctrl stats, n = 5 seeds per model
    def agg(r, k):
        v = [c[k] for c in r["climax"] if c[k] is not None]
        return float(np.mean(v)) if v else None

    cells = {}
    for (m, cond), srcs in STAT_SRC.items():
        for seed in SEEDS:
            hit = [r for r in runs if r["model"] == m and r["cond"] == cond and r["seed"] == seed and r["file"] in srcs]
            cells[(m, cond, seed)] = hit[0] if len(hit) == 1 else None
            if len(hit) != 1:
                cells[(m, cond, seed)] = {"missing_or_ambiguous": len(hit)}
    paired = {}
    for m in ("1.5B", "3B", "7B"):
        paired[m] = {}
        for metric in ("delta_sf", "d_realised_reg", "d_realised", "cos_ref"):
            a, b, used = [], [], []
            for s in SEEDS:
                rs, rc = cells[(m, "Steered", s)], cells[(m, "Ctrl", s)]
                if "climax" not in rs or "climax" not in rc:
                    continue
                x, y = agg(rs, metric), agg(rc, metric)
                if x is None or y is None:
                    continue
                a.append(x); b.append(y); used.append(s)
            a, b = np.asarray(a), np.asarray(b)
            diff = a - b
            row = {"seeds": used, "n": len(used), "steered": a.round(4).tolist(), "ctrl": b.round(4).tolist()}
            if len(used) >= 2:
                sd = float(np.std(diff, ddof=1))
                t, p = stats.ttest_rel(a, b)
                row.update({
                    "steered_mean": round(float(a.mean()), 4), "steered_sd": round(float(a.std(ddof=1)), 4),
                    "ctrl_mean": round(float(b.mean()), 4), "ctrl_sd": round(float(b.std(ddof=1)), 4),
                    "mean_diff": round(float(diff.mean()), 4), "sd_diff": round(sd, 4),
                    "cohen_d_paired": round(float(diff.mean() / sd), 3) if sd > 0 else None,
                    "t": round(float(t), 3), "df": len(used) - 1, "p": float(p),
                })
            paired[m][metric] = row
    # Steered cos_p_dq (stored; Ctrl has no released momentum vector) for reference
    cos_stored = {}
    for m in ("1.5B", "3B", "7B"):
        v = []
        for s in SEEDS:
            rs = cells[(m, "Steered", s)]
            vals = [c["cos_p_dq_stored"] for c in rs.get("climax", []) if c["cos_p_dq_stored"] is not None]
            if vals:
                v.append(float(np.mean(vals)))
        cos_stored[m] = {"n": len(v), "mean": round(float(np.mean(v)), 4) if v else None, "values": [round(x, 4) for x in v]}

    out = {
        "note": "Recomputed with A1 _direction_metrics (Climax scored regardless of discharge). Per-run value = mean over its Climax paragraphs. "
                "cos_p_dq cannot be recomputed (discharge vectors are not stored); stored Steered values are listed in steered_cos_p_dq_stored.",
        "consistency_vs_stored": consistency,
        "stat_sources": {f"{m}|{c}": v for (m, c), v in STAT_SRC.items()},
        "paired_steered_minus_ctrl": paired,
        "steered_cos_p_dq_stored": cos_stored,
        "runs": runs,
    }
    (V6 / "phase16_n5_recomputed.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"consistency": consistency, "paired": paired, "cos_p_dq_stored": cos_stored}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
