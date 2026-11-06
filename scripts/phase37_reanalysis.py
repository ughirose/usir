#!/usr/bin/env python
"""Phase 37: local re-analysis of the Phase 36 result JSONs (no new generation).

(1) Sentence length by paragraph type. The decoder runs the steady profile (genome band) on paragraphs whose director
    well is not 2 and the pulse profile on the Climax paragraph (well 2). The pre-registered Phase 36 criterion used
    the whole-text mean; this split is a post-hoc analysis. Sentence lengths use orthographic_prior.sentence_lengths,
    the same splitter as ortho[0].
(2) Phase C sensitivity to the fragment cut by the token limit: the QDA Deadzone and CV_L < 0.28 counts are
    recomputed on each text with the trailing fragment after the last sentence end removed.

Output: docs/handoff/v10/p37_reanalysis.json
Usage:  cd python && uv run python ../scripts/phase37_reanalysis.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from usir_py.extractor_24d import extract_usir_24d  # noqa: E402
from usir_py.lora_grpo_forge import strip_incomplete_tail  # noqa: E402
from usir_py.manifold_geodesic import compute_three_phase_qda_distance  # noqa: E402
from usir_py.orthographic_prior import sentence_lengths  # noqa: E402

V10 = ROOT / "docs" / "handoff" / "v10"
OUT = V10 / "p37_reanalysis.json"
ENDS = "。！？!?"


def by_paragraph_type(path: Path) -> dict:
    runs = json.loads(path.read_text(encoding="utf-8"))["runs"]
    per_run, pooled = {}, {}
    for k, r in runs.items():
        if r.get("error"):
            continue
        assert len(r["wells"]) == len(r["paragraph_texts"]), k
        S = [x for p, w in zip(r["paragraph_texts"], r["wells"]) if w != 2 for x in sentence_lengths(p)]
        C = [x for p, w in zip(r["paragraph_texts"], r["wells"]) if w == 2 for x in sentence_lengths(p)]
        per_run[k] = {"condition": r["condition"], "plot": r["plot"], "seed": r["seed"],
                      "steady_mean": st.mean(S) if S else None, "steady_n": len(S),
                      "climax_mean": st.mean(C) if C else None, "climax_n": len(C),
                      "steady_max_chars": (r.get("genome_proposal") or {}).get("steady_max_chars"),
                      "steady_over_max": sum(1 for x in S if r.get("genome_proposal") and x > r["genome_proposal"]["steady_max_chars"])}
        a = pooled.setdefault(r["condition"], {"S": [], "C": []})
        a["S"] += S
        a["C"] += C
    cond = {c: {"steady_mean": st.mean(a["S"]), "steady_n": len(a["S"]), "climax_mean": st.mean(a["C"]),
                "climax_n": len(a["C"]), "climax_sentence_share": len(a["C"]) / (len(a["S"]) + len(a["C"]))}
            for c, a in pooled.items()}
    return {"source": str(path.relative_to(ROOT)).replace("\\", "/"), "conditions": cond, "runs": per_run}


def tail_sensitivity(path: Path) -> dict:
    stages = json.loads(path.read_text(encoding="utf-8"))["stages"]
    sets = [("base", stages["base"]["records"])]
    sets += [(c, v["records"]) for c, v in stages["steering"]["conditions"].items()]
    sets.append(("lora_merged", stages["grpo_lora"]["lora_merged"]["records"]))
    out = {}
    for name, recs in sets:
        dz = lambda t: compute_three_phase_qda_distance(t, domain="novel")["phase_regime"] == "Phase_I_Deadzone"
        out[name] = {
            "n": len(recs),
            "truncated": sum(1 for r in recs if r["text"].rstrip()[-1:] not in ENDS),
            "deadzone_full": sum(1 for r in recs if dz(r["text"])),
            "deadzone_json": sum(1 for r in recs if r["metrics"]["in_deadzone"]),
            "deadzone_stripped": sum(1 for r in recs if dz(strip_incomplete_tail(r["text"]))),
            "cv_lt_028_json": sum(1 for r in recs if r["metrics"]["in_deadzone_cv"]),
            "cv_lt_028_stripped": sum(1 for r in recs if extract_usir_24d(strip_incomplete_tail(r["text"]))["len_cv"] < 0.28),
            "paragraph_break": sum(1 for r in recs if "\n\n" in r["text"]),
            "cadence_alive": [round(float(r["metrics"]["cadence_alive"]), 4) for r in recs],
        }
    return {"source": str(path.relative_to(ROOT)).replace("\\", "/"), "sets": out}


def main() -> None:
    out = {"schema": "p37_reanalysis_v1",
           "paragraph_type": {f: by_paragraph_type(V10 / f"{f}.json") for f in ("p36_q3_8b", "p36_swallow_8b", "p31b_q3_8b")},
           "phase_c_tail": tail_sensitivity(V10 / "p36c_q3_8b.json")}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for f, v in out["paragraph_type"].items():
        print(f, {c: (round(x["steady_mean"], 2), round(x["climax_mean"], 2), round(x["climax_sentence_share"], 3))
                  for c, x in v["conditions"].items()})
    for k, v in out["phase_c_tail"]["sets"].items():
        print(k, {x: v[x] for x in v if x != "cadence_alive"})
    print("wrote", OUT)


if __name__ == "__main__":
    main()
