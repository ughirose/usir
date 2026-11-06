#!/usr/bin/env python
"""Phase 43: does held-out author identification depend on sharing a transcriber (入力者) with a training work?

Rule fixed in docs/handoff/2026-11-06_phase43_preregistration.md before computing.  Uses the class-mean predictions
of Phase 41 (both held-out sets, 60 works) and the 入力者 / 校正者 columns of the official Aozora index.
Output: docs/handoff/v10/p43_transcriber_confound.json
Usage:  python scripts/transcriber_confound.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_aozora import load_index  # noqa: E402

V10 = ROOT / "docs" / "handoff" / "v10"
OUT = V10 / "p43_transcriber_confound.json"


def main() -> None:
    idx = {(r["人物ID"], r["作品ID"]): r for r in load_index()}
    manifest = json.loads((ROOT / "data" / "aozora_manifest.json").read_text(encoding="utf-8"))
    train = [e for e in (manifest["works"] if isinstance(manifest, dict) else manifest) if e.get("genre", "novel") == "novel"]
    res = json.loads((V10 / "p41_heldout2_nearest_work.json").read_text(encoding="utf-8"))["results"]
    rows = res["set1"]["A_class_mean"]["rows"] + res["set2"]["A_class_mean"]["rows"]

    def people(pid: str, wid: str, col: str) -> set:
        r = idx.get((pid, str(wid)))
        if r is None:
            return set()
        return {x.strip() for x in r[col].replace("、", ",").split(",") if x.strip()}

    out = {"schema": "p43_transcriber_confound_v1", "n_works": len(rows), "missing_index": 0, "by_column": {}}
    for col in ("入力者", "校正者"):
        tr = {}
        for e in train:
            tr.setdefault(e["author_id"], set()).update(people(e["author_id"], e["work_id"], col))
        grp = {"shared": [], "not_shared": []}
        for r in rows:
            p = people(r["author_id"], r["work_id"], col)
            if not p:
                out["missing_index"] += 1 if col == "入力者" else 0
            g = "shared" if p & tr.get(r["author_id"], set()) else "not_shared"
            grp[g].append(r)
        summ = {g: {"n": len(v), "n_correct": sum(x["correct"] for x in v),
                    "accuracy": (sum(x["correct"] for x in v) / len(v)) if v else None} for g, v in grp.items()}
        ok = summ["shared"]["n"] >= 10 and summ["not_shared"]["n"] >= 10
        diff = (summ["shared"]["accuracy"] - summ["not_shared"]["accuracy"]) if ok else None
        summ["difference"] = diff
        summ["verdict"] = ("undecidable" if not ok else "confound_suspected" if diff >= 0.15 else "no_indication")
        summ["per_author"] = {a: {g: f"{sum(x['correct'] for x in v if x['author_id'] == a)}/{sum(x['author_id'] == a for x in v)}"
                                  for g, v in grp.items()} for a in sorted({x["author_id"] for x in rows})}
        out["by_column"][col] = summ
    assert train, "no training works"
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for col, s in out["by_column"].items():
        print(col, {g: s[g] for g in ("shared", "not_shared")}, s["difference"], s["verdict"])
    print("missing index rows:", out["missing_index"], "wrote", OUT)


if __name__ == "__main__":
    main()
