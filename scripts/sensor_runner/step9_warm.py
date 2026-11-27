"""Phase 1.5 Stage 2 Step 9: warm run of the committed seed with SENSOR_ANNOTATOR_MODEL_ID UNSET (code default only).
Copies the committed cache to a sandbox, runs production prepare (expect 0 packs = 0 annotation calls) and merge + nishiki diagnose for all 7 works,
and compares the output sha256 with the Step 8 cold run (reference file passed with --ref, default step2_work/s8a/cold/summary.json).
Writes results/phase15_stage9 warm json. Measurement only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

assert "SENSOR_ANNOTATOR_MODEL_ID" not in os.environ, "this check must run with the env var unset"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from cache import Cache  # noqa: E402
from common import all_work_ids, dumps, find_nishiki, load_work, segment, write_json  # noqa: E402
from merge import merge_work  # noqa: E402
from prepare import prepare_work  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=str(HERE / "step2_work" / "s8a" / "cold" / "summary.json"))
    ap.add_argument("--out", default=str(HERE / "results" / "phase15_stage2_step9_warm.json"))
    a = ap.parse_args()
    ref = json.loads(Path(a.ref).read_text(encoding="utf-8"))
    nis = find_nishiki(None)
    tmp = Path(tempfile.mkdtemp(prefix="step9_warm_"))
    shutil.copytree(HERE / "cache", tmp / "cache")
    cache = Cache(tmp / "cache", legacy_adopt=False)
    res = {"model_id_used": common.ANNOTATION_MODEL_ID, "env_unset": True, "works": {}, "annotation_calls": 0}
    for w in all_work_ids():
        work = load_work(w)
        seg = segment(nis, work["text_abs"])
        r = prepare_work(w, cache, seg=seg, packs_dir=tmp / "packs", defer_blocked=True)
        recs, rep = merge_work(w, cache, seg)
        ann = dumps(recs).encode("utf-8")
        write_json(tmp / f"{w}.characters.json", work["characters"])
        (tmp / f"{w}.annotations.json").write_bytes(ann)
        d = subprocess.run([nis, "diagnose", str(work["text_abs"]), "--format", "json", "--characters", str(tmp / f"{w}.characters.json"),
                            "--annotations", str(tmp / f"{w}.annotations.json")], capture_output=True, check=True).stdout
        row = {"packs_written": len(r["packs_written"]), "pending": r["pending"], "l2": rep["l2"], "errors": len(rep["errors"]),
               "annotations_sha256": hashlib.sha256(ann).hexdigest(), "diagnose_sha256": hashlib.sha256(d).hexdigest()}
        row["annotations_equal_ref"] = row["annotations_sha256"] == ref[w]["annotations_sha256"]
        row["diagnose_equal_ref"] = row["diagnose_sha256"] == ref[w]["diagnose_sha256"]
        res["annotation_calls"] += row["packs_written"]
        res["works"][w] = row
    res["all_equal"] = all(v["annotations_equal_ref"] and v["diagnose_equal_ref"] for v in res["works"].values())
    shutil.rmtree(tmp, ignore_errors=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "works"}, ensure_ascii=False))
    for w, v in res["works"].items():
        print(w, v["packs_written"], v["l2"], v["annotations_equal_ref"], v["diagnose_equal_ref"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
