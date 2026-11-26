"""Phase 1.5 Stage 2 Step 6, experiment 2 (no LLM): does the first-writer-wins baseline of a character change the diagnosis?

Uses the Step 5 `ep` run (run dir step2_work/s5a): its L2 entries (annotations) stay as they are; only the ledger baseline that merge
applies to trait_changes is replaced by the vectors the annotators gave in scene 0 / 1 / 2 (taken from phase15_stage2_step5_results.json:
kept vectors = scene 0, discarded repeats = scenes 1 and 2). For each variant: merge_work (production code) -> annotations -> `nishiki diagnose`.
Compares the annotations and the diagnose output per scene. Writes results/phase15_stage2_step6_exp2.json."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("SENSOR_ANNOTATOR_MODEL_ID", "claude-code-subagent:alias=sonnet(exact-model-id-unverified)")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cache import Cache  # noqa: E402
from common import dumps, find_nishiki, load_work, segment, write_json  # noqa: E402
from merge import merge_work  # noqa: E402

W = "001801_001786"
RUN = HERE / "step2_work" / "s5a"
OUT = HERE / "step2_work" / "s6b"


class LedgerOverride(Cache):
    """Resolution (keys) keeps using the committed ledger; only the ledger handed to merge_scene is replaced."""
    override: dict = {}

    def ledger(self, work_id):  # noqa: D401
        led = dict(super().ledger(work_id))
        led.update(self.override)
        return led


def vectors() -> dict:
    d = json.loads((HERE / "results" / "phase15_stage2_step5_results.json").read_text(encoding="utf-8"))
    hist = d["epoch_state"][f"{W}.epoch.json"]["history"][0]
    v = {"onna": {}, "shinichi": {}}
    for r in hist["discarded_repeats"]:
        v[r["character"]][r["scene_hash"][:12]] = r["vector"]
        v[r["character"]][r["kept_scene_hash"][:12]] = r["kept_vector"]
    return v


def main() -> int:
    nis = find_nishiki(None)
    work = load_work(W)
    seg = segment(nis, work["text_abs"])
    hashes = [s["scene_hash"][:12] for s in seg["scenes"]]
    vec = vectors()
    by_scene = {ch: [vec[ch].get(h) for h in hashes] for ch in vec}
    print(json.dumps(by_scene))
    cache = LedgerOverride(RUN / "cache", legacy_adopt=False)
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"work": W, "vectors_by_scene": by_scene, "variants": {}}
    outs = {}
    for k in range(3):
        LedgerOverride.override = {ch: by_scene[ch][k] for ch in by_scene}
        recs, rep = merge_work(W, cache, seg)
        assert not rep["errors"] and not rep["uncached"], rep
        ann = OUT / f"base{k}.annotations.json"
        ann.write_bytes(dumps(recs).encode("utf-8"))
        chars = OUT / f"{W}.characters.json"
        write_json(chars, work["characters"])
        d = subprocess.run([nis, "diagnose", str(work["text_abs"]), "--format", "json", "--characters", str(chars), "--annotations", str(ann)], capture_output=True, check=True)
        (OUT / f"base{k}.diagnose.json").write_bytes(d.stdout)
        outs[k] = {"recs": recs, "diag": json.loads(d.stdout.decode("utf-8")), "ann_bytes": ann.read_bytes(), "diag_bytes": d.stdout}
        res["variants"][f"scene{k}"] = {"baseline": LedgerOverride.override, "annotations_bytes": len(outs[k]["ann_bytes"]), "diagnose_bytes": len(d.stdout)}
    res["byte_equal_annotations"] = {f"0v{k}": outs[0]["ann_bytes"] == outs[k]["ann_bytes"] for k in (1, 2)}
    res["byte_equal_diagnose"] = {f"0v{k}": outs[0]["diag_bytes"] == outs[k]["diag_bytes"] for k in (1, 2)}

    def diff(a, b, path=""):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                yield from diff(a.get(key), b.get(key), f"{path}/{key}")
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for i, (x, y) in enumerate(zip(a, b)):
                yield from diff(x, y, f"{path}[{i}]")
        elif a != b:
            yield {"path": path, "a": a, "b": b}

    res["annotation_diffs"] = {f"0v{k}": list(diff(outs[0]["recs"], outs[k]["recs"])) for k in (1, 2)}
    res["diagnose_diffs"] = {f"0v{k}": list(diff(outs[0]["diag"], outs[k]["diag"])) for k in (1, 2)}
    res["diagnose_top_keys"] = list(outs[0]["diag"]) if isinstance(outs[0]["diag"], dict) else None
    write_json(HERE / "results" / "phase15_stage2_step6_exp2.json", res)
    print(json.dumps({k: (v if not isinstance(v, dict) else {a: (len(b) if isinstance(b, list) else b) for a, b in v.items()}) for k, v in res.items() if k.startswith(("byte", "annotation_d", "diagnose_d"))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
