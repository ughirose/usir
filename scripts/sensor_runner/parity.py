"""Phase 1.5 Stage 1 parity harness: warm-cache partial recomputation vs cold-cache full recomputation.

For one work and one edited scene:
  warm-up : cold pipeline on the ORIGINAL text into cache C0 (fills L2 and L1)
  warm    : copy of C0, pipeline on the EDITED text
  cold    : empty cache (ledger only), pipeline on the EDITED text
Pipeline = nishiki segment -> [prepare -> annotate -> ingest]* until nothing pending -> merge (L1) -> nishiki diagnose.
The annotator is a deterministic stub (a pure function of the pack: scene index, scene text and the structured prior
state), so any wrong reuse of a stale entry changes the output bytes. Real strong-model calls are external sessions and
are only counted (one pack ingested = one call), never timed here.
Compared: merged sensor.v2 annotations (bytes as written by write_json) and `nishiki diagnose --format json` stdout (bytes).

Usage: python parity.py [--work ID] [--where tail|mid] [--variant same|state_change] [--json-out PATH]
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cache import Cache  # noqa: E402
from common import dumps, find_nishiki, load_work, read_json, segment, write_json  # noqa: E402
from ingest import ingest_file  # noqa: E402
from keys import scene_text_hash  # noqa: E402
from merge import merge_work  # noqa: E402
from prepare import prepare_work  # noqa: E402

INSERT = "そのとき、遠くで風の音が少しだけ強くなった。"


def edit_text(file_text: str, scene_text: str, where: str) -> str:
    """Insert one sentence into a scene: 'tail' = at the end of the scene, 'mid' = after its first sentence."""
    at = file_text.index(scene_text)
    if where == "tail":
        pos = at + len(scene_text)
    else:
        pos = at + scene_text.index("。") + 1
    return file_text[:pos] + INSERT + file_text[pos:]


def make_stub(seed_by_index: dict[int, dict[str, Any]], edited_text_hash: str | None, variant: str):
    def stub(pack: dict[str, Any]) -> dict[str, Any]:
        e = copy.deepcopy(seed_by_index[pack["scene_index"]])
        e["scene_hash"] = pack["scene_hash"]
        e["provenance"] = "seed:stub"
        e["annotator"] = "stub:" + pack["l2_parts"]["prior_state_hash"][:12]
        if variant == "state_change" and pack["l2_parts"]["scene_text_hash"] == edited_text_hash:
            if len(e["present"]) >= 2:
                e["present"] = e["present"][:-1]
            else:
                e["pov_character"] = None if e["pov_character"] else e["pov_character"]
                e["present"] = []
        return e

    return stub


def run_pipeline(nis: str, work_id: str, text_path: Path, cache: Cache, local_path: Path, stub, tmp: Path, use_l1: bool = True) -> dict[str, Any]:
    work = load_work(work_id)
    t: dict[str, float] = {}
    t0 = time.perf_counter()
    seg = segment(nis, text_path)
    t["segment"] = time.perf_counter() - t0

    calls, rounds, recomputed = 0, 0, []
    t0 = time.perf_counter()
    while True:
        r = prepare_work(work_id, cache, seg=seg, packs_dir=tmp / "packs", local_path=local_path, defer_blocked=True)
        if not r["packs_written"]:
            break
        rounds += 1
        for p in r["packs_written"]:
            pack = read_json(Path(p["pack"]))
            out = tmp / "out" / f"{pack['scene_hash'][:12]}.strong.json"
            write_json(out, stub(pack))
            res = ingest_file(out, cache, tmp / "packs", tmp / "rejects")
            if not res["accepted"]:
                raise RuntimeError(f"stub entry rejected for scene {pack['scene_index']}: {res['reasons']}")
            calls += 1
            recomputed.append(pack["scene_index"])
    t["prepare_annotate_ingest"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    recs, rep = merge_work(work_id, cache, seg, local_path=local_path, use_l1=use_l1)
    t["merge"] = time.perf_counter() - t0
    ann_bytes = dumps(recs).encode("utf-8")

    chars = tmp / "characters.json"
    ann = tmp / "annotations.json"
    write_json(chars, work["characters"])
    ann.write_bytes(ann_bytes)
    t0 = time.perf_counter()
    d = subprocess.run([nis, "diagnose", str(text_path), "--format", "json", "--characters", str(chars), "--annotations", str(ann)], capture_output=True, check=True)
    t["diagnose"] = time.perf_counter() - t0
    t["total"] = sum(t.values())
    return {"annotations": ann_bytes, "diagnose": d.stdout, "llm_calls": calls, "rounds": rounds, "l2_recomputed_scenes": sorted(recomputed),
            "l1": rep["l1"], "l2": rep["l2"], "seconds": {k: round(v, 4) for k, v in t.items()}, "scene_count": len(seg["scenes"])}


def run_case(nis: str, work_id: str, scene_index: int, where: str, variant: str) -> dict[str, Any]:
    work = load_work(work_id)
    orig_path = work["text_abs"]
    orig_text = orig_path.read_text(encoding="utf-8")
    seg0 = segment(nis, orig_path)
    scene = seg0["scenes"][scene_index]
    new_text = edit_text(orig_text, scene["scene_text"], where)
    base = Path(tempfile.mkdtemp(prefix="phase15_"))
    try:
        edited = base / "edited.txt"
        edited.write_text(new_text, encoding="utf-8", newline="")
        seg1 = segment(nis, edited)
        assert len(seg1["scenes"]) == len(seg0["scenes"]), "edit changed the scene count"
        changed = [i for i, (a, b) in enumerate(zip(seg0["scenes"], seg1["scenes"])) if a["scene_hash"] != b["scene_hash"]]
        assert changed == [scene_index], f"edit must change exactly scene {scene_index}, changed={changed}"

        # local-side utterances: edited scene inherits the original scene's local annotations (the edit adds no quotes)
        lp = Path(__file__).resolve().parent / "local" / f"{work_id}.utterances.json"
        local = read_json(lp)
        local["scenes"][seg1["scenes"][scene_index]["scene_hash"]] = local["scenes"][scene["scene_hash"]]
        local_path = base / "local.json"
        write_json(local_path, local)

        seeds = Cache()  # committed seed cache -> per-index seed entries for the stub
        seed_by_index = {}
        for rv in __import__("resolve").resolve_work(work_id, seeds, seg0["scenes"], work["characters"]):
            seed_by_index[rv.index] = rv.entry
        ledger = seeds.ledger(work_id)
        stub = make_stub(seed_by_index, scene_text_hash(seg1["scenes"][scene_index]["scene_text"]), variant)

        def fresh_cache(name: str) -> Cache:
            c = Cache(base / name, legacy_adopt=False)
            c.put_ledger(work_id, ledger, note="parity harness")
            return c

        c0 = fresh_cache("c0")
        warmup = run_pipeline(nis, work_id, orig_path, c0, local_path, stub, base / "t0")
        shutil.copytree(base / "c0", base / "cw")
        warm = run_pipeline(nis, work_id, edited, Cache(base / "cw", legacy_adopt=False), local_path, stub, base / "t1")
        cold = run_pipeline(nis, work_id, edited, fresh_cache("cc"), local_path, stub, base / "t2")
        no_l1 = run_pipeline(nis, work_id, edited, fresh_cache("cn"), local_path, stub, base / "t3", use_l1=False)
        # original text, warm re-run (nothing changed): must recompute nothing
        again = run_pipeline(nis, work_id, orig_path, Cache(base / "c0", legacy_adopt=False), local_path, stub, base / "t4")

        res = {
            "work_id": work_id, "edited_scene": scene_index, "where": where, "variant": variant, "scene_count": warm["scene_count"],
            "parity_annotations_bytes_equal": warm["annotations"] == cold["annotations"],
            "parity_diagnose_bytes_equal": warm["diagnose"] == cold["diagnose"],
            "parity_cold_l1off_equal": no_l1["annotations"] == cold["annotations"] and no_l1["diagnose"] == cold["diagnose"],
            "unchanged_rerun_equal_warmup": again["annotations"] == warmup["annotations"] and again["diagnose"] == warmup["diagnose"],
            "unchanged_rerun_llm_calls": again["llm_calls"],
            "annotations_bytes": len(warm["annotations"]), "diagnose_bytes": len(warm["diagnose"]),
        }
        for name, r in (("warm", warm), ("cold", cold)):
            res[name] = {k: r[k] for k in ("llm_calls", "rounds", "l2_recomputed_scenes", "l1", "l2", "seconds")}
        return res
    finally:
        shutil.rmtree(base, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="000035_000301")
    ap.add_argument("--scene", type=int, default=3)
    ap.add_argument("--where", choices=["tail", "mid"], default="tail")
    ap.add_argument("--variant", choices=["same", "state_change"], default="same")
    ap.add_argument("--json-out")
    ap.add_argument("--nishiki")
    a = ap.parse_args()
    res = run_case(find_nishiki(a.nishiki), a.work, a.scene, a.where, a.variant)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.json_out:
        write_json(Path(a.json_out), res)
    ok = res["parity_annotations_bytes_equal"] and res["parity_diagnose_bytes_equal"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
