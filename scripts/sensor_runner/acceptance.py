"""Acceptance measurement -> results/acceptance.json.

1. every known scene is cached (manifest + cache lookup)
2. merge + nishiki diagnose on every work: 6 lenses ok
3. delta behaviour: copy the cache, remove one scene, prepare must emit exactly that scene's pack
Usage: python acceptance.py [--nishiki BIN]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cache import Cache  # noqa: E402
from common import CACHE_ROOT, all_work_ids, find_nishiki, load_work, segment, write_json  # noqa: E402
from diagnose import all_ok, run_diagnose  # noqa: E402
from merge import merge_work  # noqa: E402
from prepare import prepare_work  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nishiki", default=None)
    a = ap.parse_args()
    nis = find_nishiki(a.nishiki)
    cache = Cache()
    res: dict = {"nishiki": nis, "works": {}}
    tmp = Path(tempfile.mkdtemp())
    tot = {"scenes": 0, "cached": 0, "sasae_spans": 0, "merge_errors": 0, "works_all_lenses_ok": 0}
    for w in all_work_ids():
        work = load_work(w)
        seg = segment(nis, work["text_abs"])
        scenes = seg["scenes"]
        cached = sum(cache.has(s["scene_hash"]) for s in scenes)
        recs, rep = merge_work(w, cache, seg)
        write_json(tmp / f"{w}.annotations.json", recs)
        write_json(tmp / f"{w}.characters.json", work["characters"])
        d = run_diagnose(nis, work["text_abs"], tmp / f"{w}.characters.json", tmp / f"{w}.annotations.json")
        sasae = next(l for l in d["raw"]["lenses"] if l["id"] == "sasae")
        res["works"][w] = {
            "scenes": len(scenes), "cached": cached, "merge_errors": len(rep["errors"]),
            "missing_local_utterances": rep["missing_local_utterances"],
            "local_ids_absent_from_present": rep["local_ids_absent_from_present"],
            "lenses": {k: v["status"] for k, v in d["lenses"].items()}, "all_lenses_ok": all_ok(d),
            "sasae_spans": len(sasae["spans"]),
        }
        tot["scenes"] += len(scenes)
        tot["cached"] += cached
        tot["sasae_spans"] += len(sasae["spans"])
        tot["merge_errors"] += len(rep["errors"])
        tot["works_all_lenses_ok"] += int(all_ok(d))
    res["totals"] = tot

    # delta demo on a copy of the cache
    cc = tmp / "cache"
    shutil.copytree(CACHE_ROOT, cc)
    c2 = Cache(cc)
    w = "000035_000301"
    seg = segment(nis, load_work(w)["text_abs"])
    victim = seg["scenes"][3]
    c2.delete(victim["scene_hash"])
    r = prepare_work(w, c2, seg=seg, packs_dir=tmp / "packs")
    res["delta_demo"] = {
        "work": w, "removed_scene_index": 3, "removed_scene_hash": victim["scene_hash"],
        "pending_after_removal": r["pending"], "packs_written": [p["scene_index"] for p in r["packs_written"]],
        "pack_hashes": [p["scene_hash"] for p in r["packs_written"]],
        "other_works_packs": sum(len(prepare_work(x, c2, seg=segment(nis, load_work(x)["text_abs"]), packs_dir=tmp / "packs")["packs_written"]) for x in all_work_ids() if x != w),
    }
    write_json(Path(__file__).resolve().parent / "results" / "acceptance.json", res)
    print(f"scenes={tot['scenes']} cached={tot['cached']} works_all_lenses_ok={tot['works_all_lenses_ok']}/{len(res['works'])} sasae_spans={tot['sasae_spans']}")
    print("delta:", res["delta_demo"]["packs_written"], "other_works_packs:", res["delta_demo"]["other_works_packs"])
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if tot["scenes"] == tot["cached"] and tot["works_all_lenses_ok"] == len(res["works"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
