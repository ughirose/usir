"""Phase 1.5 Stage 2 Step 2: measurement tooling for an end-to-end run with real LLM annotators (measurement only).

No cache / key / resolver / epoch / CLI code is changed. This tool drives the production functions (prepare_work, ingest_file,
commit_epoch, merge_work, run_diagnose) against a sandbox cache under step2_work/<run>/cache, so the committed seed cache is
never touched. Annotators are external subagents; they call `stamp` and `ingest` of this file.

  init   --run R [--from-run R0]             create the sandbox cache (ledger files copied from the seed cache, or a copy of run R0's cache)
  prep   --run R [--work W ...] [--text T --local L]   prepare packs (production pack format, defer_blocked) and print them as JSON
  stamp  --run R PID start|end                 timestamp (called by annotators)
  ingest --run R PID                           ingest the annotator's output exactly like `cli.py ingest`; logs the attempt
  epoch  --run R commit|status [--work W ...]  apply pending ledger additions (production commit_epoch)
  final  --run R --name N [--work W ...] [--text T --local L]   merge + diagnose; write bytes and sha256
  flags  --run R [--work W ...]                dump stale records (premise_changed / roster_changed) and counts
  stats  --run R [--work W ...]                per-annotation durations, attempts, location rate
"""
from __future__ import annotations

import os

MODEL_ID = "claude-code-subagent:alias=sonnet(exact-model-id-unverified)"
os.environ.setdefault("SENSOR_ANNOTATOR_MODEL_ID", MODEL_ID)  # must be set before common is imported

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import statistics as st  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WORK = HERE / "step2_work"

from cache import Cache  # noqa: E402
from common import REPO, all_work_ids, dumps, find_nishiki, load_work, read_json, segment, write_json  # noqa: E402
import common as _common  # noqa: E402

if os.environ.get("STEP2_WORKS_DIR"):  # measurement B: work/character files of a sandbox copy (the committed seed files stay untouched)
    _common.WORKS_DIR = Path(os.environ["STEP2_WORKS_DIR"])
from epoch import commit_epoch, epoch_status  # noqa: E402
from ingest import ingest_file  # noqa: E402
from merge import merge_work  # noqa: E402
from prepare import prepare_work  # noqa: E402
from resolve import resolve_work  # noqa: E402


def rdir(run: str) -> Path:
    return WORK / run


def cache_of(run: str) -> Cache:
    return Cache(rdir(run) / "cache", legacy_adopt=False)


from jsonl_append import append_json  # noqa: E402


def jl_append(p: Path, obj) -> None:
    append_json(p, obj)


def jl_read(p: Path) -> list:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def cmd_init(a) -> int:
    d = rdir(a.run)
    if d.exists():
        print(f"run {a.run} exists", file=sys.stderr)
        return 2
    if a.from_run:
        shutil.copytree(rdir(a.from_run) / "cache", d / "cache")
    else:
        (d / "cache" / "ledger").mkdir(parents=True)
        for f in sorted((HERE / "cache" / "ledger").glob("*.traits.json")):
            shutil.copy2(f, d / "cache" / "ledger" / f.name)  # the committed baseline ledger is input data; the L2/L1 caches start empty
    print(f"init {a.run} from {'run ' + a.from_run if a.from_run else 'seed ledger only'}; model_id={os.environ['SENSOR_ANNOTATOR_MODEL_ID']}")
    return 0


def _seg(w: str, a):
    nis = find_nishiki(None)
    text = Path(a.text) if getattr(a, "text", None) else load_work(w)["text_abs"]
    return segment(nis, text)


def _local(a):
    return Path(a.local) if getattr(a, "local", None) else None


def cmd_prep(a) -> int:
    run, cache = rdir(a.run), cache_of(a.run)
    rounds = jl_read(run / "rounds.jsonl")
    rnd = (rounds[-1]["round"] + 1) if rounds else 1
    pids = json.loads((run / "pids.json").read_text(encoding="utf-8")) if (run / "pids.json").exists() else {}
    out, per = [], {}
    for w in a.work or all_work_ids():
        r = prepare_work(w, cache, seg=_seg(w, a), packs_dir=run / "packs", local_path=_local(a), defer_blocked=True)
        per[w] = {k: r[k] for k in ("scene_count", "cached", "pending", "l2", "premise_changed", "roster_changed", "forced")}
        for p in r["packs_written"]:
            pack = read_json(Path(p["pack"]))
            pid = f"{a.run}.{w}.s{p['scene_index']}.r{rnd}"
            o = REPO / pack["output_path"]
            if o.exists():
                o.unlink()  # a previous annotation of the same scene (other run/epoch) must not be visible or block the Write tool
            pids[pid] = {"work": w, "scene_index": p["scene_index"], "scene_hash": p["scene_hash"], "pack": p["pack"], "output": str(o), "round": rnd}
            out.append({"pid": pid, "work": w, "scene_index": p["scene_index"], "pack": p["pack"], "output": str(o)})
    write_json(run / "pids.json", pids)
    jl_append(run / "rounds.jsonl", {"round": rnd, "t": time.time(), "packs": len(out), "per_work": per})
    print(json.dumps({"round": rnd, "packs": out, "per_work": per}, ensure_ascii=False, indent=1))
    return 0


def cmd_stamp(a) -> int:
    jl_append(rdir(a.run) / "timing.jsonl", {"pid": a.pid, "event": a.event, "t": time.time()})
    return 0


def cmd_ingest(a) -> int:
    run = rdir(a.run)
    m = json.loads((run / "pids.json").read_text(encoding="utf-8"))[a.pid]
    res = ingest_file(Path(m["output"]), cache_of(a.run), run / "packs", run / "rejects")
    jl_append(run / "attempts.jsonl", {"pid": a.pid, "t": time.time(), "accepted": res["accepted"], "reasons": res["reasons"],
                                       "ledger_additions_pending": res.get("ledger_additions_pending", [])})
    if res["accepted"]:
        print("ACCEPT")
        return 0
    print("REJECT")
    for r in res["reasons"]:
        print("  -", r)
    return 1


def cmd_epoch(a) -> int:
    cache = cache_of(a.run)
    rows = {}
    for w in a.work or all_work_ids():
        if a.action == "commit":
            rows[w] = commit_epoch(cache, w)
        else:
            rows[w] = epoch_status(cache, w)
    jl_append(rdir(a.run) / "epochs.jsonl", {"action": a.action, "t": time.time(), "rows": rows})
    print(json.dumps(rows, ensure_ascii=False, indent=1))
    return 0


def cmd_final(a) -> int:
    run, cache, nis = rdir(a.run), cache_of(a.run), find_nishiki(None)
    outd = run / a.name
    outd.mkdir(parents=True, exist_ok=True)
    res = {}
    for w in a.work or all_work_ids():
        work = load_work(w)
        text = Path(a.text) if a.text else work["text_abs"]
        seg = segment(nis, text)
        recs, rep = merge_work(w, cache, seg, local_path=_local(a))
        ann_bytes = dumps(recs).encode("utf-8")
        (outd / f"{w}.annotations.json").write_bytes(ann_bytes)
        write_json(outd / f"{w}.characters.json", work["characters"])
        d = subprocess.run([nis, "diagnose", str(text), "--format", "json", "--characters", str(outd / f"{w}.characters.json"),
                            "--annotations", str(outd / f"{w}.annotations.json")], capture_output=True, check=True)
        (outd / f"{w}.diagnose.json").write_bytes(d.stdout)
        res[w] = {"merged": rep["merged"], "scenes": rep["scenes"], "errors": len(rep["errors"]), "l1": rep["l1"], "l2": rep["l2"],
                  "premise_changed": len(rep.get("premise_changed", [])), "roster_changed": len(rep.get("roster_changed", [])),
                  "annotations_sha256": hashlib.sha256(ann_bytes).hexdigest(), "diagnose_sha256": hashlib.sha256(d.stdout).hexdigest(),
                  "annotations_bytes": len(ann_bytes), "diagnose_bytes": len(d.stdout)}
    write_json(outd / "summary.json", res)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


def cmd_flags(a) -> int:
    cache, out = cache_of(a.run), {}
    for w in a.work or all_work_ids():
        s = cache.stale_state(w)
        out[w] = {"premise_changed": s["premise_changed"], "roster_changed": s["roster_changed"], "forced": s["forced"]}
    write_json(rdir(a.run) / "flags.json", out)
    print(json.dumps({w: {"premise_changed": len(v["premise_changed"]), "roster_changed": len(v["roster_changed"])} for w, v in out.items()}, indent=1))
    return 0


def cmd_stats(a) -> int:
    run, cache = rdir(a.run), cache_of(a.run)
    pids = json.loads((run / "pids.json").read_text(encoding="utf-8")) if (run / "pids.json").exists() else {}
    tm: dict[str, dict] = {}
    for r in jl_read(run / "timing.jsonl"):
        tm.setdefault(r["pid"], {})[r["event"]] = r["t"]
    att = jl_read(run / "attempts.jsonl")
    dur = {p: round(v["end"] - v["start"], 2) for p, v in tm.items() if "start" in v and "end" in v}
    per = {}
    for p, m in pids.items():
        rows = [x for x in att if x["pid"] == p]
        per[p] = {**{k: m[k] for k in ("work", "scene_index", "round")}, "seconds": dur.get(p), "attempts": len(rows),
                  "accepted": any(x["accepted"] for x in rows), "ledger_additions": sorted({c for x in rows if x["accepted"] for c in x["ledger_additions_pending"]})}
    ds = sorted(v for v in dur.values())
    loc = {}
    for w in a.work or all_work_ids():
        work = load_work(w)
        seg = segment(find_nishiki(None), work["text_abs"])
        es = [r.entry for r in resolve_work(w, cache, seg["scenes"], work["characters"]) if r.entry]
        loc[w] = {"entries": len(es), "location_non_null": sum(1 for e in es if e.get("location"))}
    summ = {"invocations": len(pids), "invocations_with_timing": len(ds), "accepted": sum(1 for v in per.values() if v["accepted"]),
            "attempts_total": len(att), "rejected_attempts": sum(1 for x in att if not x["accepted"]),
            "seconds": ({"mean": round(st.mean(ds), 2), "median": round(st.median(ds), 2), "min": ds[0], "max": ds[-1]} if ds else None),
            "location": loc}
    write_json(run / "stats.json", {"summary": summ, "per_invocation": per})
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("init"); p.add_argument("--run", required=True); p.add_argument("--from-run"); p.set_defaults(f=cmd_init)
    p = sp.add_parser("prep"); p.add_argument("--run", required=True); p.add_argument("--work", nargs="*"); p.add_argument("--text"); p.add_argument("--local"); p.set_defaults(f=cmd_prep)
    p = sp.add_parser("stamp"); p.add_argument("--run", required=True); p.add_argument("pid"); p.add_argument("event", choices=["start", "end"]); p.set_defaults(f=cmd_stamp)
    p = sp.add_parser("ingest"); p.add_argument("--run", required=True); p.add_argument("pid"); p.set_defaults(f=cmd_ingest)
    p = sp.add_parser("epoch"); p.add_argument("--run", required=True); p.add_argument("action", choices=["commit", "status"]); p.add_argument("--work", nargs="*"); p.set_defaults(f=cmd_epoch)
    p = sp.add_parser("final"); p.add_argument("--run", required=True); p.add_argument("--name", required=True); p.add_argument("--work", nargs="*"); p.add_argument("--text"); p.add_argument("--local"); p.set_defaults(f=cmd_final)
    p = sp.add_parser("flags"); p.add_argument("--run", required=True); p.add_argument("--work", nargs="*"); p.set_defaults(f=cmd_flags)
    p = sp.add_parser("stats"); p.add_argument("--run", required=True); p.add_argument("--work", nargs="*"); p.set_defaults(f=cmd_stats)
    a = ap.parse_args()
    return a.f(a)


if __name__ == "__main__":
    raise SystemExit(main())
