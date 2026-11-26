"""sensor_runner CLI.

  python scripts/sensor_runner/cli.py prepare [--work ID ...] [--limit N]
  python scripts/sensor_runner/cli.py ingest PATH...            (files or dirs of *.strong.json)
  python scripts/sensor_runner/cli.py merge [--work ID ...] [--out DIR] [--diagnose]
  python scripts/sensor_runner/cli.py status
  python scripts/sensor_runner/cli.py migrate [--work ID ...]   (adopt legacy strong/ entries under strict L2 keys)
  python scripts/sensor_runner/cli.py gc [--days N] [--apply]    (explicit only; lists l1/l2 files older than N days)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cache import Cache  # noqa: E402
from common import REPO, all_work_ids, find_nishiki, load_work, segment, write_json  # noqa: E402
from diagnose import all_ok, run_diagnose  # noqa: E402
from ingest import ingest_paths  # noqa: E402
from merge import merge_work  # noqa: E402
from prepare import prepare_work  # noqa: E402
from resolve import resolve_work  # noqa: E402

DEFAULT_MERGE_OUT = Path(__file__).resolve().parent / "merged"


def _works(a) -> list[str]:
    return a.work or all_work_ids()


def cmd_prepare(a) -> int:
    nishiki, cache = find_nishiki(a.nishiki), Cache(a.cache)
    tot_p = 0
    for w in _works(a):
        r = prepare_work(w, cache, nishiki, limit=a.limit, defer_blocked=not a.emit_blocked)
        tot_p += len(r["packs_written"])
        print(f"{w}: scenes={r['scene_count']} cached={r['cached']} pending={r['pending']} packs_written={len(r['packs_written'])} deferred_blocked={len(r['deferred_blocked'])} l2={r['l2']}")
        for p in r["packs_written"]:
            print(f"   s{p['scene_index']} {p['scene_hash'][:12]} {p['pack']}")
    print(f"total packs written: {tot_p}")
    return 0


def cmd_ingest(a) -> int:
    results = ingest_paths([Path(p) for p in a.paths], Cache(a.cache))
    bad = 0
    for r in results:
        if r["accepted"]:
            print(f"ACCEPT {r['scene_id']}  {r['file']}")
        else:
            bad += 1
            print(f"REJECT {r['file']}")
            for x in r["reasons"]:
                print(f"   - {x}")
    print(f"accepted={len(results) - bad} rejected={bad}")
    return 1 if bad else 0


def cmd_merge(a) -> int:
    nishiki, cache = find_nishiki(a.nishiki), Cache(a.cache)
    out = Path(a.out) if a.out else DEFAULT_MERGE_OUT
    rc = 0
    for w in _works(a):
        work = load_work(w)
        seg = segment(nishiki, work["text_abs"])
        recs, rep = merge_work(w, cache, seg, partial=a.partial)
        write_json(out / f"{w}.annotations.json", recs)
        write_json(out / f"{w}.characters.json", work["characters"])
        line = f"{w}: merged={rep['merged']}/{rep['scenes']} errors={len(rep['errors'])} missing_local_utt={rep['missing_local_utterances']} local_ids_absent_from_present={rep['local_ids_absent_from_present']}"
        if a.diagnose:
            d = run_diagnose(nishiki, work["text_abs"], out / f"{w}.characters.json", out / f"{w}.annotations.json")
            line += " lenses=" + ",".join(f"{k}:{v['status']}" for k, v in d["lenses"].items())
            if not all_ok(d):
                rc = 1
        print(line)
        if rep["errors"]:
            rc = 1
            for e in rep["errors"][:5]:
                print("   ", e)
    return rc


def cmd_migrate(a) -> int:
    nishiki, cache = find_nishiki(a.nishiki), Cache(a.cache, legacy_adopt=True)
    for w in _works(a):
        work = load_work(w)
        seg = segment(nishiki, work["text_abs"])
        res = resolve_work(w, cache, seg["scenes"], work["characters"])
        cache.write_manifest(w, seg["scenes"], seg.get("splitter", ""), resolved=[r.entry for r in res])
        print(f"{w}: " + " ".join(f"{k}={sum(r.how == k for r in res)}" for k in ("hit", "adopted", "miss", "blocked")))
    return 0


def cmd_gc(a) -> int:
    stale = Cache(a.cache).gc(a.days, apply=a.apply)
    print(f"{'deleted' if a.apply else 'would delete'} {len(stale)} file(s) older than {a.days} days")
    for p in stale:
        print("  ", p)
    return 0


def cmd_status(a) -> int:
    cache = Cache(a.cache)
    for w in _works(a):
        m = cache.manifest(w)
        print(f"{w}: " + (f"cached {m['cached_count']}/{m['scene_count']}" if m else "no manifest"))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sensor_runner")
    ap.add_argument("--nishiki", default=None)
    ap.add_argument("--cache", default=None, help="cache root (default: scripts/sensor_runner/cache)")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prepare"); p.add_argument("--work", nargs="*"); p.add_argument("--limit", type=int); p.add_argument("--emit-blocked", action="store_true", help="also write packs for scenes whose predecessor is unannotated"); p.set_defaults(f=cmd_prepare)
    p = sp.add_parser("ingest"); p.add_argument("paths", nargs="+"); p.set_defaults(f=cmd_ingest)
    p = sp.add_parser("merge"); p.add_argument("--work", nargs="*"); p.add_argument("--out"); p.add_argument("--diagnose", action="store_true"); p.add_argument("--partial", action="store_true"); p.set_defaults(f=cmd_merge)
    p = sp.add_parser("migrate"); p.add_argument("--work", nargs="*"); p.set_defaults(f=cmd_migrate)
    p = sp.add_parser("gc"); p.add_argument("--days", type=float, default=30.0); p.add_argument("--apply", action="store_true"); p.set_defaults(f=cmd_gc)
    p = sp.add_parser("status"); p.add_argument("--work", nargs="*"); p.set_defaults(f=cmd_status)
    a = ap.parse_args(argv)
    return a.f(a)


if __name__ == "__main__":
    raise SystemExit(main())
