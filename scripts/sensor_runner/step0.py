"""Phase 1.5 Stage 2 Step 0: measurement tooling for real-LLM annotation variability (measurement only).

  build    write packs (production pack format, via prepare_work on a scratch copy of the seed cache) + manifest
  stamp    record a start/end timestamp of one annotation        (stdlib only; called by annotators)
  check    validate one annotator output exactly like `ingest` does (without touching any cache); logs the attempt
  analyze  compute all metrics from manifest + outputs, write results/phase15_stage2_step0_*.json

Packs for measurement A use the seed cache entry of the previous scene as `prior` (identical bytes for all 5 repeats
apart from `output_path`). Packs for measurement B append one inserted fact (E1 trivial / E2 major) to the previous
scene's facts, which is the only difference to the 'base' pack of the same scene.
No cache, key, resolver, threshold or output-schema code is modified by this tool.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import random
import shutil
import statistics as st
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from jsonl_append import append_json  # noqa: E402
WORK = HERE / "step0_work"
RESULTS = HERE / "results"

# measurement A scenes (registry has no work with >7 scenes, so 4 works in 2 categories give 20 scenes)
SELECT = {
    "000035_000301": ("transition", list(range(7))),
    "000081_000456": ("transition", list(range(6))),
    "001875_059897": ("dialogue", list(range(4))),
    "001798_056779": ("dialogue", [1, 2, 3]),
}
# measurement B: (work, k) -> target scene is k+1. E1 trivial / E2 major, appended to scene k's facts.
def _sc(subj, target, text):
    return {"subject": subj, "predicate": "STATE_CHANGED", "target": target, "category": "situation", "text": text, "evidence": "explicit", "basis": None}


def _lr(subj, pred, obj):
    return {"subject": subj, "predicate": pred, "object": obj, "evidence": "explicit", "basis": None}


PAIRS = {
    ("000035_000301", 1): (_sc("yozo", "yozo", "缶入りの煙草を一本吸った"), _lr("yozo", "LEARNED", "娘が自分の自殺未遂の件を警察へ届け出た")),
    ("000035_000301", 3): (_sc("yozo", "yozo", "屋台で熱いお茶を一杯飲んだ"), _lr("yozo", "LEARNED", "ツネ子が自分を警察に突き出す相談をしていた")),
    ("000035_000301", 5): (_sc("yozo", "yozo", "カフエの勘定書を一枚拾った"), _lr("yozo", "LEARNED", "ツネ子は既に別の男と所帯を持ったと堀木から聞いた")),
    ("000081_000456", 1): (_sc("giovanni", "giovanni", "河原の小石をひとつ拾った"), _lr("giovanni", "LEARNED", "カムパネルラは自分と同じ列車に長くは乗っていられない")),
    ("000081_000456", 3): (_sc("giovanni", "giovanni", "大学士の差し出した菓子を一つ食べた"), _lr("daigakushi", "LEARNED", "二人が切符を持たない無賃の乗客であること")),
    ("001875_059897", 1): (_sc("bunkichi", "bunkichi", "縁側で番茶を一杯すすった"), _lr("yasue", "LEARNED", "文吉が貸した二千円は返済の当てがなく、兵六は夜逃げの支度をしている")),
    ("001875_059897", 2): (_sc("susumu", "susumu", "玄関で靴の紐を結び直した"), _lr("susumu", "LEARNED", "兵六に預けた金は既に使い果たされている")),
    ("001798_056779", 1): (_sc("boku", "boku", "朝の茶を一杯飲んだ"), _lr("boku", "LEARNED", "不破は借金取りから逃れるために夜逃げしたのだ")),
}
N_RUNS_A = 5
N_RUNS_B = 3


# ---------------------------------------------------------------------------------------------------------------
def _pid(tag: str) -> str:
    return "p" + hashlib.sha1(f"step0|{tag}".encode()).hexdigest()[:8]


def _load_manifest() -> dict[str, Any]:
    return json.loads((WORK / "manifest.json").read_text(encoding="utf-8"))


def cmd_build(a) -> int:
    from cache import Cache
    from common import CACHE_ROOT, find_nishiki, load_work, segment, write_json
    from prepare import prepare_work
    from resolve import resolve_work

    nis = find_nishiki(a.nishiki)
    for d in ("packs", "out", "tmp"):
        shutil.rmtree(WORK / d, ignore_errors=True)
        (WORK / d).mkdir(parents=True, exist_ok=True)
    for f in ("timing.jsonl", "attempts.jsonl"):
        (WORK / f).unlink(missing_ok=True)

    segs = {w: segment(nis, load_work(w)["text_abs"]) for w in SELECT}
    seq = 0

    def make(work: str, scene: int, extra: dict | None) -> dict[str, Any]:
        nonlocal seq
        seq += 1
        tmp = WORK / "tmp" / f"c{seq}"
        shutil.copytree(CACHE_ROOT, tmp / "cache")
        c = Cache(tmp / "cache", legacy_adopt=False)
        wk, seg = load_work(work), segs[work]
        res = resolve_work(work, c, seg["scenes"], wk["characters"])
        assert all(r.how == "hit" for r in res), (work, [r.how for r in res])
        if extra is not None:
            k = scene - 1
            e = copy.deepcopy(res[k].entry)
            e["facts"].append(extra)
            c.put_l2(seg["scenes"][k]["scene_hash"], res[k].key, e, res[k].parts, source="step0-insert")
        c.delete(seg["scenes"][scene]["scene_hash"])
        r = prepare_work(work, c, seg=seg, packs_dir=tmp / "packs", defer_blocked=True)
        hit = [p for p in r["packs_written"] if p["scene_index"] == scene]
        assert len(hit) == 1, (work, scene, r["packs_written"])
        pack = json.loads(Path(hit[0]["pack"]).read_text(encoding="utf-8"))
        shutil.rmtree(tmp, ignore_errors=True)
        return pack

    manifest: dict[str, Any] = {"packs": {}, "batches": []}
    rng = random.Random(20261126)

    def emit(kind: str, cond: str, work: str, scene: int, run: int, pack: dict[str, Any]) -> str:
        pid = _pid(f"{kind}|{cond}|{work}|{scene}|{run}")
        p = copy.deepcopy(pack)
        p["output_path"] = str((WORK / "out" / f"{pid}.strong.json").as_posix())
        write_json(WORK / "packs" / f"{pid}.pack.json", p)
        manifest["packs"][pid] = {"kind": kind, "cond": cond, "work": work, "scene": scene, "run": run,
                                  "category": SELECT[work][0], "pack_sha256": hashlib.sha256(json.dumps({k: v for k, v in p.items() if k != "output_path"}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
        return pid

    # --- A: base pack per scene, 5 runs
    base_pack: dict[tuple[str, int], dict] = {}
    for w, (_, scenes) in SELECT.items():
        for s in scenes:
            base_pack[(w, s)] = make(w, s, None)
    a_items = [(w, s) for w, (_, scenes) in SELECT.items() for s in scenes]
    for run in range(1, N_RUNS_A + 1):
        order = a_items[:]
        rng.shuffle(order)
        for i in range(0, len(order), 5):
            pids = [emit("A", "base", w, s, run, base_pack[(w, s)]) for w, s in order[i : i + 5]]
            manifest["batches"].append({"id": f"A{run}-{i // 5 + 1}", "pids": pids})
    # --- B: E1 / E2 packs for each target scene (k+1), 3 runs each; old condition = A runs 1..3
    for ci, cond in enumerate(("E1", "E2")):
        packs = {}
        for (w, k), extras in PAIRS.items():
            packs[(w, k + 1)] = make(w, k + 1, extras[ci])
        for run in range(1, N_RUNS_B + 1):
            order = list(packs)
            rng.shuffle(order)
            pids = [emit("B", cond, w, s, run, packs[(w, s)]) for w, s in order]
            manifest["batches"].append({"id": f"B{cond}-{run}", "pids": pids})
    shutil.rmtree(WORK / "tmp", ignore_errors=True)
    write_json(WORK / "manifest.json", manifest)
    print(f"packs={len(manifest['packs'])} batches={len(manifest['batches'])}")
    for b in manifest["batches"]:
        print(b["id"], " ".join(b["pids"]))
    return 0


def cmd_stamp(a) -> int:
    append_json(WORK / "timing.jsonl", {"pid": a.pid, "event": a.event, "t": time.time()})
    return 0


def cmd_check(a) -> int:
    from cache import Cache
    from common import load_work
    from merge import merge_scene
    from strong import validate_strong

    m = _load_manifest()["packs"][a.pid]
    pack = json.loads((WORK / "packs" / f"{a.pid}.pack.json").read_text(encoding="utf-8"))
    out = WORK / "out" / f"{a.pid}.strong.json"
    reasons: list[str] = []
    try:
        entry = json.loads(out.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        entry, reasons = None, [f"json_parse_error: {e}"]
    if entry is not None:
        ledger = Cache(legacy_adopt=False).ledger(pack["work_id"])
        reasons = validate_strong(entry, pack["scene_text"], pack["roster_all_ids"], ledger, pack["scene_hash"])
        if not reasons:
            full = {**ledger, **(entry.get("ledger_additions") or {})}
            scene = {"scene_text": pack["scene_text"], "utterances": [], "sentences": [], "index": pack["scene_index"],
                     "scene_hash": pack["scene_hash"], "char_len": pack["char_len"]}
            _, st_ = merge_scene(pack["work_id"], scene, pack["scene_count"], entry, None, full, pack["roster_all_ids"])
            reasons = [f"sensor.v2: {e}" for e in st_["errors"]]
    append_json(WORK / "attempts.jsonl", {"pid": a.pid, "t": time.time(), "accepted": not reasons, "reasons": reasons})
    if reasons:
        print("REJECT")
        for r in reasons:
            print("  -", r)
        return 1
    print("ACCEPT")
    return 0


# ---------------------------------------------------------------------------------------------------------------
def norm_text(s: str) -> str:
    """NFKC, then drop whitespace and punctuation/symbol characters (Unicode categories Z*, P*, S*, C*)."""
    s = unicodedata.normalize("NFKC", s or "")
    return "".join(ch for ch in s if unicodedata.category(ch)[0] not in "ZPSC")


def fact_raw(f: dict[str, Any]) -> dict[str, Any]:
    return {k: f.get(k) for k in ("subject", "predicate", "object", "target", "category", "text")}


def fact_norm(f: dict[str, Any]) -> str:
    body = norm_text(f.get("target") or "") + "|" + norm_text((f.get("object") if f.get("object") is not None else f.get("text")) or "")
    return f"{f.get('predicate')}|{f.get('subject') or ''}|{body}"


def h(o: Any) -> str:
    return hashlib.sha256(json.dumps(o, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def facts_raw_hash(e: dict[str, Any]) -> str:
    return h([fact_raw(f) for f in e["facts"]])  # as emitted: order and every content field


def facts_norm_hash(e: dict[str, Any]) -> str:
    return h(sorted(set(fact_norm(f) for f in e["facts"])))  # sorted set of normalised fact sentences


def prior_hash(e: dict[str, Any]) -> str:
    from keys import prior_state_hash

    return prior_state_hash(e)


def items(e: dict[str, Any], part: str = "all") -> set[tuple]:
    s: set[tuple] = set()
    if part in ("all", "state"):
        s.add(("location", e.get("location")))
        s.add(("pov", e.get("pov_character")))
        s |= {("present", p["id"]) for p in e["present"]}
    if part in ("all", "facts"):
        s |= {("fact", fact_norm(f)) for f in e["facts"]}
    if part == "all":
        s.add(("time", e["time"]["relative_order"]))
        s |= {("event", c["code"]) for c in e["event_top3"]}
        s |= {("mood", c["code"]) for c in e["mood_top3"]}
        s |= {("awase", k, v) for k, v in e["awase"].items() if k in ("F_valence", "F_arousal", "f_valence", "f_arousal")}
        s |= {("trait", t["character_id"], c["axis"], c["delta"]) for t in e["trait_changes"] for c in (t.get("changes") or [])}
    return s


def dist(a: dict[str, Any], b: dict[str, Any], part: str = "all") -> float:
    x, y = items(a, part), items(b, part)
    u = x | y
    return 1.0 - (len(x & y) / len(u)) if u else 0.0


def pair_rate(vals: list[Any]) -> float:
    ps = list(itertools.combinations(vals, 2))
    return sum(p[0] == p[1] for p in ps) / len(ps)


def modal_freq(vals: list[Any]) -> float:
    return max(vals.count(v) for v in set(vals)) / len(vals)


def mean_pair_dist(es: list[dict[str, Any]], part: str = "all") -> float:
    ps = list(itertools.combinations(es, 2))
    return st.fmean(dist(x, y, part) for x, y in ps)


def cross_dist(xs: list[dict[str, Any]], ys: list[dict[str, Any]], part: str = "all") -> float:
    return st.fmean(dist(x, y, part) for x in xs for y in ys)


def _collect(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stamps: dict[str, dict[str, list[float]]] = {}
    tf = WORK / "timing.jsonl"
    for line in tf.read_text(encoding="utf-8").splitlines() if tf.exists() else []:
        r = json.loads(line)
        stamps.setdefault(r["pid"], {}).setdefault(r["event"], []).append(r["t"])
    atts: dict[str, list[dict]] = {}
    af = WORK / "attempts.jsonl"
    for line in af.read_text(encoding="utf-8").splitlines() if af.exists() else []:
        r = json.loads(line)
        atts.setdefault(r["pid"], []).append(r)
    rows = {}
    for pid, m in manifest["packs"].items():
        out = WORK / "out" / f"{pid}.strong.json"
        a = atts.get(pid, [])
        accepted = bool(a) and a[-1]["accepted"] and out.exists()
        s = stamps.get(pid, {})
        sec = (max(s["end"]) - min(s["start"])) if s.get("start") and s.get("end") else None
        rows[pid] = {**m, "entry": json.loads(out.read_text(encoding="utf-8")) if accepted else None, "attempts": len(a),
                     "rejects": sum(not x["accepted"] for x in a), "accepted": accepted, "seconds": sec,
                     "reject_reasons": [x["reasons"] for x in a if not x["accepted"]]}
    return rows


def _seed_entries() -> dict[tuple[str, int], dict[str, Any]]:
    from cache import Cache
    from common import find_nishiki, load_work, segment
    from resolve import resolve_work

    nis, c, out = find_nishiki(), Cache(legacy_adopt=False), {}
    for w in SELECT:
        wk = load_work(w)
        seg = segment(nis, wk["text_abs"])
        for r in resolve_work(w, c, seg["scenes"], wk["characters"]):
            out[(w, r.index)] = r.entry
    return out


def flatten(o: Any, path: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(flatten(v, f"{path}/{k}"))
    elif isinstance(o, list):
        out[path + "[]"] = json.dumps(o, ensure_ascii=False, sort_keys=True)
    else:
        out[path] = o
    return out


def diagnose_runs(rows: dict[str, dict[str, Any]], seeds: dict) -> dict[str, Any]:
    import tempfile
    import subprocess

    from cache import Cache
    from common import dumps, find_nishiki, load_work, segment
    from merge import load_local, merge_scene

    nis = find_nishiki()
    res: dict[str, Any] = {}
    for w, (_, scenes) in SELECT.items():
        wk = load_work(w)
        seg = segment(nis, wk["text_abs"])
        local = load_local(Path(__file__).resolve().parent / "local" / f"{w}.utterances.json")
        roster_ids = [c["id"] for c in wk["characters"]]
        by_run = {r: {} for r in range(1, N_RUNS_A + 1)}
        for pid, r in rows.items():
            if r["kind"] == "A" and r["work"] == w and r["entry"] is not None:
                by_run[r["run"]][r["scene"]] = r["entry"]
        variants = {"seed": {}} | {f"run{r}": d for r, d in by_run.items()}
        diags = {}
        for name, repl in variants.items():
            if name != "seed" and len(repl) != len(scenes):
                continue
            ledger = dict(Cache(legacy_adopt=False).ledger(w))
            ents = [repl.get(i, seeds[(w, i)]) for i in range(len(seg["scenes"]))]
            for e in ents:
                for k, v in (e.get("ledger_additions") or {}).items():
                    ledger.setdefault(k, v)
            recs, errs = [], 0
            for s, e in zip(seg["scenes"], ents):
                rec, stt = merge_scene(w, s, len(seg["scenes"]), e, local.get(s["scene_hash"]), ledger, roster_ids)
                errs += len(stt["errors"])
                recs.append(rec)
            with tempfile.TemporaryDirectory() as td:
                td = Path(td)
                (td / "c.json").write_text(dumps(wk["characters"]), encoding="utf-8")
                (td / "a.json").write_text(dumps(recs), encoding="utf-8")
                d = subprocess.run([nis, "diagnose", str(wk["text_abs"]), "--format", "json", "--characters", str(td / "c.json"), "--annotations", str(td / "a.json")], capture_output=True, check=True)
            diags[name] = {"merge_errors": errs, "diagnose": json.loads(d.stdout.decode("utf-8"))}
        runs = [n for n in diags if n != "seed"]
        per_lens: dict[str, Any] = {}
        for li, lens in enumerate(diags["seed"]["diagnose"]["lenses"]):
            lid = lens["id"]
            flats = {n: flatten({k: v for k, v in diags[n]["diagnose"]["lenses"][li].items() if k != "spans"}) for n in runs}
            spans = {n: {json.dumps(sp, ensure_ascii=False, sort_keys=True) for sp in diags[n]["diagnose"]["lenses"][li].get("spans", [])} for n in runs}
            keys_ = sorted(set().union(*[set(f) for f in flats.values()]))
            varying, rel = 0, []
            for k in keys_:
                vals = [flats[n].get(k) for n in runs]
                if len(set(map(str, vals))) > 1:
                    varying += 1
                    nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
                    if len(nums) == len(vals):
                        mu = st.fmean(nums)
                        rel.append((max(nums) - min(nums)) / (abs(mu) if abs(mu) > 1e-9 else 1.0))
            sp_d = [1 - len(spans[x] & spans[y]) / len(spans[x] | spans[y]) if (spans[x] | spans[y]) else 0.0 for x, y in itertools.combinations(runs, 2)]
            per_lens[lid] = {"leaves": len(keys_), "leaves_varying_across_runs": varying, "frac_varying": round(varying / max(1, len(keys_)), 4),
                             "numeric_rel_range_median": round(st.median(rel), 4) if rel else 0.0, "numeric_rel_range_max": round(max(rel), 4) if rel else 0.0,
                             "span_count_per_run": [len(spans[n]) for n in runs], "span_set_distance_mean": round(st.fmean(sp_d), 4) if sp_d else 0.0,
                             "status_per_run": [diags[n]["diagnose"]["lenses"][li]["status"] for n in runs]}
        res[w] = {"runs_compared": runs, "lenses": per_lens, "raw": {n: diags[n]["diagnose"]["lenses"] for n in diags}}
    return res


def cmd_analyze(a) -> int:
    from common import write_json

    manifest = _load_manifest()
    rows = _collect(manifest)
    seeds = _seed_entries()
    A = {k: [] for k in SELECT}
    byscene: dict[tuple[str, int], list[dict]] = {}
    for pid, r in rows.items():
        if r["kind"] == "A":
            byscene.setdefault((r["work"], r["scene"]), []).append(r)
    scene_stats = []
    for (w, s), rs in sorted(byscene.items()):
        rs = sorted(rs, key=lambda r: r["run"])
        ok = [r for r in rs if r["entry"] is not None]
        es = [r["entry"] for r in ok]
        if len(es) < 2:
            scene_stats.append({"work": w, "scene": s, "n_ok": len(es)})
            continue
        seed = seeds[(w, s)]
        loc = [e.get("location") for e in es]
        locn = [norm_text(e.get("location") or "") for e in es]
        pres = [tuple(sorted(p["id"] for p in e["present"])) for e in es]
        povs = [e.get("pov_character") for e in es]
        ph = [prior_hash(e) for e in es]
        fr = [facts_raw_hash(e) for e in es]
        fn = [facts_norm_hash(e) for e in es]
        pairs = list(itertools.combinations(range(len(es)), 2))
        loc_only = sum(1 for i, j in pairs if ph[i] != ph[j] and pres[i] == pres[j] and povs[i] == povs[j] and loc[i] != loc[j])
        fs = [set(fact_norm(f) for f in e["facts"]) for e in es]
        jac = [len(fs[i] & fs[j]) / len(fs[i] | fs[j]) if (fs[i] | fs[j]) else 1.0 for i, j in pairs]
        scene_stats.append({
            "work": w, "scene": s, "category": SELECT[w][0], "n_ok": len(es), "n_rejected_attempts": sum(r["rejects"] for r in rs),
            "prior_hash_pair_agree": round(pair_rate(ph), 4), "prior_hash_all_equal": len(set(ph)) == 1, "prior_hash_modal": round(modal_freq(ph), 4),
            "prior_hash_equal_to_seed_rate": round(sum(x == prior_hash(seed) for x in ph) / len(ph), 4),
            "location_string_pair_agree": round(pair_rate(loc), 4), "location_normalized_pair_agree": round(pair_rate(locn), 4),
            "location_none_rate": round(sum(x is None for x in loc) / len(loc), 4), "location_equal_to_seed_rate": round(sum(x == seed.get("location") for x in loc) / len(loc), 4),
            "present_pair_agree": round(pair_rate(pres), 4), "pov_pair_agree": round(pair_rate(povs), 4),
            "prior_hash_pairs_disagree_due_to_location_only": loc_only, "prior_hash_pairs_disagree_total": sum(ph[i] != ph[j] for i, j in pairs),
            "facts_raw_hash_pair_agree": round(pair_rate(fr), 4), "facts_raw_hash_all_equal": len(set(fr)) == 1,
            "facts_norm_hash_pair_agree": round(pair_rate(fn), 4), "facts_norm_hash_all_equal": len(set(fn)) == 1, "facts_norm_hash_modal": round(modal_freq(fn), 4),
            "facts_norm_jaccard_mean": round(st.fmean(jac), 4), "n_facts_mean": round(st.fmean(len(e["facts"]) for e in es), 2),
            "entry_distance_mean": round(mean_pair_dist(es), 4), "state_distance_mean": round(mean_pair_dist(es, "state"), 4), "facts_distance_mean": round(mean_pair_dist(es, "facts"), 4),
            "entry_distance_to_seed_mean": round(st.fmean(dist(e, seed) for e in es), 4),
            "seconds_mean": round(st.fmean(r["seconds"] for r in ok if r["seconds"] is not None), 1) if any(r["seconds"] is not None for r in ok) else None,
        })

    def agg(sel):
        xs = [x for x in scene_stats if x.get("n_ok", 0) >= 2 and sel(x)]
        if not xs:
            return None
        f = lambda k: round(st.fmean(x[k] for x in xs), 4)  # noqa: E731
        return {"scenes": len(xs), "prior_hash_pair_agree": f("prior_hash_pair_agree"), "prior_hash_all_equal_rate": round(sum(x["prior_hash_all_equal"] for x in xs) / len(xs), 4),
                "prior_hash_equal_to_seed_rate": f("prior_hash_equal_to_seed_rate"), "location_string_pair_agree": f("location_string_pair_agree"),
                "location_normalized_pair_agree": f("location_normalized_pair_agree"), "location_equal_to_seed_rate": f("location_equal_to_seed_rate"), "location_none_rate": f("location_none_rate"),
                "present_pair_agree": f("present_pair_agree"), "pov_pair_agree": f("pov_pair_agree"),
                "prior_hash_disagree_pairs_due_to_location_only": sum(x["prior_hash_pairs_disagree_due_to_location_only"] for x in xs), "prior_hash_disagree_pairs_total": sum(x["prior_hash_pairs_disagree_total"] for x in xs),
                "facts_raw_hash_pair_agree": f("facts_raw_hash_pair_agree"), "facts_raw_hash_all_equal_rate": round(sum(x["facts_raw_hash_all_equal"] for x in xs) / len(xs), 4),
                "facts_norm_hash_pair_agree": f("facts_norm_hash_pair_agree"), "facts_norm_hash_all_equal_rate": round(sum(x["facts_norm_hash_all_equal"] for x in xs) / len(xs), 4),
                "facts_norm_hash_pair_disagree": round(1 - st.fmean(x["facts_norm_hash_pair_agree"] for x in xs), 4), "facts_norm_jaccard_mean": f("facts_norm_jaccard_mean"),
                "entry_distance_mean": f("entry_distance_mean"), "state_distance_mean": f("state_distance_mean"), "facts_distance_mean": f("facts_distance_mean"), "entry_distance_to_seed_mean": f("entry_distance_to_seed_mean")}

    secs = [r["seconds"] for r in rows.values() if r["kind"] == "A" and r["seconds"] is not None]
    A_summary = {
        "overall": agg(lambda x: True), "by_category": {c: agg(lambda x, c=c: x["category"] == c) for c in ("transition", "dialogue")},
        "by_work": {w: agg(lambda x, w=w: x["work"] == w) for w in SELECT}, "per_scene": scene_stats,
        "timing_seconds_per_annotation": {"n": len(secs), "mean": round(st.fmean(secs), 1) if secs else None, "median": round(st.median(secs), 1) if secs else None,
                                          "min": round(min(secs), 1) if secs else None, "max": round(max(secs), 1) if secs else None,
                                          "note": "wall time from the start stamp to the last end stamp of one annotation by a subagent (read pack, write JSON, check, retries); includes tool latency"},
    }

    # ---- B
    B_rows = []
    for (w, k), (e1, e2) in PAIRS.items():
        t = k + 1
        base_all = [r["entry"] for r in sorted(byscene.get((w, t), []), key=lambda r: r["run"]) if r["entry"] is not None]
        base = base_all[:N_RUNS_B]
        cond = {c: [r["entry"] for r in sorted((x for x in rows.values() if x["kind"] == "B" and x["cond"] == c and x["work"] == w and x["scene"] == t), key=lambda r: r["run"]) if r["entry"] is not None] for c in ("E1", "E2")}
        rec = {"work": w, "k": k, "target": t, "n_base": len(base), "n_E1": len(cond["E1"]), "n_E2": len(cond["E2"])}
        for c in ("E1", "E2"):
            if len(base) < 2 or len(cond[c]) < 2:
                rec[c] = None
                continue
            out = {}
            for part in ("all", "facts", "state"):
                wb, wc = mean_pair_dist(base, part), mean_pair_dist(cond[c], part)
                W = (wb + wc) / 2
                Bt = cross_dist(base, cond[c], part)
                out[part] = {"within_base": round(wb, 4), "within_new": round(wc, 4), "within_mean": round(W, 4), "between": round(Bt, 4), "diff": round(Bt - W, 4), "influenced": Bt > W}
            fb = [facts_norm_hash(e) for e in base]
            fc = [facts_norm_hash(e) for e in cond[c]]
            out["facts_norm_hash_cross_change_rate"] = round(sum(x != y for x in fb for y in fc) / (len(fb) * len(fc)), 4)
            out["facts_norm_hash_within_change_rate"] = round(1 - st.fmean([pair_rate(fb), pair_rate(fc)]), 4)
            out["mentions_inserted_text"] = {"E1": sum(norm_text(e1.get("text") or e1.get("object") or "")[:6] in norm_text(json.dumps(e, ensure_ascii=False)) for e in cond[c]) if c == "E1" else sum(norm_text(e2.get("text") or e2.get("object") or "")[:6] in norm_text(json.dumps(e, ensure_ascii=False)) for e in cond[c])}
            rec[c] = out
        B_rows.append(rec)

    def bagg(c, part):
        xs = [r[c][part] for r in B_rows if r.get(c)]
        if not xs:
            return None
        return {"pairs": len(xs), "influenced_count": sum(x["influenced"] for x in xs), "influenced_ratio": round(sum(x["influenced"] for x in xs) / len(xs), 4),
                "influenced_margin_ge_0.05_ratio": round(sum(x["diff"] >= 0.05 for x in xs) / len(xs), 4),
                "mean_between": round(st.fmean(x["between"] for x in xs), 4), "mean_within": round(st.fmean(x["within_mean"] for x in xs), 4)}

    B_summary = {"pairs": B_rows, "aggregate": {c: {p: bagg(c, p) for p in ("all", "facts", "state")} for c in ("E1", "E2")},
                 "note": "influenced = mean between-condition distance > mean within-condition distance (n=3 per condition; distance = 1 - Jaccard over annotation items); old-prior condition = measurement A runs 1..3 of the same pack"}

    calls = {"total_attempts": sum(r["attempts"] for r in rows.values()), "accepted_final": sum(r["accepted"] for r in rows.values()),
             "packs": len(rows), "rejected_attempts": sum(r["rejects"] for r in rows.values()),
             "A_packs": sum(r["kind"] == "A" for r in rows.values()), "B_packs": sum(r["kind"] == "B" for r in rows.values())}
    diag = diagnose_runs(rows, seeds) if all(r["accepted"] for r in rows.values() if r["kind"] == "A") else {"skipped": "not all A runs accepted"}

    write_json(RESULTS / "phase15_stage2_step0_raw.json", {"calls": calls, "packs": {pid: {k: v for k, v in r.items()} for pid, r in rows.items()}})
    write_json(RESULTS / "phase15_stage2_step0_A_summary.json", A_summary)
    write_json(RESULTS / "phase15_stage2_step0_B_summary.json", B_summary)
    write_json(RESULTS / "phase15_stage2_step0_diagnose.json", diag)
    print(json.dumps({"calls": calls, "A_overall": A_summary["overall"], "A_timing": A_summary["timing_seconds_per_annotation"], "B": B_summary["aggregate"]}, ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nishiki")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("build"); p.set_defaults(f=cmd_build)
    p = sp.add_parser("stamp"); p.add_argument("pid"); p.add_argument("event", choices=["start", "end"]); p.set_defaults(f=cmd_stamp)
    p = sp.add_parser("check"); p.add_argument("pid"); p.set_defaults(f=cmd_check)
    p = sp.add_parser("analyze"); p.set_defaults(f=cmd_analyze)
    a = ap.parse_args()
    return a.f(a)


if __name__ == "__main__":
    raise SystemExit(main())
