"""Analysis for Step 0.5 (measurement only). Called from step05.py analyze."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import statistics as st
from collections import Counter
from typing import Any

import step0
import step05 as S


def _kind(o: str, roster: set[str]) -> str:
    return o if o in roster else "VAL"


def typed_sets(m: dict[str, Any], roster: set[str]) -> dict[str, frozenset]:
    tf = m["typed_facts"]
    return {
        "T3": frozenset((f["subject_id"] or "", f["predicate"], f["object"] if f["object"] in roster else step0.norm_text(f["object"])) for f in tf),
        "T2": frozenset((f["subject_id"] or "", f["predicate"], _kind(f["object"], roster)) for f in tf),
        "T1": frozenset((f["subject_id"] or "", f["predicate"]) for f in tf),
        "T0": frozenset((f["predicate"],) for f in tf),
    }


def _jac(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if (a | b) else 1.0


def _hash(x: frozenset) -> str:
    return hashlib.sha256(json.dumps(sorted(map(list, x)), ensure_ascii=False).encode()).hexdigest()[:16]


def cmd_analyze(a) -> int:
    from common import write_json

    WORK, RESULTS = S.WORK, S.RESULTS
    step0.WORK = WORK
    manifest = S._manifest()
    rows = step0._collect(manifest)
    labels = {w: {i: l for i, l in S.LOCATIONS[w]} for w in S.LOCATIONS}
    recs: dict[str, dict[str, Any]] = {}
    for pid, r in rows.items():
        mp = WORK / "out" / f"{pid}.m05.json"
        pack = json.loads((WORK / "packs" / f"{pid}.pack.json").read_text(encoding="utf-8"))
        m = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else None
        ok = r["entry"] is not None and m is not None and not S.validate_m05(m, pack["roster_all_ids"], r["work"])
        recs[pid] = {**{k: r[k] for k in ("kind", "work", "scene", "run", "category", "attempts", "rejects", "seconds", "accepted")},
                     "ok": ok, "strong_entry": r["entry"], "m05": m, "roster": pack["roster_all_ids"], "reject_reasons": r["reject_reasons"]}
    by: dict[tuple[str, int], list[dict]] = {}
    for r in recs.values():
        if r["ok"]:
            by.setdefault((r["work"], r["scene"]), []).append(r)
    LV = ("T3", "T2", "T1", "T0")
    per_scene = []
    for (w, s), rs in sorted(by.items()):
        rs = sorted(rs, key=lambda r: r["run"])
        n = len(rs)
        if n < 2:
            continue
        roster = set(rs[0]["roster"])
        ts = [typed_sets(r["m05"], roster) for r in rs]
        pairs = list(itertools.combinations(range(n), 2))
        lc = [r["m05"]["location_choice"] for r in rs]
        lcl = [None if c == "none" else (labels[w][c] if c in labels[w] else c) for c in lc]
        free = [r["strong_entry"].get("location") for r in rs]
        pres = [tuple(sorted(p["id"] for p in r["strong_entry"]["present"])) for r in rs]
        pov = [r["strong_entry"].get("pov_character") for r in rs]

        def ph(i: int, mode: str) -> str:
            e = copy.deepcopy(rs[i]["strong_entry"])
            if mode == "closed":
                e["location"] = lcl[i]
            elif mode == "none":
                e["location"] = None
            return step0.prior_hash(e)

        phs = {m_: [ph(i, m_) for i in range(n)] for m_ in ("free", "closed", "none")}
        fn = [step0.facts_norm_hash(r["strong_entry"]) for r in rs]
        tf_all = [f for r in rs for f in r["m05"]["typed_facts"]]
        d: dict[str, Any] = {
            "work": w, "scene": s, "n_ok": n,
            "n_typed_mean": round(st.fmean(len(r["m05"]["typed_facts"]) for r in rs), 2),
            "typed_empty_rate": round(sum(not r["m05"]["typed_facts"] for r in rs) / n, 4),
            "other_pred_rate": round(sum(f["predicate"] == "other" for f in tf_all) / len(tf_all), 4) if tf_all else None,
            "location_closed_pair_agree": round(step0.pair_rate(lc), 4),
            "location_closed_none_rate": round(lc.count("none") / n, 4),
            "location_closed_other_rate": round(sum(str(c).startswith("other:") for c in lc) / n, 4),
            "location_freetext_pair_agree_same_run": round(step0.pair_rate(free), 4),
            "location_freetext_none_rate_same_run": round(sum(x is None for x in free) / n, 4),
            "present_pair_agree": round(step0.pair_rate(pres), 4), "pov_pair_agree": round(step0.pair_rate(pov), 4),
            "prior_hash_pair_agree_freetext": round(step0.pair_rate(phs["free"]), 4),
            "prior_hash_pair_agree_closed_location": round(step0.pair_rate(phs["closed"]), 4),
            "prior_hash_pair_agree_without_location": round(step0.pair_rate(phs["none"]), 4),
            "facts_norm_hash_pair_agree_step0_style_same_run": round(step0.pair_rate(fn), 4),
        }
        for lv in LV:
            hs = [_hash(t[lv]) for t in ts]
            d[f"{lv}_hash_pair_agree"] = round(step0.pair_rate(hs), 4)
            d[f"{lv}_hash_all_equal"] = len(set(hs)) == 1
            d[f"{lv}_jaccard_mean"] = round(st.fmean(_jac(ts[i][lv], ts[j][lv]) for i, j in pairs), 4)
        per_scene.append(d)

    keys = [k for k, v in per_scene[0].items() if k not in ("work", "scene", "n_ok") and not isinstance(v, bool)]

    def agg(xs: list[dict]) -> dict[str, Any]:
        o: dict[str, Any] = {}
        for k in keys:
            v = [x[k] for x in xs if x.get(k) is not None]
            o[k] = round(st.fmean(v), 4) if v else None
        for lv in LV:
            o[f"{lv}_hash_all_equal_rate"] = round(sum(x[f"{lv}_hash_all_equal"] for x in xs) / len(xs), 4)
        o["scenes"] = len(xs)
        return o

    overall = agg(per_scene)
    by_work = {w: agg([x for x in per_scene if x["work"] == w]) for w in S.LOCATIONS if any(x["work"] == w for x in per_scene)}
    by_cat = {c: agg([x for x in per_scene if step0.SELECT[x["work"]][0] == c]) for c in ("dialogue", "transition")}
    pc: Counter = Counter()
    lca: Counter = Counter()
    for r in recs.values():
        if r["ok"]:
            pc.update(f["predicate"] for f in r["m05"]["typed_facts"])
            c = r["m05"]["location_choice"]
            lca[("other:*" if str(c).startswith("other:") else c)] += 1
    other_texts = [r["m05"]["location_choice"] for r in recs.values() if r["ok"] and str(r["m05"]["location_choice"]).startswith("other:")]
    other_objs = [(r["work"], f["subject_id"], f["object"]) for r in recs.values() if r["ok"] for f in r["m05"]["typed_facts"] if f["predicate"] == "other"]
    secs = [r["seconds"] for r in recs.values() if r["ok"] and r["seconds"] is not None]
    timing = ({"n": len(secs), "mean": round(st.fmean(secs), 1), "median": round(st.median(secs), 1), "min": round(min(secs), 1), "max": round(max(secs), 1)} if secs else None)
    calls = {"packs": len(recs), "accepted_both": sum(r["ok"] for r in recs.values()),
             "attempts_total": sum(r["attempts"] for r in recs.values()), "rejects_total": sum(r["rejects"] for r in recs.values())}
    summary = {"overall": overall, "by_work": by_work, "by_category": by_cat, "per_scene": per_scene, "timing_seconds_per_annotation": timing,
               "calls": calls, "predicate_counts": dict(pc.most_common()), "location_choice_counts": dict(lca.most_common()),
               "location_other_texts": other_texts, "predicate_other_objects": other_objs}
    write_json(RESULTS / "phase15_stage2_step05_raw.json", {"calls": calls, "packs": recs})
    write_json(RESULTS / "phase15_stage2_step05_summary.json", summary)
    print(json.dumps({"overall": overall, "timing": timing, "calls": calls}, ensure_ascii=False, indent=1))
    return 0
