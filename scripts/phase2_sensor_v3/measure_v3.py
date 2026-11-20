"""Measure sensor v3 runs against scripts/phase2_stage2/gold/stage_b_gold.json.

Usage: python measure_v3.py <out_json> <run_dir> [<run_dir> ...]

Matching rules (unchanged from phase2_stage2/measure_stage_b.py): fact = same predicate AND object text_match
(normalised containment either way or bigram overlap >= 0.5), one-to-one greedy within a scene; strict adds equal subject.
present = micro P/R over (scene, id); relative_order equality; explicit_time (surface rule) and location as in stage B.

New in v3:
  time_iso: share of scenes with a non-null ISO explicit_time; gold ISO = the same deterministic parser applied to the
            gold time surfaces (gold has no ISO field); match = both null or equal strings.
  traits  : per (work, character) gold trait = per axis, sign with more votes among the gold scenes' non-zero values
            (tie -> 0, axis excluded). Compared with the predicted per-work trait.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "python"))
sys.path.insert(0, str(HERE.parents[0] / "phase2_stage2"))

from measure_stage_b import prf, text_match  # noqa: E402
from usir_py.sensor_eval.scene_annotator_v3 import iso_from_surface  # noqa: E402

GOLD = HERE.parents[0] / "phase2_stage2/gold/stage_b_gold.json"
MODERN = ["001798_056779", "001801_001786", "001875_059897"]
PRIOR = HERE.parents[0] / "phase2_stage2/results/stageB_metrics.json"


def sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def match_facts(preds: list[dict], golds: list[dict], strict: bool) -> int:
    used: set[int] = set()
    tp = 0
    for f in preds:
        for i, gf in enumerate(golds):
            if i in used or gf["predicate"] != f["predicate"] or not text_match(f["object"], gf["object"]):
                continue
            if strict and gf["subject"] != f["subject"]:
                continue
            used.add(i)
            tp += 1
            break
    return tp


def gold_traits(gold: dict, scene_ids: list[str]) -> dict[tuple[str, str], dict]:
    votes: dict[tuple[str, str], list[list[int]]] = {}
    nscene: dict[tuple[str, str], int] = {}
    for sid in scene_ids:
        wid = sid.split(":")[0]
        for cid, v in gold[sid]["states"].items():
            k = (wid, cid)
            nscene[k] = nscene.get(k, 0) + 1
            acc = votes.setdefault(k, [[0, 0] for _ in range(6)])
            for ax, x in enumerate(v):
                if x > 0:
                    acc[ax][0] += 1
                elif x < 0:
                    acc[ax][1] += 1
    out = {}
    for k, acc in votes.items():
        tr = [1 if p > n else -1 if n > p else 0 for p, n in acc]
        out[k] = {"trait": tr, "n_scenes": nscene[k]}
    return out


def measure(gold: dict, run_dir: Path, scene_ids: list[str]) -> dict:
    preds: dict[str, dict] = {}
    traits: dict[tuple[str, str], list[int]] = {}
    for f in sorted(run_dir.glob("*.annotations.json")):
        wid = f.name.split(".")[0]
        for a in json.loads(f.read_text(encoding="utf-8")):
            preds[a["contract"]["scene_id"]] = a
        tf = run_dir / f"{wid}.traits.json"
        for cid, t in json.loads(tf.read_text(encoding="utf-8"))["traits"].items():
            traits[(wid, cid)] = t["values_6d"]
    ids = [s for s in scene_ids if s in preds]
    pres = [0, 0, 0]
    fa = {"all_pred_obj": [0, 0, 0], "all_strict": [0, 0, 0], "core6_pred_obj": [0, 0, 0], "core6_strict": [0, 0, 0]}
    per_pred: dict[str, list[int]] = {}
    order_ok = et_ok = loc_ok = 0
    par_gold = par_pred = par_hit = 0
    iso_pred = iso_match = iso_gold_resolvable = surf_pred = 0
    n_events = 0
    for sid in ids:
        g, a = gold[sid], preds[sid]
        sm = a["scene_meta"]
        gp, pp = set(g["present"]), set(sm["present_character_ids"])
        pres[0] += len(gp & pp); pres[1] += len(pp - gp); pres[2] += len(gp - pp)
        t = sm["time"]
        order_ok += t["relative_order"] == g["relative_order"]
        par_gold += g["relative_order"] == "parallel"
        par_pred += t["relative_order"] == "parallel"
        par_hit += g["relative_order"] == "parallel" and t["relative_order"] == "parallel"
        pe = t.get("explicit_time_surface")
        surf_pred += bool(pe)
        et_ok += (not pe and not g["explicit_time"]) or (bool(pe) and any(text_match(pe, x) for x in g["explicit_time"]))
        gi = next((iso for iso in (iso_from_surface(x) for x in g["explicit_time"]) if iso), None)
        pi = t.get("explicit_time")
        iso_pred += pi is not None
        iso_gold_resolvable += gi is not None
        iso_match += pi == gi
        pl, gl = sm["location_id"], g["location_surfaces"]
        loc_ok += (not pl and not gl) or (bool(pl) and any(text_match(pl, x) for x in gl))
        n_events += len(a["value_events"])
        pf = a["fact_updates"]
        gf = g["facts"]
        for name, (p_, g_, strict) in {
            "all_pred_obj": (pf, gf, False), "all_strict": (pf, gf, True),
            "core6_pred_obj": ([f for f in pf if f["predicate"] != "STATE_CHANGED"], [f for f in gf if f["predicate"] != "STATE_CHANGED"], False),
            "core6_strict": ([f for f in pf if f["predicate"] != "STATE_CHANGED"], [f for f in gf if f["predicate"] != "STATE_CHANGED"], True),
        }.items():
            tp = match_facts(p_, g_, strict)
            fa[name][0] += tp; fa[name][1] += len(p_) - tp; fa[name][2] += len(g_) - tp
        for pr in {f["predicate"] for f in pf} | {f["predicate"] for f in gf}:
            pp_ = [f for f in pf if f["predicate"] == pr]
            gg_ = [f for f in gf if f["predicate"] == pr]
            tp = match_facts(pp_, gg_, False)
            acc = per_pred.setdefault(pr, [0, 0, 0])
            acc[0] += tp; acc[1] += len(pp_) - tp; acc[2] += len(gg_) - tp
    n = len(ids)
    gt = gold_traits(gold, ids)
    nz = hit = both = both_ok = pred_nz = pred_nz_vs_gold_nz = 0
    nz2 = hit2 = 0
    char_cov = 0
    for k, g in gt.items():
        pt = traits.get(k)
        if pt is None:
            continue
        char_cov += 1
        for ax in range(6):
            gv, pv = g["trait"][ax], pt[ax]
            if gv != 0:
                nz += 1; hit += sgn(pv) == gv
                if g["n_scenes"] >= 2:
                    nz2 += 1; hit2 += sgn(pv) == gv
            if gv != 0 and pv != 0:
                both += 1; both_ok += sgn(pv) == gv
            if pv != 0:
                pred_nz += 1
                pred_nz_vs_gold_nz += gv != 0
    gold_nz_total = sum(sum(x != 0 for x in g["trait"]) for g in gt.values())
    return {
        "scenes": n,
        "present": prf(*pres),
        "facts": {k: prf(*v) for k, v in fa.items()},
        "fact_per_predicate_pred_obj": {k: prf(*v) for k, v in sorted(per_pred.items())},
        "time_relative_order": {"match": order_ok, "n": n, "rate": order_ok / n},
        "time_parallel": {"gold": par_gold, "pred": par_pred, "hit": par_hit},
        "time_explicit_surface_rule": {"match": et_ok, "n": n, "rate": et_ok / n},
        "time_surface_extracted": {"scenes": surf_pred, "n": n, "rate": surf_pred / n},
        "time_iso": {"scenes_with_iso": iso_pred, "rate": iso_pred / n, "gold_resolvable_scenes": iso_gold_resolvable,
                     "match_with_gold_iso": iso_match, "match_rate": iso_match / n},
        "location": {"match": loc_ok, "n": n, "rate": loc_ok / n},
        "value_events_total": n_events,
        "traits": {
            "gold_characters": len(gt), "gold_characters_with_prediction": char_cov,
            "gold_trait_nonzero_axes_total": gold_nz_total,
            "gold_nonzero_axes_evaluated": nz,
            "gold_nonzero_axes_same_sign": {"match": hit, "n": nz, "rate": hit / nz if nz else None},
            "gold_nonzero_axes_same_sign_chars_ge2_scenes": {"match": hit2, "n": nz2, "rate": hit2 / nz2 if nz2 else None},
            "polarity_both_nonzero": {"match": both_ok, "n": both, "rate": both_ok / both if both else None},
            "pred_nonzero_axes": pred_nz, "pred_nonzero_axes_where_gold_trait_nonzero": pred_nz_vs_gold_nz,
            "all_zero_baseline_same_sign_rate": 0.0,
        },
    }


def flat(d: dict, pre: str = "") -> dict[str, float]:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{pre}{k}."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[f"{pre}{k}"] = v
    return out


def main() -> int:
    out = Path(sys.argv[1])
    runs = [Path(p) for p in sys.argv[2:]]
    gold = json.loads(GOLD.read_text(encoding="utf-8"))["scenes"]
    groups = {"all": list(gold), "modern_11": [s for s in gold if s.split(":")[0] in MODERN], "f9dev_24": [s for s in gold if s.split(":")[0] not in MODERN]}
    res: dict = {"runs": [str(r) for r in runs], "per_run": {}, "range": {}}
    for gname, ids in groups.items():
        per = [measure(gold, r, ids) for r in runs]
        res["per_run"][gname] = per
        fl = [flat(p) for p in per]
        res["range"][gname] = {
            k: {"values": [f[k] for f in fl if k in f], "min": min(f[k] for f in fl if k in f), "max": max(f[k] for f in fl if k in f),
                "mean": sum(f[k] for f in fl if k in f) / len([f for f in fl if k in f])}
            for k in fl[0] if all(k in f for f in fl)
        }
    if PRIOR.exists():
        pr = json.loads(PRIOR.read_text(encoding="utf-8"))
        res["prior_stage2_single_run"] = {g: {"present": pr[g]["present"], "facts": pr[g]["fact_updates_pred_obj"], "facts_strict": pr[g]["fact_updates_strict_subject"],
                                              "order": pr[g]["time_relative_order"], "explicit": pr[g]["time_explicit"], "location": pr[g]["location"],
                                              "values_6d_gold_nonzero_same_sign": pr[g]["values_6d"]["gold_nonzero_axes_same_sign"]} for g in ("all", "modern_11", "f9dev_24") if g in pr}
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    a = res["range"]["all"]
    for k in ("present.precision", "present.recall", "facts.all_pred_obj.precision", "facts.all_pred_obj.recall", "facts.core6_pred_obj.precision",
              "facts.core6_pred_obj.recall", "time_relative_order.rate", "time_iso.rate", "traits.gold_nonzero_axes_same_sign.rate", "location.rate"):
        if k in a:
            print(k, [round(x, 3) for x in a[k]["values"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
