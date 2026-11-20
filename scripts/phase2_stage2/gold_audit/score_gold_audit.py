"""Step 2: Score blind Google-model annotations against stage_b_gold.json (modern_11).

Uses the exact matching rules from scripts/phase2_stage2/measure_stage_b.py and adds
per-person values_6d aggregation + full per-scene disagreement dumps.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
STAGE2_DIR = HERE.parent
REPO = HERE.parents[2]
sys.path.insert(0, str(STAGE2_DIR))

from measure_stage_b import measure, norm, prf, sign, text_match  # noqa: E402

GOLD_PATH = STAGE2_DIR / "gold/stage_b_gold.json"
ANN_DIR = HERE / "annotations"
RESULTS_DIR = HERE / "results"
MODERN_IDS = ["001798_056779", "001801_001786", "001875_059897"]


def aggregate_person_states(
    scene_states_by_work: dict[str, dict[str, list[list[int]]]]
) -> tuple[dict[str, list[int]], dict[str, list[dict[str, Any]]]]:
    """Aggregate per-scene 6D vectors to person level via sign(sum(vals)) per axis.

    Also records any axis where a character has both +1 and -1 across scenes.
    """
    agg: dict[str, list[int]] = {}
    conflicts: dict[str, list[dict[str, Any]]] = {}
    for person_key, vecs in sorted(scene_states_by_work.items()):
        out_vec = []
        person_conflicts = []
        for ax in range(6):
            vals = [v[ax] for v in vecs]
            s = sign(sum(vals))
            out_vec.append(s)
            if any(x > 0 for x in vals) and any(x < 0 for x in vals):
                person_conflicts.append({"axis": ax, "scene_values": vals, "agg_sign": s})
        agg[person_key] = out_vec
        if person_conflicts:
            conflicts[person_key] = person_conflicts
    return agg, conflicts


def score_person_values(
    gold_person: dict[str, list[int]],
    pred_person: dict[str, list[int]],
) -> dict[str, Any]:
    common = sorted(set(gold_person) & set(pred_person))
    vec_ok = vec_n = ax_ok = ax_n = pol_ok = pol_n = nz_gold = nz_hit = 0
    zero_vec_ok = zero_ax_ok = 0
    per_person = {}
    for pk in sorted(set(gold_person) | set(pred_person)):
        gv = gold_person.get(pk)
        pv = pred_person.get(pk)
        if gv is not None:
            zero_vec_ok += all(x == 0 for x in gv)
            zero_ax_ok += sum(x == 0 for x in gv)
        if gv is not None and pv is not None:
            vec_n += 1
            exact = pv == gv
            vec_ok += exact
            for x, y in zip(pv, gv):
                ax_n += 1
                ax_ok += x == y
                if x != 0 and y != 0:
                    pol_n += 1
                    pol_ok += sign(x) == sign(y)
                if y != 0:
                    nz_gold += 1
                    nz_hit += sign(x) == sign(y)
        per_person[pk] = {
            "gold": gv,
            "pred": pv,
            "vector_exact": (pv == gv) if (gv is not None and pv is not None) else None,
        }
    n_gold = len(gold_person)
    return {
        "gold_persons": n_gold,
        "pred_persons": len(pred_person),
        "matched_persons": vec_n,
        "vector_exact": {"match": vec_ok, "n": vec_n, "rate": vec_ok / vec_n if vec_n else None},
        "axis_exact": {"match": ax_ok, "n": ax_n, "rate": ax_ok / ax_n if ax_n else None},
        "polarity_both_nonzero": {"match": pol_ok, "n": pol_n, "rate": pol_ok / pol_n if pol_n else None},
        "gold_nonzero_axes_same_sign": {"match": nz_hit, "n": nz_gold, "rate": nz_hit / nz_gold if nz_gold else None},
        "all_zero_baseline": {
            "gold_persons": n_gold,
            "vector_exact_rate": zero_vec_ok / n_gold if n_gold else None,
            "axis_exact_rate": zero_ax_ok / (n_gold * 6) if n_gold else None,
        },
        "per_person": per_person,
    }


def fact_match_details(
    gold_facts: list[dict[str, Any]],
    pred_facts: list[dict[str, Any]],
    strict: bool,
) -> dict[str, Any]:
    used: set[int] = set()
    matched = []
    unmatched_pred = []
    for pi, f in enumerate(pred_facts):
        hit = None
        for gi, gf in enumerate(gold_facts):
            if gi in used or gf["predicate"] != f["predicate"] or not text_match(f["object"], gf["object"]):
                continue
            if strict and gf["subject"] != f["subject"]:
                continue
            used.add(gi)
            hit = gi
            break
        if hit is not None:
            matched.append({"pred_index": pi, "gold_index": hit, "pred": f, "gold": gold_facts[hit]})
        else:
            unmatched_pred.append({"pred_index": pi, "fact": f})
    unmatched_gold = [
        {"gold_index": gi, "fact": gf} for gi, gf in enumerate(gold_facts) if gi not in used
    ]
    return {
        "matched": matched,
        "unmatched_gold_fn": unmatched_gold,
        "unmatched_pred_fp": unmatched_pred,
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    gold_all = json.loads(GOLD_PATH.read_text(encoding="utf-8"))["scenes"]
    modern_sids = [s for s in gold_all if s.split(":")[0] in MODERN_IDS]

    preds: dict[str, dict[str, Any]] = {}
    for wid in MODERN_IDS:
        f = ANN_DIR / f"{wid}.annotations.json"
        for a in json.loads(f.read_text(encoding="utf-8")):
            preds[a["contract"]["scene_id"]] = a

    missing = [s for s in modern_sids if s not in preds]
    assert not missing, f"Missing scenes: {missing}"

    narr = {"001798_056779": "boku"}
    for sid, a in preds.items():
        a["contract"]["_narrator"] = narr.get(sid.split(":")[0])

    # 1. Standard Stage B measurement (identical to measure_stage_b.py)
    stage_b_modern = measure(gold_all, preds, modern_sids)

    # 2. Per-person values_6d aggregation and scoring
    gold_by_person: dict[str, list[list[int]]] = defaultdict(list)
    gold_scene_vecs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pred_by_person: dict[str, list[list[int]]] = defaultdict(list)
    pred_scene_vecs: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for sid in modern_sids:
        wid = sid.split(":")[0]
        for cid, gv in gold_all[sid]["states"].items():
            pk = f"{wid}:{cid}"
            gold_by_person[pk].append(gv)
            gold_scene_vecs[pk].append({"scene_id": sid, "values_6d": gv})
        for st in preds[sid]["character_states"]:
            pk = f"{wid}:{st['character_id']}"
            pred_by_person[pk].append(st["values_6d"])
            pred_scene_vecs[pk].append({"scene_id": sid, "values_6d": st["values_6d"]})

    gold_person_agg, gold_conflicts = aggregate_person_states(gold_by_person)
    pred_scene_person_agg, pred_conflicts = aggregate_person_states(pred_by_person)

    person_raw = json.loads((ANN_DIR / "person_values_6d.json").read_text(encoding="utf-8"))
    pred_explicit_person = {k: v["values_6d"] for k, v in person_raw.items()}

    person_eval_explicit = score_person_values(gold_person_agg, pred_explicit_person)
    person_eval_scene_agg = score_person_values(gold_person_agg, pred_scene_person_agg)

    per_person_detail = {}
    for pk in sorted(set(gold_person_agg) | set(pred_explicit_person) | set(pred_scene_person_agg)):
        per_person_detail[pk] = {
            "gold_per_scene": gold_scene_vecs.get(pk, []),
            "gold_agg": gold_person_agg.get(pk),
            "gold_axis_conflicts": gold_conflicts.get(pk, []),
            "pred_explicit_person": pred_explicit_person.get(pk),
            "pred_per_scene": pred_scene_vecs.get(pk, []),
            "pred_per_scene_agg": pred_scene_person_agg.get(pk),
            "pred_axis_conflicts": pred_conflicts.get(pk, []),
            "rationale": person_raw.get(pk, {}).get("rationale"),
        }

    # 3. Per-scene disagreement dump (facts, present, time, location, values_6d)
    disagreements = {}
    for sid in modern_sids:
        g = gold_all[sid]
        a = preds[sid]
        sm = a["scene_meta"]
        gp, pp = set(g["present"]), set(sm["present_character_ids"])
        pe = sm["time"].get("explicit_time")
        ge = g["explicit_time"]
        et_ok = (not pe and not ge) or (bool(pe) and any(text_match(pe, x) for x in ge))
        pl = sm["location_id"]
        gl = g["location_surfaces"]
        loc_ok = (not pl and not gl) or (bool(pl) and any(text_match(pl, x) for x in gl))

        gst = g["states"]
        pst = {s["character_id"]: s["values_6d"] for s in a["character_states"]}

        disagreements[sid] = {
            "present": {
                "gold": sorted(gp),
                "pred": sorted(pp),
                "tp": sorted(gp & pp),
                "fp_pred_only": sorted(pp - gp),
                "fn_gold_only": sorted(gp - pp),
            },
            "time": {
                "relative_order_gold": g["relative_order"],
                "relative_order_pred": sm["time"]["relative_order"],
                "relative_order_match": sm["time"]["relative_order"] == g["relative_order"],
                "explicit_time_gold": ge,
                "explicit_time_pred": pe,
                "explicit_time_match": et_ok,
            },
            "location": {
                "gold": gl,
                "pred": pl,
                "match": loc_ok,
            },
            "states": {
                "gold": gst,
                "pred": pst,
                "fp_pred_only": sorted(set(pst) - set(gst)),
                "fn_gold_only": sorted(set(gst) - set(pst)),
            },
            "facts_pred_obj": fact_match_details(g["facts"], a["fact_updates"], strict=False),
            "facts_strict": fact_match_details(g["facts"], a["fact_updates"], strict=True),
        }

    metrics_out = {
        "audit_model": "google-gemini-independent-audit",
        "gold_path": "scripts/phase2_stage2/gold/stage_b_gold.json",
        "annotations_dir": "scripts/phase2_stage2/gold_audit/annotations",
        "modern_11": stage_b_modern,
        "person_values_6d": {
            "aggregation_rule": "sign(sum(scene_values)) across scenes where character appears in states",
            "gold_axis_conflicts": gold_conflicts,
            "pred_axis_conflicts": pred_conflicts,
            "vs_explicit_person_values": person_eval_explicit,
            "vs_scene_aggregated_values": person_eval_scene_agg,
            "per_person_detail": per_person_detail,
        },
    }

    (RESULTS_DIR / "stageB_audit_metrics.json").write_text(
        json.dumps(metrics_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (RESULTS_DIR / "fact_disagreements.json").write_text(
        json.dumps(disagreements, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    r = stage_b_modern
    print("=== Stage B (modern_11) ===")
    print(f"present: P={r['present']['precision']:.4f} R={r['present']['recall']:.4f} F1={r['present']['f1']:.4f} (tp={r['present']['tp']} fp={r['present']['fp']} fn={r['present']['fn']})")
    print(f"time.order: {r['time_relative_order']['match']}/{r['time_relative_order']['n']} ({r['time_relative_order']['rate']:.4f})")
    print(f"time.explicit: {r['time_explicit']['match']}/{r['time_explicit']['n']} ({r['time_explicit']['rate']:.4f})")
    print(f"location: {r['location']['match']}/{r['location']['n']} ({r['location']['rate']:.4f})")
    print(f"facts(pred_obj): P={r['fact_updates_pred_obj']['precision']:.4f} R={r['fact_updates_pred_obj']['recall']:.4f} F1={r['fact_updates_pred_obj']['f1']:.4f} (tp={r['fact_updates_pred_obj']['tp']} fp={r['fact_updates_pred_obj']['fp']} fn={r['fact_updates_pred_obj']['fn']})")
    print(f"facts(strict):   P={r['fact_updates_strict_subject']['precision']:.4f} R={r['fact_updates_strict_subject']['recall']:.4f} F1={r['fact_updates_strict_subject']['f1']:.4f} (tp={r['fact_updates_strict_subject']['tp']} fp={r['fact_updates_strict_subject']['fp']} fn={r['fact_updates_strict_subject']['fn']})")
    v = r["values_6d"]
    print(f"values_6d(scene): matched={v['matched_character_scenes']} vec_exact={v['vector_exact']['rate']:.4f} axis_exact={v['axis_exact']['rate']:.4f} pol_both_nz={v['polarity_both_nonzero']['rate']} gold_nz_same_sign={v['gold_nonzero_axes_same_sign']['rate']:.4f}")
    pe = person_eval_explicit
    print(f"values_6d(person_explicit): matched={pe['matched_persons']} vec_exact={pe['vector_exact']['rate']:.4f} axis_exact={pe['axis_exact']['rate']:.4f} pol_both_nz={pe['polarity_both_nonzero']['rate']} gold_nz_same_sign={pe['gold_nonzero_axes_same_sign']['rate']:.4f}")
    ps = person_eval_scene_agg
    print(f"values_6d(person_scene_agg): matched={ps['matched_persons']} vec_exact={ps['vector_exact']['rate']:.4f} axis_exact={ps['axis_exact']['rate']:.4f} pol_both_nz={ps['polarity_both_nonzero']['rate']} gold_nz_same_sign={ps['gold_nonzero_axes_same_sign']['rate']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
