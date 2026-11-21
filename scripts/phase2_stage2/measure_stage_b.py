"""Stage B: measure sensor.v2 scene annotations against the independently written gold.

Matching rules (fixed before looking at predictions):
  present        : micro P/R over (scene, character_id).
  time.order     : relative_order equality, per scene.
  time.explicit  : both empty, or prediction non-empty and matches a gold surface (containment either way).
  location       : both empty, or prediction matches a gold surface (containment either way).
  fact_updates   : one-to-one greedy match within a scene. match = same predicate AND object match, where
                   object match = normalized containment either way, or bigram overlap coefficient >= 0.5.
                   Strict variant additionally requires equal subject. P/R micro over all scenes.
  values_6d      : over (scene, character) present in both gold states and predicted states:
                   exact-vector rate; per-axis exact rate; polarity agreement on axes where both are nonzero;
                   recall of gold-nonzero axes with the same sign. Also character-set P/R of states.
                   Baseline: an all-zero predictor on the same gold characters.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANN = HERE / "annotations"


def norm(s: str) -> str:
    return re.sub(r"[\s　「」『』、。（）()]", "", s or "")


def bigrams(s: str) -> set[str]:
    return {s[i : i + 2] for i in range(len(s) - 1)}


def text_match(a: str, b: str) -> bool:
    a, b = norm(a), norm(b)
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    ba, bb = bigrams(a), bigrams(b)
    if not ba or not bb:
        return False
    return len(ba & bb) / min(len(ba), len(bb)) >= 0.5


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}


def sign(x: int) -> int:
    return (x > 0) - (x < 0)


def measure(gold: dict, preds: dict, scene_ids: list[str]) -> dict:
    pres = [0, 0, 0]
    base = [0, 0, 0]
    order_ok = order_n = 0
    et_ok = loc_ok = 0
    fact_t = [0, 0, 0]
    fact_s = [0, 0, 0]
    pred_cnt: dict[str, int] = {}
    vec_ok = vec_n = ax_ok = ax_n = pol_ok = pol_n = nz_gold = nz_hit = 0
    zero_vec_ok = zero_ax_ok = 0
    st = [0, 0, 0]
    per_scene = []
    for sid in scene_ids:
        g, a = gold[sid], preds[sid]
        sm = a["scene_meta"]
        # present
        gp, pp = set(g["present"]), set(sm["present_character_ids"])
        pres[0] += len(gp & pp); pres[1] += len(pp - gp); pres[2] += len(gp - pp)
        mech = {u["speaker_id"] for u in a["utterances"] if u["speaker_id"]} | {u["addressee_id"] for u in a["utterances"] if u["addressee_id"]}
        if a["contract"].get("_narrator"):
            mech.add(a["contract"]["_narrator"])
        base[0] += len(gp & mech); base[1] += len(mech - gp); base[2] += len(gp - mech)
        # time / location
        order_n += 1
        order_ok += sm["time"]["relative_order"] == g["relative_order"]
        pe = sm["time"].get("explicit_time")
        ge = g["explicit_time"]
        et_ok += (not pe and not ge) or (bool(pe) and any(text_match(pe, x) for x in ge))
        pl = sm["location_id"]
        gl = g["location_surfaces"]
        loc_ok += (not pl and not gl) or (bool(pl) and any(text_match(pl, x) for x in gl))
        # facts
        for variant, acc in (("pred_obj", fact_t), ("strict", fact_s)):
            used: set[int] = set()
            tp = 0
            for f in a["fact_updates"]:
                for i, gf in enumerate(g["facts"]):
                    if i in used or gf["predicate"] != f["predicate"] or not text_match(f["object"], gf["object"]):
                        continue
                    if variant == "strict" and gf["subject"] != f["subject"]:
                        continue
                    used.add(i)
                    tp += 1
                    break
            acc[0] += tp; acc[1] += len(a["fact_updates"]) - tp; acc[2] += len(g["facts"]) - tp
        for f in a["fact_updates"]:
            pred_cnt[f["predicate"]] = pred_cnt.get(f["predicate"], 0) + 1
        # values
        pst = {s["character_id"]: s["values_6d"] for s in a["character_states"]}
        gst = g["states"]
        st[0] += len(set(pst) & set(gst)); st[1] += len(set(pst) - set(gst)); st[2] += len(set(gst) - set(pst))
        for cid, gv in gst.items():
            zero_vec_ok += all(x == 0 for x in gv)
            zero_ax_ok += sum(x == 0 for x in gv)
            if cid not in pst:
                continue
            pv = pst[cid]
            vec_n += 1
            vec_ok += pv == gv
            for x, y in zip(pv, gv):
                ax_n += 1
                ax_ok += x == y
                if x != 0 and y != 0:
                    pol_n += 1
                    pol_ok += sign(x) == sign(y)
                if y != 0:
                    nz_gold += 1
                    nz_hit += sign(x) == sign(y)
        per_scene.append({
            "scene_id": sid, "present_gold": sorted(gp), "present_pred": sorted(pp),
            "order_gold": g["relative_order"], "order_pred": sm["time"]["relative_order"],
            "facts_gold": len(g["facts"]), "facts_pred": len(a["fact_updates"]),
            "states_gold": sorted(gst), "states_pred": sorted(pst),
        })
    n_gold_chars = sum(len(gold[s]["states"]) for s in scene_ids)
    n_gold_axes = n_gold_chars * 6
    return {
        "scenes": len(scene_ids),
        "present": prf(*pres),
        "present_mechanical_baseline": prf(*base),
        "time_relative_order": {"match": order_ok, "n": order_n, "rate": order_ok / order_n},
        "time_explicit": {"match": int(et_ok), "n": order_n, "rate": et_ok / order_n},
        "location": {"match": int(loc_ok), "n": order_n, "rate": loc_ok / order_n},
        "fact_updates_pred_obj": prf(*fact_t),
        "fact_updates_strict_subject": prf(*fact_s),
        "fact_pred_predicate_counts": pred_cnt,
        "states_character_set": prf(*st),
        "values_6d": {
            "matched_character_scenes": vec_n,
            "vector_exact": {"match": vec_ok, "n": vec_n, "rate": vec_ok / vec_n if vec_n else None},
            "axis_exact": {"match": ax_ok, "n": ax_n, "rate": ax_ok / ax_n if ax_n else None},
            "polarity_both_nonzero": {"match": pol_ok, "n": pol_n, "rate": pol_ok / pol_n if pol_n else None},
            "gold_nonzero_axes_same_sign": {"match": nz_hit, "n": nz_gold, "rate": nz_hit / nz_gold if nz_gold else None},
            "all_zero_baseline": {
                "gold_characters": n_gold_chars,
                "vector_exact_rate": zero_vec_ok / n_gold_chars if n_gold_chars else None,
                "axis_exact_rate": zero_ax_ok / n_gold_axes if n_gold_axes else None,
            },
        },
        "per_scene": per_scene,
    }


def main() -> int:
    ann_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ANN
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "results/stageB_metrics.json"
    gold = json.loads((HERE / "gold/stage_b_gold.json").read_text(encoding="utf-8"))["scenes"]
    modern_ids = ["001798_056779", "001801_001786", "001875_059897"]
    preds: dict[str, dict] = {}
    for f in sorted(ann_dir.glob("*.annotations.json")):
        wid = f.name.split(".")[0]
        for a in json.loads(f.read_text(encoding="utf-8")):
            preds[a["contract"]["scene_id"]] = a
    # narrator for mechanical baseline (first-person works)
    narr = {"001798_056779": "boku", "000035_000301": "yozo", "001383_056643": "hagiwara"}
    for sid, a in preds.items():
        a["contract"]["_narrator"] = narr.get(sid.split(":")[0])
    res = {"annotations_dir": str(ann_dir)}
    groups = {
        "modern_11": [s for s in gold if s.split(":")[0] in modern_ids],
        "f9dev_24": [s for s in gold if s.split(":")[0] not in modern_ids],
    }
    missing = [s for s in gold if s not in preds]
    res["missing_scenes"] = missing
    for name, ids in groups.items():
        ids = [s for s in ids if s in preds]
        if ids:
            res[name] = measure(gold, preds, ids)
    allids = [s for s in gold if s in preds]
    res["all"] = measure(gold, preds, allids)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for name in ("modern_11", "f9dev_24", "all"):
        if name in res:
            r = res[name]
            print(name, "present", r["present"]["precision"], r["present"]["recall"], "| order", r["time_relative_order"]["rate"],
                  "| loc", r["location"]["rate"], "| facts", r["fact_updates_pred_obj"]["precision"], r["fact_updates_pred_obj"]["recall"],
                  "| vec", r["values_6d"]["vector_exact"]["rate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
