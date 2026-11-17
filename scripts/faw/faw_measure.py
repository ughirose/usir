#!/usr/bin/env python3
"""Stage FAW task A: read-only measurement of Awase / Nagare sparsity and between-work variance share.
No mainline code is modified. Deterministic (no RNG). Raw text is read from data/aozora_cache (gitignored);
only numbers are written to scripts/faw/faw_measure_results.json.

Cohort AOZORA106: aozora_manifest(novel)+heldout+heldout2 (106 works); annotation ints are re-derived by
  compute_heuristic_signals and checked against data/v5/aozora_106works_scenes.jsonl.
Cohort POSTWAR: aozora_p2b_manifest + aozora_p2c_manifest (56 works, 1,877 scenes, P-2c frozen corpus).
Run: python scripts/faw/faw_measure.py
"""
from __future__ import annotations
import json, sys, math
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
from usir_py.v5 import split_work_into_scenes  # noqa: E402
from usir_py.v5 import aozora_scene_splitter as S  # noqa: E402
from usir_py.v5 import six_pole_atlas as A  # noqa: E402

OUT = Path(__file__).with_name("faw_measure_results.json")


def raw_signals(scene_text: str):
    """Replica of the pre-quantization values in compute_heuristic_signals (same module constants)."""
    total_len = max(S.effective_len(scene_text), 1)
    clean = S.strip_commas(scene_text)
    qs = S.RE_QUOTE.findall(scene_text)
    quote_chars = sum(S.effective_len(m[0] or m[1]) for m in qs)
    dlg = quote_chars / total_len
    narr = S.RE_QUOTE.sub(" ", scene_text)
    narr_len = max(S.effective_len(narr), 1)
    excl = scene_text.count("！") + scene_text.count("!")
    pe, ne = S.score_segment_polarity(narr, S.EVENT_POS_WORDS, S.EVENT_NEG_WORDS)
    ve = (pe - ne) / max(pe + ne, 2.0)
    eh = sum(narr.count(w) for w in S.EVENT_HIGH_ARO_WORDS)
    el = sum(narr.count(w) for w in S.EVENT_LOW_ARO_WORDS)
    ae = (eh * 1.5 - el) / max(narr_len / 250.0, 1.0)
    ps, ns = S.score_segment_polarity(scene_text, S.STYLE_POS_WORDS, S.STYLE_NEG_WORDS)
    vs = (ps - ns) / max(ps + ns, 2.0)
    cnp = S.re.sub(r"[。！？!?、，\s]+$", "", clean)
    sh = excl * 1.5 + (1.0 if any(cnp.endswith(x) for x in ("ぞ", "ぜ", "わい", "たまえ", "よ")) else 0.0) + (1.0 if dlg > 0.4 else 0.0)
    sl = sum(clean.count(w) for w in ["ひっそり", "静か", "ぽつねんと", "ぼんやり", "幽か", "かすか", "そっと"])
    as_ = (sh * 1.2 - sl) / max(total_len / 300.0, 1.0)
    return ve, ae, vs, as_


def norm(v):
    return max(-1.0, min(1.0, v / 2.0))


def eta2(values, groups):
    v = np.asarray(values, float); g = np.asarray(groups)
    if v.size < 3: return float("nan")
    sst = float(np.sum((v - v.mean()) ** 2))
    if sst <= 0: return float("nan")
    ssb = 0.0
    for k in np.unique(g):
        m = v[g == k]; ssb += m.size * (m.mean() - v.mean()) ** 2
    return ssb / sst


def qs(x):
    x = np.asarray(x, float)
    return {p: float(np.quantile(x, p / 100)) for p in (5, 25, 50, 75, 95)}


def load_works(cohort):
    if cohort == "AOZORA106":
        sp = S.AozoraCorpusSplitter(REPO / "data")
        for w in sp.load_manifests():
            yield f"{w['author_id']}:{w['work_id']}", sp.get_cached_text(w)
    else:
        for mf, pre in (("aozora_p2b_manifest.json", "p2b"), ("aozora_p2c_manifest.json", "p2c")):
            man = json.loads((REPO / "data" / mf).read_text(encoding="utf-8"))
            for w in man["works"]:
                t = (REPO / "data" / "aozora_cache" / f"{pre}_{w['author_id']}_{w['work_id']}.txt").read_text(encoding="utf-8")
                yield f"{w['author_id']}:{w['work_id']}", t


def measure(cohort):
    rows = []  # per scene
    works = {}
    for key, text in load_works(cohort):
        scenes = split_work_into_scenes(text)
        recs = []
        for i, sc in enumerate(scenes):
            aw, nag, _, _, _ = S.compute_heuristic_signals(sc, i, len(scenes))
            ve, ae, vs, as_ = raw_signals(sc)
            rows.append(dict(work=key, aw=aw, raw=(ve, ae, vs, as_)))
            recs.append({"awase": aw, "nagare": nag})
        works[key] = recs
    n = len(rows)
    res = {"n_scenes": n, "n_works": len(works)}
    # consistency of raw replica with stored ints
    mis = 0
    for r in rows:
        ve, ae, vs, as_ = r["raw"]
        q = (S.quantize_val(ve), S.quantize_aro(ae), S.quantize_val(vs), S.quantize_aro(as_))
        a = r["aw"]
        if q != (a["f_event_valence"], a["f_event_arousal"], a["f_style_valence"], a["f_style_arousal"]): mis += 1
    res["raw_replica_mismatch_scenes"] = mis
    grp = [r["work"] for r in rows]
    VF = np.array([norm(r["aw"]["f_event_valence"]) for r in rows]); VS = np.array([norm(r["aw"]["f_style_valence"]) for r in rows])
    AF = np.array([norm(r["aw"]["f_event_arousal"]) for r in rows]); AS = np.array([norm(r["aw"]["f_style_arousal"]) for r in rows])
    ir = np.maximum(0.0, -VF * VS); de = np.abs(VF) * np.maximum(0.0, AF - AS)
    # unclipped signed quantities (design material, not current definition)
    sir = -VF * VS; sde = np.abs(VF) * (AF - AS)
    # raw (pre-quantization) signed analogues
    R = np.array([r["raw"] for r in rows])
    rir = -R[:, 0] * R[:, 2]
    res["awase_scene"] = {
        "irony_zero_rate": float(np.mean(ir == 0)), "detachment_zero_rate": float(np.mean(de == 0)),
        "irony_distinct_values": sorted({round(float(x), 4) for x in ir}), "detachment_distinct_values": sorted({round(float(x), 4) for x in de}),
        "P_vf0": float(np.mean(VF == 0)), "P_vs0": float(np.mean(VS == 0)), "P_vf_vs_opposite": float(np.mean(VF * VS < 0)),
        "P_vf0_or_vs0": float(np.mean((VF == 0) | (VS == 0))),
        "P_same_sign_nonzero": float(np.mean(VF * VS > 0)),
        "signed_irony_zero_rate": float(np.mean(sir == 0)), "signed_detachment_zero_rate": float(np.mean(sde == 0)),
        "raw_signed_irony_zero_rate": float(np.mean(rir == 0)),
        "raw_vf_zero_rate": float(np.mean(R[:, 0] == 0)), "raw_vs_zero_rate": float(np.mean(R[:, 2] == 0)),
        "eta2_irony": eta2(ir, grp), "eta2_detachment": eta2(de, grp),
        "eta2_signed_irony": eta2(sir, grp), "eta2_signed_detachment": eta2(sde, grp), "eta2_raw_signed_irony": eta2(rir, grp),
    }
    # work level (current 4D Awase as fed to six_pole_atlas)
    W = {"irony_mean": [], "detachment_mean": [], "irony_active": [], "irony_grad": [], "n_scenes": []}
    Wsigned = {"signed_irony_mean": [], "raw_signed_irony_mean": []}
    wi = {}
    for i, g in enumerate(grp): wi.setdefault(g, []).append(i)
    for g, idx in wi.items():
        p = A.compute_awase_profile([{"awase": rows[i]["aw"]} for i in idx])
        per = p["irony"]["per_scene"]
        if len(per) >= 5:
            k = max(1, len(per) // 5); grad = float(np.mean(per[-k:]) - np.mean(per[:k]))
        elif len(per) >= 2: grad = float(per[-1] - per[0])
        else: grad = 0.0
        W["irony_mean"].append(p["irony"]["mean"]); W["detachment_mean"].append(p["detachment"]["mean"])
        W["irony_active"].append(float(np.mean(np.array(per) > 0))); W["irony_grad"].append(grad); W["n_scenes"].append(len(per))
        Wsigned["signed_irony_mean"].append(float(np.mean(sir[idx]))); Wsigned["raw_signed_irony_mean"].append(float(np.mean(rir[idx])))
    res["awase_work"] = {k: {"zero_rate": float(np.mean(np.array(v) == 0)), "quantiles": qs(v), "distinct": len({round(x, 6) for x in v})}
                         for k, v in {**W, **Wsigned}.items() if k != "n_scenes"}
    res["awase_work"]["n_scenes_quantiles"] = qs(W["n_scenes"]); res["awase_work"]["min_scenes"] = int(min(W["n_scenes"]))
    # Nagare
    feats = {"phi_zeami": [], "mood_speed_mean": [], "still_density": [], "speed_ratio": [], "tw": []}
    scene_ms, scene_ms_g, scene_es = [], [], []
    for g, recs in works.items():
        prof = A.compute_nagare_profile(recs)
        ms = prof["mood"]["speed"]; tw = len(recs)
        feats["phi_zeami"].append(prof["phi_zeami"]); feats["tw"].append(tw)
        feats["still_density"].append(len(prof["still_runs"]) / max(tw, 1))
        feats["mood_speed_mean"].append(float(np.mean(ms)) if ms else 0.5)
        if len(ms) >= 4:
            m = len(ms) // 2; feats["speed_ratio"].append((float(np.mean(ms[m:])) + 1e-4) / (float(np.mean(ms[:m])) + 1e-4))
        else: feats["speed_ratio"].append(1.0)
        scene_ms += ms; scene_ms_g += [g] * len(ms); scene_es += prof["event"]["speed"]
    res["nagare_work"] = {k: {"zero_rate": float(np.mean(np.array(v) == 0)), "quantiles": qs(v)} for k, v in feats.items() if k != "tw"}
    res["nagare_work"]["still_density_zero_count"] = int(sum(1 for v in feats["still_density"] if v == 0))
    res["nagare_work"]["speed_ratio_exact_1_count"] = int(sum(1 for v in feats["speed_ratio"] if v == 1.0))
    res["nagare_scene"] = {"mood_speed_quantiles": qs(scene_ms), "event_speed_quantiles": qs(scene_es),
                           "eta2_mood_speed": eta2(scene_ms, scene_ms_g),
                           "frac_mood_speed_below_0.25": float(np.mean(np.array(scene_ms) < 0.25)),
                           "frac_event_speed_below_0.25": float(np.mean(np.array(scene_es) < 0.25))}
    return res


def jsonl_check():
    """Reproduce 2026-11-13 numbers directly from the frozen annotation file."""
    rows = [json.loads(l) for l in (REPO / "data" / "v5" / "aozora_106works_scenes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    VF = np.array([norm(r["awase"]["f_event_valence"]) for r in rows]); VS = np.array([norm(r["awase"]["f_style_valence"]) for r in rows])
    AF = np.array([norm(r["awase"]["f_event_arousal"]) for r in rows]); AS = np.array([norm(r["awase"]["f_style_arousal"]) for r in rows])
    ir = np.maximum(0, -VF * VS); de = np.abs(VF) * np.maximum(0, AF - AS)
    return {"n": len(rows), "irony_zero_rate": float(np.mean(ir == 0)), "detachment_zero_rate": float(np.mean(de == 0))}


if __name__ == "__main__":
    out = {"frozen_jsonl": jsonl_check()}
    for c in ("AOZORA106", "POSTWAR"):
        out[c] = measure(c)
        print(c, json.dumps({k: out[c][k] for k in ("n_scenes", "n_works", "raw_replica_mismatch_scenes")}), flush=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT)
