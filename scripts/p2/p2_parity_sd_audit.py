#!/usr/bin/env python
"""Stage P-1 Task 2/3: 1SD sensitivity audit on v1.1 (439) with F-17 vs 216 canonical floors.
Per pair: abs_max under both floors, edit-operation metrics (difflib char ops), flip list, proxy discrimination (AUC of abs_max vs edit scale),
threshold-candidate table. Also Python/Rust eval_rule v2 parity (439). Read-only for frozen files."""
import json, sys, hashlib, difflib, unicodedata
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python")); sys.path.insert(0, str(REPO / "scripts"))
import run_stage_b13_analysis as b13, run_stage_b14_analysis as b14
from usir_py.v5.f14_hybrid_noise_floor import load_f14_noise_floor

EXE = Path(r"C:\dev\nishiki\target\f14\release\nishiki.exe")
DS = REPO / "data/usir_v5_eval/real_revision_pairs_v1_1.jsonl"
V10 = REPO / "docs/handoff/v10"
F17 = V10 / "paragraph_noise_floor_all_novel.json"; C216 = V10 / "paragraph_noise_floor_all_216_canonical.json"
OUT = Path(__file__).resolve().parent / "p2_parity_sd_audit.json"
rows = [json.loads(l) for l in DS.read_text(encoding="utf-8").splitlines() if l.strip()]
assert len(rows) == 439
PUNCT = set("、。，．・「」『』（）()〔〕［］…―ー！？!?")
def cclass(ch):
    if ch in PUNCT: return "punct"
    n = unicodedata.name(ch, "")
    if n.startswith("HIRAGANA"): return "hira"
    if n.startswith("KATAKANA"): return "kata"
    if n.startswith("CJK"): return "kanji"
    return "other"

def edit_metrics(a, b):
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    ins = dele = rep_a = rep_b = 0; nblocks = 0; cls = {"punct": 0, "hira": 0, "kata": 0, "kanji": 0, "other": 0}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal": continue
        nblocks += 1
        if tag == "insert": ins += j2 - j1
        elif tag == "delete": dele += i2 - i1
        else: rep_a += i2 - i1; rep_b += j2 - j1
        for ch in a[i1:i2] + b[j1:j2]: cls[cclass(ch)] += 1
    changed = ins + dele + max(rep_a, rep_b)
    tot = sum(cls.values()) or 1
    return {"len_a": len(a), "len_b": len(b), "len_delta": len(b) - len(a), "n_edit_blocks": nblocks, "ins_chars": ins, "del_chars": dele,
            "replace_chars_a": rep_a, "replace_chars_b": rep_b, "changed_chars": changed,
            "char_change_ratio": changed / max(len(a), len(b), 1), "paras_a": a.count("\n") + 1, "paras_b": b.count("\n") + 1,
            "ratio_edit_similarity": sm.ratio(), "changed_punct_share": cls["punct"] / tot, "changed_kana_share": (cls["hira"] + cls["kata"]) / tot, "changed_kanji_share": cls["kanji"] / tot}

per = {}; parity = {}
for name, p in (("F17", F17), ("C216", C216)):
    recs = b13._run_rust_harvest_batch(EXE, p, rows)
    nf, prov = load_f14_noise_floor(None if name == "C216" else p)
    bad = b14.python_rust_parity(rows, recs, nf, prov); parity[name] = len(bad)
    perdim = [{k: abs(float(x)) for k, x in (rec["noise_floor_ratio_kasane11"].get("per_dim") or {}).items() if x is not None} for rec in recs]
    per[name] = perdim; print(name, "parity mismatches", len(bad), flush=True)
am = {n: np.array([max(d.values()) if d else np.nan for d in per[n]]) for n in per}
em = [edit_metrics(r["text_before"], r["text_after"]) for r in rows]
ccr = np.array([e["char_change_ratio"] for e in em]); dice = np.array([r["dice_similarity"] for r in rows])

def auc(score, pos):
    s = np.asarray(score, float); ok = ~np.isnan(s); s, pos = s[ok], np.asarray(pos)[ok]
    P, N = s[pos], s[~pos]
    if len(P) == 0 or len(N) == 0: return None
    return float(((P[:, None] > N[None, :]).sum() + 0.5 * (P[:, None] == N[None, :]).sum()) / (len(P) * len(N)))
def spearman(a, b):
    from scipy.stats import spearmanr
    ok = ~np.isnan(a); r = spearmanr(a[ok], b[ok]); return {"rho": float(r.statistic), "p": float(r.pvalue)}

res = {"dataset_sha256": hashlib.sha256(DS.read_bytes()).hexdigest().upper(), "parity_mismatch_F17": parity["F17"], "parity_mismatch_C216": parity["C216"],
       "n_pairs": len(rows), "labels_note": "All 439 pairs are real revisions/edition variants (positives). No negative (no-revision) pairs exist in v1.1; FPR cannot be computed."}
# flips
flips = []
for i, r in enumerate(rows):
    a, b = am["F17"][i] > 1, am["C216"][i] > 1
    if a != b:
        dims = [{"dim": k, "f17": per["F17"][i][k], "c216": per["C216"][i][k]} for k in per["F17"][i] if k in per["C216"][i] and ((per["F17"][i][k] > 1) != (per["C216"][i][k] > 1))]
        flips.append({"index": i, "pair_id": r["pair_id"], "author": r["author"], "variant_relationship": r["variant_relationship"], "band": r["band"], "dice": r["dice_similarity"],
                      "abs_max_f17": float(am["F17"][i]), "abs_max_c216": float(am["C216"][i]), "direction": "gt1->le1" if a else "le1->gt1", "flipped_dims": dims, **em[i]})
res["flips14"] = flips
gt = am["C216"] > 1
res["n_gt1_f17"] = int((am["F17"] > 1).sum()); res["n_gt1_c216"] = int(gt.sum())
def stats(idx):
    idx = np.asarray(idx, bool)
    return {"n": int(idx.sum()), "char_change_ratio_median": float(np.median(ccr[idx])) if idx.any() else None, "char_change_ratio_mean": float(ccr[idx].mean()) if idx.any() else None,
            "changed_chars_median": float(np.median([e["changed_chars"] for e, k in zip(em, idx) if k])) if idx.any() else None,
            "n_edit_blocks_median": float(np.median([e["n_edit_blocks"] for e, k in zip(em, idx) if k])) if idx.any() else None,
            "dice_median": float(np.median(dice[idx])) if idx.any() else None, "len_delta_abs_median": float(np.median([abs(e["len_delta"]) for e, k in zip(em, idx) if k])) if idx.any() else None,
            "paras_changed": int(sum(e["paras_a"] != e["paras_b"] for e, k in zip(em, idx) if k)),
            "punct_share_median": float(np.median([e["changed_punct_share"] for e, k in zip(em, idx) if k])) if idx.any() else None}
flipmask = np.zeros(len(rows), bool); flipmask[[f["index"] for f in flips]] = True
res["groups"] = {"all439": stats(np.ones(len(rows), bool)), "flip14": stats(flipmask), "gt1_c216_97": stats(gt), "le1_c216": stats(~gt),
                 "gt1_both_floors": stats(gt & (am["F17"] > 1)), "gt1_f17_only_equals_flip14": stats((am["F17"] > 1) & ~gt)}
# graded dependence of abs_max>1 on edit scale (quantiles of char_change_ratio)
qs = np.quantile(ccr, [0, .25, .5, .75, 1]); res["edit_scale_quartiles"] = []
for lo, hi, k in zip(qs[:-1], qs[1:], range(4)):
    m = (ccr >= lo) & ((ccr < hi) if k < 3 else (ccr <= hi))
    res["edit_scale_quartiles"].append({"q": k + 1, "ccr_range": [float(lo), float(hi)], "n": int(m.sum()), "gt1_f17": int((am["F17"][m] > 1).sum()), "gt1_c216": int(gt[m].sum()),
                                        "median_abs_max_f17": float(np.nanmedian(am["F17"][m])), "median_abs_max_c216": float(np.nanmedian(am["C216"][m]))})
res["spearman_abs_max_vs_char_change_ratio"] = {n: spearman(am[n], ccr) for n in am}
res["spearman_abs_max_vs_1_minus_dice"] = {n: spearman(am[n], 1 - dice) for n in am}
# proxy: "substantive edit" = char_change_ratio >= population median (proxy; not a ground-truth label)
res["proxy_label"] = "substantive = char_change_ratio >= median(439); proxy only, not ground truth"
sub = ccr >= np.median(ccr)
res["proxy_auc_abs_max"] = {n: auc(am[n], sub) for n in am}
res["flag_vs_proxy"] = {}
for n in am:
    for thr in (0.8, 0.85, 0.9, 0.95, 1.0):
        f = am[n] > thr; res["flag_vs_proxy"][f"{n}@{thr}"] = {"n_flagged": int(f.sum()), "tpr_proxy": float((f & sub).sum() / sub.sum()), "precision_proxy": (float((f & sub).sum() / f.sum()) if f.sum() else None),
                                                          "flag_rate_among_nonsubstantive": float((f & ~sub).sum() / (~sub).sum())}
# which dims fall in [0.9,1.0] under 216 (pairs that would be re-flagged by an auxiliary threshold)
for thr in (0.8, 0.9, 0.95):
    m = (am["C216"] > thr) & (am["C216"] <= 1.0)
    res[f"c216_pairs_in_({thr},1.0]"] = {"n": int(m.sum()), "of_which_flip14": int((m & flipmask).sum()), "char_change_ratio_median": float(np.median(ccr[m])) if m.any() else None}
# flip14 percentile of char_change_ratio within population, and vs gt1 retained
res["flip14_ccr_percentile_in_all439"] = [float((ccr < ccr[f["index"]]).mean()) for f in flips]
# 97 table by band and by variant_relationship
res["gt1_97_by_variant"] = {}
for v in sorted({r["variant_relationship"] for r in rows}):
    m = np.array([r["variant_relationship"] == v for r in rows]); res["gt1_97_by_variant"][v] = {"n": int(m.sum()), "gt1_c216": int((gt & m).sum()), "ccr_median_all": float(np.median(ccr[m])), "ccr_median_gt1": float(np.median(ccr[gt & m])) if (gt & m).any() else None}
res["pairs97"] = [{"pair_id": rows[i]["pair_id"], "band": rows[i]["band"], "abs_max_c216": float(am["C216"][i]), "char_change_ratio": float(ccr[i]), "changed_chars": em[i]["changed_chars"], "len_delta": em[i]["len_delta"], "paras_a": em[i]["paras_a"], "paras_b": em[i]["paras_b"], "dice": float(dice[i])} for i in np.where(gt)[0]]
OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({k: res[k] for k in ("parity_mismatch_F17", "parity_mismatch_C216", "n_gt1_f17", "n_gt1_c216", "groups", "proxy_auc_abs_max", "spearman_abs_max_vs_char_change_ratio")}, indent=1, ensure_ascii=False))
