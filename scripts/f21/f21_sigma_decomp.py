#!/usr/bin/env python
"""Stage F-21 Task1 Gate1/Gate3: author diversity and sigma stability from set216_paragraphs.tsv (read-only)."""
import json, sys, collections
from pathlib import Path
import numpy as np

EV = Path(r"C:\dev\nishiki\docs\handoff\evidence\2026-11-14_b17")
TSV = EV / "set216_paragraphs.tsv"
NF = EV / "paragraph_noise_floor_all_216.json"
OUT = Path(__file__).resolve().parent / "f21_sigma_decomp.json"
AC = ["ac_mean_velocity", "ac_max_sis", "ac_coupling"]

rows = [l.rstrip("\n").split("\t") for l in TSV.read_text(encoding="utf-8").splitlines()]
hdr = rows[0]; rows = rows[1:]
ix = {h: i for i, h in enumerate(hdr)}
nf = json.loads(NF.read_text(encoding="utf-8"))
works_meta = nf["kasane11"]["source"]["works"]
wfile_to_author = {w["file"]: w["author_id"] for w in works_meta}
wfile_to_idx = {}
data = collections.defaultdict(list)
for r in rows:
    wi = int(r[ix["work_idx"]]); wfile_to_idx[wi] = r[ix["work_file"]]
    data[wi].append((int(r[ix["pos"]]), int(r[ix["chars"]]), int(r[ix["windows"]]), [float(r[ix[a]]) for a in AC]))

# pairs: both windows>=2
pairs = []  # (work_idx, band, [diffs])
for wi, ps in data.items():
    ps.sort()
    for a, b in zip(ps, ps[1:]):
        if b[0] != a[0] + 1: continue
        if a[2] >= 2 and b[2] >= 2:
            lmax = max(a[1], b[1])
            band = "S" if lmax <= 250 else ("M" if lmax <= 500 else "L")
            pairs.append((wi, band, [b[3][k] - a[3][k] for k in range(3)]))
res = {"n_works_in_tsv": len(data), "n_pairs_multi": len(pairs)}
def sd(v): return float(np.std(v)) if len(v) else None
band_arr = {}
for band in "ML":
    sel = [(wi, d) for wi, bnd, d in pairs if bnd == band]
    band_arr[band] = (np.array([s[0] for s in sel]), np.array([s[1] for s in sel]))
    res[f"n_pairs_{band}"] = len(sel)
    res[f"sigma_recomputed_{band}"] = {AC[k]: sd(band_arr[band][1][:, k]) for k in range(3)}
    res[f"sigma_json_{band}"] = nf["length_bands"]["bands"][1 if band == "M" else 2]["kasane11"]["ac_multi_window_per_dim_std"]

# subsets by origin: work_idx order vs author-id coverage
authors_all = {wi: wfile_to_author.get(wfile_to_idx[wi]) for wi in data}
first_authors = collections.Counter(authors_all[wi] for wi in range(106) if wi in authors_all)
res["authors_first106_n"] = len(first_authors)
res["authors_last110_n"] = len(set(authors_all[wi] for wi in range(106, 216) if wi in authors_all))
res["authors_overlap_first106_last110"] = len(set(first_authors) & set(authors_all[wi] for wi in range(106, 216)))
res["subset_sigma"] = {}
for name, cond in (("first106", lambda w: w < 106), ("last110", lambda w: w >= 106), ("all216", lambda w: True)):
    for band in "ML":
        wis, arr = band_arr[band]
        m = np.array([cond(w) for w in wis])
        res["subset_sigma"][f"{name}_{band}"] = {"n_pairs": int(m.sum()), "n_works": int(len(set(wis[m]))), **{AC[k]: sd(arr[m][:, k]) for k in range(3)}}

# author concentration of [>=2,>=2] pairs
res["concentration"] = {}
for band in "ML":
    wis, _ = band_arr[band]
    for label, mp in (("216", lambda w: True), ("first106", lambda w: w < 106)):
        ac = collections.Counter(authors_all[w] for w in wis if mp(w))
        wc = collections.Counter(w for w in wis if mp(w))
        n = sum(ac.values()); sh = np.array(sorted(ac.values(), reverse=True)) / n
        wsh = np.array(sorted(wc.values(), reverse=True)) / n
        res["concentration"][f"{label}_{band}"] = {
            "n_pairs": n, "n_authors": len(ac), "n_works": len(wc),
            "top1_author_share": float(sh[0]), "top3_author_share": float(sh[:3].sum()), "top10_author_share": float(sh[:10].sum()),
            "effective_n_authors_invHHI": float(1 / (sh ** 2).sum()),
            "top1_work_share": float(wsh[0]), "effective_n_works_invHHI": float(1 / (wsh ** 2).sum())}
# author count of all-216 = authors with >=1 pair in band
# leave-one-work-out and leave-one-author-out influence on sigma
res["influence"] = {}
for band in "ML":
    wis, arr = band_arr[band]
    full = np.array([sd(arr[:, k]) for k in range(3)])
    uw = sorted(set(wis))
    loo = np.array([[sd(arr[wis != w][:, k]) for k in range(3)] for w in uw])
    rel = loo / full - 1
    ua = sorted(set(authors_all[w] for w in uw))
    awis = np.array([authors_all[w] for w in wis])
    looa = np.array([[sd(arr[awis != a][:, k]) for k in range(3)] for a in ua])
    rela = looa / full - 1
    res["influence"][band] = {AC[k]: {
        "loo_work_rel_min": float(rel[:, k].min()), "loo_work_rel_max": float(rel[:, k].max()),
        "max_abs_work": uw[int(np.abs(rel[:, k]).argmax())], "max_abs_work_file": wfile_to_idx[uw[int(np.abs(rel[:, k]).argmax())]],
        "max_abs_work_author": authors_all[uw[int(np.abs(rel[:, k]).argmax())]],
        "loo_author_rel_min": float(rela[:, k].min()), "loo_author_rel_max": float(rela[:, k].max()),
        "max_abs_author": ua[int(np.abs(rela[:, k]).argmax())]} for k in range(3)}
# max_sis M +8.68% decomposition: top-|diff| pairs, tail summary, trimmed sigma
wis, arr = band_arr["M"]; x = arr[:, 1]
new = wis >= 106
res["max_sis_M_decomp"] = {
    "n_pairs_old": int((~new).sum()), "n_pairs_new": int(new.sum()),
    "sigma_old": sd(x[~new]), "sigma_new_only": sd(x[new]), "sigma_all": sd(x),
    "sigma_ratio_new_over_old": sd(x[new]) / sd(x[~new]),
    "var_all": float(x.var()), "var_old": float(x[~new].var()), "var_new": float(x[new].var()),
    "kurtosis_excess_all": float(((x - x.mean()) ** 4).mean() / x.var() ** 2 - 3),
    "kurtosis_excess_old": float(((x[~new] - x[~new].mean()) ** 4).mean() / x[~new].var() ** 2 - 3),
    "kurtosis_excess_new": float(((x[new] - x[new].mean()) ** 4).mean() / x[new].var() ** 2 - 3),
    "frac_var_from_top5_pairs_all": float(np.sort((x - x.mean()) ** 2)[::-1][:5].sum() / ((x - x.mean()) ** 2).sum()),
    "sigma_all_drop_top5_abs": sd(x[np.argsort(-np.abs(x - x.mean()))[5:]]),
    "sigma_all_trim_1pct_each": sd(np.sort(x)[int(.01 * len(x)): len(x) - int(.01 * len(x))]),
    "sigma_old_trim_1pct_each": sd(np.sort(x[~new])[int(.01 * (~new).sum()): (~new).sum() - int(.01 * (~new).sum())]),
    "mad_sigma_old": float(1.4826 * np.median(np.abs(x[~new] - np.median(x[~new])))),
    "mad_sigma_all": float(1.4826 * np.median(np.abs(x - np.median(x)))),
    "top5_pairs": [{"work_idx": int(wis[i]), "file": wfile_to_idx[int(wis[i])], "author": authors_all[int(wis[i])], "diff": float(x[i])} for i in np.argsort(-np.abs(x - x.mean()))[:5]],
}
# work-bootstrap of sigma ratio all216/first106 for M (is +8.68% within sampling noise of the old 106?)
rng = np.random.default_rng(20261114)
old_works = sorted(set(wis[~new])); by = {w: x[wis == w] for w in old_works}
bs = []
for _ in range(5000):
    pick = rng.choice(old_works, size=len(old_works), replace=True)
    bs.append(np.std(np.concatenate([by[w] for w in pick])))
res["max_sis_M_decomp"]["old106_work_bootstrap_sigma_p2.5_p97.5"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
res["max_sis_M_decomp"]["old106_work_bootstrap_note"] = "B=5000, seed=20261114, resample 'first106' contributing works with replacement"
OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n", encoding="utf-8")
print("written", OUT)
