#!/usr/bin/env python3
"""FAW-IMPL task 0: confirm-cohort screening (deterministic, no RNG). Compares p2cE / p2cA2 / heldout3 against FAW design cohorts by work ID and author ID."""
import json, sys, collections
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
for p in ("python", "scripts", "scripts/faw"):
    sys.path.insert(0, str(REPO / p))
from usir_py.v5 import AozoraCorpusSplitter, split_work_into_scenes
D = REPO / "data"
def jl(n): return json.loads((D / n).read_text(encoding="utf-8"))
sp = AozoraCorpusSplitter(D)
faw_works = {}  # work_id -> (author_id, cohort)
for w in sp.load_manifests(): faw_works[w["work_id"]] = (w["author_id"], "AOZORA106")
for mf in ("aozora_p2b_manifest.json", "aozora_p2c_manifest.json"):
    for w in jl(mf)["works"]: faw_works[w["work_id"]] = (w["author_id"], "POSTWAR")
faw_authors = {a for a, _ in faw_works.values()}
def cohort(prefix, entries):
    out = []
    for e in entries:
        p = D / "aozora_cache" / f"{prefix}_{e['author_id']}_{e['work_id']}.txt"
        sc = len(split_work_into_scenes(p.read_text(encoding="utf-8"))) if p.exists() else None
        out.append(dict(author_id=e["author_id"], work_id=e["work_id"], cached=p.exists(), n_scenes=sc,
                        dup_work=e["work_id"] in faw_works, author_seen=e["author_id"] in faw_authors))
    return out
tE = jl("aozora_p2c_taskE_centroid_manifest.json")
E = cohort("p2cE", [w for ws in tE["selection"].values() for w in ws])
em = jl("aozora_p2c_essay_manifest.json")
ew = em["works"] if "works" in em else em["selection"]
if isinstance(ew, dict): ew = [w for ws in ew.values() for w in ws]
A2 = cohort("p2cA2", ew)
def summ(rows):
    keep = [r for r in rows if r["cached"] and not r["dup_work"]]
    au = collections.Counter(r["author_id"] for r in keep)
    unk = [r for r in keep if not r["author_seen"]]
    auu = collections.Counter(r["author_id"] for r in unk)
    return dict(n_listed=len(rows), n_cached=sum(r["cached"] for r in rows), n_dup_work=sum(r["dup_work"] for r in rows),
        n_works_after_dedup=len(keep), n_authors=len(au), authors_ge2=sum(v >= 2 for v in au.values()),
        works_author_seen_in_faw=sum(r["author_seen"] for r in keep), n_authors_unseen=len(auu), unseen_authors_ge2=sum(v >= 2 for v in auu.values()),
        scenes_total=sum(r["n_scenes"] for r in keep), scenes_min=min(r["n_scenes"] for r in keep), scenes_median=sorted(r["n_scenes"] for r in keep)[len(keep)//2],
        n_works_T_ge10=sum(r["n_scenes"] >= 10 for r in keep), n_works_T_lt5=sum(r["n_scenes"] < 5 for r in keep))
h3 = jl("aozora_heldout3_manifest.json")
res = dict(faw_cohort_works=len(faw_works), faw_cohort_authors=len(faw_authors), p2cE=summ(E), p2cA2=summ(A2), p2cE_rows=E, p2cA2_rows=A2,
           heldout3=dict(n=len(h3), authors=len({e["author_id"] for e in h3}), works_per_author=dict(collections.Counter(collections.Counter(e["author_id"] for e in h3).values())),
                         authors_seen_in_faw=len({e["author_id"] for e in h3} & faw_authors), authors_in_p2cE=len({e["author_id"] for e in h3} & {r["author_id"] for r in E})))
(Path(__file__).with_name("cohort_check_results.json")).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if not k.endswith("_rows")}, ensure_ascii=False, indent=1))
