#!/usr/bin/env python3
"""P-2c step 1: per-scene non-reversible profiles (32D features + 3 surface quantities), original and perturbed.
Perturbations follow P-2b 登録 4 exactly (imported from scripts/p2b/p2b_analysis.py), applied per cohort in manifest order.
Raw text is read from data/aozora_cache/ (gitignored) and never written out. Output: data/p2c_scene_profiles.npz (numbers only)."""
from __future__ import annotations
import json, re, sys, time
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
for p in ("python", "scripts", "experiments", "scripts/p2b"): sys.path.insert(0, str(REPO / p))
import p2b_analysis as P2B  # noqa: E402  (perturb, PRIMARY, SECONDARY, FN, _SENSE)
from usir_py.v5 import split_work_into_scenes  # noqa: E402

OUT = REPO / "data" / "p2c_scene_profiles.npz"
_SENT = re.compile(r"[^。！？]+[。！？]+[」』）)]*")
SURF_NAMES = ["comma_density", "sensory_rate", "ending_diversity"]


def surface(t: str) -> np.ndarray:
    n = max(len(t), 1)
    q1 = t.count("、") / n
    q2 = sum(ch in P2B._SENSE for ch in t) / n
    ends, ns = set(), 0
    for m in _SENT.finditer(t):
        s = m.group(0).rstrip("」』）)")
        s = s.rstrip("。！？")
        if not s: continue
        ns += 1; ends.add(s[-3:])
    q3 = len(ends) / ns if ns else 0.0
    return np.array([q1, q2, q3])


def load_cohort(manifest: Path, prefix: str):
    man = json.loads(manifest.read_text(encoding="utf-8"))
    texts, aids, wids = [], [], []
    for w in man["works"]:
        t = (REPO / "data" / "aozora_cache" / f"{prefix}_{w['author_id']}_{w['work_id']}.txt").read_text(encoding="utf-8")
        for s in split_work_into_scenes(t):
            texts.append(s); aids.append(w["author_id"]); wids.append(w["work_id"])
    return texts, aids, wids


def main():
    t0 = time.time()
    out = {}
    cohorts = [("p2b", REPO / "data" / "aozora_p2b_manifest.json", "p2b", 0), ("p2c", REPO / "data" / "aozora_p2c_manifest.json", "p2c", 1)]
    A, W, C, X, Q = [], [], [], [], []
    XP = {k: [] for k in P2B.PRIMARY}; QP = {k: [] for k in P2B.PRIMARY}
    DOSE = {}
    for name, mf, prefix, cid in cohorts:
        texts, aids, wids = load_cohort(mf, prefix)
        A += aids; W += wids; C += [cid] * len(texts)
        X.append(np.stack([P2B.FN(t) for t in texts])); Q.append(np.stack([surface(t) for t in texts]))
        for k, r in P2B.PRIMARY.items():
            pt = P2B.perturb(k, texts, r)
            XP[k].append(np.stack([P2B.FN(t) for t in pt])); QP[k].append(np.stack([surface(t) for t in pt]))
        for k, rates in P2B.SECONDARY.items():
            for rt in rates:
                pt = P2B.perturb(k, texts, rt)
                DOSE.setdefault(f"{k}@{rt}", {"X": [], "Q": []})
                DOSE[f"{k}@{rt}"]["X"].append(np.stack([P2B.FN(t) for t in pt])); DOSE[f"{k}@{rt}"]["Q"].append(np.stack([surface(t) for t in pt]))
        print(f"cohort {name}: {len(texts)} scenes, elapsed {time.time()-t0:.0f}s", flush=True)
    save = dict(author=np.array(A), work=np.array(W), cohort=np.array(C), X=np.concatenate(X), Q=np.concatenate(Q))
    for k in P2B.PRIMARY:
        save[f"X_{k}"] = np.concatenate(XP[k]); save[f"Q_{k}"] = np.concatenate(QP[k])
    for k, v in DOSE.items():
        save[f"Xd_{k}"] = np.concatenate(v["X"]); save[f"Qd_{k}"] = np.concatenate(v["Q"])
    np.savez_compressed(OUT, **save)
    print("saved", OUT, "scenes", len(A), "dim", save["X"].shape[1], "elapsed", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
