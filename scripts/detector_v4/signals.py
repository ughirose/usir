#!/usr/bin/env python3
"""DETV4 T2: v3 signals + L_narr (narration char count). Output data/detv4_scene_signals.npz.
Existing 8 columns are verified identical to data/detv3_scene_signals.npz."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "detector_v3"))
import signals as S3  # noqa: E402  (also puts p2b/p2c on sys.path)
P2B, F = S3.P2B, S3.F

OUT = REPO / "data" / "detv4_scene_signals.npz"
NAMES = S3.NAMES + ["L_narr"]


def l_narr(t: str) -> float:
    t2 = S3._WS.sub("", t).replace("　", "")
    return float(len(S3._DLG.sub("", t2)))


def signals(t: str) -> np.ndarray:
    return np.concatenate([S3.signals(t), [l_narr(t)]])


def main():
    ref3 = np.load(REPO / "data" / "detv3_scene_signals.npz")
    cohorts = [("p2b", REPO / "data" / "aozora_p2b_manifest.json", "p2b"), ("p2c", REPO / "data" / "aozora_p2c_manifest.json", "p2c")]
    versions = {"orig": []}
    for k in P2B.PRIMARY: versions[k] = []
    for k, rates in P2B.SECONDARY.items():
        for rt in rates: versions[f"{k}@{rt}"] = []
    A, W = [], []
    for name, mf, prefix in cohorts:
        texts, aids, wids = F.load_cohort(mf, prefix)
        A += aids; W += wids
        versions["orig"].append(np.stack([signals(t) for t in texts]))
        for k, r in P2B.PRIMARY.items():
            versions[k].append(np.stack([signals(t) for t in P2B.perturb(k, texts, r)]))
        for k, rates in P2B.SECONDARY.items():
            for rt in rates:
                versions[f"{k}@{rt}"].append(np.stack([signals(t) for t in P2B.perturb(k, texts, rt)]))
        print("cohort", name, len(texts), flush=True)
    save = {"author": np.array(A), "work": np.array(W), "names": np.array(NAMES)}
    assert (save["author"] == ref3["author"]).all() and (save["work"] == ref3["work"]).all()
    for v, arr in versions.items():
        a = np.concatenate(arr)
        save[v] = a
        err = float(np.abs(a[:, :8] - ref3[v]).max())
        print(f"verify 8 cols vs detv3 npz [{v}]: max abs diff {err:.3e}")
        assert err < 1e-12, v
    np.savez_compressed(OUT, **save)
    print("saved", OUT, save["orig"].shape)


if __name__ == "__main__":
    main()
