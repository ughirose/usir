#!/usr/bin/env python3
"""P-4a signals: per-scene raw quantities [q1, q2, q3, L_total] for the v13 detector, for orig + every perturbed version.
q1 = ending diversity, q2 = comma density, q3 = whole-scene sensory rate, L_total = len(scene string) (design 3.4).
q1-q3 are verified bit-identical to data/detv4_scene_signals.npz (cols 2, 0, 1). Output: data/detp4_scene_signals.npz (numbers only)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "detector_v3"))
import signals as S3  # noqa: E402  (also puts p2b/p2c on sys.path)
P2B, F = S3.P2B, S3.F

OUT = REPO / "data" / "detp4_scene_signals.npz"
NAMES = ["q1", "q2", "q3", "L_total"]


def signals(t: str) -> np.ndarray:
    q2c, q3s, q1e = F.surface(t)  # surface order: comma_density, sensory_rate, ending_diversity
    return np.array([q1e, q2c, q3s, float(len(t))])


def main():
    ref4 = np.load(REPO / "data" / "detv4_scene_signals.npz")
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
    assert (save["author"] == ref4["author"]).all() and (save["work"] == ref4["work"]).all(), "scene order mismatch"
    for v, arr in versions.items():
        a = np.concatenate(arr)
        save[v] = a
        err = float(np.abs(a[:, :3] - ref4[v][:, [2, 0, 1]]).max())
        print(f"verify q1-q3 vs detv4 npz [{v}]: max abs diff {err:.3e}")
        assert err == 0.0, v
    np.savez_compressed(OUT, **save)
    print("saved", OUT, save["orig"].shape)


if __name__ == "__main__":
    main()
