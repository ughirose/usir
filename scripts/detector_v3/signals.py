#!/usr/bin/env python3
"""DETV3 T2: per-scene input quantities (q1-q3, LS, DR, S1b, S2a, S2b) for original and perturbed texts.
Perturbations/texts are regenerated exactly as scripts/p2c/p2c_features.py; q1-q3 are verified against data/p2c_scene_profiles.npz.
Output: data/detv3_scene_signals.npz (numbers only). Raw text is read from data/aozora_cache/ and never written."""
from __future__ import annotations
import re, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in ("python", "scripts", "experiments", "scripts/p2b", "scripts/p2c"):
    sys.path.insert(0, str(REPO / p))
import p2b_analysis as P2B  # noqa: E402
import p2c_features as F  # noqa: E402

OUT = REPO / "data" / "detv3_scene_signals.npz"
NAMES = ["q1", "q2", "q3", "LS", "DR", "S1b", "S2a", "S2b"]
LONG_LEN = 30
_WS = re.compile(r"\s+")
_DLG = re.compile(r"「[^」]*」|『[^』]*』")
_SENT = F._SENT
_SENSE = P2B._SENSE


def _sentences(t: str) -> list[str]:
    return [m.group(0) for m in _SENT.finditer(t)]


def signals(t: str) -> np.ndarray:
    q1, q2, q3 = F.surface(t)
    t2 = _WS.sub("", t).replace("　", "")
    sents = _sentences(t2)
    ls = (sum(len(s) for s in sents) / len(sents)) if sents else float(len(t2))
    dl = sum(len(m.group(0)) for m in _DLG.finditer(t2))
    dr = dl / len(t2) if t2 else 0.0
    nocomma = [s for s in sents if "、" not in s]
    s1b = (sum(len(s) >= LONG_LEN for s in nocomma) / len(nocomma)) if nocomma else 0.0
    nar = _DLG.sub("", t2)
    s2a = (sum(c in _SENSE for c in nar) / len(nar)) if nar else 0.0
    best = run = 0
    for s in _sentences(nar):
        if any(c in _SENSE for c in s):
            run = 0
        else:
            run += 1
            best = max(best, run)
    return np.array([q1, q2, q3, ls, dr, s1b, s2a, float(best)])


def main():
    cohorts = [("p2b", REPO / "data" / "aozora_p2b_manifest.json", "p2b"), ("p2c", REPO / "data" / "aozora_p2c_manifest.json", "p2c")]
    ref = np.load(REPO / "data" / "p2c_scene_profiles.npz")
    versions: dict[str, list] = {"orig": []}
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
    assert (save["author"] == ref["author"]).all() and (save["work"] == ref["work"]).all(), "scene order mismatch"
    for v, arr in versions.items():
        a = np.concatenate(arr)
        save[v] = a
        rq = ref["Q"] if v == "orig" else (ref[f"Q_{v}"] if v in P2B.PRIMARY else ref[f"Qd_{v}"])
        err = float(np.abs(a[:, :3] - rq).max())
        print(f"verify q1-q3 vs p2c npz [{v}]: max abs diff {err:.3e}")
        assert err < 1e-12, v
    np.savez_compressed(OUT, **save)
    print("saved", OUT, save["orig"].shape)


if __name__ == "__main__":
    main()
