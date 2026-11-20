"""Shared loaders for the fact/values analysis (scene text, roster, gold, baseline annotations)."""
from __future__ import annotations
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
S2 = REPO / "scripts/phase2_stage2"
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts/phase2"))
sys.path.insert(0, str(S2))
from run_full_annotation import SOURCES, NISHIKI, roster_entries  # noqa: E402
from build_annotations import segment  # noqa: E402


def load_works():
    """-> {work_id: {'scenes': [scene dict], 'characters': [...], 'roster_out': [...]}}"""
    out = {}
    for group, src in SOURCES.items():
        gold = json.loads(Path(src["gold"]).read_text(encoding="utf-8"))
        gw = {w["id"]: w for w in gold["works"]}
        for wid in src["ids"]:
            seg = segment(NISHIKI, Path(src["text_dir"]) / f"{wid}.txt")
            w = gw[wid]
            out[wid] = {"group": group, "scenes": seg["scenes"], "characters": w["characters"],
                        "roster_out": roster_entries(w["characters"]), "work": w}
    return out


def load_gold():
    return json.loads((S2 / "gold/stage_b_gold.json").read_text(encoding="utf-8"))["scenes"]


def load_preds(ann_dir: Path):
    preds, traces = {}, {}
    for f in sorted(Path(ann_dir).glob("*.annotations.json")):
        wid = f.name.split(".")[0]
        for a in json.loads(f.read_text(encoding="utf-8")):
            preds[a["contract"]["scene_id"]] = a
        tf = f.with_name(f"{wid}.trace.json")
        if tf.exists():
            for t in json.loads(tf.read_text(encoding="utf-8"))["traces"]:
                traces[t["scene_id"]] = t
    return preds, traces
