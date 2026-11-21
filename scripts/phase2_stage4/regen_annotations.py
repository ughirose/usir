"""Phase 2 stage 4 / work 1+3: re-derive annotations with fact_updates[].sentence_index and characters.json default_politeness.

No LLM call. Inputs are the frozen stage-2 outputs (annotations + trace.json holding the raw model responses).
The raw response of each scene is re-validated with the stage-4 validator, which maps each fact's verbatim `basis`
to the [S#] sentence of `nishiki segment` (deterministic). Facts whose basis cannot be pinned to exactly one
sentence are dropped and counted (reason in the audit). Every other section is checked to be identical to stage 2.

default_politeness (work 3): per character, the most frequent non-unknown `register.politeness` over the
utterances where that character is the speaker. Ties break by fixed order polite > plain (deterministic).

Usage: python regen_annotations.py <nishiki.exe> <out_dir>
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts/phase2"))

from build_annotations import segment  # noqa: E402
from usir_py.sensor_eval.scene_annotator import validate_scene_output  # noqa: E402

SRC = REPO / "scripts/phase2_stage2/annotations"
MODERN = REPO / "scripts/step1/modern_gold"
F9_DIR = Path("C:/dev/nishiki/crates/nishiki-discourse/tests/data/f9_gold")
MODERN_IDS = ["001798_056779", "001801_001786", "001875_059897"]
TIE_ORDER = ["polite", "plain"]


def default_politeness(anns: list[dict], char_ids: list[str]) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    cnt: dict[str, Counter] = {c: Counter() for c in char_ids}
    for a in anns:
        for u in a["utterances"]:
            sp = u["speaker_id"]
            p = u["register"]["politeness"]
            if sp in cnt and p in TIE_ORDER:
                cnt[sp][p] += 1
    out, basis = {}, {}
    for c, k in cnt.items():
        basis[c] = dict(k)
        if not k:
            continue
        best = max(k.values())
        out[c] = next(p for p in TIE_ORDER if k.get(p, 0) == best)
    return out, basis


def main() -> int:
    nishiki, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = {"works": {}, "facts_total_stage2": 0, "facts_kept": 0, "facts_dropped": 0, "drop_reasons": {}}
    for ann_path in sorted(SRC.glob("*.annotations.json")):
        wid = ann_path.name.split(".")[0]
        anns = json.loads(ann_path.read_text(encoding="utf-8"))
        chars = json.loads((SRC / f"{wid}.characters.json").read_text(encoding="utf-8"))
        trace = json.loads((SRC / f"{wid}.trace.json").read_text(encoding="utf-8"))
        txt = (MODERN if wid in MODERN_IDS else F9_DIR) / f"{wid}.txt"
        seg = segment(nishiki, txt)
        assert len(seg["scenes"]) == len(anns) == len(trace["raw"]), wid
        ids = [c["id"] for c in chars]
        w_aud = {"scenes": len(anns), "facts_stage2": 0, "facts_kept": 0, "dropped": {}, "scene_text_hash_ok": True, "other_sections_identical": True}
        for si, (ann, sc, raw) in enumerate(zip(anns, seg["scenes"], trace["raw"])):
            assert sc["scene_hash"] == ann["contract"]["scene_hash"], (wid, si)
            assert raw["scene_id"] == ann["contract"]["scene_id"]
            try:
                parsed = json.loads(raw["raw_content"]) if raw["parse_error"] is None else None
            except Exception:
                parsed = None
            sents = [(s["index"], s["local"]["start"], s["local"]["end"]) for s in sc["sentences"]]
            cleaned, aud = validate_scene_output(parsed, ids, sc["scene_text"], sentences=sents)
            old = validate_scene_output(parsed, ids, sc["scene_text"], sentences=None)[0]["sections"]
            new = cleaned["sections"]
            # stage-2 equality check for everything but fact_updates
            for k in ("time", "location_id", "present_character_ids", "character_states"):
                if new[k] != old[k] or new[k] != (ann["scene_meta"].get(k) if k in ann["scene_meta"] else ann.get(k)):
                    w_aud["other_sections_identical"] = False
            stage2_facts = ann["fact_updates"]
            it = iter(old["fact_updates"])
            assert all(any({k: v for k, v in f.items() if k != "sentence_index"} == o for o in it) for f in new["fact_updates"]), (wid, si)
            assert old["fact_updates"] == stage2_facts, (wid, si)
            w_aud["facts_stage2"] += len(stage2_facts)
            w_aud["facts_kept"] += len(new["fact_updates"])
            for r in aud["rejected"]:
                if r["kind"] == "fact" and r["reason"].startswith("sentence_unresolved"):
                    w_aud["dropped"][r["reason"]] = w_aud["dropped"].get(r["reason"], 0) + 1
            ann["fact_updates"] = new["fact_updates"]
        dp, dp_basis = default_politeness(anns, ids)
        for c in chars:
            if c["id"] in dp:
                c["default_politeness"] = dp[c["id"]]
        w_aud["default_politeness"] = dp
        w_aud["default_politeness_counts"] = dp_basis
        (out_dir / f"{wid}.annotations.json").write_text(json.dumps(anns, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out_dir / f"{wid}.characters.json").write_text(json.dumps(chars, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        audit["works"][wid] = w_aud
        audit["facts_total_stage2"] += w_aud["facts_stage2"]
        audit["facts_kept"] += w_aud["facts_kept"]
        for k, v in w_aud["dropped"].items():
            audit["drop_reasons"][k] = audit["drop_reasons"].get(k, 0) + v
    audit["facts_dropped"] = audit["facts_total_stage2"] - audit["facts_kept"]
    (out_dir / "regen_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k != "works"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
