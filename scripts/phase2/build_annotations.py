"""Phase 2 step 1: build sensor.v2 scene annotations for 3 public modern-Japanese works.

Inputs
- frozen sensor output (run_step1_f9, v5 + postprocess + speaker recheck, qwen2.5:14b): per-work parsed_utterances
- frozen gold JSON: roster (characters), narration/narrator only (no gold speaker/politeness labels are copied into annotations)
- nishiki segment: scene split, scene_hash, utterance numbering

Register politeness is NOT taken from the model. It is re-assigned by Strict Policy X
(build_modern_gold.classify_politeness_surface; Janome surface sentence-final grammar):
polite/plain/ambiguous; ambiguous is written as sensor.v2 "unknown" (the schema has no 'ambiguous' politeness value).

Usage: python build_annotations.py <sensor_result.json> <out_dir> <nishiki.exe>
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GOLD_JSON = Path("F:/マイドライブ/Claude-Sync/inbox/2026-11-20_modern_gold_frozen.json")
TEXT_DIR = REPO / "scripts/step1/modern_gold"
sys.path.insert(0, str(TEXT_DIR))
sys.path.insert(0, str(REPO / "python"))

from janome.tokenizer import Tokenizer  # noqa: E402
from build_modern_gold import classify_politeness_surface  # noqa: E402

SPLITTER = "usir_v5_splitter@1"
# Pronoun aliases are removed from the nishiki roster (plan §3 additional defence: generic pronouns cause AmbiguousAlias).
PRONOUN_ALIASES = {"彼", "彼女", "私", "俺"}
POLITE_MAP = {"polite": "polite", "plain": "plain", "ambiguous": "unknown"}


def segment(nishiki: str, txt: Path) -> dict:
    out = subprocess.run([nishiki, "segment", str(txt), "--format", "json"], capture_output=True, check=True)
    return json.loads(out.stdout.decode("utf-8"))


def roster_entries(characters: list[dict]) -> list[dict]:
    count: dict[str, int] = {}
    for c in characters:
        for a in set(c["aliases"]) | {c["name"]}:
            count[a] = count.get(a, 0) + 1
    entries = []
    for c in characters:
        al = {a for a in set(c["aliases"]) | {c["name"]} if a not in PRONOUN_ALIASES and count[a] == 1}
        entries.append(
            {
                "id": c["id"],
                "aliases": sorted(al, key=lambda s: (-len(s), s)),
                "gender": c["gender"],
                "honored": False,
            }
        )
    return entries


def main() -> int:
    sensor_path, out_dir, nishiki = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    out_dir.mkdir(parents=True, exist_ok=True)
    sensor = json.loads(sensor_path.read_text(encoding="utf-8"))
    gold = json.loads(GOLD_JSON.read_text(encoding="utf-8"))
    gold_sha = hashlib.sha256(GOLD_JSON.read_bytes()).hexdigest()
    tok = Tokenizer()
    audit = {"works": {}}
    for w in gold["works"]:
        wid = w["id"]
        txt = TEXT_DIR / f"{wid}.txt"
        text = txt.read_text(encoding="utf-8")
        seg = segment(nishiki, txt)
        scenes = seg["scenes"]
        preds = sensor["raw_outputs"][wid]["parsed_utterances"]
        # Global utterance order: concatenation of scene utterances == outer-quote order of the work.
        flat = [(si, u) for si, sc in enumerate(scenes) for u in sc["utterances"]]
        assert len(flat) == len(preds) == len(w["utterances"]), (wid, len(flat), len(preds), len(w["utterances"]))
        roster = roster_entries(w["characters"])
        ids = {c["id"] for c in w["characters"]}
        narrator = w["narrator"] if w["narration"] == "first" else None
        pov = narrator or w.get("pov")
        anns = []
        reg_rows = []
        for si, sc in enumerate(scenes):
            utts = []
            present: list[str] = []
            for k, (sj, u) in enumerate(flat):
                if sj != si:
                    continue
                p = preds[k]
                assert p["utterance_index"] == k
                scene_text = sc["scene_text"]
                chars = list(scene_text)
                quote = "".join(chars[u["local"]["start"] - 1 : u["local"]["end"] + 1])
                assert quote == w["utterances"][k]["quote_text"], (wid, k, quote, w["utterances"][k]["quote_text"])
                pol = classify_politeness_surface(quote, tok)
                sp, ad = p["speaker_id"], p["addressee_id"]
                for x in (sp, ad):
                    assert x is None or x in ids, (wid, k, x)
                    if x and x not in present:
                        present.append(x)
                utts.append(
                    {
                        "utterance_index": u["index"],
                        "speaker_id": sp,
                        "addressee_id": ad,
                        "evidence": p["evidence"],
                        "register": {"politeness": POLITE_MAP[pol["politeness"]], "first_person": p["register"]["first_person"]},
                    }
                )
                reg_rows.append(
                    {
                        "work_id": wid,
                        "scene_index": si,
                        "global_utterance": k,
                        "scene_utterance": u["index"],
                        "quote_text": quote,
                        "model_politeness": p["register"]["politeness"],
                        "policy_x": pol["politeness"],
                        "policy_x_rule": pol["rule_id"],
                    }
                )
            if narrator and narrator not in present:
                present.append(narrator)
            anns.append(
                {
                    "contract": {
                        "schema_version": "sensor.v2",
                        "splitter_version": seg["splitter"],
                        "scene_id": f"{wid}:s{si}",
                        "scene_hash": sc["scene_hash"],
                        "ledger_hash": f"gold-roster:{gold_sha[:16]}",
                        "model_id": sensor["model_spec"],
                        "prompt_hash": sensor["prompt_hash"],
                        "work_id": wid,
                        "scene_index": si,
                        "scene_count": len(scenes),
                        "char_len": sc["char_len"],
                        "seed": sensor["inference_options"]["seed"],
                    },
                    "scene_meta": {
                        "time": {"relative_order": "unknown", "evidence": "ambiguous"},
                        "location_id": None,
                        "present_character_ids": present,
                        "pov_character": pov,
                    },
                    "utterances": utts,
                    "character_states": [],
                    "fact_updates": [],
                }
            )
        (out_dir / f"{wid}.annotations.json").write_text(json.dumps(anns, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out_dir / f"{wid}.characters.json").write_text(json.dumps(roster, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        audit["works"][wid] = {
            "scene_count": len(scenes),
            "scene_utterance_counts": [len(a["utterances"]) for a in anns],
            "pov_character": pov,
            "register_rows": reg_rows,
        }
    audit["gold_json_sha256"] = gold_sha
    (out_dir / "register_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("ok", {k: v["scene_utterance_counts"] for k, v in audit["works"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
