"""Build full hybrid annotations (session-model, manual) = phase2_ceiling annotations (presence/time/place/states/facts; written before
stage_b_gold was opened) + utterances (speaker/addressee/evidence) + scene_meta.nagare (event/mood top3) + scene_meta.awase.
Politeness re-assigned by Policy X (same deterministic rule as phase2 stage2); first_person by surface regex.
Validates: counts of utterances vs segmenter, basis strings verbatim in scene text, 3 distinct codes, schema (sensor_v2).
Does not read stage_b_gold.json or event_top3_gold.json."""
import json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
sys.path.insert(0, str(REPO/"python")); sys.path.insert(0, str(REPO/"scripts/phase2")); sys.path.insert(0, str(REPO/"scripts/step1/modern_gold"))
from janome.tokenizer import Tokenizer
from build_annotations import POLITE_MAP
from build_modern_gold import classify_politeness_surface
from usir_py.sensor_eval.scene_annotator import validate_annotation_schema
CEIL = REPO/"scripts/phase2_ceiling"
inp = json.loads((CEIL/"scenes_input.json").read_text(encoding="utf-8"))
utt = json.loads((CEIL/"scenes_utterances.json").read_text(encoding="utf-8"))
schema = json.loads((REPO/"schemas/sensor_v2.schema.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(REPO/"scripts/phase2_nagare_awase_ref")) if False else None
EVENTS = {f"E{i:02d}" for i in range(1,17)}; MOODS = {f"M{i:02d}" for i in range(1,17)}
norm = lambda s: re.sub(r"\s+", "", s)
FP = [("僕","僕"),("ぼく","ぼく"),("わたくし","わたくし"),("私","私"),("わたし","わたし"),("俺","俺"),("おれ","おれ"),("あたし","あたし")]
tok = Tokenizer()
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
errs = []
for wid, w in inp.items():
    ceil = json.loads((CEIL/"annotations"/f"{wid}.annotations.json").read_text(encoding="utf-8"))
    src = json.loads((HERE/"src"/f"{wid}.json").read_text(encoding="utf-8"))["scenes"]
    ids = [r["id"] for r in w["roster"]]
    assert len(ceil) == len(src) == len(w["scenes"]), wid
    anns, basis_trace = [], []
    for i, (a, s, sc, us) in enumerate(zip(ceil, src, w["scenes"], utt[wid])):
        nt = norm(sc["text"]); sid = sc["scene_id"]
        quotes = us["utterances"]
        if len(quotes) != len(s["utt"]): errs.append((sid, "utt_count", len(quotes), len(s["utt"])))
        # quote text for politeness/first-person
        text = sc["text"]
        ul = []
        for k, (q, (spk, adr, ev)) in enumerate(zip(quotes, s["utt"])):
            quote = text[q["local"]["start"]-1: q["local"]["end"]+1]
            for x in (spk, adr):
                if x is not None and x not in ids: errs.append((sid, "utt_id", k, x))
            pol = POLITE_MAP[classify_politeness_surface(quote, tok)["politeness"]]
            fp = next((v for p, v in FP if p in quote), None) if spk else None
            ul.append({"utterance_index": q["index"], "speaker_id": spk, "addressee_id": adr, "evidence": ev,
                       "register": {"politeness": pol, "first_person": fp}})
        ev3, mo3 = s["event"], s["mood"]
        for kind, items, table in (("event", ev3, EVENTS), ("mood", mo3, MOODS)):
            if len(items) != 3 or len({c for c, _ in items}) != 3 or any(c not in table for c, _ in items): errs.append((sid, kind, "codes"))
            for c, b in items:
                if norm(b) not in nt: errs.append((sid, kind, "basis", c, b))
        Fv, Fa, fv, fa, Fb, fb = s["awase"]
        for v in (Fv, Fa, fv, fa):
            if not isinstance(v, int) or not -2 <= v <= 2: errs.append((sid, "awase_val", v))
        for b in (Fb, fb):
            if norm(b) not in nt: errs.append((sid, "awase_basis", b))
        c = dict(a["contract"])
        c.update({"schema_version": "sensor.v2", "splitter_version": "usir_v5_splitter@1",
                  "ledger_hash": "gold-roster:hybrid-manual", "model_id": "claude-sonnet-5-5 (session model, manual annotation)",
                  "prompt_hash": "manual:hybrid-full-run:v1", "work_id": wid, "scene_index": i, "scene_count": len(src)})
        sm = dict(a["scene_meta"])
        sm["nagare"] = {"event_top3": [{"code": c_, "rank_weight": rw} for (c_, _), rw in zip(ev3, (3,2,1))],
                        "mood_top3": [{"code": c_, "rank_weight": rw} for (c_, _), rw in zip(mo3, (3,2,1))]}
        sm["awase"] = {"f_event_valence": Fv, "f_event_arousal": Fa, "f_style_valence": fv, "f_style_arousal": fa}
        ann = {"contract": c, "scene_meta": sm, "utterances": ul, "character_states": a["character_states"], "fact_updates": a["fact_updates"]}
        for e in validate_annotation_schema(ann, schema, ids): errs.append((sid, "schema", e))
        anns.append(ann)
        basis_trace.append({"scene_id": sid, "event": ev3, "mood": mo3, "awase_basis": {"F": Fb, "f": fb}})
    (OUT/f"{wid}.annotations.json").write_text(json.dumps(anns, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (OUT/f"{wid}.basis.json").write_text(json.dumps(basis_trace, ensure_ascii=False, indent=1)+"\n", encoding="utf-8")
print("errors", len(errs))
for e in errs: print(e)
