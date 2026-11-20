"""Convert hand-written annotations_src/*.json (session model, written before opening gold) into
stage-A style annotation files (same layout as scripts/phase2_stage2/annotations) and validate
against scene text: ids in roster, closed predicate set, 6-axis values in {-1,0,1},
location / explicit_time / elapsed_hint must be verbatim substrings of the scene text."""
import json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
inp = json.loads((HERE/"scenes_input.json").read_text(encoding="utf-8"))
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
PRED = {"LEARNED","ACQUIRED","LOST","MOVED_TO","STATE_CHANGED","HOOK_OPENED","HOOK_RESOLVED"}
ORD = {"continuous","flashback","parallel","unknown"}
norm = lambda s: re.sub(r"\s+","",s)
errs = []; tot = 0
for wid, w in inp.items():
    src = json.loads((HERE/"annotations_src"/f"{wid}.json").read_text(encoding="utf-8"))["scenes"]
    assert len(src) == len(w["scenes"]), wid
    ids = [r["id"] for r in w["roster"]]
    pov = w["narrator"] if w["narration"] == "first" else w.get("pov")
    anns = []
    for i, (a, sc) in enumerate(zip(src, w["scenes"])):
        nt = norm(sc["text"]); sid = sc["scene_id"]
        assert a["order"] in ORD, sid
        for k in ("location","explicit_time","elapsed_hint"):
            v = a.get(k)
            if v and norm(v) not in nt: errs.append((sid,k,v))
        for p in a["present"]:
            if p not in ids: errs.append((sid,"present",p))
        for c, v in a["states"].items():
            if c not in ids or len(v)!=6 or any(x not in (-1,0,1) for x in v): errs.append((sid,"state",c,v))
        for f in a["facts"]:
            if f[1] not in PRED or (f[0] is not None and f[0] not in ids) or not f[2]: errs.append((sid,"fact",f))
        tot += len(a["facts"])
        time = {"relative_order": a["order"], "evidence": "explicit" if a.get("explicit_time") else "inferred"}
        if a.get("explicit_time"): time["explicit_time"] = a["explicit_time"]
        if a.get("elapsed_hint"): time["elapsed_hint"] = a["elapsed_hint"]
        anns.append({
            "contract": {"schema_version":"sensor.v2","scene_id":sid,"scene_hash":sc["scene_hash"],"work_id":wid,"scene_index":i,"scene_count":len(src),
                         "model_id":"claude-sonnet-5-5 (session model, manual annotation)","char_len":len(sc["text"])},
            "scene_meta": {"time": time, "location_id": a.get("location"), "present_character_ids": a["present"], "pov_character": pov},
            "utterances": [],
            "character_states": [{"character_id": c, "values_6d": v, "evidence": "inferred"} for c, v in a["states"].items()],
            "fact_updates": [{"subject": s, "predicate": p, "object": o, "evidence": e} for s, p, o, e in a["facts"]],
        })
    (OUT/f"{wid}.annotations.json").write_text(json.dumps(anns, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
print("facts total", tot, "errors", len(errs))
for e in errs: print(e)
