"""Stage A: per-scene sensor.v2 annotation (scene_meta / character_states / fact_updates).

The utterance part (speaker / addressee / evidence) is produced by the frozen
v5 + postprocess + speaker-recheck pipeline and is NOT re-run here. This module adds one
scene-level LLM call (same model, seed, temperature as the frozen configuration) and a
deterministic validator that rejects anything not grounded in the scene text or the roster.

Deterministic rules enforced by `validate_scene_output`:
  * present_character_ids: ids must be in the roster; each needs a `basis` string that is a
    verbatim substring of the scene text (no mechanical derivation from speakers).
  * character_states: roster id, values_6d = 6 ints in {-1,0,1}, evidence enum, basis substring.
  * fact_updates: predicate in the closed 7-set, roster-or-null subject, non-empty object,
    evidence enum, basis substring.
  * location: surface must be a verbatim substring of the scene text, else null.
  * time.relative_order enum; explicit_time / elapsed_hint must be substrings if given.
Rejected items are dropped and counted in the audit (never silently repaired).
"""

from __future__ import annotations

import json
import re
from typing import Any

PREDICATES = ["LEARNED", "ACQUIRED", "LOST", "MOVED_TO", "STATE_CHANGED", "HOOK_OPENED", "HOOK_RESOLVED"]
EVIDENCE = ["explicit", "inferred", "ambiguous"]
RELATIVE_ORDER = ["continuous", "flashback", "parallel", "unknown"]

SYSTEM_PROMPT_SCENE_V1 = """あなたは小説の場面注釈器です。与えられた1場面の本文と登場人物名簿だけを根拠に、次を JSON で出力します。名簿外の人物は出力しません。本文に根拠が無い項目は出力しない（空配列／null）。

1. time: この場面の時間。relative_order は直前の場面との関係（continuous=直後に続く／flashback=回想／parallel=並行／unknown=本文から判断できない）。explicit_time は本文に書かれた時刻・時期の表現（本文からそのまま抜く。無ければ null）。elapsed_hint は前場面からの経過を示す本文中の表現（無ければ null）。evidence は explicit（本文に明記）／inferred（推測）／ambiguous。
2. location: この場面の主な場所を表す本文中の語句（本文からそのまま抜く。判断できなければ surface=null）。
3. present_characters: この場面にその場で居合わせる人物（名簿の id）。台詞の話し手だけでなく、描写・動作の主体として現場にいる人物も含める。他人の話題に出るだけの人物、回想・伝聞・電話の相手方は含めない。各人物に basis として、居合わせることを示す本文中の語句（本文からそのまま抜く）を付ける。
4. character_states: 場面で能動的な人物（話す・動く・内面が描かれる）ごとに、価値観6軸を {-1,0,+1} で出力する。軸の順序と意味:
   [0]生死観（-1=死・虚無・滅び／+1=生・生命力・再生）
   [1]秩序・規範（-1=反逆・逸脱／+1=秩序・義理・遵法）
   [2]世俗・神秘（-1=世俗・金銭・打算／+1=神秘・聖性・超越）
   [3]利己・献身（-1=利己・支配／+1=献身・自己犠牲・庇護）
   [4]理性・情念（-1=冷徹・打算・抑制／+1=情念・耽美・激情）
   [5]宿命・自由意志（-1=宿命・諦念／+1=自由意志・開拓・抗い）
   その場面の言動・内面から本文で裏付けられる軸だけ ±1 にし、裏付けが無い軸は 0 にする。evidence は explicit／inferred／ambiguous。basis は裏付けとなる本文中の語句（本文からそのまま抜く）。
5. fact_updates: 場面内で起きた事実の更新。predicate は次の7種のみ:
   LEARNED=人物が新しい事実を知る・悟る／ACQUIRED=人物が物・立場・関係を得る／LOST=人物が物・立場・関係・人を失う／MOVED_TO=人物が場所へ移動する／STATE_CHANGED=人物や物の状態が変わる（病気・感情の転換・変身など）／HOOK_OPENED=未解決の謎・約束・懸案が新たに提示される／HOOK_RESOLVED=既出の謎・約束・懸案が解決する。
   subject は名簿の id か null、object は対象を表す短い語句、basis は根拠となる本文中の語句（本文からそのまま抜く）。単なる会話や反復動作は更新に含めない。

出力は指定の JSON スキーマに厳密に従うこと。"""


def scene_prompt_hash() -> str:
    import hashlib

    return hashlib.sha256(SYSTEM_PROMPT_SCENE_V1.encode("utf-8")).hexdigest()


def build_scene_format_schema(roster_ids: list[str]) -> dict[str, Any]:
    idenum = {"type": "string", "enum": roster_ids}
    ev = {"type": "string", "enum": EVIDENCE}
    nstr = {"type": ["string", "null"]}
    return {
        "type": "object",
        "required": ["time", "location", "present_characters", "character_states", "fact_updates"],
        "properties": {
            "time": {
                "type": "object",
                "required": ["relative_order", "explicit_time", "elapsed_hint", "evidence"],
                "properties": {
                    "relative_order": {"type": "string", "enum": RELATIVE_ORDER},
                    "explicit_time": nstr,
                    "elapsed_hint": nstr,
                    "evidence": ev,
                },
            },
            "location": {
                "type": "object",
                "required": ["surface"],
                "properties": {"surface": nstr},
            },
            "present_characters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "basis"],
                    "properties": {"id": idenum, "basis": {"type": "string"}},
                },
            },
            "character_states": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["character_id", "values_6d", "evidence", "basis"],
                    "properties": {
                        "character_id": idenum,
                        "values_6d": {
                            "type": "array",
                            "items": {"type": "integer", "enum": [-1, 0, 1]},
                            "minItems": 6,
                            "maxItems": 6,
                        },
                        "evidence": ev,
                        "basis": {"type": "string"},
                    },
                },
            },
            "fact_updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["subject", "predicate", "object", "evidence", "basis"],
                    "properties": {
                        "subject": {"anyOf": [idenum, {"type": "null"}]},
                        "predicate": {"type": "string", "enum": PREDICATES},
                        "object": {"type": "string"},
                        "evidence": ev,
                        "basis": {"type": "string"},
                    },
                },
            },
        },
    }


def build_scene_messages(
    roster: list[dict[str, Any]], scene_text: str, scene_index: int, scene_count: int, prev_tail: str | None
) -> list[dict[str, str]]:
    lines = []
    for c in roster:
        al = "、".join(c["aliases"]) if c["aliases"] else "（別名なし）"
        lines.append(f"- id={c['id']} 表記: {al} 性別: {c['gender']}")
    user = "【登場人物名簿】\n" + "\n".join(lines) + "\n\n"
    user += f"【場面 {scene_index + 1}/{scene_count}】\n"
    if prev_tail:
        user += f"（参考: 直前の場面の末尾）{prev_tail}\n\n"
    elif scene_index == 0:
        user += "（先頭の場面。直前の場面は無い）\n\n"
    user += "【本文】\n" + scene_text + "\n\n上記の場面を注釈してください。"
    return [{"role": "system", "content": SYSTEM_PROMPT_SCENE_V1}, {"role": "user", "content": user}]


_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub("", s)


def _in_text(snippet: str | None, norm_text: str) -> bool:
    return bool(snippet) and _norm(snippet) in norm_text


def validate_scene_output(
    parsed: dict[str, Any] | None, roster_ids: list[str], scene_text: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (clean_sections, audit). Never repairs; drops invalid items and counts why."""
    audit: dict[str, Any] = {"parse_ok": parsed is not None, "rejected": [], "kept": {}}
    norm_text = _norm(scene_text)
    ids = set(roster_ids)
    out: dict[str, Any] = {
        "time": {"relative_order": "unknown", "evidence": "ambiguous"},
        "location_id": None,
        "present_character_ids": [],
        "character_states": [],
        "fact_updates": [],
    }
    trace: dict[str, Any] = {"present": [], "states": [], "facts": [], "location_surface": None}

    def rej(kind: str, reason: str, item: Any) -> None:
        audit["rejected"].append({"kind": kind, "reason": reason, "item": item})

    if parsed is None:
        audit["kept"] = {"present": 0, "states": 0, "facts": 0}
        return {"sections": out, "trace": trace}, audit

    t = parsed.get("time") or {}
    ro = t.get("relative_order")
    if ro in RELATIVE_ORDER:
        out["time"] = {"relative_order": ro}
        if t.get("evidence") in EVIDENCE:
            out["time"]["evidence"] = t["evidence"]
        for key in ("explicit_time", "elapsed_hint"):
            v = t.get(key)
            if v is None:
                continue
            if _in_text(v, norm_text):
                out["time"][key] = v
            else:
                rej("time", f"{key}_not_in_text", v)
    else:
        rej("time", "bad_relative_order", ro)

    loc = (parsed.get("location") or {}).get("surface")
    if loc:
        if _in_text(loc, norm_text):
            out["location_id"] = loc
            trace["location_surface"] = loc
        else:
            rej("location", "surface_not_in_text", loc)

    seen = set()
    for p in parsed.get("present_characters") or []:
        cid, basis = p.get("id"), p.get("basis")
        if cid not in ids:
            rej("present", "id_not_in_roster", p)
        elif not _in_text(basis, norm_text):
            rej("present", "basis_not_in_text", p)
        elif cid not in seen:
            seen.add(cid)
            out["present_character_ids"].append(cid)
            trace["present"].append({"id": cid, "basis": basis})

    seen_state = set()
    for s in parsed.get("character_states") or []:
        cid, v, ev, basis = s.get("character_id"), s.get("values_6d"), s.get("evidence"), s.get("basis")
        if cid not in ids:
            rej("state", "id_not_in_roster", s)
        elif not (isinstance(v, list) and len(v) == 6 and all(isinstance(x, int) and not isinstance(x, bool) and -1 <= x <= 1 for x in v)):
            rej("state", "bad_values_6d", s)
        elif ev not in EVIDENCE:
            rej("state", "bad_evidence", s)
        elif not _in_text(basis, norm_text):
            rej("state", "basis_not_in_text", s)
        elif cid in seen_state:
            rej("state", "duplicate_character", s)
        else:
            seen_state.add(cid)
            out["character_states"].append({"character_id": cid, "values_6d": v, "evidence": ev})
            trace["states"].append({"id": cid, "basis": basis})

    for f in parsed.get("fact_updates") or []:
        sub, pr, ob, ev, basis = f.get("subject"), f.get("predicate"), f.get("object"), f.get("evidence"), f.get("basis")
        if sub is not None and sub not in ids:
            rej("fact", "subject_not_in_roster", f)
        elif pr not in PREDICATES:
            rej("fact", "bad_predicate", f)
        elif not (isinstance(ob, str) and ob.strip()):
            rej("fact", "empty_object", f)
        elif ev not in EVIDENCE:
            rej("fact", "bad_evidence", f)
        elif not _in_text(basis, norm_text):
            rej("fact", "basis_not_in_text", f)
        else:
            out["fact_updates"].append({"subject": sub, "predicate": pr, "object": ob.strip(), "evidence": ev})
            trace["facts"].append({"basis": basis})

    audit["kept"] = {
        "present": len(out["present_character_ids"]),
        "states": len(out["character_states"]),
        "facts": len(out["fact_updates"]),
    }
    return {"sections": out, "trace": trace}, audit


def validate_annotation_schema(ann: dict[str, Any], schema: dict[str, Any], roster_ids: list[str]) -> list[str]:
    """JSON Schema (sensor.v2) + roster membership across every id-bearing field."""
    import jsonschema

    errs = [f"schema: {e.message} @ {list(e.absolute_path)}" for e in jsonschema.Draft202012Validator(schema).iter_errors(ann)]
    ids = set(roster_ids)
    sm = ann["scene_meta"]
    for cid in sm["present_character_ids"]:
        if cid not in ids:
            errs.append(f"roster: present {cid}")
    if sm["pov_character"] is not None and sm["pov_character"] not in ids:
        errs.append(f"roster: pov {sm['pov_character']}")
    for u in ann["utterances"]:
        for k in ("speaker_id", "addressee_id"):
            if u[k] is not None and u[k] not in ids:
                errs.append(f"roster: utterance {u['utterance_index']} {k}={u[k]}")
    for s in ann["character_states"]:
        if s["character_id"] not in ids:
            errs.append(f"roster: state {s['character_id']}")
        if not (len(s["values_6d"]) == 6 and all(-1 <= x <= 1 for x in s["values_6d"])):
            errs.append(f"range: values_6d {s['values_6d']}")
    for f in ann["fact_updates"]:
        if f["subject"] is not None and f["subject"] not in ids:
            errs.append(f"roster: fact subject {f['subject']}")
    idx = [u["utterance_index"] for u in ann["utterances"]]
    if len(idx) != len(set(idx)):
        errs.append("duplicate utterance_index")
    return errs


def dumps(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, indent=2) + "\n"
