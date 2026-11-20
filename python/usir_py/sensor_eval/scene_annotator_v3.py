"""sensor v3: per-scene annotation (time ISO / facts with sentence_index / value-change events)
and per-work character traits (values_6d, 3-level, estimated once per character).

Revision points (2026-11-20 evening judgment):
  1. time: explicit_time is ISO (YYYY-MM-DD or YYYY-MM-DDTHH:MM) or null. The ISO value must be derivable from the
     verbatim time surface by a deterministic parser (digits/kanji numerals, era names); otherwise it is rejected -> null.
     relative_order includes parallel.
  2. facts: LEARNED / ACQUIRED / LOST / MOVED_TO / HOOK_OPENED / HOOK_RESOLVED, plus STATE_CHANGED only as key=value.
     Subject (the person who learned / received / lost / moved) is required for the four person predicates.
     basis (verbatim) and sentence_index ([S#]) are required; basis must lie inside that sentence.
  3. values: no per-scene character_states. `character_traits` is estimated once per character per work;
     scenes only emit explicit value-change events (value_events).
Invalid items are dropped and counted (never repaired).
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from typing import Any

PREDICATES_V3 = ["LEARNED", "ACQUIRED", "LOST", "MOVED_TO", "HOOK_OPENED", "HOOK_RESOLVED", "STATE_CHANGED"]
PERSON_PREDICATES = {"LEARNED", "ACQUIRED", "LOST", "MOVED_TO"}
EVIDENCE = ["explicit", "inferred", "ambiguous"]
RELATIVE_ORDER = ["continuous", "flashback", "parallel", "unknown"]

SYSTEM_PROMPT_SCENE_V3 = """あなたは小説の場面注釈器です。与えられた1場面の本文（文頭に [S0] [S1] … の文番号、台詞行には [U0] … の番号が付いている）と登場人物名簿だけを根拠に、JSON で出力します。名簿外の人物は出力しません。本文に根拠が無い項目は出力しない（空配列／null）。

1. time:
   - relative_order: 直前の場面との関係（continuous=直後に続く／flashback=回想／parallel=並行（同じ頃・その頃に別の場所で）／unknown=判断できない）。
   - time_surface: 本文に書かれた時刻・日付・時期の表現を本文からそのまま抜く。無ければ null。
   - explicit_time: time_surface を比較できる形に直したもの。年月日が本文に全て書かれている場合のみ YYYY-MM-DD（時刻まで書かれていれば YYYY-MM-DDTHH:MM）。年・月・日のどれかが本文に無い、または相対的な表現（翌朝・三日目・夜 など）は null。年・月・日を推測で補ってはならない。
   - evidence: explicit／inferred／ambiguous。
2. location: この場面の主な場所を表す本文中の語句（本文からそのまま抜く。判断できなければ surface=null）。
3. present_characters: この場面にその場で居合わせる人物（名簿の id）。台詞の話し手だけでなく、描写・動作の主体として現場にいる人物も含める。話題・回想・伝聞・電話の相手のみの人物は含めない。basis は本文からそのまま抜く。
4. fact_updates: 場面内で起きた事実の更新。predicate は次のみ。方向を厳密に守る:
   - LEARNED: subject=新しく知った人物（知った側）。object=知った内容。教えた側ではない。
   - ACQUIRED: subject=物・立場・関係を受け取った人物（受け取った側）。object=得たもの。渡した側ではない。
   - LOST: subject=失った人物。object=失ったもの・人。
   - MOVED_TO: subject=移動した本人。object=移動先の場所。
   - HOOK_OPENED: 物語上の謎・約束・対立など未解決の問いが生じた。subject は関係する人物か null。object=その内容。
   - HOOK_RESOLVED: 既出の謎・約束・対立が解決した。subject は関係する人物か null。object=その内容。
   - STATE_CHANGED: 所有・所在・知識のいずれでも表せない大きな変化（死亡・結婚・病気・身分の変化など）に限り、object を「項目=値」（例: 生死=死亡、婚姻=既婚）の形で書く。
   感情・口調・表情・単なる動作・会話の叙述は事実に入れない。繰り返しの動作も入れない。
   各事実に sentence_index（根拠のある文の番号。[S3] なら 3 の整数）と basis（その [S#] 文の中にある語句を本文からそのまま抜く）を必ず付ける。[U#] だけが付いた台詞行は根拠にできない。[S#] の付いた文（地の文、および [S#] と同じ行にある台詞）から根拠を選ぶ。
5. value_events: 人物の価値観が場面内で明示的に変わった出来事のみ（本文に変化が書かれている場合に限る。無ければ空配列）。axis は 0〜5、to_sign は変化後の向き(-1/+1)、sentence_index と basis を必ず付ける。軸:
   [0]生死観（-1=死・虚無／+1=生・生命力）[1]秩序・規範（-1=反逆／+1=秩序・義理）[2]世俗・神秘（-1=世俗・打算／+1=神秘・聖性）[3]利己・献身（-1=利己／+1=献身）[4]理性・情念（-1=理性・冷徹／+1=情念・激情）[5]宿命・自由意志（-1=宿命・諦念／+1=自由意志・抗い）。

出力は指定の JSON スキーマに厳密に従うこと。"""

SYSTEM_PROMPT_TRAIT_V3 = """あなたは小説の人物分析器です。1人の人物について、作品全体から集めた抜粋（その人物が登場する文、場面番号付き）だけを根拠に、その人物の恒常的な価値観の特性を6軸で出力します。1つの場面だけの気分や状況ではなく、複数の箇所で一貫して見える傾向を読みます。裏付けが無い軸は 0 にする（無理に ±1 にしない）。
軸（値は -1／0／+1）:
[0]生死観（-1=死・虚無・滅び／+1=生・生命力・再生）
[1]秩序・規範（-1=反逆・逸脱／+1=秩序・義理・遵法）
[2]世俗・神秘（-1=世俗・金銭・打算／+1=神秘・聖性・超越）
[3]利己・献身（-1=利己・支配／+1=献身・自己犠牲・庇護）
[4]理性・情念（-1=冷徹・打算・抑制／+1=情念・耽美・激情）
[5]宿命・自由意志（-1=宿命・諦念／+1=自由意志・開拓・抗い）
evidence は explicit／inferred／ambiguous。basis は抜粋の中の語句を抜粋からそのまま抜く。出力は指定の JSON スキーマに厳密に従うこと。"""


def prompt_hash() -> str:
    return hashlib.sha256((SYSTEM_PROMPT_SCENE_V3 + "\n---\n" + SYSTEM_PROMPT_TRAIT_V3).encode("utf-8")).hexdigest()


def scene_format_schema(roster_ids: list[str], n_sent: int) -> dict[str, Any]:
    idenum = {"type": "string", "enum": roster_ids}
    ev = {"type": "string", "enum": EVIDENCE}
    nstr = {"type": ["string", "null"]}
    sidx = {"type": "integer", "minimum": 0, "maximum": max(0, n_sent - 1)}
    return {
        "type": "object",
        "required": ["time", "location", "present_characters", "fact_updates", "value_events"],
        "properties": {
            "time": {
                "type": "object",
                "required": ["relative_order", "time_surface", "explicit_time", "evidence"],
                "properties": {
                    "relative_order": {"type": "string", "enum": RELATIVE_ORDER},
                    "time_surface": nstr,
                    "explicit_time": nstr,
                    "evidence": ev,
                },
            },
            "location": {"type": "object", "required": ["surface"], "properties": {"surface": nstr}},
            "present_characters": {
                "type": "array",
                "items": {"type": "object", "required": ["id", "basis"], "properties": {"id": idenum, "basis": {"type": "string"}}},
            },
            "fact_updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["subject", "predicate", "object", "sentence_index", "evidence", "basis"],
                    "properties": {
                        "subject": {"anyOf": [idenum, {"type": "null"}]},
                        "predicate": {"type": "string", "enum": PREDICATES_V3},
                        "object": {"type": "string"},
                        "sentence_index": sidx,
                        "evidence": ev,
                        "basis": {"type": "string"},
                    },
                },
            },
            "value_events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["character_id", "axis", "to_sign", "sentence_index", "basis"],
                    "properties": {
                        "character_id": idenum,
                        "axis": {"type": "integer", "minimum": 0, "maximum": 5},
                        "to_sign": {"type": "integer", "enum": [-1, 1]},
                        "sentence_index": sidx,
                        "basis": {"type": "string"},
                    },
                },
            },
        },
    }


def trait_format_schema(cid: str) -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["values_6d", "evidence", "basis"],
        "properties": {
            "values_6d": {"type": "array", "items": {"type": "integer", "enum": [-1, 0, 1]}, "minItems": 6, "maxItems": 6},
            "evidence": {"type": "string", "enum": EVIDENCE},
            "basis": {"type": "string"},
        },
    }


def scene_messages(roster: list[dict[str, Any]], numbered_text: str, si: int, n_scenes: int, prev_tail: str | None) -> list[dict[str, str]]:
    lines = []
    for c in roster:
        al = "、".join(c["aliases"]) if c["aliases"] else "（別名なし）"
        lines.append(f"- id={c['id']} 表記: {al} 性別: {c['gender']}")
    user = "【登場人物名簿】\n" + "\n".join(lines) + "\n\n" + f"【場面 {si + 1}/{n_scenes}】\n"
    if prev_tail:
        user += f"（参考: 直前の場面の末尾）{prev_tail}\n\n"
    elif si == 0:
        user += "（先頭の場面。直前の場面は無い）\n\n"
    user += "【本文】\n" + numbered_text + "\n\n上記の場面を注釈してください。"
    return [{"role": "system", "content": SYSTEM_PROMPT_SCENE_V3}, {"role": "user", "content": user}]


def trait_messages(char: dict[str, Any], excerpts: str) -> list[dict[str, str]]:
    al = "、".join(char["aliases"]) if char["aliases"] else char["id"]
    user = f"【人物】id={char['id']} 表記: {al} 性別: {char['gender']}\n\n【作品からの抜粋】\n{excerpts}\n\n上記の人物の恒常的な価値観の特性を出力してください。"
    return [{"role": "system", "content": SYSTEM_PROMPT_TRAIT_V3}, {"role": "user", "content": user}]


_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub("", s)


def in_text(snippet: str | None, text: str) -> bool:
    return bool(snippet) and _norm(snippet) in _norm(text)


# ---- ISO normalisation (deterministic) --------------------------------------------------------------
_KD = {"〇": 0, "零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_ERA = {"明治": 1867, "大正": 1911, "昭和": 1925, "平成": 1988, "令和": 2018}
_NUM = r"[0-9０-９〇零一二三四五六七八九十百]+"


def parse_num(s: str) -> int | None:
    s = s.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    if s.isdigit():
        return int(s)
    if s == "元":
        return 1
    if set(s) <= set(_KD) | {"十", "百"}:
        if "百" in s or "十" in s:
            total, cur = 0, 0
            for ch in s:
                if ch == "百":
                    total += (cur or 1) * 100
                    cur = 0
                elif ch == "十":
                    total += (cur or 1) * 10
                    cur = 0
                else:
                    cur = _KD[ch]
            return total + cur
        return int("".join(str(_KD[c]) for c in s))
    return None


def iso_from_surface(surface: str) -> str | None:
    """Deterministic: full year+month+day (Gregorian digits or an era name) required; optional HH時[MM分]."""
    s = _norm(surface)
    y = mo = d = None
    m = re.search(r"(明治|大正|昭和|平成|令和)(元|" + _NUM + r")年", s)
    if m:
        n = parse_num(m.group(2))
        y = _ERA[m.group(1)] + n if n else None
    else:
        m = re.search(r"(" + _NUM + r")年", s)
        n = parse_num(m.group(1)) if m else None
        y = n if n and n >= 1000 else None
    m = re.search(r"(" + _NUM + r")月", s)
    mo = parse_num(m.group(1)) if m else None
    m = re.search(r"(" + _NUM + r")日", s)
    d = parse_num(m.group(1)) if m else None
    if not (y and mo and d):
        return None
    try:
        date(y, mo, d)
    except ValueError:
        return None
    out = f"{y:04d}-{mo:02d}-{d:02d}"
    m = re.search(r"(" + _NUM + r")時(?:(" + _NUM + r")分|半)?", s)
    if m:
        hh = parse_num(m.group(1))
        mm = parse_num(m.group(2)) if m.group(2) else (30 if "時半" in s else 0)
        if hh is not None and 0 <= hh < 24 and mm is not None and 0 <= mm < 60:
            out += f"T{hh:02d}:{mm:02d}"
    return out


_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2})?$")
_KV_RE = re.compile(r"^[^=\s]{1,20}=[^=]{1,30}$")


def validate_scene_v3(parsed: dict[str, Any] | None, roster_ids: list[str], scene_text: str, sentences: list[tuple[int, int, int]]) -> tuple[dict[str, Any], dict[str, Any]]:
    audit: dict[str, Any] = {"parse_ok": parsed is not None, "rejected": [], "kept": {}}
    ids = set(roster_ids)
    chars = list(scene_text)
    sent_text = {i: _norm("".join(chars[a:b])) for i, a, b in sentences}
    out: dict[str, Any] = {
        "time": {"relative_order": "unknown", "explicit_time": None, "explicit_time_surface": None, "evidence": "ambiguous"},
        "location_id": None, "present_character_ids": [], "fact_updates": [], "value_events": [],
    }
    trace: dict[str, Any] = {"present": [], "facts": [], "events": []}

    def rej(kind: str, reason: str, item: Any) -> None:
        audit["rejected"].append({"kind": kind, "reason": reason, "item": item})

    if parsed is None:
        audit["kept"] = {"present": 0, "facts": 0, "events": 0}
        return {"sections": out, "trace": trace}, audit
    t = parsed.get("time") or {}
    if t.get("relative_order") in RELATIVE_ORDER:
        out["time"]["relative_order"] = t["relative_order"]
    else:
        rej("time", "bad_relative_order", t.get("relative_order"))
    if t.get("evidence") in EVIDENCE:
        out["time"]["evidence"] = t["evidence"]
    surf = t.get("time_surface")
    if surf:
        if in_text(surf, scene_text):
            out["time"]["explicit_time_surface"] = surf
        else:
            rej("time", "time_surface_not_in_text", surf)
            surf = None
    iso = t.get("explicit_time")
    if iso is not None:
        det = iso_from_surface(surf) if surf else None
        if not _ISO_RE.match(iso):
            rej("time", "iso_bad_format", iso)
        elif det is None:
            rej("time", "iso_not_derivable_from_surface", iso)
        elif det != iso:
            rej("time", f"iso_differs_from_parser:{det}", iso)
        else:
            out["time"]["explicit_time"] = iso
    loc = (parsed.get("location") or {}).get("surface")
    if loc:
        if in_text(loc, scene_text):
            out["location_id"] = loc
        else:
            rej("location", "surface_not_in_text", loc)
    seen = set()
    for p in parsed.get("present_characters") or []:
        cid, basis = p.get("id"), p.get("basis")
        if cid not in ids:
            rej("present", "id_not_in_roster", p)
        elif not in_text(basis, scene_text):
            rej("present", "basis_not_in_text", p)
        elif cid not in seen:
            seen.add(cid)
            out["present_character_ids"].append(cid)
            trace["present"].append({"id": cid, "basis": basis})
    for f in parsed.get("fact_updates") or []:
        sub, pr, ob, si, ev, basis = (f.get(k) for k in ("subject", "predicate", "object", "sentence_index", "evidence", "basis"))
        if pr not in PREDICATES_V3:
            rej("fact", "bad_predicate", f)
        elif sub is not None and sub not in ids:
            rej("fact", "subject_not_in_roster", f)
        elif pr in PERSON_PREDICATES and sub is None:
            rej("fact", "subject_required", f)
        elif not (isinstance(ob, str) and ob.strip()):
            rej("fact", "empty_object", f)
        elif pr == "STATE_CHANGED" and not _KV_RE.match(ob.strip()):
            rej("fact", "state_changed_not_key_value", f)
        elif ev not in EVIDENCE:
            rej("fact", "bad_evidence", f)
        elif si not in sent_text:
            rej("fact", "sentence_index_out_of_range", f)
        elif not basis or _norm(basis) not in sent_text[si]:
            rej("fact", "basis_not_in_sentence", f)
        else:
            out["fact_updates"].append({"sentence_index": si, "subject": sub, "predicate": pr, "object": ob.strip(), "evidence": ev})
            trace["facts"].append({"basis": basis})
    for e in parsed.get("value_events") or []:
        cid, ax, sg, si, basis = (e.get(k) for k in ("character_id", "axis", "to_sign", "sentence_index", "basis"))
        if cid not in ids:
            rej("event", "id_not_in_roster", e)
        elif si not in sent_text or not basis or _norm(basis) not in sent_text[si]:
            rej("event", "basis_not_in_sentence", e)
        else:
            out["value_events"].append({"character_id": cid, "axis": ax, "to_sign": sg, "sentence_index": si})
            trace["events"].append({"basis": basis})
    audit["kept"] = {"present": len(out["present_character_ids"]), "facts": len(out["fact_updates"]), "events": len(out["value_events"])}
    return {"sections": out, "trace": trace}, audit


def validate_trait(parsed: dict[str, Any] | None, excerpts: str) -> tuple[dict[str, Any] | None, str | None]:
    if parsed is None:
        return None, "parse_failed"
    v = parsed.get("values_6d")
    if not (isinstance(v, list) and len(v) == 6 and all(isinstance(x, int) and not isinstance(x, bool) and -1 <= x <= 1 for x in v)):
        return None, "bad_values_6d"
    if parsed.get("evidence") not in EVIDENCE:
        return None, "bad_evidence"
    if any(v):  # an all-zero trait needs no evidence; a non-zero trait needs a verbatim basis from the excerpts
        plain = re.sub(r"\(場面\d+\)\s*", "", excerpts)
        frags = [f for f in re.split(r"[。\n]",parsed.get("basis") or "") if f.strip()]
        if not frags or not all(in_text(f, plain) for f in frags):  # basis may quote several sentences; every fragment must be verbatim
            return None, "basis_not_in_excerpts"
    return {"values_6d": v, "evidence": parsed["evidence"]}, None


def dumps(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, indent=2) + "\n"
