"""Nagare/Awase extra call: prompt, format schema, deterministic validator (sensor.v2 scene_meta.nagare / awase).

Definitions: usir_v5_nishiki_complete_spec_and_plan_20261110.md section 3.2 (E01-E16 / M01-M16, HW-01) and
section 3.3 / 4.2 (Awase F+f, 5-level ordinal -2..2). Nothing is repaired: a slot whose items fail validation is dropped (null) and counted.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

EVENTS = {
    "E01": "遭遇・邂逅", "E02": "移動・旅立ち", "E03": "対話・交渉", "E04": "対立・口論",
    "E05": "暴力・戦闘・傷害", "E06": "死・葬送・別離", "E07": "発見・露見・覚醒", "E08": "隠蔽・欺瞞・密謀",
    "E09": "労働・制作・日常所作", "E10": "祝祭・飲食・享楽", "E11": "恋愛・接触・情交", "E12": "病気・衰弱・狂気",
    "E13": "契約・誓約・審判", "E14": "逃走・追跡・潜伏", "E15": "救済・赦し・和解", "E16": "停滞・待機・反復",
}
MOODS = {
    "M01": "静寂・澄明・冷気", "M02": "陰翳・幽玄・薄暮", "M03": "焦燥・逼迫・窒息感", "M04": "哀愁・無常・喪失感",
    "M05": "滑稽・道化・自嘲", "M06": "官能・耽美・陶酔", "M07": "畏怖・崇高・怪異", "M08": "倦怠・憂鬱・アンニュイ",
    "M09": "憤怒・憎悪・殺気", "M10": "慈愛・温もり・安らぎ", "M11": "孤独・疎外・寂寥", "M12": "狂騒・祝祭・熱狂",
    "M13": "虚無・白昼夢・非現実", "M14": "決意・凛然・緊張", "M15": "郷愁・追憶・ノスタルジア", "M16": "乾いた非情・無機質",
}


def _defs(d: dict[str, str]) -> str:
    return " / ".join(f"{k}={v}" for k, v in d.items())


SYSTEM_PROMPT_NAGARE_AWASE_V1 = f"""あなたは小説の場面注釈器です。与えられた1場面の本文だけを根拠に、次を JSON で出力します。

1. event: この場面で起きている出来事に最も当てはまる出来事コードを、当てはまりの強い順に、重複なく3つ。各コードに basis（その出来事を示す本文中の語句。本文からそのまま抜く）を付ける。
   出来事コード: {_defs(EVENTS)}
2. mood: この場面の風情（場の雰囲気・情調）に最も当てはまる風情コードを、強い順に、重複なく3つ。各コードに basis（その風情を示す本文中の語句。本文からそのまま抜く）を付ける。
   風情コード: {_defs(MOODS)}
3. awase: 夏目漱石『文学論』の F+f に基づく4つの評定。すべて整数 -2〜+2。
   - F（焦点的出来事）: 場面で起きている出来事そのものの情動的性質。f（文体情緒）: 語り口・語彙・情景描写・リズムが帯びる調子。F と f は別々に評価する（出来事が悲痛でも、文体が淡々と乾いていれば f は 0 付近）。
   - valence（価数）: -2=強く不快・悲痛・凄惨／-1=やや不快／0=中立／+1=やや快／+2=強く快・歓喜。
   - arousal（覚醒度）: -2=極めて静か・沈静／-1=やや静か／0=中庸／+1=やや高ぶり／+2=激しい・高揚。
   - F_basis: F の根拠となる本文中の語句。f_basis: f の根拠となる本文中の語句（本文からそのまま抜く）。

basis の規則（全項目共通）: 本文中に実在する連続した文字列を、一字も変えずにコピーする。コードの名称や定義語（例:「孤独」「憤怒」）の言い換え・要約・複数箇所の連結は不可。本文にその語が無ければ、風情・出来事を最もよく示す本文中の別の語句（4〜20字程度）を選ぶ。

本文に根拠が見つからないコードは選ばない。出力は指定の JSON スキーマに厳密に従うこと。"""


def prompt_hash() -> str:
    return hashlib.sha256(SYSTEM_PROMPT_NAGARE_AWASE_V1.encode("utf-8")).hexdigest()


def format_schema() -> dict[str, Any]:
    def ranked(codes: list[str]) -> dict[str, Any]:
        return {
            "type": "array", "minItems": 3, "maxItems": 3,
            "items": {"type": "object", "required": ["code", "basis"],
                      "properties": {"code": {"type": "string", "enum": codes}, "basis": {"type": "string"}}},
        }

    r = {"type": "integer", "enum": [-2, -1, 0, 1, 2]}
    return {
        "type": "object", "required": ["event", "mood", "awase"],
        "properties": {
            "event": ranked(list(EVENTS)), "mood": ranked(list(MOODS)),
            "awase": {"type": "object",
                      "required": ["F_valence", "F_arousal", "f_valence", "f_arousal", "F_basis", "f_basis"],
                      "properties": {"F_valence": r, "F_arousal": r, "f_valence": r, "f_arousal": r,
                                     "F_basis": {"type": "string"}, "f_basis": {"type": "string"}}},
        },
    }


def build_messages(scene_text: str, scene_index: int, scene_count: int) -> list[dict[str, str]]:
    user = f"【場面 {scene_index + 1}/{scene_count}】\n【本文】\n{scene_text}\n\n上記の場面を注釈してください。"
    return [{"role": "system", "content": SYSTEM_PROMPT_NAGARE_AWASE_V1}, {"role": "user", "content": user}]


_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub("", s)


def _in(s: Any, nt: str) -> bool:
    return isinstance(s, str) and bool(s) and _norm(s) in nt


def validate(parsed: dict[str, Any] | None, scene_text: str):
    """Return (nagare_slot|None, awase_slot|None, audit). rank_weight = 3,2,1 by list order."""
    nt = _norm(scene_text)
    audit: dict[str, Any] = {"parse_ok": parsed is not None, "rejected": [], "trace": {}}
    if parsed is None:
        return None, None, audit
    slot: dict[str, Any] = {}
    ok_nagare = True
    for key, table, out in (("event", EVENTS, "event_top3"), ("mood", MOODS, "mood_top3")):
        items = parsed.get(key) or []
        codes = [i.get("code") for i in items]
        good = True
        if len(items) != 3:
            good = False
            audit["rejected"].append({"kind": key, "reason": "not_3_items"})
        elif len(set(codes)) != 3:
            good = False
            audit["rejected"].append({"kind": key, "reason": "duplicate_codes", "codes": codes})
        else:
            for i in items:
                if i.get("code") not in table:
                    good = False
                    audit["rejected"].append({"kind": key, "reason": "bad_code", "item": i})
                elif not _in(i.get("basis"), nt):
                    good = False
                    audit["rejected"].append({"kind": key, "reason": "basis_not_in_text", "item": i})
        audit["trace"][key] = [{"code": i.get("code"), "basis": i.get("basis")} for i in items]
        if good:
            slot[out] = [{"code": i["code"], "rank_weight": w} for i, w in zip(items, (3, 2, 1))]
        else:
            ok_nagare = False
    aw = parsed.get("awase") or {}
    vals = [aw.get(k) for k in ("F_valence", "F_arousal", "f_valence", "f_arousal")]
    ok_awase = all(isinstance(v, int) and not isinstance(v, bool) and -2 <= v <= 2 for v in vals)
    if not ok_awase:
        audit["rejected"].append({"kind": "awase", "reason": "bad_rating", "values": vals})
    for k in ("F_basis", "f_basis"):
        if not _in(aw.get(k), nt):
            ok_awase = False
            audit["rejected"].append({"kind": "awase", "reason": f"{k}_not_in_text", "value": aw.get(k)})
    audit["trace"]["awase"] = {k: aw.get(k) for k in ("F_basis", "f_basis")}
    awase_slot = ({"f_event_valence": vals[0], "f_event_arousal": vals[1], "f_style_valence": vals[2], "f_style_arousal": vals[3]}
                  if ok_awase else None)
    return (slot if ok_nagare else None), awase_slot, audit
