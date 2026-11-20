"""Prompt variants for the single-shot improvement trial.

fact_v2: only item 5 (fact_updates) of SYSTEM_PROMPT_SCENE_V1 is replaced; items 1-4 are byte-identical.
values_ctx: system prompt = V1 + one addendum line; the user message gets a 【既知情報】 block.
"""
from __future__ import annotations

import hashlib

from usir_py.sensor_eval.scene_annotator import SYSTEM_PROMPT_SCENE_V1, build_scene_messages

_HEAD, _SEP, _TAIL = SYSTEM_PROMPT_SCENE_V1.partition("5. fact_updates:")
assert _SEP, "item 5 marker not found"

FACT_V2_ITEM5 = """5. fact_updates: 本文に明示の根拠がある「状態変化」だけを出力する。迷ったら出さない（出力は空配列でよい）。1場面で多くても3件。
   出力してよい条件（すべて満たすこと）:
   (a) 場面の中で実際に起きた変化である。本文の語句を basis にそのまま抜けること（basis は変化そのものを述べた語句。会話の感想・評価・比喩・情景描写は不可）。
   (b) 変化の主体が分かる。subject は変化を起こした／受けた本人の名簿 id。主体が本文で特定できない変化は出さない（HOOK_OPENED / HOOK_RESOLVED のみ subject=null 可）。
   (c) 他人の発言・行為を、その聞き手や見ている人物の変化として書き換えない。変化した本人を subject にする。
   predicate の定義（この意味に当てはまるものだけ）:
   LEARNED=その人物が、これまで知らなかった事実を新たに知らされた・悟った（話す側ではなく聞く側／気づいた側が subject。会話があっただけ、質問されただけ、は含めない）／
   ACQUIRED=その人物が物・金・立場・関係を新たに手に入れた（受け取った側が subject。渡した側、運ぶ・見る・持つだけの場合、遊びの配役は含めない）／
   LOST=その人物が物・金・立場・関係・人を失った（死別・喪失・手放した）／
   MOVED_TO=その人物が別の場所へ移動した（subject は移動した本人。場所名は本文の語句で）／
   STATE_CHANGED=人物や物の状態が、場面内で明確に別の状態になった（病気・失踪・変身・死・身分の変化など）。一時的な気分・表情・口調・内心の揺れ、単なる動作の反復や描写は含めない／
   HOOK_OPENED=本文で明示的に提示された未解決の謎・約束・依頼・企み（地名や物の言及、感想、情景は含めない）／
   HOOK_RESOLVED=既出の謎・約束・懸案が、この場面で明示的に解決した。
   object は変化の内容を表す短い語句（名詞句）。"""

SYSTEM_PROMPT_FACT_V2 = _HEAD + FACT_V2_ITEM5 + "\n\n出力は指定の JSON スキーマに厳密に従うこと。"

VALUES_CTX_ADDENDUM = (
    "\n\n補足: 利用者メッセージに【既知情報】がある場合、それは各人物の背景と、前の場面までに本文で確定した内容の要約である。"
    "軸の判断に参照してよいが、軸を ±1 にする場合も basis はこの場面の本文中の語句にすること。"
)
SYSTEM_PROMPT_VALUES_CTX = SYSTEM_PROMPT_SCENE_V1 + VALUES_CTX_ADDENDUM

AXIS = ["生死観", "秩序・規範", "世俗・神秘", "利己・献身", "理性・情念", "宿命・自由意志"]


def prompt_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def known_info_block(characters: list[dict], history: list[dict]) -> str:
    """history: per previous scene, in order: {'scene': k, 'states': [validated], 'facts': [validated]}.
    Only the sensor's own validated outputs with evidence == explicit are summarized; no gold is used."""
    lines = []
    for c in characters:
        parts = []
        role = c.get("role")
        if role:
            parts.append(f"名簿の説明: {role}")
        prev = []
        for h in history:
            for s in h["states"]:
                if s["character_id"] != c["id"] or s.get("evidence") != "explicit":
                    continue
                ax = [f"{AXIS[i]}{'+' if v > 0 else '-'}" for i, v in enumerate(s["values_6d"]) if v]
                if ax:
                    prev.append(f"場面{h['scene'] + 1}で価値観: " + "・".join(ax))
            for f in h["facts"]:
                if f["subject"] == c["id"] and f.get("evidence") == "explicit":
                    prev.append(f"場面{h['scene'] + 1}で {f['predicate']}: {f['object']}")
        parts.extend(prev[-6:])
        lines.append(f"- id={c['id']}: " + (" / ".join(parts) if parts else "（既知情報なし）"))
    return "【既知情報】\n" + "\n".join(lines) + "\n\n"


def build_messages(variant, roster, scene_text, si, n, prev_tail, characters=None, history=None):
    m = build_scene_messages(roster, scene_text, si, n, prev_tail)
    if variant == "control":
        return m
    if variant == "fact_v2":
        m[0] = {"role": "system", "content": SYSTEM_PROMPT_FACT_V2}
        return m
    if variant == "values_ctx":
        m[0] = {"role": "system", "content": SYSTEM_PROMPT_VALUES_CTX}
        u = m[1]["content"]
        k = u.index(f"【場面 {si + 1}/{n}】")
        m[1]["content"] = u[:k] + known_info_block(characters, history) + u[k:]
        return m
    raise ValueError(variant)
