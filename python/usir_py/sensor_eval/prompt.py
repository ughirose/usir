"""Prompt construction and Ollama structured-output schema for sensor.v2 utterances."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from usir_py.sensor_eval.f9_loader import F9Work

PROMPT_VERSION = "step1_utterances_v1"

SYSTEM_PROMPT = """あなたは日本文学テクストの話者帰属・口調注釈センサー（sensor.v2）です。
入力された場面本文には、最外のカギ括弧の直前に `[U0]`, `[U1]`, ... という連番マーカーが振られています。
すべての `[U0]` 〜 `[U{N-1}]` について、以下の JSON スキーマに厳密に従った `utterances` 配列のみを JSON オブジェクトとして出力してください。余計な説明文やコードブロック記法は出力しないでください。

## 出力規則（sensor.v2 準拠）
1. **欠落・重複の禁止**:
   - `utterance_index` は `0` から `N-1` までのすべての連番を昇順で 1 回ずつ必ず含めること。
2. **`speaker_id` と `addressee_id` の制約**:
   - 必ず与えられた「人物名簿」の `id`（英数字識別子。例: `yozo`, `musume`）または `null` のみを出力すること。
   - 日本語の別名・漢字表記・名簿に無い文字列は決して出力しないこと。
   - 語り手が一人称（`narration == "first"`）で `narrator` が指定されている場合、語り手自身の発話（地の文の語り手の一人称「私」「僕」「おれ」や、語り手自身の内言・発話）は `narrator` の `id` に帰属すること。
3. **`evidence` の3段階基準**:
   - `"explicit"`: 発話の直前・直後の伝達節や地の文（例: 「〜と〇〇が言った」、同一話者の明示的継続描写、発話内の決定的呼称など）により、本文中の直接的な根拠から話者が確実に特定できる場合。
   - `"inferred"`: 明示的な伝達節は無いが、会話の交互交替、直前発話への応答関係、人物の口調・性別・文脈などから話者が合理的に推定できる場合。
   - `"ambiguous"`: 以下のいずれかに該当し、話者を確定できない場合。
     (a) 複数人の同時発話、または 2 人以上のどちらが話したか判断できない断片
     (b) 発話ではなく、地の文中の語句の強調・術語・書名・単なる引用符（例: `「生活」`、`「聯絡」` 等）
     (c) 抜粋範囲外に根拠があり本文から話者を特定できない場合
   - `evidence` が `"ambiguous"` のとき、`speaker_id` は原則 `null` とすること。根拠が薄いのに無理に誰かへ当てはめないこと。
4. **`addressee_id`**:
   - その発話が向けられた相手の `id`。呼びかけ・対話文脈から特定できる場合はその `id`、独り言・心内語・強調引用・相手不明の場合は `null`。
5. **`register.politeness`（5値）**:
   - `"plain"`: 常体・普通体（だ／である調、親しい間の自然な口語、敬語でも粗暴でもない発話）
   - `"polite"`: 丁寧語（です／ます／でしょう／ませんか 等）
   - `"honorific"`: 尊敬語・謙譲語・丁重な敬語（ございます／おっしゃる／下さいまし／存じます 等）
   - `"rough"`: ぞんざい・荒っぽい・威圧的・命令口調（貴様／お前／〜やがる／黙れ／てめえ 等）
   - `"unknown"`: 1語のみの断片や非発話引用（強調語句等）で口調を判定できない場合
6. **`register.first_person`**:
   - その発話のセリフ内に明示的に含まれる一人称代名詞（例: `"ぼく"`, `"私"`, `"わたくし"`, `"おれ"`, `"あたし"`）を文字列で記入。発話内に一人称が含まれない場合は `null`（地の文から補わない）。"""


def build_ollama_utterances_format_schema() -> dict[str, Any]:
    """Build a flat JSON Schema (no $ref/$defs/anyOf) for Ollama structured outputs."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["utterances"],
        "properties": {
            "utterances": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "utterance_index",
                        "speaker_id",
                        "addressee_id",
                        "evidence",
                        "register",
                    ],
                    "properties": {
                        "utterance_index": {"type": "integer", "minimum": 0},
                        "speaker_id": {"type": ["string", "null"]},
                        "addressee_id": {"type": ["string", "null"]},
                        "evidence": {
                            "type": "string",
                            "enum": ["explicit", "inferred", "ambiguous"],
                        },
                        "register": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["politeness", "first_person"],
                            "properties": {
                                "politeness": {
                                    "type": "string",
                                    "enum": [
                                        "plain",
                                        "polite",
                                        "honorific",
                                        "rough",
                                        "unknown",
                                    ],
                                },
                                "first_person": {"type": ["string", "null"]},
                            },
                        },
                    },
                },
            }
        },
    }


def compute_prompt_hash(system_prompt: str = SYSTEM_PROMPT) -> str:
    """Return deterministic SHA-256 hex digest identifying prompt version and text."""
    payload = f"{PROMPT_VERSION}\n{system_prompt}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_work_messages(work: F9Work) -> list[dict[str, str]]:
    """Build Ollama chat messages for one F9 work excerpt."""
    roster_lines: list[str] = []
    for char in work.characters:
        aliases_str = ", ".join(char.aliases) if char.aliases else "(別名なし)"
        roster_lines.append(
            f"- id: `{char.id}` | aliases: [{aliases_str}] | gender: {char.gender}"
        )
    roster_block = "\n".join(roster_lines)

    n_utts = len(work.utterances)
    user_prompt = (
        f"## 場面メタ情報\n"
        f"- 作品ID: {work.work_id}\n"
        f"- 語り形式 (`narration`): {work.narration}\n"
        f"- 一人称語り手 (`narrator`): {work.narrator or 'null'}\n"
        f"- 視点人物 (`pov`): {work.pov or 'null'}\n"
        f"- 発話総数 (`[U0]`〜`[U{n_utts - 1}]`): {n_utts} 件\n\n"
        f"## 人物名簿（出力の speaker_id / addressee_id は必ずこの `id` のみ使用）\n"
        f"{roster_block}\n\n"
        f"## 番号付き本文\n"
        f"{work.numbered_text}\n\n"
        f"上記 `[U0]` から `[U{n_utts - 1}]` までの全 {n_utts} 件について JSON を出力してください。"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
