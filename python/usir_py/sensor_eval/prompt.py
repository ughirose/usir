"""Prompt construction and Ollama structured-output schema for sensor.v2 utterances."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from usir_py.sensor_eval.f9_loader import F9Work
from usir_py.sensor_eval.features import (
    extract_work_utterance_features,
    format_utterance_features_block,
)

PROMPT_VERSION_V1 = "step1_utterances_v1"
PROMPT_VERSION_V2 = "step1_utterances_v2"
PROMPT_VERSION_V3 = "step1_utterances_v3"
PROMPT_VERSION_V4 = "step1_utterances_v4"

SYSTEM_PROMPT_V1 = """あなたは日本文学テクストの話者帰属・口調注釈センサー（sensor.v2）です。
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

SYSTEM_PROMPT_V2 = """あなたは日本文学テクストの話者帰属・口調注釈センサー（sensor.v2）です。
入力された場面本文には、最外のカギ括弧の直前に `[U0]`, `[U1]`, ... という連番マーカーが振られています。
すべての `[U0]` 〜 `[U{N-1}]` について、以下の JSON スキーマに厳密に従った `utterances` 配列のみを JSON オブジェクトとして出力してください。余計な説明文やコードブロック記法は出力しないでください。

## 出力規則（sensor.v2 準拠）

1. **欠落・重複の禁止**:
   - `utterance_index` は `0` から `N-1` までのすべての連番を昇順で 1 回ずつ必ず含めること。

2. **`speaker_id` と人物名簿の照合**:
   - 必ず与えられた「人物名簿」の `id`（英数字識別子）または JSON の `null` のみを出力すること。日本語名・別名・名簿外の文字列は出力しないこと。
   - 名簿の `aliases`（別名・肩書・呼び名）と `gender` を丁寧に照合すること。
   - 発話内の呼びかけ語（例: 「〇〇さん、〜」「先生、〜」）は話者ではなく聞き手（`addressee_id`）を指すため、呼びかけられた相手を話者に誤認しないこと。
   - 一人称語り（`narration == "first"`）で `narrator` がある場合、語り手（「私」「僕」「自分」「おれ」等）が実際に発したセリフ・心内語は `narrator` に帰属する。ただし、地の文中の強調語句・引用語句（後述の非発話）は語り手の発話ではない。

3. **日本文学の会話構造に関する重要規則（誤帰属の防止）**:
   - **後置伝達節（発話の後ろに話者が来る形）**:
     カギ括弧の直後に続く「〜と〇〇は言った」「〇〇は答えました」「〇〇は〜しながら云った」「〇〇はおじぎしました」のような伝達節・動作描写は、**直前の発話 `[Uk]` の話者**を示すものであり、次の発話 `[U(k+1)]` の導入ではない。各発話の直前と直後の両方を必ず確認すること。
     - 例1: `[U0]「雨が降ってきたよ。」太郎は窓の外を見て言った。[U1]「傘を持っていこう。」花子が玄関で答えた。`
       → `[U0]` は `taro`（`explicit`）、`[U1]` は `hanako`（`explicit`）。`太郎は…` を `[U1]` に結びつけない。
     - 例2: `[U0]「ここで少し休みませんか。」と、客は息をついて尋ねた。`
       → `[U0]` は `kyaku`（`explicit`）。
   - **地の文を挟んだ同一話者の継続（単純な交互交替を仮定しない）**:
     発話と発話の間に動作・仕草・聞き手の無言の反応（首を振る、うなずく、微笑む、立ち止まる等）や情景描写が挟まっても、話者が交代せず**同一話者が続けて話す**場合がある。前後の口調・方言・内容の連続性・無言反応を確認すること。
     - 例3: `[U0]「もう日も暮れるな、」と老人は微笑み、[U1]「宿を探そう。」`
       → `[U0]` は `rojin`（`explicit`）、`[U1]` も同一話者継続で `rojin`（`inferred`）。
     - 例4: `[U0]「この道で合っているかな。」案内人は地図を広げた。旅人は黙ってうなずいた。[U1]「それなら、もう少し急ごう。」`
       → 「黙ってうなずいた」は無言動作であり、`[U1]` も続けて `annainin`（`inferred`）。
     - 例5: 地の文で別の人物が単に紹介・描写されただけで発話動詞が無い場合、直後の発話がその人物のものとは限らない（口調や後続の伝達節で確認する）。

4. **非発話引用（`not_speech`）の除外**:
   - 地の文に埋め込まれた語句の強調・比喩・慣用句・言葉自体への言及・名称の引用は、誰かの発話ではない。
   - 一人称語りの地の文中にあっても語り手へ帰属せず、必ず `speaker_id: null`, `addressee_id: null`, `evidence: "ambiguous"`, `register.politeness: "unknown"` とすること。
   - 例6（いずれも非発話 → `speaker_id: null`, `evidence: "ambiguous"`）:
     - `彼は[U0]「自由」という言葉を嫌った。`
     - `いわゆる[U1]「常識」で測れる話ではない。`
     - `[U2]「冬の星」のように冷たい光だった。`
     - `実際に[U3]「怖い」と口に出したわけではない。`

5. **`evidence` の厳格な 3 段階基準**:
   - **`"explicit"`（厳格基準・直近地の文での明示のみ）**:
     発話の**直前または直後の地の文**に、名簿の人物（別名や一人称語り手の「私・僕・自分」等を含む）が発話・応答・呼びかけの主体として**直接明示されている場合のみ** `"explicit"` とする。
     - **次のような場合は、話者に自信があっても `"explicit"` にしてはならない（必ず `"inferred"` とする）**:
       - 地の文が無くセリフが連続し、会話の交互交替から判断した場合
       - 口調・方言・敬語・性別・呼称などのセリフ内容から推定した場合
       - 離れた前段落や無言動作を挟んだ同一話者継続を文脈で判断した場合
       - 直近の地の文に人物名は出るが、その人物が話したとは明記されていない場合
   - **`"inferred"`（文脈推定）**:
     直近の地の文に話者明示は無いが、会話の交互交替、直前発話への応答関係、口調・方言・性別・呼称・場面の動作描写から 1 人の話者を合理的に推定できる場合。合理的な根拠があるときは `"ambiguous"` に逃げず、最も妥当な `speaker_id` と `"inferred"` を出力すること。
   - **`"ambiguous"`（判定不能・非発話）**:
     以下のいずれかに該当し、話者を 1 人に確定できない場合は、無理に誰かへ帰属せず `speaker_id: null`, `evidence: "ambiguous"` とすること。
     (a) 上記 4 の非発話引用（強調語句・比喩・語句への言及など）
     (b) 複数人の同時発話・合唱・斉唱（例: 「〜」と二人は同時に叫んだ／三人は笑った）
     (c) 複数の候補が同等にあり得て、本文中に話者を絞り込む手がかり（伝達節・交互交替の起点・口調差・内容の固有性）が無い場合（例: 「二人は語り合った」直後の未帰属の会話断片など）
   - `evidence` が `"ambiguous"` のとき、`speaker_id` は必ず JSON の `null` とすること。

6. **`addressee_id`**:
   - 発話が向けられた相手の `id`。呼びかけ・対話文脈から特定できる場合はその `id`、独り言・心内語・非発話引用・複数向け・相手不明の場合は `null`。

7. **`register.politeness`（5値）と `register.first_person`**:
   - `"plain"`: 常体・普通体（だ／である調、通常の口語）
   - `"polite"`: 丁寧語（です／ます／でしょう／ませんか 等）
   - `"honorific"`: 尊敬語・謙譲語・丁重な敬語（ございます／おっしゃる／下さいまし／存じます 等）
   - `"rough"`: ぞんざい・荒っぽい・威圧的・命令口調（貴様／お前／〜やがる／黙れ／てめえ 等）
   - `"unknown"`: 1語のみの断片や非発話引用で口調を判定できない場合
   - `first_person`: セリフ内に明示的に含まれる一人称代名詞（例: `"ぼく"`, `"私"`, `"わたくし"`, `"おれ"`, `"あたし"`, `"うち"`）を文字列で記入。含まれない場合は `null`（地の文から補わない）。"""

SYSTEM_PROMPT_V3 = """あなたは日本文学テクストの話者帰属・口調注釈センサー（sensor.v2）です。
本文の最外カギ括弧の直前には `[U0]`, `[U1]`, ... が振られています。
`0` から `N-1` まで欠落・重複なく全件の `utterances` 配列のみを JSON オブジェクトで出力してください。

## 判定手順（各 `[Uk]` を次の順に判定する）

### 手順1: 非発話引用（`not_speech`）の除外（最優先）
- 地の文の**文の途中**に単語・語句・比喩がカギ括弧で挟まれ、独立した発話ではないものは「非発話」である。
  - 例: `彼は[U0]「自由」を嫌った。` / `いわゆる[U1]「常識」だ。` / `[U2]「冬の星」のように光る。` / `口で[U3]「怖い」とは言わない。`
- 一人称語り（`narration == "first"`）の地の文中であっても、語り手へ帰属してはならない。
- 非発話は必ず `speaker_id: null`, `addressee_id: null`, `evidence: "ambiguous"`, `register.politeness: "unknown"` とする。

### 手順2: 話者（`speaker_id`）の特定
- 人物名簿の `id`（英数字）または JSON の `null` のみ使用する。名簿の `aliases`（別名・肩書）と `gender` を必ず照合する。
- **前後両方の伝達節を確認（後置伝達節に注意）**:
  カギ括弧の直後に続く「〜と〇〇は言った」「〇〇は答えました」「〇〇は話しかけた」等は、**直前の `[Uk]` の話者**を示す（次の `[U(k+1)]` の導入ではない）。
  - 例: `[U0]「雨だね。」太郎が言った。[U1]「傘を持とう。」花子が答えた。` → `[U0]=taro`, `[U1]=hanako`。
- **対話の原則と例外**:
  - **原則（交互交替）**: 2 人の会話で改行後に次のセリフが続くとき、相手への応答・質問・相槌であれば話者は**交互に交代する**（口調・性別・敬語の変化も確認する）。
  - **例外（同一話者継続）**: (a)「〜と老人は微笑み、[U1]「…」」のように同じ話者の動作でセリフが分割された場合、(b) 聞き手が無言動作（首を振る、うなずく等）しかせず同じ話者が続けて話す場合のみ同一話者が継続する。
- **口調・方言・呼びかけ語の照合**:
  - 発話内の呼びかけ語（「〇〇さん」「先生」等）は聞き手（`addressee_id`）を指し、話者ではない。
  - 地の文に別人物が紹介・描写されただけで発話動詞が無い場合、直後の発話がその人物のものとは限らない。直前・直後の口調や方言の一貫性で判定する。
  - 一人称語り（`narration == "first"`）では、語り手（「私」「僕」「自分」「おれ」等）が実際に発したセリフ・心内語のみ `narrator` に帰属する（他人物のセリフを `narrator` にしない）。
- **判定不能は無理に帰属しない**:
  複数人の同時発話・斉唱（例:「二人は同時に叫んだ」）、または 2 人以上の候補から手がかり無く絞れない発話断片は、無理に当てはめず `speaker_id: null`, `evidence: "ambiguous"` とする。

### 手順3: `evidence` の厳格判定（基本は `"inferred"`）
- **`"explicit"` は厳格条件を満たす場合のみ使用**:
  その発話に**直接隣接する直前または直後の地の文**（他の発話を挟まない）に、**話者を示す語**（名前・別名・一人称の「私／僕／自分」等）と**発話・応答の動詞**（言う・答える・尋ねる・叫ぶ・囁く・話す・呼ぶ等）が**明記されている場合のみ** `"explicit"` とする。
- **`"inferred"`（標準・文脈推定）**:
  上記 `"explicit"` の条件を満たさないが、交互交替・応答関係・口調・方言・性別・呼称・動作描写から話者を 1 人に推定できる場合は、**話者に自信があっても必ず `"inferred"` にする**。
  - 地の文が無い連続セリフ、口調や方言からの推定、無言動作を挟んだ同一話者継続、人物名はあるが発話動詞が無い地の文の直後はすべて `"inferred"`。
- **`"ambiguous"`**:
  手順1の非発話、同時発話、または判定不能の場合のみ（`speaker_id` は必ず `null`）。合理的に推定できる発話を `"ambiguous"` に逃がさないこと。

### 手順4: `addressee_id` と `register`
- `addressee_id`: 呼びかけ・対話相手が分かる場合はその `id`、独り言・心内語・非発話・複数向け・不明は `null`。
- `register.politeness`: `"plain"`（常体・普通口語）/ `"polite"`（です・ます）/ `"honorific"`（ございます・おっしゃる等）/ `"rough"`（貴様・命令・粗暴）/ `"unknown"`（1語断片・非発話）。
- `register.first_person`: セリフ内に明示された一人称代名詞（`"ぼく"`, `"私"`, `"おれ"`, `"うち"` 等）のみ記入。無ければ `null`。"""

SYSTEM_PROMPT_V4 = SYSTEM_PROMPT_V3

PROMPT_REGISTRY: dict[str, tuple[str, str]] = {
    "v1": (PROMPT_VERSION_V1, SYSTEM_PROMPT_V1),
    "v2": (PROMPT_VERSION_V2, SYSTEM_PROMPT_V2),
    "v3": (PROMPT_VERSION_V3, SYSTEM_PROMPT_V3),
    "v4": (PROMPT_VERSION_V4, SYSTEM_PROMPT_V4),
}
DEFAULT_PROMPT_KEY = "v2"
PROMPT_VERSION = PROMPT_VERSION_V2
SYSTEM_PROMPT = SYSTEM_PROMPT_V2


def get_prompt(prompt_version: str = DEFAULT_PROMPT_KEY) -> tuple[str, str]:
    """Return `(version_label, system_prompt_text)` for short key or full label."""
    if prompt_version in PROMPT_REGISTRY:
        return PROMPT_REGISTRY[prompt_version]
    for label, text in PROMPT_REGISTRY.values():
        if prompt_version == label:
            return label, text
    supported = sorted(list(PROMPT_REGISTRY.keys()) + [v[0] for v in PROMPT_REGISTRY.values()])
    raise ValueError(f"Unsupported prompt_version {prompt_version!r}; expected one of {supported}")


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


def compute_prompt_hash(
    system_prompt: str | None = None,
    prompt_version: str | None = None,
) -> str:
    """Return deterministic SHA-256 hex digest identifying prompt version and text."""
    if system_prompt is None:
        ver_label, text = get_prompt(prompt_version or DEFAULT_PROMPT_KEY)
    else:
        if prompt_version is None:
            ver_label = PROMPT_VERSION
        elif prompt_version in PROMPT_REGISTRY:
            ver_label = PROMPT_REGISTRY[prompt_version][0]
        else:
            ver_label = prompt_version
        text = system_prompt
    payload = f"{ver_label}\n{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_work_messages(
    work: F9Work,
    prompt_version: str = DEFAULT_PROMPT_KEY,
) -> list[dict[str, str]]:
    """Build Ollama chat messages for one F9 work excerpt."""
    ver_label, sys_prompt = get_prompt(prompt_version)
    roster_lines: list[str] = []
    for char in work.characters:
        aliases_str = ", ".join(char.aliases) if char.aliases else "(別名なし)"
        roster_lines.append(
            f"- id: `{char.id}` | aliases: [{aliases_str}] | gender: {char.gender}"
        )
    roster_block = "\n".join(roster_lines)

    n_utts = len(work.utterances)
    features_section = ""
    if ver_label == PROMPT_VERSION_V4:
        feats = extract_work_utterance_features(work)
        features_section = f"\n\n{format_utterance_features_block(feats)}"

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
        f"{work.numbered_text}"
        f"{features_section}\n\n"
        f"上記 `[U0]` から `[U{n_utts - 1}]` までの全 {n_utts} 件について JSON を出力してください。"
    )
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
