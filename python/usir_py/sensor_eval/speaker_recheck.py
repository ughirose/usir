"""Stage 3 speaker-only recheck for inferred utterances (zero gold-label access)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Sequence

from usir_py.sensor_eval.f9_loader import F9Character, F9Dataset, F9Work
from usir_py.sensor_eval.features import (
    UtteranceFeatures,
    extract_work_utterance_features,
    find_speakers_in_text,
)
from usir_py.sensor_eval.metrics import (
    evaluate_model_predictions,
    evaluate_step1_gate,
    validate_speaker_id,
)
from usir_py.sensor_eval.ollama_client import call_ollama_chat, parse_model_spec
from usir_py.sensor_eval.postprocess import _copy_utterance_item

RECHECK_PROMPT_VERSION_V1 = "step1_speaker_recheck_v1"
RECHECK_PROMPT_VERSION_V2 = "step1_speaker_recheck_v2"
RECHECK_WORK_PROMPT_VERSION_V1 = "step1_speaker_recheck_work_v1"

RECHECK_SYSTEM_PROMPT_V1 = """あなたは日本文学テクストの話者検証センサー（Stage 3: 話者だけの再確認）です。
第1段・第2段で暫定的に `"inferred"`（文脈推定）と判定された1件の発話 `[Uk]` について、**話者（`speaker_id`）が誰かだけ**を会話の流れと周辺本文から検証してください。

出力は必ず次の JSON オブジェクトのみとし、説明文やコードブロック記法は出さないでください：
`{"reason": "80文字以内の簡潔な根拠", "speaker_id": "人物名簿のid"}`

## 判定規則
1. **保守的検証（原則維持）**:
   - 暫定話者に文脈・口調・地の文上の明確な矛盾がない限り、無理に変更せず暫定話者を維持する。
   - ただし、以下の明確な矛盾がある場合は、最も整合する人物名簿の `id` へ修正する。
2. **口調・方言・敬語・役割の一貫性**:
   - 発話内の敬語・謙譲語・方言・命令口調・専門知識などの話しぶりが、話者の立場と一致しているか確認する。
   - 相手への敬意や称賛・懇願を含む敬語発話を、その敬意を受け取る相手側の話者に帰属してはならない。逆に、指導・説明・命令する発話を聞き手側へ帰属してはならない。
3. **直前・直後の地の文との整合**:
   - カギ括弧直後の地の文が、直前の発話内容に対する動作や身体的状態（傷・表情・行動など）を示している場合、誰が話したかを確認する。
   - 冒頭発話（`[U0]` 等）は直前文脈が無いため、直後の応答・行動描写・呼びかけから逆算して話者を判定する。
   - 発話直前に別の人物が紹介・描写されていても発話動詞が無い場合、直後の発話が必ずしもその新出人物のものとは限らない。
4. **会話の流れ（交互交替・同一話者継続・三者会話）**:
   - **2人の対話の交互交替**: 質問に対する短い応答（「そう」「うん」「はい」等）や、一方が発見・疑問を示しもう一方が説明する応答は話者が交代していることが多い。
   - **同一話者の継続**: 発話末尾が「……」や「、」で途中切れしている場合や、直後に「〜と言った」が入っても次の発話が同じ話題・同じ敬語/方言・同じ語り口の続きである場合は、機械的に交代させず同一話者継続を検討する。
   - **3人以上の場**: 「呼びかけ・短い合いの手・訂正/説明」が連続する場合、誰が誰の言葉を引き継いでいるかを前後の口調と反応で追い、1つ前の取り違えに引きずられない。
5. **呼びかけ・聞き手との逆転・不在人物・一人称語り手の注意**:
   - 発話内の呼びかけ語（「〇〇さん」「先生」等）は聞き手（`addressee_id`）であり話者ではない。第1段の `speaker_id` と `addressee_id` が逆転していないか必ず確認する。
   - 周辺本文にも前後の会話の流れにも登場していない人物を話者に選ばない。
   - 一人称語り（`narration == "first"`）であっても、他の登場人物のセリフを安易に語り手（`narrator`）に寄せない。
6. **出力制約**:
   - `speaker_id` は必ず与えられた人物名簿の `id` から1つだけ選ぶこと（`null` や名簿外の文字列は禁止）。"""

RECHECK_SYSTEM_PROMPT_V2 = """あなたは日本文学テクストの話者矛盾検証センサー（Stage 3: 話者だけの再確認）です。
第1段・第2段で暫定的に `"inferred"`（文脈推定）と判定された1件の発話 `[Uk]` について、**暫定話者が本文・会話の流れと明確に矛盾していないか**だけを検証してください。

出力は必ず次の JSON オブジェクトのみとし、説明文やコードブロック記法は出さないでください：
`{"provisional_speaker_check": "80文字以内で暫定話者の妥当性または明確な矛盾を簡潔に検証", "contradiction_found": false, "speaker_id": "人物名簿のid"}`

## 判定手順（厳守）
### 第1手順（暫定話者の優先確認・保守原則）
- まず「直前の本文」「対象発話」「直後の本文」「会話の流れ」に照らし、第1段・第2段の暫定話者が話したとして成立するか検証する。
- 直前・直後の地の文や発話内容と矛盾無く成立する場合、または少しでも迷う場合は、**必ず `contradiction_found: false` とし、`speaker_id` に暫定話者をそのまま出力する**。

### 第2手順（明確な反証があるときのみ修正）
以下のいずれかの明確な反証がある場合に限り `contradiction_found: true` とし、最も整合する人物名簿の `id` を `speaker_id` に出力する：
1. **不在人物**: 暫定話者が周辺本文にも会話の流れにも登場していない。
2. **直前・直後の地の文との明確な矛盾**:
   - 直後の地の文に発話と直結する動作・身体的描写の主体が明示されているのに別人物になっている。
   - 冒頭発話（直前テキストなし）で安易に一人称語り手へ帰属され、直後の応答や地の文と食い違う。
   - 直前の地の文が発話動詞の無い単なる人物紹介なのに、前話者の語りを分断してその新出人物へ誤帰属している。
3. **口調・敬語・方言・役割の明確な反転**:
   - 相手への敬語・謙譲語・嘆願を、敬われている相手自身に帰属している。
   - 専門解説や作業指示・命令を聞き手側へ帰属している。
   - 標準語・丁寧語の人物と方言・訛りの人物が取り違えられている。
4. **不自然な会話構造の破綻**:
   - 2人の短い問答（質問→「そう」「うん」「はい」等の短い返答→反応）で、同一話者が不自然に3連続している。
   - 直前発話の途中切れ（「……」「、」）からの同一話者継続を機械的に交代させている。
5. **呼びかけ語・`addressee_id` との反転**:
   - 発話内の呼びかけ語が話者本人になっている、または暫定話者と `addressee_id` が逆転している。

## 厳禁事項
- `provisional_speaker_check` に書いた人物と矛盾する `speaker_id` を出力しないこと。
- `speaker_id` は必ず与えられた人物名簿の `id` のみを選ぶこと（`null`・名簿外の文字列は禁止）。"""

RECHECK_WORK_SYSTEM_PROMPT_V1 = """あなたは日本文学テクストの場面単位・話者監査センサー（Stage 3: 話者だけの再確認）です。
入力には「人物名簿」「番号付き本文」および第1段・第2段の「会話フロー表」が与えられます。
表の中で `状態 == 対象`（`inferred` かつ話者が非 `null`）となっている発話についてのみ、本文全体と前後の会話の流れに照らし、**話者（`speaker_id`）が明確に誤っているものだけ**を列挙してください。

出力は必ず次の JSON オブジェクトのみとし、説明文やコードブロック記法は出さないでください：
`{"corrections": [{"utterance_index": 0, "reason": "80文字以内の明確な矛盾と根拠", "speaker_id": "人物名簿のid"}]}`
明確な誤りが無い場合は `{"corrections": []}` を出力してください。

## 監査規則（厳守）
1. **保守原則（過剰修正の厳禁）**:
   - 明確な根拠・矛盾が無い発話は `corrections` に含めない（迷う場合は修正しない）。
   - `状態 == 確定`（`explicit`）および `状態 == 棄権`（`null`）の発話は絶対に含めない。
2. **不在人物チェック**:
   - その発話の前後の場面に登場していない人物（別名を含め文脈に現れない人物）が `仮話者` になっていないか確認する。
3. **口調・方言・敬語・役割の矛盾チェック**:
   - 強い敬語・謙譲語・感謝・嘆願で相手を立てている発話が、敬われている相手自身に帰属されていないか確認する。
   - 標準語・丁寧語の人物と方言・訛りの強い人物が取り違えられていないか確認する。
   - 専門的な解説や指示・命令を発する話者を、質問側・参観者側へ誤帰属していないか確認する。
4. **直前・直後の地の文との対応チェック**:
   - 冒頭発話（`[U0]` 等）が安易に一人称語り手へ帰属され、直後の地の文や次発話と矛盾していないか確認する。
   - 発話直後の地の文にある身体的描写・動作の主体と、`仮話者` が食い違っていないか確認する。
   - 発話直前の地の文が「発話動詞の無い単なる人物紹介」なのに、その新出人物へ誤帰属して前話者の語りを分断していないか確認する（表の `前後動詞` が `-/-` の箇所は特に注意）。
5. **会話の流れ・交互交替・同一話者継続・連鎖誤りの整合チェック**:
   - 2人の対話における質問→短い返答（「そう」「うん」「はい」等）→反応の並びで、同一話者が不自然に連続していないか確認する。
   - 直前発話で疑問や未知の発見を述べた直後の説明・回答が同一話者のままになっていないか確認する。
   - 逆に、直前発話が「……」「、」等で途切れた同一話者の語りの続きなのに、直前の伝達節に引かれて機械的に話者を交代させていないか確認する。
   - ある発話の取り違えが直後の発話に連鎖していないか、前後をまとめて整合確認する。
6. **呼びかけ語・`addressee_id` 反転チェック**:
   - 発話内の呼びかけ語（表の `発話内呼称`）は原則として聞き手であり、話者本人ではない。
   - 表の `仮話者` と `聞き手`（`addressee_id`）が逆転していないか点検する。
7. **出力制約**:
   - `speaker_id` は必ず人物名簿の `id` のみを使うこと（`null`・名簿外は禁止）。各項目では `reason` を簡潔に先に述べてから `speaker_id` を確定すること。"""

RECHECK_PROMPT_REGISTRY: dict[str, tuple[str, str]] = {
    "recheck_v1": (RECHECK_PROMPT_VERSION_V1, RECHECK_SYSTEM_PROMPT_V1),
    "recheck_v2": (RECHECK_PROMPT_VERSION_V2, RECHECK_SYSTEM_PROMPT_V2),
    "recheck_work_v1": (RECHECK_WORK_PROMPT_VERSION_V1, RECHECK_WORK_SYSTEM_PROMPT_V1),
}
DEFAULT_RECHECK_PROMPT_KEY = "recheck_v1"
ALLOWED_RECHECK_MODES = ("per_utterance", "work_audit")

_ALLOWED_CONFIG_KEYS = frozenset(
    {
        "enabled",
        "mode",
        "model",
        "prompt_version",
        "flow_window_before",
        "flow_window_after",
        "context_chars_before",
        "context_chars_after",
        "sequential",
        "num_ctx",
        "num_predict",
        "seed",
        "timeout",
    }
)


def get_recheck_prompt(prompt_version: str = DEFAULT_RECHECK_PROMPT_KEY) -> tuple[str, str]:
    """Return `(version_label, system_prompt_text)` for short key or full label."""
    if prompt_version in RECHECK_PROMPT_REGISTRY:
        return RECHECK_PROMPT_REGISTRY[prompt_version]
    for label, text in RECHECK_PROMPT_REGISTRY.values():
        if prompt_version == label:
            return label, text
    supported = sorted(
        list(RECHECK_PROMPT_REGISTRY.keys()) + [v[0] for v in RECHECK_PROMPT_REGISTRY.values()]
    )
    raise ValueError(
        f"Unsupported recheck prompt_version {prompt_version!r}; expected one of {supported}"
    )


def compute_recheck_prompt_hash(
    system_prompt: str | None = None,
    prompt_version: str | None = None,
) -> str:
    """Return deterministic SHA-256 hex digest identifying Stage 3 prompt version and text."""
    if system_prompt is None:
        ver_label, text = get_recheck_prompt(prompt_version or DEFAULT_RECHECK_PROMPT_KEY)
    else:
        if prompt_version is None:
            ver_label = RECHECK_PROMPT_VERSION_V1
        elif prompt_version in RECHECK_PROMPT_REGISTRY:
            ver_label = RECHECK_PROMPT_REGISTRY[prompt_version][0]
        else:
            ver_label = prompt_version
        text = system_prompt
    payload = f"{ver_label}\n{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_non_negative_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")


def _validate_positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


@dataclass(frozen=True)
class SpeakerRecheckConfig:
    """Configuration controlling Stage 3 speaker-only recheck."""

    enabled: bool = False
    mode: str = "per_utterance"
    model: str = "qwen2.5:14b"
    prompt_version: str = DEFAULT_RECHECK_PROMPT_KEY
    flow_window_before: int = 12
    flow_window_after: int = 6
    context_chars_before: int = 1200
    context_chars_after: int = 900
    sequential: bool = True
    num_ctx: int = 6144
    num_predict: int = 384
    seed: int = 42
    timeout: float = 120.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError(f"enabled must be a bool, got {type(self.enabled).__name__}")
        if not isinstance(self.mode, str) or self.mode not in ALLOWED_RECHECK_MODES:
            raise ValueError(
                f"mode must be one of {ALLOWED_RECHECK_MODES}, got {self.mode!r}"
            )
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError(f"model must be a non-empty string, got {self.model!r}")
        model_name, _, _ = parse_model_spec(self.model)
        if not model_name.strip():
            raise ValueError(f"model must specify a non-empty model name, got {self.model!r}")
        if not isinstance(self.prompt_version, str):
            raise ValueError(
                f"prompt_version must be a string, got {type(self.prompt_version).__name__}"
            )
        get_recheck_prompt(self.prompt_version)

        _validate_non_negative_int(self.flow_window_before, "flow_window_before")
        _validate_non_negative_int(self.flow_window_after, "flow_window_after")
        _validate_non_negative_int(self.context_chars_before, "context_chars_before")
        _validate_non_negative_int(self.context_chars_after, "context_chars_after")
        if not isinstance(self.sequential, bool):
            raise ValueError(f"sequential must be a bool, got {type(self.sequential).__name__}")
        _validate_positive_int(self.num_ctx, "num_ctx")
        _validate_positive_int(self.num_predict, "num_predict")
        _validate_non_negative_int(self.seed, "seed")
        if isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float)) or self.timeout <= 0:
            raise ValueError(f"timeout must be a positive number, got {self.timeout!r}")

    @property
    def is_enabled(self) -> bool:
        """Return True when Stage 3 speaker recheck is enabled."""
        return self.enabled

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> SpeakerRecheckConfig:
        """Parse and validate a config mapping; None yields the disabled default."""
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError(f"speaker_recheck config must be a dict, got {type(raw).__name__}")
        unknown = set(raw.keys()) - _ALLOWED_CONFIG_KEYS
        if unknown:
            raise ValueError(f"Unknown speaker_recheck config keys: {sorted(unknown)}")
        return cls(
            enabled=raw.get("enabled", False),
            mode=raw.get("mode", "per_utterance"),
            model=raw.get("model", "qwen2.5:14b"),
            prompt_version=raw.get("prompt_version", DEFAULT_RECHECK_PROMPT_KEY),
            flow_window_before=raw.get("flow_window_before", 12),
            flow_window_after=raw.get("flow_window_after", 6),
            context_chars_before=raw.get("context_chars_before", 1200),
            context_chars_after=raw.get("context_chars_after", 900),
            sequential=raw.get("sequential", True),
            num_ctx=raw.get("num_ctx", 6144),
            num_predict=raw.get("num_predict", 384),
            seed=raw.get("seed", 42),
            timeout=raw.get("timeout", 120.0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a plain JSON-compatible dictionary."""
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "flow_window_before": self.flow_window_before,
            "flow_window_after": self.flow_window_after,
            "context_chars_before": self.context_chars_before,
            "context_chars_after": self.context_chars_after,
            "sequential": self.sequential,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "seed": self.seed,
            "timeout": float(self.timeout),
        }


def is_recheck_target(item: Any) -> bool:
    """Return True iff `item` is an inferred utterance with a non-null speaker_id."""
    if not isinstance(item, dict):
        return False
    idx = item.get("utterance_index")
    if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0:
        return False
    return item.get("evidence") == "inferred" and item.get("speaker_id") is not None


def extract_in_quote_aliases(
    quote_text: str,
    characters: Sequence[F9Character],
) -> list[tuple[str, str]]:
    """Find character aliases mentioned inside `quote_text` (earliest position, deduplicated)."""
    inner = quote_text
    if len(inner) >= 2 and inner[0] in "「『" and inner[-1] in "」』":
        inner = inner[1:-1]
    if not inner:
        return []

    hits: list[tuple[int, int, str, str]] = []
    for char_order, char in enumerate(characters):
        best_pos: int | None = None
        best_alias: str | None = None
        for alias in sorted(char.aliases, key=len, reverse=True):
            if not alias:
                continue
            pos = inner.find(alias)
            if pos != -1 and (best_pos is None or pos < best_pos):
                best_pos = pos
                best_alias = alias
        if best_pos is not None and best_alias is not None:
            hits.append((best_pos, char_order, char.id, best_alias))

    hits.sort(key=lambda item: (item[0], item[1]))
    seen_ids: set[str] = set()
    result: list[tuple[str, str]] = []
    for _, _, cid, alias in hits:
        if cid not in seen_ids:
            seen_ids.add(cid)
            result.append((cid, alias))
    return result


def compute_numbered_quote_spans(work: F9Work) -> list[tuple[int, int]]:
    """Locate `[Ui]` + quote_text spans sequentially inside `work.numbered_text`."""
    spans: list[tuple[int, int]] = []
    text = work.numbered_text
    text_len = len(text)
    cursor = 0
    for idx, u in enumerate(work.utterances):
        marker = f"[U{idx}]"
        pos = text.find(marker, cursor)
        if pos == -1:
            pos = text.find(marker)
        if pos == -1:
            start = max(0, min(u.open_offset, text_len))
            end = max(start, min(u.close_offset, text_len))
            spans.append((start, end))
            continue
        end = min(text_len, pos + len(marker) + len(u.quote_text))
        spans.append((pos, end))
        cursor = end
    return spans


def extract_local_numbered_context(
    work: F9Work,
    numbered_spans: Sequence[tuple[int, int]],
    target_idx: int,
    chars_before: int,
    chars_after: int,
) -> str:
    """Slice `work.numbered_text` around `[U{target_idx}]` preserving the full target quote."""
    if not work.numbered_text or target_idx < 0 or target_idx >= len(numbered_spans):
        return work.numbered_text
    start, end = numbered_spans[target_idx]
    lo = max(0, start - chars_before)
    hi = min(len(work.numbered_text), end + chars_after)
    snippet = work.numbered_text[lo:hi]
    prefix = "……\n" if lo > 0 else ""
    suffix = "\n……" if hi < len(work.numbered_text) else ""
    return f"{prefix}{snippet}{suffix}"


def _normalize_allowed_roster(roster_ids: Sequence[str]) -> list[str]:
    allowed = [rid for rid in dict.fromkeys(roster_ids) if isinstance(rid, str) and rid]
    if not allowed:
        raise ValueError("roster_ids must contain at least one non-empty character id")
    return allowed


def build_ollama_recheck_format_schema(roster_ids: Sequence[str]) -> dict[str, Any]:
    """Build strict Ollama JSON Schema permitting only roster speaker_id values."""
    allowed = _normalize_allowed_roster(roster_ids)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["reason", "speaker_id"],
        "properties": {
            "reason": {"type": "string"},
            "speaker_id": {"type": "string", "enum": allowed},
        },
    }


def build_ollama_recheck_v2_format_schema(roster_ids: Sequence[str]) -> dict[str, Any]:
    """Build strict Ollama JSON Schema for Stage 3 v2 binary contradiction verification."""
    allowed = _normalize_allowed_roster(roster_ids)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["provisional_speaker_check", "contradiction_found", "speaker_id"],
        "properties": {
            "provisional_speaker_check": {"type": "string"},
            "contradiction_found": {"type": "boolean"},
            "speaker_id": {"type": "string", "enum": allowed},
        },
    }


def build_ollama_work_audit_format_schema(roster_ids: Sequence[str]) -> dict[str, Any]:
    """Build strict Ollama JSON Schema for whole-work Stage 3 audit corrections."""
    allowed = _normalize_allowed_roster(roster_ids)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["corrections"],
        "properties": {
            "corrections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["utterance_index", "reason", "speaker_id"],
                    "properties": {
                        "utterance_index": {"type": "integer", "minimum": 0},
                        "reason": {"type": "string"},
                        "speaker_id": {"type": "string", "enum": allowed},
                    },
                },
            }
        },
    }


def _truncate_quote_snippet(quote_text: str, max_chars: int = 36) -> str:
    flat = quote_text.replace("\r\n", " ").replace("\n", " ").replace("|", "｜").strip()
    if len(flat) <= max_chars:
        return flat
    return flat[: max_chars - 1] + "…"


def _build_shared_work_prefix(work: F9Work) -> tuple[str, list[str]]:
    n_utts = len(work.utterances)
    roster_lines: list[str] = []
    allowed_ids: list[str] = []
    for char in work.characters:
        allowed_ids.append(char.id)
        aliases_str = ", ".join(char.aliases) if char.aliases else "(別名なし)"
        roster_lines.append(
            f"- id: `{char.id}` | aliases: [{aliases_str}] | gender: {char.gender}"
        )
    roster_block = "\n".join(roster_lines)
    shared_prefix = (
        f"## 場面メタ情報\n"
        f"- 作品ID: {work.work_id}\n"
        f"- 語り形式 (`narration`): {work.narration}\n"
        f"- 一人称語り手 (`narrator`): {work.narrator or 'null'}\n"
        f"- 視点人物 (`pov`): {work.pov or 'null'}\n"
        f"- 発話総数 (`[U0]`〜`[U{n_utts - 1}]`): {n_utts} 件\n\n"
        f"## 人物名簿（出力の speaker_id は必ずこの `id` のみ使用）\n"
        f"{roster_block}"
    )
    return shared_prefix, allowed_ids


def _format_flow_status(
    idx: int,
    target_idx: int,
    pred: dict[str, Any] | None,
    rechecked_indices: set[int] | frozenset[int],
) -> str:
    if idx == target_idx:
        return "★再確認対象"
    if pred is None:
        return "欠落"
    ev = pred.get("evidence")
    spk = pred.get("speaker_id")
    if ev == "explicit":
        return "確定(explicit)"
    if spk is None or ev == "ambiguous":
        return "棄権(null)"
    if idx in rechecked_indices:
        return "再確認済(inferred)"
    return "仮推定(inferred)"


def _nearest_non_null_neighbor(
    current_by_idx: dict[int, dict[str, Any]],
    start_idx: int,
    step: int,
    total_utts: int,
) -> tuple[int, dict[str, Any]] | None:
    idx = start_idx + step
    while 0 <= idx < total_utts:
        pred = current_by_idx.get(idx)
        if pred is not None and pred.get("speaker_id") is not None and pred.get("evidence") != "ambiguous":
            return idx, pred
        idx += step
    return None


def build_speaker_recheck_messages(
    work: F9Work,
    target_idx: int,
    current_by_idx: dict[int, dict[str, Any]],
    rechecked_indices: set[int] | frozenset[int],
    *,
    config: SpeakerRecheckConfig | None = None,
    numbered_spans: Sequence[tuple[int, int]] | None = None,
    features: Sequence[UtteranceFeatures] | None = None,
    prompt_version: str | None = None,
) -> list[dict[str, str]]:
    """Build Stage 3 speaker-only recheck messages for `[U{target_idx}]` (zero gold access)."""
    cfg = config or SpeakerRecheckConfig(enabled=True)
    eff_pv = prompt_version or cfg.prompt_version
    ver_label, sys_prompt = get_recheck_prompt(eff_pv)

    n_utts = len(work.utterances)
    if target_idx < 0 or target_idx >= n_utts:
        raise IndexError(f"target_idx={target_idx} out of range for {work.work_id} ({n_utts} utts)")

    # Shared prefix for Ollama KV-cache reuse across utterances in the same work.
    shared_prefix, allowed_ids = _build_shared_work_prefix(work)

    flow_lo = max(0, target_idx - cfg.flow_window_before)
    flow_hi = min(n_utts - 1, target_idx + cfg.flow_window_after)
    flow_lines = [
        "| id | 状態 | 話者(speaker_id) | 聞き手(addressee_id) | 発話冒頭 |",
        "|---|---|---|---|---|",
    ]
    for i in range(flow_lo, flow_hi + 1):
        pred = current_by_idx.get(i)
        status_label = _format_flow_status(i, target_idx, pred, rechecked_indices)
        spk = (pred.get("speaker_id") or "null") if pred is not None else "null"
        addr = (pred.get("addressee_id") or "null") if pred is not None else "null"
        snippet = _truncate_quote_snippet(work.utterances[i].quote_text)
        flow_lines.append(f"| U{i} | {status_label} | {spk} | {addr} | {snippet} |")
    flow_block = "\n".join(flow_lines)

    spans = compute_numbered_quote_spans(work) if numbered_spans is None else numbered_spans
    local_context = extract_local_numbered_context(
        work,
        spans,
        target_idx,
        cfg.context_chars_before,
        cfg.context_chars_after,
    )

    work_features = extract_work_utterance_features(work) if features is None else features
    feat = work_features[target_idx] if target_idx < len(work_features) else None

    target_u = work.utterances[target_idx]
    target_pred = current_by_idx.get(target_idx) or {}
    prov_spk = target_pred.get("speaker_id") or "null"
    prov_addr = target_pred.get("addressee_id") or "null"

    prev_nn = _nearest_non_null_neighbor(current_by_idx, target_idx, -1, n_utts)
    next_nn = _nearest_non_null_neighbor(current_by_idx, target_idx, 1, n_utts)
    prev_nn_str = (
        f"U{prev_nn[0]} (speaker={prev_nn[1].get('speaker_id')}, "
        f"evidence={prev_nn[1].get('evidence')}, "
        f"addressee={prev_nn[1].get('addressee_id') or 'null'})"
        if prev_nn is not None
        else "なし"
    )
    next_nn_str = (
        f"U{next_nn[0]} (speaker={next_nn[1].get('speaker_id')}, "
        f"evidence={next_nn[1].get('evidence')}, "
        f"addressee={next_nn[1].get('addressee_id') or 'null'})"
        if next_nn is not None
        else "なし"
    )

    in_quote_hits = extract_in_quote_aliases(target_u.quote_text, work.characters)
    in_quote_str = (
        ", ".join(f"「{alias}」→ {cid}" for cid, alias in in_quote_hits)
        if in_quote_hits
        else "なし"
    )

    local_speakers = find_speakers_in_text(
        local_context,
        work.characters,
        work.narration,
        work.narrator,
    )
    local_speakers_str = ", ".join(local_speakers) if local_speakers else "なし"

    pos_str = feat.position if feat is not None else "unknown"
    pre_verb_str = (feat.pre_speech_verb or "なし") if feat is not None else "なし"
    post_verb_str = (feat.post_speech_verb or "なし") if feat is not None else "なし"
    allowed_str = ", ".join(allowed_ids)

    cues_block = (
        f"## 再確認対象 [U{target_idx}] の詳細手がかり\n"
        f"- 対象発話全文: {target_u.quote_text}\n"
        f"- 第1段/第2段の暫定話者 (`speaker_id`): `{prov_spk}`\n"
        f"- 第1段の聞き手推定 (`addressee_id`): `{prov_addr}`\n"
        f"- 直前の非null発話: {prev_nn_str}\n"
        f"- 直後の非null発話: {next_nn_str}\n"
        f"- 機械抽出特徴: pos=`{pos_str}`, pre_speech_verb=`{pre_verb_str}`, post_speech_verb=`{post_verb_str}`\n"
        f"- 発話内の名簿語句ヒット（呼びかけ・言及候補）: {in_quote_str}\n"
        f"- 周辺本文に登場する人物候補: {local_speakers_str}\n"
        f"- 選択可能な人物ID: [{allowed_str}]"
    )

    if ver_label == RECHECK_PROMPT_VERSION_V2:
        start_span, end_span = (
            spans[target_idx]
            if 0 <= target_idx < len(spans)
            else (0, len(work.numbered_text))
        )
        imm_pre = work.numbered_text[max(0, start_span - 180) : start_span].strip() or "（冒頭・直前テキストなし）"
        imm_post = (
            work.numbered_text[end_span : min(len(work.numbered_text), end_span + 180)].strip()
            or "（末尾・直後テキストなし）"
        )
        imm_block = (
            f"## 直前・対象・直後の本文（最優先確認）\n"
            f"- 直前（最大180文字）:\n{imm_pre}\n\n"
            f"- 対象発話 `[U{target_idx}]`:\n{target_u.quote_text}\n\n"
            f"- 直後（最大180文字）:\n{imm_post}"
        )
        user_prompt = (
            f"{shared_prefix}\n\n"
            f"{imm_block}\n\n"
            f"## 会話の流れ（前後発話の話者・聞き手）\n"
            f"{flow_block}\n\n"
            f"## 周辺本文（広域抜粋）\n"
            f"{local_context}\n\n"
            f"{cues_block}\n\n"
            f"暫定話者 `{prov_spk}` が `[U{target_idx}]` の話者として成立するか検証し、"
            f"JSON（`provisional_speaker_check`, `contradiction_found`, `speaker_id`）のみを出力してください。"
        )
    else:
        user_prompt = (
            f"{shared_prefix}\n\n"
            f"## 会話の流れ（前後発話の話者・聞き手）\n"
            f"{flow_block}\n\n"
            f"## 周辺本文（対象発話 [U{target_idx}] の前後抜粋）\n"
            f"{local_context}\n\n"
            f"{cues_block}\n\n"
            f"`[U{target_idx}]` の話者は人物名簿の誰ですか？ JSON（`reason`, `speaker_id`）のみを出力してください。"
        )
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _format_work_audit_status(pred: dict[str, Any] | None) -> str:
    if pred is None:
        return "欠落"
    if pred.get("evidence") == "explicit":
        return "確定"
    if is_recheck_target(pred):
        return "対象"
    return "棄権"


def build_work_audit_recheck_messages(
    work: F9Work,
    stage2_by_idx: dict[int, dict[str, Any]],
    *,
    features: Sequence[UtteranceFeatures] | None = None,
    prompt_version: str = "recheck_work_v1",
) -> list[dict[str, str]]:
    """Build whole-work Stage 3 speaker audit messages (zero gold-label access)."""
    _, sys_prompt = get_recheck_prompt(prompt_version)
    shared_prefix, allowed_ids = _build_shared_work_prefix(work)
    work_features = extract_work_utterance_features(work) if features is None else features

    flow_lines = [
        "| id | 状態 | 仮話者 | 聞き手 | 発話内呼称 | 前後動詞 | 冒頭 |",
        "|---|---|---|---|---|---|---|",
    ]
    target_labels: list[str] = []
    for idx, u in enumerate(work.utterances):
        pred = stage2_by_idx.get(idx)
        status_label = _format_work_audit_status(pred)
        if status_label == "対象":
            target_labels.append(f"U{idx}")
        spk = (pred.get("speaker_id") or "null") if pred is not None else "null"
        addr = (pred.get("addressee_id") or "null") if pred is not None else "null"
        in_quote_hits = extract_in_quote_aliases(u.quote_text, work.characters)
        in_quote_str = (
            ", ".join(f"{alias}→{cid}" for cid, alias in in_quote_hits)
            if in_quote_hits
            else "-"
        )
        feat = work_features[idx] if idx < len(work_features) else None
        verbs_str = (
            f"{feat.pre_speech_verb or '-'}/{feat.post_speech_verb or '-'}"
            if feat is not None
            else "-/-"
        )
        snippet = _truncate_quote_snippet(u.quote_text, max_chars=24)
        flow_lines.append(
            f"| U{idx} | {status_label} | {spk} | {addr} | {in_quote_str} | {verbs_str} | {snippet} |"
        )

    flow_block = "\n".join(flow_lines)
    targets_str = ", ".join(target_labels) if target_labels else "なし"
    allowed_str = ", ".join(allowed_ids)

    user_prompt = (
        f"{shared_prefix}\n\n"
        f"## 番号付き本文\n"
        f"{work.numbered_text}\n\n"
        f"## 第1段＋第2段の会話フロー表（全発話）\n"
        f"{flow_block}\n\n"
        f"- 監査対象発話（`状態 == 対象`）: [{targets_str}]\n"
        f"- 選択可能な人物ID: [{allowed_str}]\n"
        f"上記のうち話者が明確に誤っているものだけを `corrections` 配列で出力してください（明確な誤りが無ければ `[]`）。"
    )
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _empty_work_audit(work_id: str, config: SpeakerRecheckConfig) -> dict[str, Any]:
    return {
        "mode": config.mode,
        "work_id": work_id,
        "enabled": config.enabled,
        "model": config.model,
        "target_count": 0,
        "rechecked_count": 0,
        "changed_count": 0,
        "rejected_count": 0,
        "elapsed_sec": 0.0,
        "prompt_eval_count": 0,
        "eval_count": 0,
        "any_context_window_warning": False,
        "decisions": [],
    }


def _validate_recheck_candidate(
    payload: dict[str, Any],
    roster_set: set[str],
) -> tuple[str | None, str | None]:
    """Validate `speaker_id` field against null/roster rules; returns `(speaker, rejected_reason)`."""
    if "speaker_id" not in payload:
        return None, "missing_speaker_id"
    candidate = payload.get("speaker_id")
    if candidate is None:
        return None, "null_speaker_id"
    violation = validate_speaker_id(candidate, roster_set)
    if violation is not None:
        return None, violation
    return str(candidate), None


def _apply_work_audit_recheck(
    work: F9Work,
    stage2_by_idx: dict[int, dict[str, Any]],
    working_by_idx: dict[int, dict[str, Any]],
    target_indices: list[int],
    config: SpeakerRecheckConfig,
    *,
    ollama_url: str,
    work_features: Sequence[UtteranceFeatures],
) -> dict[str, Any]:
    roster_ids = [c.id for c in work.characters]
    roster_set = set(roster_ids)
    audit_schema = build_ollama_work_audit_format_schema(roster_ids)
    model_name, think, _ = parse_model_spec(config.model)
    messages = build_work_audit_recheck_messages(
        work,
        stage2_by_idx,
        features=work_features,
        prompt_version=config.prompt_version,
    )
    resp = call_ollama_chat(
        model=model_name,
        messages=messages,
        format_schema=audit_schema,
        ollama_url=ollama_url,
        num_ctx=config.num_ctx,
        num_predict=config.num_predict,
        seed=config.seed,
        temperature=0.0,
        timeout_sec=config.timeout,
        think=think,
    )

    p_tok = int(resp.prompt_eval_count or 0)
    e_tok = int(resp.eval_count or 0)
    ctx_warn = bool((p_tok + e_tok) >= config.num_ctx or e_tok >= config.num_predict)

    decisions: list[dict[str, Any]] = []
    changed_count = 0
    rejected_count = 0
    parse_ok = (
        isinstance(resp.parsed_json, dict)
        and isinstance(resp.parsed_json.get("corrections"), list)
    )

    if not parse_ok:
        rejected_count += 1
        decisions.append(
            {
                "utterance_index": None,
                "before_speaker": None,
                "after_speaker": None,
                "addressee_id": None,
                "changed": False,
                "accepted": False,
                "rejected_reason": (
                    "parse_error" if resp.parsed_json is None else "invalid_corrections_payload"
                ),
                "reason": None,
                "parse_error": resp.parse_error,
            }
        )
    else:
        n_utts = len(work.utterances)
        seen_indices: set[int] = set()
        for entry in resp.parsed_json["corrections"]:
            if not isinstance(entry, dict):
                rejected_count += 1
                decisions.append(
                    {
                        "utterance_index": None,
                        "before_speaker": None,
                        "after_speaker": None,
                        "addressee_id": None,
                        "changed": False,
                        "accepted": False,
                        "rejected_reason": "invalid_entry_type",
                        "reason": None,
                    }
                )
                continue

            raw_idx = entry.get("utterance_index")
            raw_reason = entry.get("reason")
            reason_text = str(raw_reason) if raw_reason is not None else None

            if isinstance(raw_idx, bool) or not isinstance(raw_idx, int):
                rejected_count += 1
                decisions.append(
                    {
                        "utterance_index": raw_idx,
                        "before_speaker": None,
                        "after_speaker": None,
                        "addressee_id": None,
                        "changed": False,
                        "accepted": False,
                        "rejected_reason": "invalid_utterance_index",
                        "reason": reason_text,
                    }
                )
                continue

            if raw_idx < 0 or raw_idx >= n_utts:
                rejected_count += 1
                decisions.append(
                    {
                        "utterance_index": raw_idx,
                        "before_speaker": None,
                        "after_speaker": None,
                        "addressee_id": None,
                        "changed": False,
                        "accepted": False,
                        "rejected_reason": "out_of_range",
                        "reason": reason_text,
                    }
                )
                continue

            orig_item = stage2_by_idx.get(raw_idx)
            before_speaker = orig_item.get("speaker_id") if orig_item is not None else None
            addressee_id = orig_item.get("addressee_id") if orig_item is not None else None

            if raw_idx in seen_indices:
                rejected_count += 1
                decisions.append(
                    {
                        "utterance_index": raw_idx,
                        "before_speaker": before_speaker,
                        "after_speaker": before_speaker,
                        "addressee_id": addressee_id,
                        "changed": False,
                        "accepted": False,
                        "rejected_reason": "duplicate_utterance_index",
                        "reason": reason_text,
                    }
                )
                continue
            seen_indices.add(raw_idx)

            if orig_item is None or not is_recheck_target(orig_item):
                rejected_count += 1
                decisions.append(
                    {
                        "utterance_index": raw_idx,
                        "before_speaker": before_speaker,
                        "after_speaker": before_speaker,
                        "addressee_id": addressee_id,
                        "changed": False,
                        "accepted": False,
                        "rejected_reason": "not_recheck_target",
                        "reason": reason_text,
                    }
                )
                continue

            candidate, rej_reason = _validate_recheck_candidate(entry, roster_set)
            if rej_reason is not None or candidate is None:
                rejected_count += 1
                decisions.append(
                    {
                        "utterance_index": raw_idx,
                        "before_speaker": before_speaker,
                        "after_speaker": before_speaker,
                        "addressee_id": addressee_id,
                        "changed": False,
                        "accepted": False,
                        "rejected_reason": rej_reason,
                        "reason": reason_text,
                    }
                )
                continue

            working_by_idx[raw_idx]["speaker_id"] = candidate
            changed = candidate != before_speaker
            if changed:
                changed_count += 1
            decisions.append(
                {
                    "utterance_index": raw_idx,
                    "before_speaker": before_speaker,
                    "after_speaker": candidate,
                    "addressee_id": addressee_id,
                    "changed": changed,
                    "accepted": True,
                    "rejected_reason": None,
                    "reason": reason_text,
                }
            )

    return {
        "mode": config.mode,
        "work_id": work.work_id,
        "enabled": config.enabled,
        "model": config.model,
        "target_count": len(target_indices),
        "rechecked_count": len(target_indices) if parse_ok else 0,
        "changed_count": changed_count,
        "rejected_count": rejected_count,
        "elapsed_sec": resp.elapsed_sec,
        "prompt_eval_count": p_tok,
        "eval_count": e_tok,
        "any_context_window_warning": ctx_warn,
        "format_mode": resp.format_mode,
        "parse_error": resp.parse_error,
        "decisions": decisions,
    }


def apply_speaker_recheck_to_work(
    work: F9Work,
    utterances: list[dict[str, Any]] | None,
    config: SpeakerRecheckConfig,
    *,
    ollama_url: str = "http://127.0.0.1:11434",
    features: Sequence[UtteranceFeatures] | None = None,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    """Apply Stage 3 speaker-only recheck to `inferred` non-null utterances in `work`."""
    if utterances is None:
        return None, _empty_work_audit(work.work_id, config)

    processed = [_copy_utterance_item(item) for item in utterances]
    if not config.enabled or not work.characters:
        return processed, _empty_work_audit(work.work_id, config)

    n_utts = len(work.utterances)
    stage2_by_idx: dict[int, dict[str, Any]] = {}
    working_by_idx: dict[int, dict[str, Any]] = {}
    for orig_item, work_item in zip(utterances, processed):
        if not isinstance(orig_item, dict) or not isinstance(work_item, dict):
            continue
        idx = orig_item.get("utterance_index")
        if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0 or idx >= n_utts:
            continue
        if idx not in working_by_idx:
            stage2_by_idx[idx] = orig_item
            working_by_idx[idx] = work_item

    target_indices = sorted(
        idx for idx, item in stage2_by_idx.items() if is_recheck_target(item)
    )
    if not target_indices:
        return processed, _empty_work_audit(work.work_id, config)

    work_features = extract_work_utterance_features(work) if features is None else features
    if config.mode == "work_audit":
        audit = _apply_work_audit_recheck(
            work,
            stage2_by_idx,
            working_by_idx,
            target_indices,
            config,
            ollama_url=ollama_url,
            work_features=work_features,
        )
        return processed, audit

    roster_ids = [c.id for c in work.characters]
    roster_set = set(roster_ids)
    is_v2 = get_recheck_prompt(config.prompt_version)[0] == RECHECK_PROMPT_VERSION_V2
    recheck_schema = (
        build_ollama_recheck_v2_format_schema(roster_ids)
        if is_v2
        else build_ollama_recheck_format_schema(roster_ids)
    )
    model_name, think, _ = parse_model_spec(config.model)
    numbered_spans = compute_numbered_quote_spans(work)

    rechecked_indices: set[int] = set()
    decisions: list[dict[str, Any]] = []
    changed_count = 0
    rejected_count = 0
    total_elapsed = 0.0
    total_prompt_tokens = 0
    total_eval_tokens = 0
    any_ctx_warn = False

    for idx in target_indices:
        flow_source = working_by_idx if config.sequential else stage2_by_idx
        shown_rechecked = rechecked_indices if config.sequential else set()
        messages = build_speaker_recheck_messages(
            work,
            idx,
            flow_source,
            shown_rechecked,
            config=config,
            numbered_spans=numbered_spans,
            features=work_features,
            prompt_version=config.prompt_version,
        )
        resp = call_ollama_chat(
            model=model_name,
            messages=messages,
            format_schema=recheck_schema,
            ollama_url=ollama_url,
            num_ctx=config.num_ctx,
            num_predict=config.num_predict,
            seed=config.seed,
            temperature=0.0,
            timeout_sec=config.timeout,
            think=think,
        )

        target_item = working_by_idx[idx]
        before_speaker = target_item.get("speaker_id")
        addressee_id = target_item.get("addressee_id")

        reason_text: str | None = None
        accepted = False
        rejected_reason: str | None = None
        contradiction_val: bool | None = None
        if is_v2:
            if resp.parsed_json is None:
                rejected_reason = "parse_error"
                after_speaker = before_speaker
            else:
                raw_reason = resp.parsed_json.get("provisional_speaker_check")
                reason_text = str(raw_reason) if raw_reason is not None else None
                raw_cf = resp.parsed_json.get("contradiction_found")
                if not isinstance(raw_cf, bool):
                    rejected_reason = "invalid_contradiction_flag"
                    after_speaker = before_speaker
                else:
                    contradiction_val = raw_cf
                    candidate, rej_reason = _validate_recheck_candidate(resp.parsed_json, roster_set)
                    if rej_reason is not None or candidate is None:
                        rejected_reason = rej_reason
                        after_speaker = before_speaker
                    elif not raw_cf:
                        accepted = True
                        after_speaker = before_speaker
                    else:
                        accepted = True
                        after_speaker = candidate
                        target_item["speaker_id"] = candidate
        else:
            if resp.parsed_json is None:
                rejected_reason = "parse_error"
                after_speaker = before_speaker
            else:
                raw_reason = resp.parsed_json.get("reason")
                reason_text = str(raw_reason) if raw_reason is not None else None
                candidate, rej_reason = _validate_recheck_candidate(resp.parsed_json, roster_set)
                if rej_reason is not None or candidate is None:
                    rejected_reason = rej_reason
                    after_speaker = before_speaker
                else:
                    accepted = True
                    after_speaker = candidate
                    target_item["speaker_id"] = candidate

        if not accepted:
            rejected_count += 1
        changed = after_speaker != before_speaker
        if changed:
            changed_count += 1
        rechecked_indices.add(idx)

        p_tok = int(resp.prompt_eval_count or 0)
        e_tok = int(resp.eval_count or 0)
        total_elapsed += resp.elapsed_sec
        total_prompt_tokens += p_tok
        total_eval_tokens += e_tok
        ctx_warn = bool((p_tok + e_tok) >= config.num_ctx or e_tok >= config.num_predict)
        if ctx_warn:
            any_ctx_warn = True

        decision_entry: dict[str, Any] = {
            "utterance_index": idx,
            "before_speaker": before_speaker,
            "after_speaker": after_speaker,
            "addressee_id": addressee_id,
            "changed": changed,
            "accepted": accepted,
            "rejected_reason": rejected_reason,
            "reason": reason_text,
            "format_mode": resp.format_mode,
            "elapsed_sec": resp.elapsed_sec,
            "prompt_eval_count": resp.prompt_eval_count,
            "eval_count": resp.eval_count,
            "context_window_warning": ctx_warn,
            "parse_error": resp.parse_error,
        }
        if is_v2:
            decision_entry["contradiction_found"] = contradiction_val
        decisions.append(decision_entry)

    audit = {
        "mode": config.mode,
        "work_id": work.work_id,
        "enabled": config.enabled,
        "model": config.model,
        "target_count": len(target_indices),
        "rechecked_count": len(decisions),
        "changed_count": changed_count,
        "rejected_count": rejected_count,
        "elapsed_sec": round(total_elapsed, 3),
        "prompt_eval_count": total_prompt_tokens,
        "eval_count": total_eval_tokens,
        "any_context_window_warning": any_ctx_warn,
        "decisions": decisions,
    }
    return processed, audit


def _index_utterances(utts: list[dict[str, Any]] | None) -> dict[int, dict[str, Any]]:
    idx_map: dict[int, dict[str, Any]] = {}
    for item in utts or []:
        if isinstance(item, dict):
            idx = item.get("utterance_index")
            if not isinstance(idx, bool) and isinstance(idx, int) and idx not in idx_map:
                idx_map[idx] = item
    return idx_map


def replay_saved_result_with_speaker_recheck(
    dataset: F9Dataset,
    saved_report: dict[str, Any],
    config: SpeakerRecheckConfig,
    *,
    ollama_url: str = "http://127.0.0.1:11434",
) -> dict[str, Any]:
    """Run Stage 3 speaker recheck offline on saved Stage 1+2 `parsed_utterances`."""
    raw_outputs = saved_report.get("raw_outputs") or {}
    before_preds: dict[str, list[dict[str, Any]] | None] = {
        work.work_id: (raw_outputs.get(work.work_id) or {}).get("parsed_utterances")
        for work in dataset.works
    }
    model_id = str(saved_report.get("model_spec") or "unknown-model")
    prompt_hash = str(saved_report.get("prompt_hash") or "")

    before_metrics = evaluate_model_predictions(
        works=dataset.works,
        predictions_by_work=before_preds,
        model_id=model_id,
        prompt_hash=prompt_hash,
    )
    before_gate = evaluate_step1_gate(before_metrics)

    after_preds: dict[str, list[dict[str, Any]] | None] = {}
    per_work_audits: dict[str, dict[str, Any]] = {}
    target_total = 0
    rechecked_total = 0
    changed_total = 0
    rejected_total = 0
    elapsed_total = 0.0
    prompt_tokens_total = 0
    eval_tokens_total = 0
    any_ctx_warn = False

    for work in dataset.works:
        work_after, w_audit = apply_speaker_recheck_to_work(
            work,
            before_preds.get(work.work_id),
            config,
            ollama_url=ollama_url,
        )
        after_preds[work.work_id] = work_after
        per_work_audits[work.work_id] = w_audit
        target_total += int(w_audit["target_count"])
        rechecked_total += int(w_audit["rechecked_count"])
        changed_total += int(w_audit["changed_count"])
        rejected_total += int(w_audit["rejected_count"])
        elapsed_total += float(w_audit["elapsed_sec"])
        prompt_tokens_total += int(w_audit["prompt_eval_count"])
        eval_tokens_total += int(w_audit["eval_count"])
        if w_audit["any_context_window_warning"]:
            any_ctx_warn = True

    after_metrics = evaluate_model_predictions(
        works=dataset.works,
        predictions_by_work=after_preds,
        model_id=model_id,
        prompt_hash=prompt_hash,
    )
    after_gate = evaluate_step1_gate(after_metrics)

    fixed_errors: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    wrong_to_different_wrong: list[dict[str, Any]] = []
    unchanged_errors: list[dict[str, Any]] = []
    non_attributed_changes: list[dict[str, Any]] = []

    for work in dataset.works:
        b_map = _index_utterances(before_preds.get(work.work_id))
        a_map = _index_utterances(after_preds.get(work.work_id))
        for gold_u in work.utterances:
            b_u = b_map.get(gold_u.n)
            a_u = a_map.get(gold_u.n)
            b_spk = b_u.get("speaker_id") if b_u is not None else None
            a_spk = a_u.get("speaker_id") if a_u is not None else None
            b_ev = b_u.get("evidence") if b_u is not None else None
            a_ev = a_u.get("evidence") if a_u is not None else None

            if gold_u.status != "attributed":
                if b_spk != a_spk:
                    non_attributed_changes.append(
                        {
                            "work_id": work.work_id,
                            "title": work.title,
                            "utterance_index": gold_u.n,
                            "gold_status": gold_u.status,
                            "before_speaker": b_spk,
                            "after_speaker": a_spk,
                            "quote_text": gold_u.quote_text,
                        }
                    )
                continue

            before_ok = b_u is not None and b_ev != "ambiguous" and b_spk == gold_u.speaker
            after_ok = a_u is not None and a_ev != "ambiguous" and a_spk == gold_u.speaker
            row = {
                "work_id": work.work_id,
                "title": work.title,
                "utterance_index": gold_u.n,
                "gold_speaker": gold_u.speaker,
                "before_speaker": b_spk,
                "after_speaker": a_spk,
                "quote_text": gold_u.quote_text,
            }
            if not before_ok and after_ok:
                fixed_errors.append(row)
            elif before_ok and not after_ok:
                regressions.append(row)
            elif not before_ok and not after_ok:
                if b_spk != a_spk:
                    wrong_to_different_wrong.append(row)
                else:
                    unchanged_errors.append(row)

    return {
        "speaker_recheck": config.to_dict(),
        "recheck_prompt_version": get_recheck_prompt(config.prompt_version)[0],
        "recheck_prompt_hash": compute_recheck_prompt_hash(prompt_version=config.prompt_version),
        "recheck_summary": {
            "target_count": target_total,
            "rechecked_count": rechecked_total,
            "changed_count": changed_total,
            "rejected_count": rejected_total,
            "elapsed_sec": round(elapsed_total, 3),
            "prompt_eval_count": prompt_tokens_total,
            "eval_count": eval_tokens_total,
            "any_context_window_warning": any_ctx_warn,
        },
        "before_metrics": before_metrics,
        "before_gate": before_gate,
        "after_metrics": after_metrics,
        "after_gate": after_gate,
        "diff": {
            "fixed_errors": fixed_errors,
            "regressions": regressions,
            "wrong_to_different_wrong": wrong_to_different_wrong,
            "unchanged_errors": unchanged_errors,
            "non_attributed_changes": non_attributed_changes,
        },
        "per_work_audits": per_work_audits,
        "after_predictions": after_preds,
    }
