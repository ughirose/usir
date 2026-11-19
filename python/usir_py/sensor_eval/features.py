"""Deterministic pre-features for quoted utterances (zero LLM/gold-label access)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from usir_py.sensor_eval.f9_loader import F9Character, F9Work

LOCAL_WINDOW_CHARS = 55

_HORIZONTAL_WS = " \t\u3000"
_ALL_WS = " \t\r\n\u3000"

FIRST_PERSON_NARRATION_PRONOUNS: tuple[str, ...] = (
    "わたくし",
    "わたし",
    "あたし",
    "自分",
    "吾輩",
    "ぼく",
    "おれ",
    "私",
    "僕",
    "俺",
)

_SPEECH_VERB_PATTERNS: tuple[str, ...] = (
    (
        r"話しかけ|声を?かけ|呼び止め|耳う?ち(?:を)?[しすせ]|耳打ち(?:を)?[しすせ]"
        r"|言い?出|云い?出|言いつけ|指図(?:を)?[しすせ]|挨拶(?:を)?[しすせ]"
        r"|辞退(?:を)?[しすせ]|返事(?:を)?[しすせ]|質問(?:を)?[しすせ]|談し合|話し合"
    ),
    r"(?:言|云|謂)(?!わな|いま(?:せ|す)ん|わず|わぬ)[わいうえっ]",
    r"尋ね|訊[かきくけい]|(?<!を)聞(?:い[たて]|き(?:まし|ながら|、|。)|く(?:と|の|が|。|$))|問[わいうえい]|答え|応[じえ]",
    (
        r"話[さしすせ]|語[らりるれっ]|物語[らりるれっ]|叫[ばびぶべん]|呼[ばびぶべん]"
        r"|囁[かきくけい]|ささや[かきくけい]|呟[かきくけい]|つぶや[かきくけい]"
        r"|怒鳴[らりるれっ]|喚[かきくけい]|わめ[かきくけい]|叱[らりるれっ]"
        r"|誘[わいうえっ]|申[さしすせ]|仰[らりるれっ]|仰有|おっしゃ[らりるれっ]"
        r"|告げ|伝え|唱え|述べ|漏ら[さしすせ]"
    ),
)
_SPEECH_VERB_RE = re.compile("|".join(_SPEECH_VERB_PATTERNS))
_NEGATED_AFTER_RE = re.compile(r"^(?:な[いかくけ]|ま[せす]ん|ず|ぬ|なかっ)")
_QUOTATIVE_START_RE = re.compile(
    r"^[\s\u3000]*(?:と(?!ころ|ても|にかく|うとう|っく|んでも|もかく)|こう|そう|こんな|そんな)"
)
_ANAPHORA_RE = re.compile(r"(?:こう|そう|こんな|そんな)")


@dataclass(frozen=True)
class UtteranceFeatures:
    """Deterministic structural and lexical features around a single quote."""

    utterance_index: int
    position: str  # "standalone_line" | "line_head_with_trailing" | "inline_embedded"
    quote_chars: int
    pre_has_narration: bool
    pre_speakers: tuple[str, ...]
    pre_speech_verb: str | None
    pre_speaker_and_verb: bool
    post_speakers: tuple[str, ...]
    post_speech_verb: str | None
    post_speaker_and_verb: bool


def find_speech_verb(text: str) -> str | None:
    """Return the first non-negated Japanese speech/reporting verb in `text`, or None."""
    if not text:
        return None
    for match in _SPEECH_VERB_RE.finditer(text):
        tail = text[match.end() : match.end() + 6]
        if _NEGATED_AFTER_RE.match(tail):
            continue
        return match.group(0)
    return None


def _find_first_unblocked_pronoun(text: str, pronoun: str) -> int | None:
    start = 0
    while True:
        idx = text.find(pronoun, start)
        if idx == -1:
            return None
        if pronoun == "私":
            next_bad = idx + 1 < len(text) and text[idx + 1] in "的立有用"
            prev_bad = idx > 0 and text[idx - 1] in "公無"
            if next_bad or prev_bad:
                start = idx + 1
                continue
        return idx


def find_speakers_in_text(
    text: str,
    characters: Sequence[F9Character],
    narration: str,
    narrator: str | None,
) -> tuple[str, ...]:
    """Match roster aliases (and 1st-person narrator pronouns) in `text` by earliest position."""
    if not text:
        return ()
    hits: list[tuple[int, str]] = []
    for char in characters:
        best_pos: int | None = None
        for alias in sorted(char.aliases, key=len, reverse=True):
            if not alias:
                continue
            pos = text.find(alias)
            if pos != -1 and (best_pos is None or pos < best_pos):
                best_pos = pos
        if best_pos is not None:
            hits.append((best_pos, char.id))

    if narration == "first" and narrator:
        narr_pos: int | None = None
        for pronoun in FIRST_PERSON_NARRATION_PRONOUNS:
            pos = _find_first_unblocked_pronoun(text, pronoun)
            if pos is not None and (narr_pos is None or pos < narr_pos):
                narr_pos = pos
        if narr_pos is not None and not any(cid == narrator for _, cid in hits):
            hits.append((narr_pos, narrator))

    hits.sort(key=lambda item: item[0])
    seen: set[str] = set()
    ordered: list[str] = []
    for _, cid in hits:
        if cid not in seen:
            seen.add(cid)
            ordered.append(cid)
    return tuple(ordered)


def _classify_position(text: str, open_offset: int, close_offset: int) -> str:
    line_start = text.rfind("\n", 0, open_offset) + 1
    pre_on_line = text[line_start:open_offset].strip(_HORIZONTAL_WS + "\r")
    if pre_on_line:
        return "inline_embedded"

    nl_after = text.find("\n", close_offset)
    line_end = len(text) if nl_after == -1 else nl_after
    suf_on_line = text[close_offset:line_end].strip(_HORIZONTAL_WS + "\r")
    if suf_on_line:
        return "line_head_with_trailing"
    return "standalone_line"


def _extract_post_clause(gap_after: str, has_next_quote: bool) -> tuple[str, int]:
    """Extract post-quote reporting clause owned by the preceding quote and its consumed span."""
    if not gap_after:
        return "", 0

    idx = 0
    while idx < len(gap_after) and gap_after[idx] in _HORIZONTAL_WS:
        idx += 1
    if idx >= len(gap_after):
        return "", 0

    if not gap_after.startswith(("\n", "\r\n"), idx):
        nl = gap_after.find("\n", idx)
        line = (gap_after[idx:] if nl == -1 else gap_after[idx:nl]).rstrip("\r")
        if not line.strip(_ALL_WS):
            return "", 0
        period = line.find("。")
        if period != -1:
            end_in_gap = idx + period + 1
            consumed = end_in_gap
            if nl != -1 and not gap_after[end_in_gap:nl].strip(_HORIZONTAL_WS + "\r"):
                consumed = nl + 1
            return line[: period + 1].strip(), consumed
        consumed = nl + 1 if nl != -1 else len(gap_after)
        return line.strip(), consumed

    after_first_nl = idx + (2 if gap_after.startswith("\r\n", idx) else 1)
    rest = gap_after[after_first_nl:]
    if not rest or rest.startswith(("\n", "\r\n")):
        return "", 0

    nl = rest.find("\n")
    line = (rest if nl == -1 else rest[:nl]).rstrip("\r")
    if not line.strip(_ALL_WS):
        return "", 0

    period = line.find("。")
    first_sent = line[: period + 1] if period != -1 else line

    is_post = bool(_QUOTATIVE_START_RE.match(first_sent)) or (
        bool(_ANAPHORA_RE.search(first_sent)) and find_speech_verb(first_sent) is not None
    )
    if not is_post:
        return "", 0

    if period != -1:
        end_in_rest = period + 1
        consumed = after_first_nl + end_in_rest
        if nl != -1 and not rest[end_in_rest:nl].strip(_HORIZONTAL_WS + "\r"):
            consumed = after_first_nl + nl + 1
        return first_sent.strip(), consumed

    if has_next_quote and nl != -1 and not rest[nl + 1 :].strip(_ALL_WS):
        first_verb_match: re.Match[str] | None = None
        for match in _SPEECH_VERB_RE.finditer(line):
            tail = line[match.end() : match.end() + 6]
            if not _NEGATED_AFTER_RE.match(tail):
                first_verb_match = match
                break
        if first_verb_match is not None:
            comma_after = line.find("、", first_verb_match.end())
            if comma_after != -1 and comma_after < len(line) - 1 and line[comma_after + 1 :].strip():
                return line[: comma_after + 1].strip(), after_first_nl + comma_after + 1

    consumed = after_first_nl + (nl + 1 if nl != -1 else len(rest))
    return line.strip(), consumed


def _extract_pre_clause(remaining_gap_before: str) -> str:
    stripped = remaining_gap_before.rstrip(_ALL_WS)
    if not stripped:
        return ""
    trailing_ws = remaining_gap_before[len(stripped) :]
    if trailing_ws.count("\n") >= 2:
        return ""

    nl = stripped.rfind("\n")
    last_line = stripped[nl + 1 :].strip() if nl != -1 else stripped.strip()
    if not last_line:
        return ""

    body = last_line[:-1] if last_line.endswith("。") else last_line
    last_period = body.rfind("。")
    if last_period != -1:
        return last_line[last_period + 1 :].strip()
    return last_line


def extract_utterance_features_from_text(
    text: str,
    spans: Sequence[tuple[int, int]],
    characters: Sequence[F9Character],
    narration: str,
    narrator: str | None,
) -> list[UtteranceFeatures]:
    """Extract deterministic positional and lexical features for each quote span."""
    total = len(spans)
    post_clauses: list[str] = [""] * total
    post_consumed: list[int] = [0] * total

    for idx, (_open_off, close_off) in enumerate(spans):
        next_open = spans[idx + 1][0] if idx + 1 < total else len(text)
        gap_after = text[close_off:next_open]
        clause, consumed = _extract_post_clause(gap_after, has_next_quote=(idx + 1 < total))
        post_clauses[idx] = clause
        post_consumed[idx] = consumed

    features: list[UtteranceFeatures] = []
    for idx, (open_off, close_off) in enumerate(spans):
        position = _classify_position(text, open_off, close_off)
        quote_chars = max(0, (close_off - open_off) - 2)

        prev_end = spans[idx - 1][1] + post_consumed[idx - 1] if idx > 0 else 0
        if prev_end > open_off:
            prev_end = spans[idx - 1][1] if idx > 0 else 0
        rem_gap_before = text[prev_end:open_off]
        pre_clause = _extract_pre_clause(rem_gap_before)

        pre_window = pre_clause[-LOCAL_WINDOW_CHARS:] if pre_clause else ""
        post_clause = post_clauses[idx]
        post_window = post_clause[:LOCAL_WINDOW_CHARS] if post_clause else ""

        pre_speakers = find_speakers_in_text(pre_window, characters, narration, narrator)
        pre_verb = find_speech_verb(pre_window)
        post_speakers = find_speakers_in_text(post_window, characters, narration, narrator)
        post_verb = find_speech_verb(post_window)

        features.append(
            UtteranceFeatures(
                utterance_index=idx,
                position=position,
                quote_chars=quote_chars,
                pre_has_narration=bool(pre_window),
                pre_speakers=pre_speakers,
                pre_speech_verb=pre_verb,
                pre_speaker_and_verb=bool(pre_speakers) and pre_verb is not None,
                post_speakers=post_speakers,
                post_speech_verb=post_verb,
                post_speaker_and_verb=bool(post_speakers) and post_verb is not None,
            )
        )
    return features


def extract_work_utterance_features(work: F9Work) -> list[UtteranceFeatures]:
    """Compute deterministic utterance features for an F9Work without reading gold labels."""
    spans = [(u.open_offset, u.close_offset) for u in work.utterances]
    return extract_utterance_features_from_text(
        text=work.text,
        spans=spans,
        characters=work.characters,
        narration=work.narration,
        narrator=work.narrator,
    )


def format_utterance_features_block(
    features: Sequence[UtteranceFeatures],
    *,
    include_speaker_ids: bool = True,
) -> str:
    """Format extracted features as a compact Markdown table for prompt v4/v5."""
    if include_speaker_ids:
        lines = [
            "## 各引用の機械抽出特徴（参考前情報）",
            (
                "各 `[Uk]` の決定論的特徴（`pos`: `standalone_line`=行頭独立発話行 / "
                "`line_head_with_trailing`=行頭開始+同一行後続地の文 / `inline_embedded`=地の文途中埋込、"
                "`chars`=括弧内文字数、`pre_*`/`post_*`=直前/直後の地の文の話者名候補・発話動詞・両方あり判定）。"
                "判定自体は本文と人物名簿に基づきモデルが行うこと。"
            ),
            "| id | pos | chars | pre_spk | pre_verb | pre_spk_verb | post_spk | post_verb | post_spk_verb |",
            "|---|---|---:|---|---|---|---|---|---|",
        ]
        for feat in features:
            pre_spk = ",".join(feat.pre_speakers) if feat.pre_speakers else "-"
            post_spk = ",".join(feat.post_speakers) if feat.post_speakers else "-"
            pre_verb = feat.pre_speech_verb or "-"
            post_verb = feat.post_speech_verb or "-"
            pre_sv = "true" if feat.pre_speaker_and_verb else "false"
            post_sv = "true" if feat.post_speaker_and_verb else "false"
            lines.append(
                f"| U{feat.utterance_index} | {feat.position} | {feat.quote_chars} "
                f"| {pre_spk} | {pre_verb} | {pre_sv} "
                f"| {post_spk} | {post_verb} | {post_sv} |"
            )
        return "\n".join(lines)

    lines = [
        "## 各引用の機械抽出特徴（参考前情報）",
        (
            "各 `[Uk]` の決定論的特徴（`pos`: `standalone_line`=行頭独立発話行 / "
            "`line_head_with_trailing`=行頭開始+同一行後続地の文 / `inline_embedded`=地の文途中埋込、"
            "`chars`=括弧内文字数、`pre_verb`/`post_verb`=直前/直後の地の文の発話動詞、"
            "`pre_spk_verb`/`post_spk_verb`=直前/直後の地の文に話者名と発話動詞の両方があるか）。"
            "判定自体は本文と人物名簿に基づきモデルが行うこと。"
        ),
        "| id | pos | chars | pre_verb | pre_spk_verb | post_verb | post_spk_verb |",
        "|---|---|---:|---|---|---|---|",
    ]
    for feat in features:
        pre_verb = feat.pre_speech_verb or "-"
        post_verb = feat.post_speech_verb or "-"
        pre_sv = "true" if feat.pre_speaker_and_verb else "false"
        post_sv = "true" if feat.post_speaker_and_verb else "false"
        lines.append(
            f"| U{feat.utterance_index} | {feat.position} | {feat.quote_chars} "
            f"| {pre_verb} | {pre_sv} "
            f"| {post_verb} | {post_sv} |"
        )
    return "\n".join(lines)
