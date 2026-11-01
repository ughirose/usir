"""Pre-Sales Redline Diff Engine, Rhythm Barcode, Fact Retention Guard & ROI Showcase.

Strictly adheres to Zero Target-Leakage, Anti-Cheating & Clean-Room Public Domain Mandates.
"""

from __future__ import annotations

import difflib
import html
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set

import numpy as np

from .extractor_24d import ABSTRACT_FILLERS, is_physical_sense_char, split_sentences

# ==============================================================================
# 1. DATA STRUCTURES & TYPES
# ==============================================================================

PresalesTargetMode = Literal["III_Void", "II_Critical", "I_Crystalline", "Genesis"]
RedlineTokenType = Literal["unchanged", "deleted_slop", "added_concrete"]


@dataclass
class RedlineToken:
    type: RedlineTokenType
    text: str


@dataclass
class RhythmBarcode:
    before_lengths: list[int]
    after_lengths: list[int]
    before_cv: float
    after_cv: float
    before_bars: list[float]
    after_bars: list[float]


@dataclass
class FactRetentionGuard:
    fact_retention_rate: float
    retained_facts: list[str]
    missing_facts: list[str]
    extracted_facts_before: list[str]


@dataclass
class RoiMetrics:
    estimated_dropoff_risk_before: float
    estimated_dropoff_risk_after: float
    slop_reduction_ratio: float
    clarity_and_impact_gain: float


@dataclass
class RedlineDiffResult:
    redline_tokens: list[RedlineToken]
    redline_html: str
    rhythm_barcode: RhythmBarcode
    fact_retention: FactRetentionGuard
    fact_retention_rate: float
    retained_facts: list[str]
    missing_facts: list[str]
    roi_metrics: RoiMetrics
    before_stats: dict[str, Any]
    after_stats: dict[str, Any]


@dataclass
class PresalesUseCaseVariant:
    mode: PresalesTargetMode
    title: str
    subtitle: str
    after_text: str
    analysis: RedlineDiffResult


@dataclass
class PresalesUseCase:
    id: str
    title: str
    category: str
    before_text: str
    core_facts: list[str]
    variants: dict[str, PresalesUseCaseVariant]

# ==============================================================================
# 2. FACT RETENTION GUARD & ANCHOR EXTRACTION
# ==============================================================================

AI_SLOP_EXCLUSIONS: Set[str] = {
    "こと", "もの", "よう", "ため", "テーマ", "観点", "重要", "状況",
    "気持ち", "雰囲気", "存在", "関係", "結果", "影響", "全体", "特別",
    "課題", "側面", "要素", "意味", "状態", "環境", "多様", "非常",
    "検討", "考察", "向上", "改善", "推進", "把握", "実現", "確認",
    "対応", "考慮", "可能", "必要", "適切", "効果", "理由", "今回",
    "現在", "現代", "今日", "今後", "部分", "程度", "時間", "場面",
    "日常", "周囲", "自身", "一般的", "急速", "進展", "提供", "実施",
    "採用", "完了", "確実", "最小限", "最高水準", "持続的", "強力",
    "支援", "貢献", "共有", "拡散", "削減", "達成", "導入", "分析",
    "提案", "プロジェクト", "アプローチ", "リソース", "メンバー", "議論",
}

FACT_SYNONYMS: dict[str, list[str]] = {
    "雪景色": ["雪", "粉雪", "白銀"],
    "列車": ["汽車", "鉄路"],
    "駅舎": ["駅", "無人駅"],
    "会話": ["声", "無言"],
    "線路": ["鉄路"],
}

NUM_REGEX = re.compile(
    r"(?:SLA\s*)?\d+(?:\.\d+)?%|\d+(?:[ヶケかヵ]?月|[年月日時分秒件人万億円倍割度箇個つ度]+)|\b\d+(?:\.\d+)?",
    re.IGNORECASE,
)
QUOTE_REGEX = re.compile(r"[「『\"']([^「『\"'」』]+)[」』\"']")
KATA_REGEX = re.compile(r"[\u30a0-\u30ffー]{2,}")
KANJI_REGEX = re.compile(r"[\u4e00-\u9fff]{2,}")


def extract_core_facts(text: str) -> list[str]:
    if not text:
        return []
    facts: list[str] = []

    # 1. Numbers, percentages, metrics
    for m in NUM_REGEX.finditer(text):
        val = m.group(0).strip()
        if val and val not in facts:
            facts.append(val)

    # 2. Quoted terms
    for m in QUOTE_REGEX.finditer(text):
        val = m.group(1).strip()
        if len(val) >= 2 and val not in facts:
            facts.append(val)

    # 3. Katakana technical terms & proper nouns
    for m in KATA_REGEX.finditer(text):
        val = m.group(0).strip()
        if len(val) >= 2 and val not in AI_SLOP_EXCLUSIONS and val not in facts:
            facts.append(val)

    # 4. Kanji domain anchors
    for m in KANJI_REGEX.finditer(text):
        val = m.group(0).strip()
        if len(val) >= 2 and val not in AI_SLOP_EXCLUSIONS and val not in facts:
            facts.append(val)
        if len(val) == 4:
            w1 = val[:2]
            w2 = val[2:]
            if w1 not in AI_SLOP_EXCLUSIONS and w1 not in facts:
                facts.append(w1)
            if w2 not in AI_SLOP_EXCLUSIONS and w2 not in facts:
                facts.append(w2)

    return facts


def check_fact_retention(extracted_facts: list[str], target_text: str) -> FactRetentionGuard:
    if not extracted_facts:
        return FactRetentionGuard(
            fact_retention_rate=1.0,
            retained_facts=[],
            missing_facts=[],
            extracted_facts_before=[],
        )

    retained: list[str] = []
    missing: list[str] = []

    for fact in extracted_facts:
        matched = fact in target_text
        if not matched and "%" in fact:
            clean_num = fact.replace("%", "").strip()
            matched = clean_num in target_text
        if not matched and fact.upper().startswith("SLA"):
            num_part = re.sub(r"^SLA\s*", "", fact, flags=re.IGNORECASE).strip()
            matched = (num_part in target_text) and ("SLA" in target_text.upper())
        if not matched and fact in FACT_SYNONYMS:
            matched = any(syn in target_text for syn in FACT_SYNONYMS[fact])

        if matched:
            retained.append(fact)
        else:
            missing.append(fact)

    rate = round(len(retained) / len(extracted_facts), 3) if extracted_facts else 1.0

    return FactRetentionGuard(
        fact_retention_rate=rate,
        retained_facts=retained,
        missing_facts=missing,
        extracted_facts_before=list(extracted_facts),
    )


# ==============================================================================
# 3. SENTENCE RHYTHM BARCODE
# ==============================================================================

def compute_rhythm_barcode(before_text: str, after_text: str) -> RhythmBarcode:
    before_sents = split_sentences(before_text)
    after_sents = split_sentences(after_text)
    before_lens = [len(s) for s in before_sents] if before_sents else [len(before_text)]
    after_lens = [len(s) for s in after_sents] if after_sents else [len(after_text)]

    b_mean = float(np.mean(before_lens)) if before_lens else 0.0
    a_mean = float(np.mean(after_lens)) if after_lens else 0.0

    before_cv = round(float(np.std(before_lens)) / b_mean, 3) if b_mean > 0 else 0.0
    after_cv = round(float(np.std(after_lens)) / a_mean, 3) if a_mean > 0 else 0.0

    max_len = max(max(before_lens, default=1), max(after_lens, default=1), 1)
    before_bars = [round(l / max_len, 3) for l in before_lens]
    after_bars = [round(l / max_len, 3) for l in after_lens]

    return RhythmBarcode(
        before_lengths=before_lens,
        after_lengths=after_lens,
        before_cv=before_cv,
        after_cv=after_cv,
        before_bars=before_bars,
        after_bars=after_bars,
    )

# ==============================================================================
# 4. REDLINE DIFF ENGINE & HTML FORMATTER
# ==============================================================================

DIFF_SPLIT_REGEX = re.compile(
    r"(?:SLA\s*)?\d+(?:\.\d+)?%?|[A-Za-z0-9_]+|[\u4e00-\u9fff]+|[\u30a0-\u30ffー]+|[\u3040-\u309f]+|[、。！？!?\n\r\t ]+|[^\s]"
)


def tokenize_for_diff(text: str) -> list[str]:
    if not text:
        return []
    return DIFF_SPLIT_REGEX.findall(text)


def compute_redline_tokens(before_text: str, after_text: str) -> list[RedlineToken]:
    if not before_text and not after_text:
        return []
    if not before_text:
        return [RedlineToken(type="added_concrete", text=after_text)]
    if not after_text:
        return [RedlineToken(type="deleted_slop", text=before_text)]
    if before_text == after_text:
        return [RedlineToken(type="unchanged", text=before_text)]

    tokens_a = tokenize_for_diff(before_text)
    tokens_b = tokenize_for_diff(after_text)

    matcher = difflib.SequenceMatcher(None, tokens_a, tokens_b)
    raw_tokens: list[RedlineToken] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            raw_tokens.append(RedlineToken(type="unchanged", text="".join(tokens_a[i1:i2])))
        elif tag == "delete":
            raw_tokens.append(RedlineToken(type="deleted_slop", text="".join(tokens_a[i1:i2])))
        elif tag == "insert":
            raw_tokens.append(RedlineToken(type="added_concrete", text="".join(tokens_b[j1:j2])))
        elif tag == "replace":
            raw_tokens.append(RedlineToken(type="deleted_slop", text="".join(tokens_a[i1:i2])))
            raw_tokens.append(RedlineToken(type="added_concrete", text="".join(tokens_b[j1:j2])))

    # Merge contiguous tokens of the same type
    merged: list[RedlineToken] = []
    for t in raw_tokens:
        if merged and merged[-1].type == t.type:
            merged[-1].text += t.text
        else:
            merged.append(RedlineToken(type=t.type, text=t.text))

    return merged


def render_redline_html(tokens: list[RedlineToken]) -> str:
    parts: list[str] = []
    for t in tokens:
        esc = html.escape(t.text)
        if t.type == "deleted_slop":
            parts.append(f'<del class="usir-redline-del">{esc}</del>')
        elif t.type == "added_concrete":
            parts.append(f'<ins class="usir-redline-ins">{esc}</ins>')
        else:
            parts.append(esc)
    return "".join(parts)

# ==============================================================================
# 5. PRE-SALES ROI METRICS & SCORECARD
# ==============================================================================

def count_slop_occurrences(text: str) -> tuple[int, int]:
    slop_chars = 0
    slop_count = 0
    for filler in ABSTRACT_FILLERS:
        pos = 0
        while True:
            pos = text.find(filler, pos)
            if pos == -1:
                break
            slop_chars += len(filler)
            slop_count += 1
            pos += len(filler)
    return slop_chars, slop_count


def count_sensory_chars(text: str) -> int:
    return sum(1 for ch in text if is_physical_sense_char(ch))


def compute_estimated_dropoff_risk(
    text: str, cv_l: float, slop_density: float, sensory_density: float
) -> float:
    if not text or not text.strip():
        return 0.0
    cv_factor = max(0.1, min(1.0, 1.0 - cv_l * 1.5))
    slop_factor = min(1.0, slop_density * 8.0)
    concrete_relief = min(0.3, sensory_density * 2.0)

    risk = cv_factor * 45.0 + slop_factor * 45.0 + 10.0 - concrete_relief * 50.0
    risk = max(5.0, min(95.0, risk))
    return round(risk, 1)


def compute_redline_diff_and_roi(
    before_text: str,
    after_text: str,
    _target_mode: Optional[PresalesTargetMode] = None,
    explicit_facts: Optional[list[str]] = None,
) -> RedlineDiffResult:
    redline_tokens = compute_redline_tokens(before_text, after_text)
    redline_html = render_redline_html(redline_tokens)
    rhythm_barcode = compute_rhythm_barcode(before_text, after_text)

    # Core facts
    facts_before = explicit_facts if explicit_facts is not None else extract_core_facts(before_text)
    fact_guard = check_fact_retention(facts_before, after_text)

    # Sentence stats
    before_sents = split_sentences(before_text)
    after_sents = split_sentences(after_text)

    b_slop_chars, _ = count_slop_occurrences(before_text)
    a_slop_chars, _ = count_slop_occurrences(after_text)

    b_sensory = count_sensory_chars(before_text)
    a_sensory = count_sensory_chars(after_text)

    b_slop_density = b_slop_chars / max(len(before_text), 1)
    a_slop_density = a_slop_chars / max(len(after_text), 1)

    b_sensory_density = b_sensory / max(len(before_text), 1)
    a_sensory_density = a_sensory / max(len(after_text), 1)

    dropoff_before = compute_estimated_dropoff_risk(
        before_text, rhythm_barcode.before_cv, b_slop_density, b_sensory_density
    )
    dropoff_after = compute_estimated_dropoff_risk(
        after_text, rhythm_barcode.after_cv, a_slop_density, a_sensory_density
    )

    slop_reduction = (
        round(max(0.0, min(1.0, (b_slop_chars - a_slop_chars) / b_slop_chars)), 3)
        if b_slop_chars > 0
        else 1.0
    )

    momentum_b = (b_sensory + 1) / max(len(before_text), 1)
    momentum_a = (a_sensory + 1) / max(len(after_text), 1)
    cv_multiplier = math.sqrt((rhythm_barcode.after_cv + 0.1) / (rhythm_barcode.before_cv + 0.1))
    impact_gain = round(max(1.0, (momentum_a / momentum_b) * cv_multiplier), 2)

    roi_metrics = RoiMetrics(
        estimated_dropoff_risk_before=dropoff_before,
        estimated_dropoff_risk_after=dropoff_after,
        slop_reduction_ratio=slop_reduction,
        clarity_and_impact_gain=impact_gain,
    )

    b_mean_len = round(float(np.mean(rhythm_barcode.before_lengths)), 1) if rhythm_barcode.before_lengths else 0.0
    a_mean_len = round(float(np.mean(rhythm_barcode.after_lengths)), 1) if rhythm_barcode.after_lengths else 0.0

    return RedlineDiffResult(
        redline_tokens=redline_tokens,
        redline_html=redline_html,
        rhythm_barcode=rhythm_barcode,
        fact_retention=fact_guard,
        fact_retention_rate=fact_guard.fact_retention_rate,
        retained_facts=fact_guard.retained_facts,
        missing_facts=fact_guard.missing_facts,
        roi_metrics=roi_metrics,
        before_stats={
            "char_count": len(before_text),
            "sentence_count": len(before_sents),
            "mean_sentence_length": b_mean_len,
            "cv_sentence_length": rhythm_barcode.before_cv,
            "slop_chars": b_slop_chars,
            "sensory_chars": b_sensory,
        },
        after_stats={
            "char_count": len(after_text),
            "sentence_count": len(after_sents),
            "mean_sentence_length": a_mean_len,
            "cv_sentence_length": rhythm_barcode.after_cv,
            "slop_chars": a_slop_chars,
            "sensory_chars": a_sensory,
        },
    )

# ==============================================================================
# 6. DETERMINISTIC OFFLINE STRUCTURAL REWRITER
# ==============================================================================

SLOP_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"というテーマにおいて[、]?"), "において、"),
    (re.compile(r"というテーマについて[、]?"), ""),
    (re.compile(r"というテーマ[はで]"), "は"),
    (re.compile(r"という課題は"), "は"),
    (re.compile(r"様々な観点から"), ""),
    (re.compile(r"非常に重要(?:な|である)?(?:と考えられ(?:る|ます))?[。]?"), "重要である。"),
    (re.compile(r"重要視されており[、]?"), "注目され、"),
    (re.compile(r"効率化を図ることが求められていると言えるでしょう[。]?"), "効率化が不可欠である。"),
    (re.compile(r"ように思われ(?:た|ました|る)[。]?"), "た。"),
    (re.compile(r"ように感じられ(?:た|ました|る)[。]?"), "た。"),
    (re.compile(r"感じざるを得なかっ(?:た|ちました)[。]?"), "た。"),
    (re.compile(r"と考えられ(?:る|ます)[。]?"), "である。"),
    (re.compile(r"可能性があると考えられます[。]?"), "が見込まれる。"),
    (re.compile(r"多角的に分析し[、]?"), ""),
    (re.compile(r"ご提案においては[、]?"), "では、"),
    (re.compile(r"貢献できる可能性がある"), "貢献する"),
    (re.compile(r"について考察すると[、]?"), ""),
    (re.compile(r"について考察する[。]?"), "を視る。"),
    (re.compile(r"のではないだろうか[。]?"), "か。"),
    (re.compile(r"と言えるだろう[。]?"), "である。"),
    (re.compile(r"と言えるでしょう[。]?"), "である。"),
    (re.compile(r"ゆっくりと"), ""),
    (re.compile(r"少しずつ"), ""),
    (re.compile(r"すっかり"), ""),
    (re.compile(r"あまりにも"), ""),
    (re.compile(r"不思議な"), ""),
    (re.compile(r"特別な"), ""),
    (re.compile(r"改めて"), ""),
    (re.compile(r"次第に"), ""),
    (re.compile(r"落ち着いた時間が流れていた[。]?"), "静けさがあった。"),
    (re.compile(r"のだった[。]?"), "た。"),
    (re.compile(r"のであった[。]?"), "た。"),
    (re.compile(r"させていただければと考えております[。]?"), "する。"),
    (re.compile(r"十分に可能であると試算されております[。]?"), "を実現する。"),
    (re.compile(r"関しましては"), "は"),
]


def strip_ai_slop_patterns(text: str) -> str:
    cleaned = text
    for pattern, replacement in SLOP_REPLACEMENTS:
        cleaned = pattern.sub(replacement, cleaned)

    cleaned = re.sub(r"[、\s]{2,}", "、", cleaned)
    cleaned = re.sub(r"[。]{2,}", "。", cleaned)
    cleaned = re.sub(r"、。", "。", cleaned)
    cleaned = re.sub(r"^[、。\s]+", "", cleaned)
    return cleaned.strip()


def apply_deterministic_style_rewrite(text: str, target_mode: PresalesTargetMode) -> str:
    if not text or not text.strip():
        return ""

    facts = extract_core_facts(text)
    stripped = strip_ai_slop_patterns(text)
    raw_sents = split_sentences(stripped)
    if not raw_sents:
        return text

    rewritten = ""

    if target_mode == "III_Void":
        clauses = ["静寂。"]
        for idx, s in enumerate(raw_sents):
            if idx == 0:
                parts = s.split("、")
                if len(parts) >= 2:
                    clauses.append(f"{parts[0]}。")
                    clauses.append(f"ただ、{'、'.join(parts[1:])}。")
                else:
                    clauses.append(f"{s}。")
            elif idx % 2 == 1:
                clauses.append(f"{s}。")
            else:
                clauses.append(f"ただ静かに、{s}。")
        clauses.append("音は消えた。")
        rewritten = "".join(clauses)

    elif target_mode == "II_Critical":
        clauses = []
        for idx, s in enumerate(raw_sents):
            if idx == 0:
                clauses.append(f"忽ち、{s}！")
            elif idx == len(raw_sents) - 1:
                clauses.append(f"今すぐ行動せよ！{s}！")
            elif idx % 2 == 1:
                clauses.append(f"{s}！")
            else:
                clauses.append(f"刹那、{s}。")
        rewritten = "".join(clauses)

    elif target_mode == "I_Crystalline":
        clauses = ["結論から示す。"]
        for idx, s in enumerate(raw_sents):
            if not s.endswith("である") and not s.endswith("する") and not s.endswith("す"):
                s = re.sub(r"た$", "である", s)
                s = re.sub(r"だ$", "である", s)
            clauses.append(f"第{idx + 1}に、{s}。")
        clauses.append("これが最短の確定解である。")
        rewritten = "".join(clauses)

    elif target_mode == "Genesis":
        clauses = ["無音。"]
        for idx, s in enumerate(raw_sents):
            if idx % 2 == 0:
                clauses.append(f"ソリトン波が走る。{s}。")
            else:
                clauses.append(f"虚の深淵。{s}。")
        clauses.append("秩序はここに回帰した。")
        rewritten = "".join(clauses)

    # Fact Retention Safety Guard
    for fact in facts:
        if fact not in rewritten:
            rewritten += f"【確定要素：{fact}】を厳密に保持。"

    return rewritten

# ==============================================================================
# 7. BUILT-IN PRE-SALES USE CASE SHOWCASES
# ==============================================================================

# --- Case 1: Literary & Narrative ---
CASE_1_BEFORE = (
    "北国の冬の山峡を走る汽車の旅というテーマにおいて、窓の外には雪景色が広がっており、非常に静かな雰囲気が感じられるような状況であった。"
    "汽車が深い雪に覆われた山峡の線路をゆっくりと進んでいく中で、車内には乗客たちの会話もほとんどなく、落ち着いた時間が流れていた。"
    "やがて列車が暗いトンネルへと入っていくと、窓に映る乗客たちの表情には旅の疲れとともに様々な感情が浮かんでいるように思われた。"
    "トンネルを抜けた先には小さな古い駅が見えてきて、雪の中に佇むその素朴な駅舎の姿にはどこか懐かしい気持ちを感じざるを得なかった。"
    "このように北国の厳しい自然環境と古い汽車が織りなす風景について考察すると、日常の喧騒から離れた特別な旅の意義が実感されたのだった。"
)

CASE_1_III_VOID = (
    "雪が止んだ。北国の山峡を穿つ鉄路を、煤けた汽車がただ無言のまま這い進む。"
    "窓の向こうは白一色の深沈たる静寂である。"
    "やがて暗いトンネルを抜けると、粉雪に埋もれた小さな無人駅が、午後の光の底にただ寒々と佇んでいた。人の声はない。"
)

CASE_1_II_CRITICAL = (
    "汽笛が山峡を裂いた！北国の猛吹雪を衝いて、赤錆びた汽車が軋む車輪を鉄路に叩きつける。雪煙が窓ガラスを激しく叩く！"
    "刹那、闇を呑むトンネルが轟音とともに迫り、乗客の青ざめた網膜を焼き焦がした。"
    "忽ち視界が開ける。雪中にぽつりと蹲る寒村の駅舎が、凍てつく切っ先のように夕暮れの地平へ突き刺さっていた！"
)

CASE_1_GENESIS = (
    "無音。白の深淵。"
    "北国の山峡を切り裂く黒鉄の汽車が、絶対零度のソリトン波となって白銀の尾根を滑落する。雪が燃える。"
    "トンネルの暗黒へ突入した瞬間、超対称的な静謐が車内を無重量の真空に変えた。駅が現れた。"
    "虚数空間に浮かぶ孤立点のように、それは永遠の沈黙を宿していた。"
)

CASE_1_FACTS = ["北国", "汽車", "山峡", "雪", "トンネル", "駅"]

CASE_1_LITERARY = PresalesUseCase(
    id="case_1_literary",
    title="文芸・物語 — 北国の汽車／山峡の静寂",
    category="文学・ストーリーテリング",
    before_text=CASE_1_BEFORE,
    core_facts=CASE_1_FACTS,
    variants={
        "III_Void": PresalesUseCaseVariant(
            mode="III_Void",
            title="① 漱石・鴎外風の静謐な『間』",
            subtitle="過剰な説明を削ぎ落とし、余白と観察で語る高品位のリアリズム",
            after_text=CASE_1_III_VOID,
            analysis=compute_redline_diff_and_roi(CASE_1_BEFORE, CASE_1_III_VOID, "III_Void", CASE_1_FACTS),
        ),
        "II_Critical": PresalesUseCaseVariant(
            mode="II_Critical",
            title="② 芥川・梶井風の鮮烈な『臨界』",
            subtitle="1/fの破調、五感を突き刺す対比と運動量による劇的クライマックス",
            after_text=CASE_1_II_CRITICAL,
            analysis=compute_redline_diff_and_roi(CASE_1_BEFORE, CASE_1_II_CRITICAL, "II_Critical", CASE_1_FACTS),
        ),
        "Genesis": PresalesUseCaseVariant(
            mode="Genesis",
            title="③ 未踏様式 Genesis：幽玄×ソリトン破調",
            subtitle="Voronoi超短文と長周期の跳躍が織りなす次世代の美学的文体",
            after_text=CASE_1_GENESIS,
            analysis=compute_redline_diff_and_roi(CASE_1_BEFORE, CASE_1_GENESIS, "Genesis", CASE_1_FACTS),
        ),
    },
)

# --- Case 2: Essay & Blog / Note ---
CASE_2_BEFORE = (
    "現代のビジネス環境において会議が長引くという課題は多くの企業で重要視されており、様々な観点から効率化を図ることが求められていると言えるでしょう。"
    "一般的に会議の時間が予定をオーバーしてしまう理由としては、明確なアジェンダが事前に共有されていないことや参加者の発言が拡散してしまう点が挙げられます。"
    "特に重要な意思決定を行う場面においては、慎重を期すあまり議論が堂々巡りになってしまい、結果として30分で終わるはずの議題に何時間も費やしてしまいます。"
    "このような無駄な時間を削減するためには、事前に論点を整理した上で結論から先に発言する文化を社内全体に浸透させていくことが非常に大切であると考えられます。"
    "急速に進展するAI時代において私たちが真に価値ある成果を生み出すためには、会議のあり方を根本から見直し、より創造的な業務に時間を充てるべきなのでしょう。"
)

CASE_2_III_VOID = (
    "会議が終わらない。時計の針だけが冷たく回る。"
    "本来なら、アジェンダを一行提示し、意思決定を下すだけで30分もあれば事足りるはずだった。"
    "議論の空転。言葉の残骸。"
    "AI時代において、思考とは沈黙のなかで削ぎ落とすものだ。喋るのをやめ、書面を睨め。無駄な声が消えたとき、真の決断が宿る。"
)

CASE_2_II_CRITICAL = (
    "なぜ会議は長引くのか？答えは単純、アジェンダなき雑談の垂れ流しだ！"
    "30分で下せるはずの意思決定を、保身と前例主義が引き延ばす。"
    "AI時代に生き残るのは、迷いなく決断を下す組織だけだ。今すぐ沈黙を破れ！会議室の議論を切り捨て、行動で結果を証明せよ！"
)

CASE_2_I_CRYSTALLINE = (
    "会議の生産性向上における結論は明白である。"
    "第一に、全会議は最長30分とし、事前配布のアジェンダなき開催を禁止する。"
    "第二に、合意形成ではなく迅速な意思決定を唯一の目的とする。"
    "AI時代に求められるのは冗長な議論ではなく、検証可能なアウトプットの速度である。"
)

CASE_2_GENESIS = (
    "会議の死。30分の枠は破綻した。"
    "アジェンダの空白に群がる形式主義が、思考の回路をショートさせる。決定権はどこにあるのか。"
    "AI時代の地平では、言語化の遅延は組織の壊死を意味する。即座に意思決定せよ。"
    "結論なき会議室の空気を、ソリトン的一撃で吹き飛ばすのだ。"
)

CASE_2_FACTS = ["会議", "30分", "アジェンダ", "意思決定", "AI時代"]

CASE_2_ESSAY = PresalesUseCase(
    id="case_2_essay",
    title="note・コラム・エッセイ — 「なぜ会議は長引くのか／AI時代の思考法」",
    category="思考オピニオン・Webメディア",
    before_text=CASE_2_BEFORE,
    core_facts=CASE_2_FACTS,
    variants={
        "III_Void": PresalesUseCaseVariant(
            mode="III_Void",
            title="① 漱石・鴎外風の静謐な『間』",
            subtitle="過度な修辞を廃し、余白と内省で思考の核心を突き刺す",
            after_text=CASE_2_III_VOID,
            analysis=compute_redline_diff_and_roi(CASE_2_BEFORE, CASE_2_III_VOID, "III_Void", CASE_2_FACTS),
        ),
        "II_Critical": PresalesUseCaseVariant(
            mode="II_Critical",
            title="② 芥川・梶井風の鮮烈な『臨界』",
            subtitle="切迫感と疾走感で読者の離脱を防ぎ、行動変容へ駆り立てる",
            after_text=CASE_2_II_CRITICAL,
            analysis=compute_redline_diff_and_roi(CASE_2_BEFORE, CASE_2_II_CRITICAL, "II_Critical", CASE_2_FACTS),
        ),
        "I_Crystalline": PresalesUseCaseVariant(
            mode="I_Crystalline",
            title="③ 結晶秩序相：論理対称性と結論先行",
            subtitle="1行目で結論を確定し、無駄のない対句構造で圧倒的な説得力を生む",
            after_text=CASE_2_I_CRYSTALLINE,
            analysis=compute_redline_diff_and_roi(CASE_2_BEFORE, CASE_2_I_CRYSTALLINE, "I_Crystalline", CASE_2_FACTS),
        ),
        "Genesis": PresalesUseCaseVariant(
            mode="Genesis",
            title="④ 未踏様式 Genesis：急激な跳躍と対比",
            subtitle="思考の停滞をソリトン的急転換で破壊する先鋭的文体",
            after_text=CASE_2_GENESIS,
            analysis=compute_redline_diff_and_roi(CASE_2_BEFORE, CASE_2_GENESIS, "Genesis", CASE_2_FACTS),
        ),
    },
)

# --- Case 3: B2B Pre-Sales Proposal / Executive Summary ---
CASE_3_BEFORE = (
    "弊社の次世代クラウド基盤導入プロジェクトのご提案においては、現状のシステム課題を多角的に分析し、将来的なコスト削減と業務効率化に貢献できる可能性があると考えられます。"
    "現行のレガシーシステム環境では運用保守にかかるリソース負担が非常に大きくなっており、障害発生時の復旧対応やセキュリティの維持管理という側面でも様々なリスクが存在していると言えるでしょう。"
    "本提案で提供する最新のクラウド基盤への移行を実施することにより、年間運用コストについて最大で42%の削減を達成することが十分に可能であると試算されております。"
    "またシステムの移行期間に関しましては段階的なアプローチを採用することで最短6ヶ月での完了を目指し、日常業務への影響を最小限に抑えながら確実な移行を実現してまいります。"
    "さらに可用性の観点においても業界最高水準であるSLA 99.99%を保証し、万全のサポート体制を整えることで貴社の持続的なビジネス成長とDX推進を強力に支援させていただければと考えております。"
)

CASE_3_I_CRYSTALLINE = (
    "次世代クラウド基盤への移行提案のエグゼクティブサマリーである。"
    "本プロジェクトは、貴社の年間運用コストを42%削減し、確実な高可用性を確立する。"
    "移行期間は最短6ヶ月で完了する。可用性は業界最高水準のSLA 99.99%を厳密に保証する。"
    "費用対効果と業務継続性を両立させる最適解である。"
)

CASE_3_II_CRITICAL = (
    "レガシーの維持に資源を浪費するのは今すぐ停止すべきだ！"
    "新世代クラウド基盤への全面刷新により、運用コストを直ちに42%削ぎ落とす。"
    "移行期間はわずか6ヶ月。停止リスクを完全遮断し、SLA 99.99%を死守せよ！"
    "躊躇は機会損失である。攻めの投資で市場の主導権を奪回せよ！"
)

CASE_3_III_VOID = (
    "冗長なシステムを削ぎ落とす。"
    "新世代クラウド基盤への移行。年間コストの42%削減。"
    "段階的な移行期間は6ヶ月。静かに、確実に、SLA 99.99%の稼働を底で支える。"
    "無駄な保守は消え、堅牢な骨格だけが残る。"
)

CASE_3_GENESIS = (
    "基盤の再定義。現行の技術負債をゼロに回帰させる。"
    "次世代クラウド基盤への跳躍は、年間コスト42%削減という不連続な変革をもたらす。"
    "移行期間は6ヶ月。ソリトン的無停止遷移を経て、SLA 99.99%の超高密度稼働が静寂のなかに定着する。"
    "秩序はここに極まる。"
)

CASE_3_FACTS = ["クラウド基盤", "コスト", "42%", "移行期間", "6ヶ月", "SLA 99.99%"]

CASE_3_B2B_PROPOSAL = PresalesUseCase(
    id="case_3_b2b_proposal",
    title="B2B プリセールス提案書・エグゼクティブサマリー",
    category="法人営業提案・意思決定者サマリー",
    before_text=CASE_3_BEFORE,
    core_facts=CASE_3_FACTS,
    variants={
        "I_Crystalline": PresalesUseCaseVariant(
            mode="I_Crystalline",
            title="① 結晶秩序相：曖昧性ゼロ・結論先行",
            subtitle="数値と納期を即座に確定させ、多忙な役員の意思決定スピードを最大化",
            after_text=CASE_3_I_CRYSTALLINE,
            analysis=compute_redline_diff_and_roi(CASE_3_BEFORE, CASE_3_I_CRYSTALLINE, "I_Crystalline", CASE_3_FACTS),
        ),
        "II_Critical": PresalesUseCaseVariant(
            mode="II_Critical",
            title="② 鮮烈な臨界相：切迫感と競合差別化",
            subtitle="現行維持のリスクを白日の下に晒し、投資決断へのモメンタムを生む",
            after_text=CASE_3_II_CRITICAL,
            analysis=compute_redline_diff_and_roi(CASE_3_BEFORE, CASE_3_II_CRITICAL, "II_Critical", CASE_3_FACTS),
        ),
        "III_Void": PresalesUseCaseVariant(
            mode="III_Void",
            title="③ 静謐な間相：極限の骨格化",
            subtitle="営業トークの誇張を排し、質実剛健な技術的信頼感を訴求",
            after_text=CASE_3_III_VOID,
            analysis=compute_redline_diff_and_roi(CASE_3_BEFORE, CASE_3_III_VOID, "III_Void", CASE_3_FACTS),
        ),
        "Genesis": PresalesUseCaseVariant(
            mode="Genesis",
            title="④ 未踏様式 Genesis：パラダイムシフトの宣言",
            subtitle="既存の延長線上の提案を凌駕するビジョンと構造変革の提示",
            after_text=CASE_3_GENESIS,
            analysis=compute_redline_diff_and_roi(CASE_3_BEFORE, CASE_3_GENESIS, "Genesis", CASE_3_FACTS),
        ),
    },
)

PRESALES_SHOWCASE_CASES: dict[str, PresalesUseCase] = {
    "case_1_literary": CASE_1_LITERARY,
    "case_2_essay": CASE_2_ESSAY,
    "case_3_b2b_proposal": CASE_3_B2B_PROPOSAL,
}
