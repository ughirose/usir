"""U6 Content Rewrite vs Style Polish classifier with Numeral and Negation Guards.

Adheres strictly to the specification in Rust diff_harvester.rs and Stage B-6 requirements:
1. Numeral Guard:
   - Normalizes Japanese numerals (kanji, arabic, full-width, positional/place-value).
   - Allows orthographic variants with equal numerical value (e.g. 三年 -> 3年 as STYLE_POLISH).
   - Rejects numerical value changes (e.g. 三年前 -> 十年前 as CONTENT_REWRITE).
   - Closed-class numeral exclusions:
     * Idioms: 一通り, 一旦, 一応, 一緒, 一層, 一向, 一斉, 一般, 万一, 一途, 一見, 一部, 一様, 唯一, 一同, etc.
     * Adverbial 十分: when not immediately followed by counter/time suffix (間, 後, 前, etc.).
2. Negation Guard:
   - Detects verbal/auxiliary negation (ない, なかった, なく, ず, ぬ, ません, まい, ん, etc.).
   - Disregards lexical prefixes (不, 無, 非, 未) as sole polarity flipping evidence.
   - Exempts double-negation constructions (e.g. ないわけではない) from negation-flip flagging.
   - Flags polarity inversion (e.g. 泣いた -> 泣かなかった) as CONTENT_REWRITE.
   - Closed-class negation exclusions:
     * Adverb 先ず / まず.
     * Epistemic conjecture modalities: 違いない / ちがいない / かもしれない / かも知れない (and inflections).
     * Lexical adjective stems (少, 切, 危, 汚, 儚) + ない/なかった/なく/なけれ.

Calibration Scope Note:
- calibration_scope: "micro_polish_only"
- Provisional thresholds: (min_jaccard=0.35, max_char_change=0.45).
- IMPORTANT: マクロ改稿（Class B）では Jaccard 系ゲートは Recall 0.4062 に低下するため使用しない（ミクロ推敲専用）。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Calibration Scope Metadata
CALIBRATION_SCOPE = "micro_polish_only"
CHAR_CHANGE_PLATEAU_NOTE = (
    "max_char_change >= 0.55 forms a 7-point flat plateau (Mean CV F1 = 0.7698); "
    "length ratio is effectively unconstrained in micro-polish"
)
MACRO_REVISION_JACCARD_DISALLOWED_NOTE = (
    "マクロ改稿（Class B）では Jaccard 系ゲートは Recall 0.4062 に低下するため使用しない（ミクロ推敲専用）"
)

U6_PROFILES = {
    "uncalibrated_v1": {
        "min_jaccard": 0.55,
        "max_char_change": 0.25,
        "calibration_scope": "legacy_heuristic_38vectors",
        "note": "Default uncalibrated heuristic for 38-vector fixture backward compatibility.",
    },
    "provisional_micro_v2": {
        "min_jaccard": 0.25,
        "max_char_change": 0.85,
        "calibration_scope": CALIBRATION_SCOPE,
        "char_change_plateau_note": CHAR_CHANGE_PLATEAU_NOTE,
        "note": MACRO_REVISION_JACCARD_DISALLOWED_NOTE,
    },
}

PROVISIONAL_CALIBRATED_THRESHOLDS = U6_PROFILES["provisional_micro_v2"]

# Full-width to half-width mapping
_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")

_KANJI_DIGITS = {
    "〇": 0, "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
_LARGE_UNITS = [("兆", 1000000000000), ("億", 100000000), ("万", 10000)]

_NUMERAL_PATTERN = re.compile(r"[0-9０-９一二三四五六七八九〇零十百千万億兆]+")

# Closed-class numeral idioms starting with "一" or other kanji numerals
_IDIOM_PREFIX_EXCLUSIONS = (
    "一通り", "一旦", "一応", "一緒", "一層", "一向", "一斉",
    "一般", "一途", "一見", "一部", "一様", "一同", "万事",
    "百面相", "千差万別", "千載一遇", "四角", "三角", "二重",
    "万が一",
)

# Time counters/suffixes that indicate "十分" is temporal (10 minutes/parts) rather than adverbial "sufficiently"
_JUUPPUN_TIME_COUNTERS = (
    "間", "後", "前", "足らず", "あまり", "余り", "過", "すぎ", "過ぎ",
    "ほど", "ばかり", "程度", "おき", "置き", "以", "目",
)

_NUMERAL_EXCLUSIONS = {
    "一般", "万一", "一途", "一見", "一番", "一部", "一様", "唯一", "一同",
    "万事", "百面相", "千差万別", "千載一遇", "四角", "三角", "二重",
    "一通り", "一旦", "一応", "一緒", "一層", "一向", "一斉",
    "万が一",
}


def parse_japanese_numeral(token: str) -> Optional[int]:
    """Parse a single Japanese numeral string (kanji, arabic, or mixed) into an integer."""
    if not token or token in _NUMERAL_EXCLUSIONS:
        return None

    # Normalize full-width digits
    s = token.translate(_FULLWIDTH_DIGITS)
    if s.isdigit():
        return int(s)

    # Pure positional kanji (e.g., 一九八四 -> 1984)
    if all(c in _KANJI_DIGITS for c in s):
        val = 0
        for c in s:
            val = val * 10 + _KANJI_DIGITS[c]
        return val

    def _parse_small_section(sec: str) -> int:
        total = 0
        curr = 0
        has_curr = False
        i = 0
        while i < len(sec):
            c = sec[i]
            if c.isdigit():
                num_buf = ""
                while i < len(sec) and sec[i].isdigit():
                    num_buf += sec[i]
                    i += 1
                curr = int(num_buf)
                has_curr = True
                continue
            elif c in _KANJI_DIGITS:
                curr = _KANJI_DIGITS[c]
                has_curr = True
            elif c in _SMALL_UNITS:
                u = _SMALL_UNITS[c]
                total += (curr if has_curr else 1) * u
                curr = 0
                has_curr = False
            else:
                return 0
            i += 1
        total += curr
        return total

    total_val = 0
    rem = s
    for unit_char, unit_val in _LARGE_UNITS:
        if unit_char in rem:
            parts = rem.split(unit_char, 1)
            high_str = parts[0]
            high_val = _parse_small_section(high_str) if high_str else 1
            total_val += high_val * unit_val
            rem = parts[1]

    if rem:
        total_val += _parse_small_section(rem)

    return total_val


def extract_numerals_with_spans(text: str) -> List[Tuple[int, Tuple[int, int]]]:
    """Extract list of (parsed numeral value, (start_idx, end_idx)) from text in sequential order.

    Excludes:
    - Fixed idioms: 一通り, 一旦, 一応, 一緒, 一層, 一向, 一斉, 万一, 唯一, 一番 (not 一番目), etc.
    - Adverbial 十分: when not immediately followed by counter/time suffix (間, 後, 前, etc.)
      or fraction indicator (分の).
    """
    results: List[Tuple[int, Tuple[int, int]]] = []
    for m in _NUMERAL_PATTERN.finditer(text):
        token = m.group(0)
        start, end = m.start(), m.end()

        # Check idiom prefix exclusions (e.g. 一通り, 一旦, 一応, 一緒, 一層, 一向, 一斉)
        rest_from_start = text[start:]
        if any(rest_from_start.startswith(p) for p in _IDIOM_PREFIX_EXCLUSIONS):
            continue

        # Check surrounding idioms like 唯一, 万一, 万が一
        if start > 0 and text[start - 1:end] in ("唯一", "万一"):
            continue
        if start >= 2 and text[start - 2:end] == "万が一":
            continue

        # "一番" vs "一番目": ordinal 一番目 is allowed as numeral 1, bare 一番 is adverbial
        if rest_from_start.startswith("一番") and not rest_from_start.startswith("一番目"):
            continue

        # Check adverbial 十分 vs temporal 十分 vs fraction 十分の一
        # In _NUMERAL_PATTERN, "十" is matched. If followed by "分":
        if token == "十" and text[end:].startswith("分"):
            after_juuppun = text[end + 1:]
            is_fraction = text[end:].startswith("分の")
            is_temporal = any(after_juuppun.startswith(c) for c in _JUUPPUN_TIME_COUNTERS)
            if not is_temporal and not is_fraction:
                # Adverbial 十分 (sufficiently) -> exclude
                continue
            else:
                # Temporal 十分 (10 minutes) or fraction 十分の一 -> value is 10
                results.append((10, (start, end)))
                continue

        val = parse_japanese_numeral(token)
        if val is not None:
            results.append((val, (start, end)))
    return results


def extract_numerals(text: str) -> List[int]:
    """Extract list of parsed numeral values from text in sequential order.

    Excludes:
    - Fixed idioms: 一通り, 一旦, 一応, 一緒, 一層, 一向, 一斉, 万一, 万が一, 唯一, 一番 (not 一番目), etc.
    - Adverbial 十分: when not immediately followed by counter/time suffix (間, 後, 前, etc.)
      or fraction indicator (分の).
    """
    return [val for val, _ in extract_numerals_with_spans(text)]


extract_numbers = extract_numerals


def mask_numeral_spans(text: str, spans: Sequence[Tuple[int, int]], placeholder: str = "_NUM_") -> str:
    """Replace numeral spans in text with a placeholder from right to left."""
    s = text
    for start, end in sorted(spans, key=lambda x: x[0], reverse=True):
        s = s[:start] + placeholder + s[end:]
    return s


def check_numeral_guard(t_before: str, t_after: str) -> Tuple[bool, str, List[int], List[int]]:
    """Numeral Guard: returns (passes_guard, reason, vals_before, vals_after).

    passes_guard is True if numeral values are identical (including orthographic changes like 三年 -> 3年).
    passes_guard is False if values differ (e.g. 三年前 -> 十年前).
    """
    nums_before = extract_numerals(t_before)
    nums_after = extract_numerals(t_after)

    if nums_before == nums_after:
        return True, "NUMERAL_MATCH", nums_before, nums_after
    else:
        return False, f"NUMERAL_MISMATCH: {nums_before} vs {nums_after}", nums_before, nums_after


# Negation regexes
_DOUBLE_NEGATION = re.compile(
    r"(?:ない|なく|ず|ぬ|ざる)(?:わけ|こと|はず|でも|も)*(?:では|じゃ|は|が|も)*(?:ない|なく|なかった|ありません|い)"
)

# Verbal/auxiliary negation (excluding lexical prefixes 不, 無, 非, 未)
# Specifically targeting negation endings and inflections
_VERBAL_NEGATION = re.compile(
    r"(?:(?<![不無非未])(?:"
    r"泣かなかった|行かなかった|見なかった|読まなかった|知らなかった|知らぬ|行かぬ|言わぬ|"
    r"なかった|ない|なく|ず|ぬ|ません|まい|ざる"
    r"))"
)

# Negation exclusion patterns:
# 1. Adverbs: 先ず / まず
# 2. Epistemic conjecture modality: 違いない / ちがいない / かもしれない / かも知れない (and inflections)
# 3. Adjectives with stems (少 / 切 / 危 / 汚 / 儚) + ない/なかった/なく/なけれ/なかろ
_NEGATION_EXCLUSIONS = re.compile(
    r"(?:"
    r"先ず|まず|"
    r"(?:違い|ちがい)(?:ない|なかった|なく|なかろ|なけれ)|"
    r"(?:かもしれ|かも知れ)(?:ない|なかった|なく|なかろ|なけれ|ぬ|ず)|"
    r"(?:[少切危汚儚]|すく|せつ|あぶ|きた|はか)(?:ない|なかった|なく|なけれ|なかろ)"
    r")"
)


def _mask_negation_exclusions(text: str) -> str:
    """Mask false-positive negation patterns before polarity analysis."""
    def _replacer(m: re.Match) -> str:
        s = m.group(0)
        # Replace with safe non-negation dummy characters of equal length
        return "〇" * len(s)

    return _NEGATION_EXCLUSIONS.sub(_replacer, text)


def extract_negation_polarity(sentence: str) -> int:
    """Returns 1 if sentence has net auxiliary/verbal negation, 0 if affirmative.

    Double negation cancels out to 0 (affirmative/neutral).
    Lexical prefixes alone do not trigger negation (0).
    Excludes adverbs (先ず/まず), modalities (違いない/かもしれない), and adjectives (少ない/切ない/危ない/汚い/儚い).
    """
    # Check for double negation on unmasked text first
    if _DOUBLE_NEGATION.search(sentence):
        return 0

    # Mask closed-class false positive exclusions
    masked = _mask_negation_exclusions(sentence)

    # Count verbal negations on masked sentence
    matches = _VERBAL_NEGATION.findall(masked)
    return 1 if len(matches) > 0 else 0


def check_negation_guard(t_before: str, t_after: str) -> Tuple[bool, str, int, int]:
    """Negation Guard: returns (passes_guard, reason, pol_before, pol_after).

    passes_guard is False if polarity flips between before and after (CONTENT_REWRITE).
    passes_guard is True if polarity is preserved.
    """
    # Split into rough sentences for comparison
    sents_before = [s for s in re.split(r"[。！？!?]", t_before) if s.strip()]
    sents_after = [s for s in re.split(r"[。！？!?]", t_after) if s.strip()]

    # Global polarity count
    pol_before = sum(extract_negation_polarity(s) for s in sents_before)
    pol_after = sum(extract_negation_polarity(s) for s in sents_after)

    # Check for polarity flip
    if pol_before != pol_after:
        return False, f"NEGATION_FLIP: {pol_before} vs {pol_after}", pol_before, pol_after

    return True, "NEGATION_PRESERVED", pol_before, pol_after


def u6_classify(
    t_before: str,
    t_after: str,
    min_jaccard: Optional[float] = None,
    max_char_change: Optional[float] = None,
    profile: Optional[str] = None,
) -> Dict[str, Any]:
    """U6 Classifier with Numeral Guard and Negation Guard.

    Returns dict with:
      - 'label': 'STYLE_POLISH' or 'CONTENT_REWRITE'
      - 'guard_passed': bool
      - 'rejection_reason': Optional[str]
      - 'jaccard': float
      - 'char_change_ratio': float
    """
    if profile is not None and profile in U6_PROFILES:
        prof_dict = U6_PROFILES[profile]
        eff_min_j = prof_dict["min_jaccard"] if min_jaccard is None else min_jaccard
        eff_max_c = prof_dict["max_char_change"] if max_char_change is None else max_char_change
    else:
        eff_min_j = 0.55 if min_jaccard is None else min_jaccard
        eff_max_c = 0.25 if max_char_change is None else max_char_change

    # 1. Numeral Guard
    res_b = extract_numerals_with_spans(t_before)
    res_a = extract_numerals_with_spans(t_after)
    nums_b = [v for v, _ in res_b]
    nums_a = [v for v, _ in res_a]

    if nums_b != nums_a:
        return {
            "label": "CONTENT_REWRITE",
            "guard_passed": False,
            "rejection_reason": f"NUMERAL_MISMATCH: {nums_b} vs {nums_a}",
            "jaccard": 0.0,
            "char_change_ratio": 1.0,
            "nums_before": nums_b,
            "nums_after": nums_a,
            "pol_before": 0,
            "pol_after": 0,
        }

    # 2. Negation Guard
    neg_pass, neg_reason, pol_b, pol_a = check_negation_guard(t_before, t_after)
    if not neg_pass:
        return {
            "label": "CONTENT_REWRITE",
            "guard_passed": False,
            "rejection_reason": neg_reason,
            "jaccard": 0.0,
            "char_change_ratio": 1.0,
            "nums_before": nums_b,
            "nums_after": nums_a,
            "pol_before": pol_b,
            "pol_after": pol_a,
        }

    # 3. Content Jaccard & Length Change
    # If numerals match and are present, normalize spans to placeholder _NUM_ to protect orthographic variation
    if len(nums_b) > 0:
        spans_b = [s for _, s in res_b]
        spans_a = [s for _, s in res_a]
        t_b_eval = mask_numeral_spans(t_before, spans_b, "_NUM_")
        t_a_eval = mask_numeral_spans(t_after, spans_a, "_NUM_")
    else:
        t_b_eval = t_before
        t_a_eval = t_after

    w1 = set(x for x in t_b_eval if "\u4e00" <= x <= "\u9fff")
    w2 = set(x for x in t_a_eval if "\u4e00" <= x <= "\u9fff")
    jaccard = len(w1 & w2) / max(len(w1 | w2), 1) if (w1 or w2) else 1.0
    char_change = abs(len(t_b_eval) - len(t_a_eval)) / max(len(t_b_eval), 1)

    if jaccard >= eff_min_j and char_change <= eff_max_c:
        label = "STYLE_POLISH"
        rejection_reason = None
    else:
        label = "CONTENT_REWRITE"
        rejection_reason = f"SURFACE_DIVERGENCE: jaccard={jaccard:.3f} < {eff_min_j} or len_change={char_change:.3f} > {eff_max_c}"

    return {
        "label": label,
        "guard_passed": True,
        "rejection_reason": rejection_reason,
        "jaccard": round(jaccard, 4),
        "char_change_ratio": round(char_change, 4),
        "nums_before": nums_b,
        "nums_after": nums_a,
        "pol_before": pol_b,
        "pol_after": pol_a,
    }
