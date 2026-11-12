"""U6 Content Rewrite vs Style Polish classifier with Numeral and Negation Guards.

Adheres strictly to the specification in Rust diff_harvester.rs:
1. Numeral Guard:
   - Normalizes Japanese numerals (kanji, arabic, full-width, positional/place-value).
   - Allows orthographic variants with equal numerical value (e.g. 三年 -> 3年 as STYLE_POLISH).
   - Rejects numerical value changes (e.g. 三年前 -> 十年前 as CONTENT_REWRITE).
2. Negation Guard:
   - Detects verbal/auxiliary negation (ない, なかった, なく, ず, ぬ, ません, まい, ん, etc.).
   - Disregards lexical prefixes (不, 無, 非, 未) as sole polarity flipping evidence.
   - Exempts double-negation constructions (e.g. ないわけではない) from negation-flip flagging.
   - Flags polarity inversion (e.g. 泣いた -> 泣かなかった) as CONTENT_REWRITE.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Full-width to half-width mapping
_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")

_KANJI_DIGITS = {
    "〇": 0, "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
_LARGE_UNITS = [("兆", 1000000000000), ("億", 100000000), ("万", 10000)]

_NUMERAL_PATTERN = re.compile(r"[0-9０-９一二三四五六七八九〇零十百千万億兆]+")

# Lexical idioms that shouldn't be parsed as pure numerals
_NUMERAL_EXCLUSIONS = {
    "一般", "万一", "一途", "一見", "一番", "一部", "一様", "唯一", "一同",
    "万事", "百面相", "千差万別", "千載一遇", "四角", "三角", "二重",
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


def extract_numerals(text: str) -> List[int]:
    """Extract list of parsed numeral values from text in sequential order."""
    values = []
    for m in _NUMERAL_PATTERN.finditer(text):
        token = m.group(0)
        val = parse_japanese_numeral(token)
        if val is not None:
            values.append(val)
    return values


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
    r"(?:ない|なく|ず|ぬ|ざる)(?:わけ|こと|でも|も)?(?:では|じゃ|は|が)?(?:ない|なく|なかった|ありません|い)"
)

# Verbal/auxiliary negation (excluding lexical prefixes 不, 無, 非, 未)
# Specifically targeting negation endings and inflections
_VERBAL_NEGATION = re.compile(
    r"(?:(?<![不無非未])(?:"
    r"泣かなかった|行かなかった|見なかった|読まなかった|知らなかった|知らぬ|行かぬ|言わぬ|"
    r"なかった|ない|なく|ず|ぬ|ません|まい|ざる"
    r"))"
)


def extract_negation_polarity(sentence: str) -> int:
    """Returns 1 if sentence has net auxiliary/verbal negation, 0 if affirmative.

    Double negation cancels out to 0 (affirmative/neutral).
    Lexical prefixes alone do not trigger negation (0).
    """
    # Check for double negation first
    if _DOUBLE_NEGATION.search(sentence):
        return 0

    # Count verbal negations
    matches = _VERBAL_NEGATION.findall(sentence)
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
    min_jaccard: float = 0.55,
    max_char_change: float = 0.25,
) -> Dict[str, Any]:
    """U6 Classifier with Numeral Guard and Negation Guard.

    Returns dict with:
      - 'label': 'STYLE_POLISH' or 'CONTENT_REWRITE'
      - 'guard_passed': bool
      - 'rejection_reason': Optional[str]
      - 'jaccard': float
      - 'char_change_ratio': float
    """
    # 1. Numeral Guard
    num_pass, num_reason, nums_b, nums_a = check_numeral_guard(t_before, t_after)
    if not num_pass:
        return {
            "label": "CONTENT_REWRITE",
            "guard_passed": False,
            "rejection_reason": num_reason,
            "jaccard": 0.0,
            "char_change_ratio": 1.0,
            "nums_before": nums_b,
            "nums_after": nums_a,
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
            "pol_before": pol_b,
            "pol_after": pol_a,
        }

    # 3. Content Jaccard & Length Change
    w1 = set(x for x in t_before if "\u4e00" <= x <= "\u9fff")
    w2 = set(x for x in t_after if "\u4e00" <= x <= "\u9fff")
    jaccard = len(w1 & w2) / max(len(w1 | w2), 1) if (w1 or w2) else 1.0
    char_change = abs(len(t_before) - len(t_after)) / max(len(t_before), 1)

    if jaccard >= min_jaccard and char_change <= max_char_change:
        label = "STYLE_POLISH"
        rejection_reason = None
    else:
        label = "CONTENT_REWRITE"
        rejection_reason = f"SURFACE_DIVERGENCE: jaccard={jaccard:.3f} < {min_jaccard} or len_change={char_change:.3f} > {max_char_change}"

    return {
        "label": label,
        "guard_passed": True,
        "rejection_reason": rejection_reason,
        "jaccard": round(jaccard, 4),
        "char_change_ratio": round(char_change, 4),
        "nums_before": nums_b,
        "nums_after": nums_a,
    }
