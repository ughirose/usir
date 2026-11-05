"""Surface text metrics for rhythm / punctuation / short-range collapse (Phase 23). Pure functions, no model needed."""

from __future__ import annotations

import collections
import re
from typing import Dict, List

_SENT_END = "。！？!?"
_SENT_RE = re.compile("[^" + _SENT_END + "]+[" + _SENT_END + "]*")
_WORD_CH = "[一-鿿ァ-ヺ々ー]"
_PAIR_RE = re.compile(_WORD_CH + "{2}")
# Heuristic only: a 終止/連体-ambiguous ending (た/る/い) followed by '。' and then 1-4 kanji + の/が/を/は/に
# (also kanji + くの, so 「残した。多くの工具」 matches).
# True 連体 splits (「残した。多くの工具」) match, but so do legitimate sentence pairs ("閉じた。彼が...").
# Treat the count as an upper-bound suspect count, not a precision figure.
_ADNOMINAL_SPLIT_RE = re.compile("(?<=[たるい])。(?=[一-鿿]{1,4}(?:く?の|[がをはに]))")


def split_sentences(text: str) -> List[str]:
    """Sentences with whitespace removed; a sentence ends at 。！？!? (closing quote chars stay in the next piece's
    head only if no terminator preceded them; empty pieces are dropped)."""
    out = []
    for m in _SENT_RE.finditer(re.sub(r"\s+", "", text)):
        s = m.group(0).strip()
        if s:
            out.append(s)
    return out


def max_kanji_repeat(sent: str) -> int:
    """Max count of one kanji inside `sent`."""
    cnt = collections.Counter(ch for ch in sent if "一" <= ch <= "鿿")
    return max(cnt.values()) if cnt else 0


def max_word2_repeat(sent: str) -> int:
    """Max count of an overlapping 2-char kanji/katakana string inside `sent`."""
    cnt = collections.Counter(sent[i:i + 2] for i in range(len(sent) - 1) if _PAIR_RE.fullmatch(sent[i:i + 2]))
    return max(cnt.values()) if cnt else 0


def mean_sentence_len(text: str) -> float:
    s = split_sentences(text)
    return float(sum(len(x) for x in s) / len(s)) if s else 0.0


def comma_per_100(text: str) -> float:
    n = len(re.sub(r"\s+", "", text))
    return float(text.count("、") * 100.0 / n) if n else 0.0


def max_kanji_repeat_in_sentence(text: str) -> int:
    return max((max_kanji_repeat(s) for s in split_sentences(text)), default=0)


def repeat_2gram_in_sentence_max(text: str) -> int:
    return max((max_word2_repeat(s) for s in split_sentences(text)), default=0)


def adnominal_split_suspects(text: str) -> int:
    return len(_ADNOMINAL_SPLIT_RE.findall(re.sub(r"\s+", "", text)))


def text_metrics(text: str) -> Dict[str, float]:
    return {
        "mean_sentence_len": round(mean_sentence_len(text), 2),
        "comma_per_100": round(comma_per_100(text), 2),
        "max_kanji_repeat_in_sentence": max_kanji_repeat_in_sentence(text),
        "repeat_2gram_in_sentence_max": repeat_2gram_in_sentence_max(text),
        "adnominal_split_suspects": adnominal_split_suspects(text),
    }
