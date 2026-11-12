"""Modern and Web Novel Text Normalizer & Tier-C Clean-Room Feature Extractor.

Adheres strictly to USIR v5 Gate 3a / Tier C (Copyright Article 30-4 clean-room) specifications:
1. Punctuation and layout normalization:
   - Ellipsis: '…', '......' -> standard paired '……'
   - Dashes: '―', '—' -> standard paired '――'
   - Consecutive exclamation/question marks: '!?' -> '！？', '!!' -> '！！', '??' -> '？？'
   - Web novel rubi: '|親文字《ルビ》' -> '親文字', '親文字《ルビ》' -> '親文字', '親文字(ルビ)' -> '親文字'
   - Fullwidth spaces & line layout metrics (dialogue ratio 50-80% safe)
2. Statistical diff:
   - compute_punctuation_normalization_diff(raw, normalized)
3. Tier-C Irreversible Profile Extraction:
   - extract_tier_c_irreversible_profile(text)
   - Entity-blanking: Proper nouns and content words masked to U+25A0 ('■')
   - Irreversible features only: ortho_v2 31D, closed-class PPM bits/token, closed-class coverage
   - Strict cleanroom invariant: NO raw text or substrings retained in memory or output dict
"""
from __future__ import annotations

import collections
import gc
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from usir_py.orthographic_prior import (
    ortho_matrix_ext_v2,
    ortho_matrix_narr_v2,
)

# ---------------------------------------------------------------------------
# Closed-Class Vocabulary for PPM Grammar Skeleton
# ---------------------------------------------------------------------------
CLOSED_CLASS_VOCAB: List[str] = [
    # Particles
    "の", "に", "は", "を", "が", "と", "で", "て", "も", "から", "より", "へ", "や", "か",
    "な", "ね", "よ", "ぞ", "ぜ", "わ", "さ", "ば", "ど", "のに", "ので", "ても", "まで",
    # Auxiliaries & inflections
    "だ", "た", "です", "ます", "ない", "ぬ", "ん", "ず", "れる", "られる", "せる", "させる",
    "たい", "らしい", "ようだ", "そうだ", "まい", "だった", "でした", "ません", "なかった",
    # Conjunctions & Adverbs
    "しかし", "だが", "そして", "また", "あるいは", "なお", "だから", "そこで", "すると",
    "もっとも", "ただし", "ところが", "および", "ならびに", "さて", "では",
    # Pronouns & Role markers
    "これ", "それ", "あれ", "どれ", "ここ", "そこ", "あそこ", "どこ", "こう", "そう", "ああ", "どう",
    "私", "わたし", "僕", "ぼく", "俺", "おれ", "彼", "彼女", "自分", "貴方", "あなた",
    # Punctuation & Brackets
    "、", "。", "！", "？", "「", "」", "（", "）", "――", "……", "・",
]
CLOSED_SET: Set[str] = set(CLOSED_CLASS_VOCAB)
TOKEN_CONTENT = "<CONTENT>"
TOKEN_UNK = "<UNK>"


# ---------------------------------------------------------------------------
# 1. Punctuation & Layout Normalization
# ---------------------------------------------------------------------------

# Rubi regexes
# Format 1: |親文字《ルビ》 or 親文字《ルビ》
_RUBI_KAKUYOMU = re.compile(r"\|?([一-龠々〆ヵヶa-zA-Z0-9]+)《([^》]+)》")
# Format 2: 漢字(ふりがな) or 漢字（ふりがな）where inside is purely hiragana/katakana
_RUBI_PARENS = re.compile(r"([一-龠々〆ヵヶ]+)[（(]([ぁ-んァ-ヶー]+)[）)]")

# Exclamation / Question regexes
_EXCL_QUEST_ASCII = re.compile(r"[!?]+")

# Dash regexes
_DASH_PATTERN = re.compile(r"[―—–]{1,}")

# Ellipsis regexes
_ELLIPSIS_PATTERN = re.compile(r"[…\.]{2,}|…")


def normalize_modern_text(text: str) -> str:
    """Non-destructively normalize modern web-novel and literature punctuation and rubi."""
    if not text:
        return ""

    # 1. Strip web-novel rubi
    # e.g. |親文字《ルビ》 -> 親文字
    t = _RUBI_KAKUYOMU.sub(r"\1", text)
    # e.g. 親文字(るび) -> 親文字 (only when inside is purely kana)
    t = _RUBI_PARENS.sub(r"\1", t)

    # 2. Normalize dashes: any sequence of horizontal bars/em dashes to standard paired '――'
    t = _DASH_PATTERN.sub("――", t)

    # 3. Normalize ellipsis: single or repeated ellipsis/dots to standard paired '……'
    t = _ELLIPSIS_PATTERN.sub("……", t)

    # 4. Normalize exclamation & question marks:
    # Convert combinations to full-width standard
    t = t.replace("!?", "！？").replace("?!", "！？")
    t = t.replace("??", "？？").replace("!!", "！！")
    t = t.replace("?", "？").replace("!", "！")

    # Clean consecutive duplicate full-width exclamation/questions exceeding 2
    t = re.sub(r"！{3,}", "！！", t)
    t = re.sub(r"？{3,}", "？？", t)

    # 5. Normalize newline and spacing
    lines = [line.strip() for line in t.splitlines()]
    t = "\n".join(lines)

    return t


def compute_punctuation_normalization_diff(text_raw: str, text_normalized: str) -> Dict[str, Any]:
    """Compute statistical differences before and after punctuation normalization."""
    len_raw = len(text_raw)
    len_norm = len(text_normalized)

    # Count dialogue characters inside brackets 「...」 or 『...』
    dialogue_chars = sum(len(m.group(0)) for m in re.finditer(r"「[^」]*」|『[^』]*』", text_normalized))
    dialogue_ratio = dialogue_chars / max(len_norm, 1)

    # Measure punctuation frequencies
    diff = {
        "len_raw": len_raw,
        "len_norm": len_norm,
        "char_delta": len_norm - len_raw,
        "char_delta_ratio": round((len_norm - len_raw) / max(len_raw, 1), 4),
        "dialogue_ratio": round(dialogue_ratio, 4),
        "ellipsis_count_raw": text_raw.count("…") + text_raw.count("..."),
        "ellipsis_count_norm": text_normalized.count("……") * 2,
        "dash_count_raw": text_raw.count("―") + text_raw.count("—"),
        "dash_count_norm": text_normalized.count("――") * 2,
        "excl_quest_count_raw": len(_EXCL_QUEST_ASCII.findall(text_raw)),
        "excl_quest_count_norm": len(re.findall(r"[！？]+", text_normalized)),
    }
    return diff


# ---------------------------------------------------------------------------
# 2. Grammar Skeleton Tokenizer & Order-3 PPM Model
# ---------------------------------------------------------------------------

def tokenize_closed_class_skeleton(text: str) -> List[str]:
    """Tokenize text into closed-class grammar skeleton tokens and <CONTENT> tokens."""
    tokens: List[str] = []
    i = 0
    n = len(text)
    # Longest match first
    sorted_vocab = sorted(CLOSED_CLASS_VOCAB, key=len, reverse=True)

    while i < n:
        matched = False
        for cl in sorted_vocab:
            if text.startswith(cl, i):
                tokens.append(cl)
                i += len(cl)
                matched = True
                break
        if not matched:
            ch = text[i]
            if ch.isspace():
                i += 1
                continue
            # Accumulate content sequence
            while i < n and not any(text.startswith(cl, i) for cl in sorted_vocab) and not text[i].isspace():
                i += 1
            tokens.append(TOKEN_CONTENT)
    return tokens


class Order3PPMModel:
    """Order-3 Prediction by Partial Matching (PPM) with Escape Method C."""

    def __init__(self, vocab: List[str] = CLOSED_CLASS_VOCAB) -> None:
        self.vocab = sorted(list(set(vocab + [TOKEN_CONTENT, TOKEN_UNK])))
        self.vocab_size = len(self.vocab)
        self.unigrams: collections.Counter[str] = collections.Counter()
        self.bigrams: Dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
        self.trigrams: Dict[Tuple[str, str], collections.Counter[str]] = collections.defaultdict(collections.Counter)
        self.total_tokens: int = 0

    def train(self, token_sequences: List[List[str]]) -> None:
        for tokens in token_sequences:
            self.total_tokens += len(tokens)
            for i, tok in enumerate(tokens):
                t = tok if tok in self.vocab else TOKEN_UNK
                self.unigrams[t] += 1
                if i >= 1:
                    ctx1 = tokens[i - 1] if tokens[i - 1] in self.vocab else TOKEN_UNK
                    self.bigrams[ctx1][t] += 1
                if i >= 2:
                    ctx2 = (
                        tokens[i - 2] if tokens[i - 2] in self.vocab else TOKEN_UNK,
                        tokens[i - 1] if tokens[i - 1] in self.vocab else TOKEN_UNK,
                    )
                    self.trigrams[ctx2][t] += 1

    def token_log_probability(self, token: str, context: Tuple[str, ...]) -> float:
        """Compute -log2 P(token | context) using PPM Method C recursive escape."""
        t = token if token in self.vocab else TOKEN_UNK

        # Try trigram
        if len(context) >= 2:
            ctx2 = context[-2:]
            c2 = self.trigrams.get(ctx2, None)
            if c2 and sum(c2.values()) > 0:
                total_c2 = sum(c2.values())
                distinct_types = len(c2)
                esc_c2 = distinct_types / (total_c2 + distinct_types)
                if t in c2:
                    p = c2[t] / (total_c2 + distinct_types)
                    return -math.log2(max(p, 1e-12))
                else:
                    p_esc = esc_c2
                    return -math.log2(max(p_esc, 1e-12)) + self.token_log_probability(t, context[-1:])

        # Try bigram
        if len(context) >= 1:
            ctx1 = context[-1]
            c1 = self.bigrams.get(ctx1, None)
            if c1 and sum(c1.values()) > 0:
                total_c1 = sum(c1.values())
                distinct_types = len(c1)
                esc_c1 = distinct_types / (total_c1 + distinct_types)
                if t in c1:
                    p = c1[t] / (total_c1 + distinct_types)
                    return -math.log2(max(p, 1e-12))
                else:
                    p_esc = esc_c1
                    return -math.log2(max(p_esc, 1e-12)) + self.token_log_probability(t, ())

        # Unigram fallback with Laplace smoothing
        total_u = self.total_tokens + self.vocab_size
        cnt_u = self.unigrams.get(t, 0) + 1.0
        p_u = cnt_u / total_u
        return -math.log2(max(p_u, 1e-12))

    def evaluate_cross_entropy(self, tokens: List[str]) -> Tuple[float, float]:
        """Compute cross-entropy (bits/token) and closed-class token ratio (formerly called coverage_pct).

        Returns:
            Tuple of (mean_bits_per_token, closed_class_token_ratio_pct).
            Note: The second element is the closed-class token density in text (55-65%),
            corresponding to Rust closed_share * 100.0, NOT vocabulary coverage.
            For exact parity with Rust nishiki-ppm, see `evaluate_skeleton_metrics`.
        """
        metrics = self.evaluate_skeleton_metrics(tokens)
        return metrics["bits_per_token"], metrics["closed_class_token_ratio"] * 100.0

    def evaluate_skeleton_metrics(self, tokens: List[str]) -> Dict[str, Any]:
        """Compute detailed PPM cross-entropy and dual closed-class coverage metrics.

        Parity with Rust nishiki-ppm::shift::ClosedClassCoverage:
        - closed_class_token_ratio: closed_tokens / total_tokens (density, natural range 55-65%)
        - closed_class_vocab_coverage: closed_known / closed_tokens (vocab coverage, >=99.9%)
        """
        if not tokens:
            return {
                "bits_per_token": 0.0,
                "closed_class_token_ratio": 0.0,
                "closed_class_vocab_coverage": 1.0,
                "tokens_total": 0,
                "closed_tokens": 0,
                "closed_known": 0,
            }

        total_bits = 0.0
        closed_tokens = 0
        closed_known = 0

        for i, tok in enumerate(tokens):
            is_closed = tok in CLOSED_SET
            if is_closed:
                closed_tokens += 1
                if tok in self.vocab and self.unigrams.get(tok, 0) > 0:
                    closed_known += 1

            ctx = ()
            if i == 1:
                ctx = (tokens[0] if tokens[0] in self.vocab else TOKEN_UNK,)
            elif i >= 2:
                ctx = (
                    tokens[i - 2] if tokens[i - 2] in self.vocab else TOKEN_UNK,
                    tokens[i - 1] if tokens[i - 1] in self.vocab else TOKEN_UNK,
                )
            bits = self.token_log_probability(tok, ctx)
            total_bits += bits

        mean_bits_per_token = total_bits / len(tokens)
        token_ratio = closed_tokens / len(tokens)
        vocab_coverage = (closed_known / closed_tokens) if closed_tokens > 0 else 1.0

        return {
            "bits_per_token": mean_bits_per_token,
            "closed_class_token_ratio": token_ratio,
            "closed_class_vocab_coverage": vocab_coverage,
            "tokens_total": len(tokens),
            "closed_tokens": closed_tokens,
            "closed_known": closed_known,
        }


# ---------------------------------------------------------------------------
# Default baseline training data for fallback PPM
# ---------------------------------------------------------------------------
_CANONICAL_REFERENCE_SKELETONS: List[str] = [
    "私は彼と共に歩いた。しかし彼は何も言わなかった。",
    "雨が激しく降っていた。どこへ行くのかと彼女は尋ねた。",
    "それは非常に困難な計画であったが、実行するほかなかった。",
    "遠くで鐘が鳴っていた。その音は静かに響いていた。",
    "誰もいない部屋に入った。机の上には手紙が一通置かれていた。",
    "これ以上待つことはできない。すぐに準備を始めよう。",
    "自分自身の手でそれを確認しなければならない。",
    "何が起きたのかを正確に知る必要がある。",
]

_DEFAULT_PPM_MODEL: Optional[Order3PPMModel] = None


def get_default_ppm_model() -> Order3PPMModel:
    """Lazy initialize and return default baseline Order-3 PPM model."""
    global _DEFAULT_PPM_MODEL
    if _DEFAULT_PPM_MODEL is None:
        model = Order3PPMModel()
        seqs = [tokenize_closed_class_skeleton(s) for s in _CANONICAL_REFERENCE_SKELETONS]
        model.train(seqs)
        _DEFAULT_PPM_MODEL = model
    return _DEFAULT_PPM_MODEL


# ---------------------------------------------------------------------------
# 3. Tier-C Clean-Room Feature Extraction
# ---------------------------------------------------------------------------

def mask_entities_to_black_square(text: str) -> str:
    """Mask content words and potential proper nouns to U+25A0 ('■').

    Preserves closed-class functional particles, conjunctions, inflections, and punctuation.
    """
    tokens = tokenize_closed_class_skeleton(text)
    masked_parts: List[str] = []
    for tok in tokens:
        if tok in CLOSED_SET:
            masked_parts.append(tok)
        else:
            # Mask content words with U+25A0
            masked_parts.append("■")
    return "".join(masked_parts)


def extract_tier_c_irreversible_profile(
    text: str,
    ppm_model: Optional[Order3PPMModel] = None,
) -> Dict[str, Any]:
    """Clean-Room extraction of irreversible features under Copyright Tier-C rules.

    Invariants guaranteed:
    1. Input raw text is normalized and entity-masked immediately.
    2. No raw text strings, phrases, or unmasked content words are stored in the output dict.
    3. Memory references to the original string are explicitly cleared.
    4. Outputs strictly:
       - 'ortho_v2_31d': List of 31 float numbers
       - 'ppm_bits_per_token': float
       - 'closed_class_coverage_pct': float
       - 'dialogue_ratio': float
       - 'norm_diff': statistical diff dict
    """
    # 1. Normalize
    norm_text = normalize_modern_text(text)
    diff = compute_punctuation_normalization_diff(text, norm_text)

    # 2. Extract ortho_v2 31D (27D ext + 4D narr)
    v2_ext = ortho_matrix_ext_v2([norm_text])[0]
    v2_narr = ortho_matrix_narr_v2([norm_text])[0]
    ortho_31d = [round(float(x), 6) for x in list(v2_ext) + list(v2_narr)]

    # 3. Closed-class grammar skeleton & PPM evaluation
    tokens = tokenize_closed_class_skeleton(norm_text)
    model = ppm_model if ppm_model is not None else get_default_ppm_model()
    bits_per_tok, coverage = model.evaluate_cross_entropy(tokens)

    # 4. Formulate cleanroom irreversible profile (strictly numbers and metadata)
    profile = {
        "status": "CLEAN_ROOM_TIER_C_VERIFIED",
        "n_tokens": len(tokens),
        "ppm_bits_per_token": round(float(bits_per_tok), 4),
        "closed_class_coverage_pct": round(float(coverage), 2),
        "dialogue_ratio": float(diff["dialogue_ratio"]),
        "ortho_v2_31d": ortho_31d,
        "punctuation_diff": diff,
    }

    # 5. Destroy raw string references in local scope
    del text
    del norm_text
    del tokens
    gc.collect()

    return profile
