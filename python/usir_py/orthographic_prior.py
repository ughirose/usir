"""Orthographic prior (Phase 30): author-agnostic grapheme statistics of a text window.

These are counts of universal character classes and punctuation only (no lexicon, no names), so they carry
no copyrighted expression and no target vocabulary (invariant 3).  Vector layout (ORTHO_NAMES):

  0 log_mean_sent_len   ln of the mean sentence length in characters (sentences split on 。！？!?)
  1 sent_len_cv         sigma_L / mu_L of sentence lengths
  2 commas_per_100      「、」 per 100 characters
  3 commas_per_sent     「、」 per sentence
  4 kanji_ratio         CJK ideographs / non-space characters
  5 hiragana_ratio      hiragana / non-space characters
  6 quote_ratio         「」『』 brackets / characters
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Sequence

import numpy as np

ORTHO_NAMES = ("log_mean_sent_len", "sent_len_cv", "commas_per_100", "commas_per_sent",
               "kanji_ratio", "hiragana_ratio", "quote_ratio")
ORTHO_DIM = len(ORTHO_NAMES)

_SENT_SPLIT = re.compile(r"(?<=[。！？!?])")
_KANJI = re.compile(r"[一-鿿々〆]")
_HIRA = re.compile(r"[ぁ-ゖ]")


def sentence_lengths(text: str) -> List[int]:
    body = re.sub(r"\s", "", text)
    return [len(s) for s in _SENT_SPLIT.split(body) if len(s) >= 2]


def ortho_vector(text: str) -> np.ndarray:
    body = re.sub(r"\s", "", text)
    n = max(len(body), 1)
    L = np.asarray(sentence_lengths(body), dtype=float)
    if L.size == 0:
        L = np.asarray([float(n)])
    mu = float(L.mean())
    cv = float(L.std() / mu) if mu > 0 else 0.0
    commas = body.count("、")
    return np.array([
        math.log(max(mu, 1.0)),
        cv,
        100.0 * commas / n,
        commas / max(L.size, 1),
        len(_KANJI.findall(body)) / n,
        len(_HIRA.findall(body)) / n,
        sum(body.count(c) for c in "「」『』") / n,
    ], dtype=float)


# Extended prior: frequency of single function kana (per 100 characters) and of the character before each sentence
# terminal.  Both are universal character classes (classic function-word stylometry at the character level).
FUNCTION_KANA = ("は", "が", "の", "に", "を", "と", "で", "も", "て", "た")
FINAL_CHARS = ("た", "る", "だ", "い", "う", "す", "ぬ", "か", "」")
ORTHO_EXT_NAMES = ORTHO_NAMES + tuple(f"kana_{c}" for c in FUNCTION_KANA) + tuple(f"final_{c}" for c in FINAL_CHARS) + ("final_other",)
ORTHO_EXT_DIM = len(ORTHO_EXT_NAMES)


def ortho_vector_ext(text: str) -> np.ndarray:
    body = re.sub(r"\s", "", text)
    n = max(len(body), 1)
    kana = [100.0 * body.count(c) / n for c in FUNCTION_KANA]
    sents = [s for s in _SENT_SPLIT.split(body) if len(s) >= 2]
    finals = [s.rstrip("。！？!?")[-1:] for s in sents]
    m = max(len(finals), 1)
    fin = [sum(f == c for f in finals) / m for c in FINAL_CHARS]
    fin.append(1.0 - sum(fin))
    return np.concatenate([ortho_vector(body), np.asarray(kana), np.asarray(fin)])


def ortho_matrix_ext(windows: Sequence[str]) -> np.ndarray:
    return np.array([ortho_vector_ext(w) for w in windows], dtype=float).reshape(-1, ORTHO_EXT_DIM)


def ortho_matrix(windows: Sequence[str]) -> np.ndarray:
    return np.array([ortho_vector(w) for w in windows], dtype=float).reshape(-1, ORTHO_DIM)


def ortho_summary(windows: Sequence[str]) -> Dict[str, float]:
    """Author-level prior: means over windows, plus the pooled sentence-length mean (characters)."""
    M = ortho_matrix(windows)
    allL = [x for w in windows for x in sentence_lengths(w)]
    out = {name: float(M[:, i].mean()) for i, name in enumerate(ORTHO_NAMES)}
    out["mean_sent_len_chars"] = float(np.mean(allL)) if allL else 0.0
    out["n_windows"] = int(len(M))
    return out
