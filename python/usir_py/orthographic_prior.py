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


# Phase 36: narrative-channel statistics.  Dialogue spans 「...」 are replaced by one placeholder character so that
# short utterances do not enter the sentence-length denominator; the first fragment of a window and an unterminated
# last fragment are dropped (they are cut by the window boundary).  Universal character classes only (invariant 3).
NARR_NAMES = ("narr_log_mean_sent_len", "narr_log_p90_sent_len", "narr_commas_per_sent", "dialogue_char_ratio")
NARR_DIM = len(NARR_NAMES)
_DIALOGUE = re.compile(r"「[^「」]*」")


def _narrative_sentences(body: str) -> List[str]:
    parts = _SENT_SPLIT.split(_DIALOGUE.sub("□", body))
    if parts and not re.search(r"[。！？!?]$", parts[-1]):
        parts = parts[:-1]
    return [s for s in parts[1:] if len(s) >= 2]


def narrative_sentence_lengths(text: str) -> List[int]:
    return [len(s) for s in _narrative_sentences(re.sub(r"\s", "", text))]


def ortho_vector_narr(text: str) -> np.ndarray:
    body = re.sub(r"\s", "", text)
    n = max(len(body), 1)
    dia = sum(len(m) for m in _DIALOGUE.findall(body))
    sents = _narrative_sentences(body)
    if sents:
        L = np.asarray([len(s) for s in sents], dtype=float)
        commas = sum(s.count("、") for s in sents) / len(sents)
    else:
        narr = _DIALOGUE.sub("□", body)
        L = np.asarray([float(max(len(narr), 1))])
        commas = float(narr.count("、"))
    return np.array([
        math.log(max(float(L.mean()), 1.0)),
        math.log(max(float(np.percentile(L, 90)), 1.0)),
        commas,
        dia / n,
    ], dtype=float)


def ortho_matrix_narr(windows: Sequence[str]) -> np.ndarray:
    return np.array([ortho_vector_narr(w) for w in windows], dtype=float).reshape(-1, NARR_DIM)


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


def proposal_from_prior(author: Dict[str, float], reference: Sequence[Dict[str, float]], kappa: float = 2.0,
                        band: Sequence[float] = (0.55, 1.6), comma_slack: float = 1.0, hold_bias: float = -12.0,
                        push_bias: float = 3.0) -> Dict[str, float]:
    """Phase 31: decoder proposal parameters from an author's measured orthographic prior.

    The author's deviation from the atlas mean (unweighted over authors) is amplified by kappa:
      mu*  = exp(r_logL + kappa (logL_a - r_logL))            target mean sentence length (chars)
      c*   = max(0, r_c + kappa (c_a - r_c))                  target commas per sentence
      kanji_bias = clip(10 kappa (k_a - r_k), -0.5, 0.5)
    and mapped to the steady band: min = round(0.55 mu*), max = round(1.6 mu*), max_commas = max(1, round(c* + 1)).
    Deterministic; documented in docs/handoff/2026-11-06_phase31_genome_proposal.md.
    Phase 36 (docs/handoff/2026-11-06_phase36_pinpoint.md) exposes the band factors, the comma slack
    (max_commas = max(1, round(c* + comma_slack))) and the steady period hold / push biases; defaults reproduce Phase 31.
    """
    r_logL = float(np.mean([r["log_mean_sent_len"] for r in reference]))
    r_c = float(np.mean([r["commas_per_sent"] for r in reference]))
    r_k = float(np.mean([r["kanji_ratio"] for r in reference]))
    mu = math.exp(r_logL + kappa * (author["log_mean_sent_len"] - r_logL))
    c = max(0.0, r_c + kappa * (author["commas_per_sent"] - r_c))
    kb = float(np.clip(10.0 * kappa * (author["kanji_ratio"] - r_k), -0.5, 0.5))
    return {
        "target_sent_len": mu, "comma_target": c, "kanji_bias": kb, "comma_push": 1.5,
        "steady_min_chars": int(round(float(band[0]) * mu)), "steady_max_chars": int(round(float(band[1]) * mu)),
        "steady_max_commas": int(max(1, round(c + float(comma_slack)))), "kappa": float(kappa),
        "hold_bias": float(hold_bias), "push_bias": float(push_bias),
    }
