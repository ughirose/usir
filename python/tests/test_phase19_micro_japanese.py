"""Phase 19 micro-Japanese gates: kanji i-adjective 連用止め, demonstrative decay, short-prev noun-ending gate."""
from __future__ import annotations

import numpy as np
import pytest

from usir_py.realtime_decoder import (
    _FRAGMENT_RE,
    BoundaryBiasLogits,
    ClauseTwistedSMCDecoder,
    ContentRepLogits,
    _is_connective_end,
    case_frame_ok,
)

VOCAB = ["<eos>", "。", "、", "その", "それ", "この", "光", "が", "あ", "る", "先", "ほど", "の", "景", "彼", "は", "窓", "を", "見", "た"]
TID = {s: i for i, s in enumerate(VOCAB)}
N_CTX = 2


class MockTok:
    eos_token_id = 0

    def __init__(self):
        self._by_len = sorted(range(len(VOCAB)), key=lambda i: -len(VOCAB[i]))

    def __len__(self):
        return len(VOCAB)

    def encode(self, text, add_special_tokens=False):
        out, i = [], 0
        while i < len(text):
            for t in self._by_len:
                if t and text.startswith(VOCAB[t], i):
                    out.append(t)
                    i += len(VOCAB[t])
                    break
            else:
                raise ValueError(text[i])
        return out

    def decode(self, ids, skip_special_tokens=True):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return "".join(VOCAB[i] for i in ids if not (skip_special_tokens and i == 0))

    def batch_decode(self, seqs, skip_special_tokens=False):
        return [self.decode(s, skip_special_tokens=skip_special_tokens) for s in seqs]


TK = MockTok()


def _ids(*tails):
    return np.array([[0, 0] + TK.encode(t) for t in tails], dtype=np.int64)


def _scores(P):
    return np.zeros((P, len(VOCAB) + 2), dtype=np.float32)


@pytest.mark.parametrize("s", ["薄く", "白く", "高く", "窓が低く", "空が青く"])
def test_kanji_i_adjective_is_connective(s):
    assert _is_connective_end(s)
    assert _FRAGMENT_RE.search(s + "。")


@pytest.mark.parametrize("s", ["泣く", "置く", "行く", "書く", "届く", "吹く", "近く", "遠く"])
def test_godan_verb_and_noun_not_blocked(s):
    assert not _is_connective_end(s)
    assert not _FRAGMENT_RE.search(s + "。")


def test_existing_hiragana_ku_unchanged():
    assert _is_connective_end("よく") and _is_connective_end("なく")
    assert _FRAGMENT_RE.search("よく。")


def _gate_period_score(ctx_tail):
    proc = BoundaryBiasLogits(TK, N_CTX, [4], [0], [TID["。"]], [TID["、"]], gate_on=True, gate_val=-12.0, ctx_tail=ctx_tail)
    sc = _scores(1)
    proc(np.array([[0, 0]], dtype=np.int64), sc)
    return sc[0, TID["。"]]


def test_gate_bias_on_period_after_adj_renyou():
    assert _gate_period_score("空が青く") == -12.0
    assert _gate_period_score("本を置く") == 0.0


def _demo_proc(prefix):
    dec = ClauseTwistedSMCDecoder(dim=12, seed=0, demo_decay=True)
    dec.tokenizer = TK
    return ContentRepLogits(TK, N_CTX, [], {}, prefix, 1.0, 0.0, demo_ids=dec._demo_token_ids(), demo_pen1=4.0, demo_pen2=10.0)


def test_demo_token_ids_precomputed():
    dec = ClauseTwistedSMCDecoder(dim=12, seed=0, demo_decay=True)
    dec.tokenizer = TK
    assert dec._demo_token_ids() == {"それ": [TID["それ"]], "その": [TID["その"]], "この": [TID["この"]]}


def test_demo_decay_counts_per_demonstrative():
    sc = _scores(1)
    _demo_proc("")(_ids("光"), sc)
    assert np.all(sc == 0.0)
    sc = _scores(1)
    _demo_proc("その光が")(_ids("光"), sc)
    assert sc[0, TID["その"]] == -4.0 and sc[0, TID["それ"]] == 0.0 and sc[0, TID["この"]] == 0.0
    sc = _scores(1)
    _demo_proc("その光が")(_ids("その"), sc)           # the row tail counts too -> 2 occurrences
    assert sc[0, TID["その"]] == -10.0
    sc = _scores(1)
    _demo_proc("その光。それが")(_ids("その"), sc)
    assert sc[0, TID["その"]] == -10.0 and sc[0, TID["それ"]] == -4.0


def test_flags_default_off():
    dec = ClauseTwistedSMCDecoder(dim=12, seed=0)
    assert dec.demo_decay is False and dec.short_prev_gate is False


def test_case_frame_ok_prev_len():
    # unchanged without prev_len
    assert not case_frame_ok("先ほどの") and case_frame_ok("先ほどの光景") and case_frame_ok("彼は窓を見た")
    assert case_frame_ok("先ほどの光景", prev_len=20)
    # short previous sentence: a noun-ended sentence is rejected even with a subject; all other cases are unchanged
    assert not case_frame_ok("先ほどの光景", prev_len=6)
    assert not case_frame_ok("彼は窓の光景", prev_len=15)
    assert case_frame_ok("彼は窓を見た", prev_len=6)
    assert case_frame_ok("窓を見た", prev_len=6) == case_frame_ok("窓を見た")
    assert not case_frame_ok("彼は部屋の中", prev_len=6)


def _bb(tail0, prev_len, short):
    return BoundaryBiasLogits(TK, N_CTX, [5], [3.0], [TID["。"]], [TID["、"]], noun_req=True, sent_tail0=tail0,
                              case_frame_gate=True, prev_len=prev_len, short_prev_gate=short)


def test_boundary_short_prev_gate_threading():
    ids = _ids("光景")
    sc = _scores(1)
    _bb("先ほどの", 6, True)(ids, sc)
    assert sc[0, TID["。"]] < 0.0               # row bias suppressed + short-sentence penalty
    sc = _scores(1)
    _bb("先ほどの", 6, False)(ids, sc)
    assert sc[0, TID["。"]] > 0.0               # flag off: unchanged behaviour
