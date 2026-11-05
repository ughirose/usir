"""Unit tests for the module-level logits processors of the realtime decoder (numpy arrays, no torch needed)."""

from __future__ import annotations

import numpy as np
import pytest

from usir_py.realtime_decoder import (
    BoundaryBiasLogits,
    ClauseTwistedSMCDecoder,
    ContentRepLogits,
    ContextNgramBan,
    NarrationStyleLogitsMask,
    _is_connective_end,
    _is_terminal_end,
    case_frame_ok,
    count_non_jis_kanji,
    has_non_jis_kanji,
    kanji_bigrams,
    lex_bigram_targets,
    narration_row_flags,
)

VOCAB = [
    "<eos>", "。", "、", "た", "まし", "ょう", "ん", "です", "ます", "静", "寂", "山", "の", "は", "て", "窓",
    "が", "雨", "降", "っ", "空", "青", "い", "「", "」", "寂れ", "寂び", "に", "を", "あ", "る", "広", "外", "や", "と",
]
TID = {s: i for i, s in enumerate(VOCAB)}


class MockTokenizer:
    eos_token_id = 0

    def __init__(self, vocab=VOCAB):
        self.vocab = list(vocab)
        self._by_len = sorted(range(len(self.vocab)), key=lambda i: -len(self.vocab[i]))

    def __len__(self):
        return len(self.vocab)

    def encode(self, text, add_special_tokens=False):
        out, i = [], 0
        while i < len(text):
            for t in self._by_len:
                s = self.vocab[t]
                if t != 0 and s and text.startswith(s, i):
                    out.append(t)
                    i += len(s)
                    break
            else:
                raise ValueError(f"cannot encode {text[i]!r}")
        return out

    def decode(self, ids, skip_special_tokens=True):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return "".join(self.vocab[i] for i in ids if not (skip_special_tokens and i == 0))

    def batch_decode(self, seqs, skip_special_tokens=False):
        if hasattr(seqs, "tolist"):
            seqs = seqs.tolist()
        return [self.decode(s, skip_special_tokens=skip_special_tokens) for s in seqs]

    def convert_tokens_to_ids(self, tok):
        return self.vocab.index(tok) if tok in self.vocab else -1


TK = MockTokenizer()
N_CTX = 2


def _ids(*tails):
    """ids matrix: 2 context tokens + each row's tail (must have the same token count)."""
    rows = [[0, 0] + TK.encode(t) for t in tails]
    return np.array(rows, dtype=np.int64)


def _scores(P, V=None):
    return np.zeros((P, V or len(VOCAB) + 3), dtype=np.float32)


def _decoder():
    dec = ClauseTwistedSMCDecoder(dim=12, seed=0)
    dec.tokenizer = TK
    return dec


# ---------------------------------------------------------------------------------------------------------------
def test_narration_row_flags():
    narr, stem = narration_row_flags(["", "「静", "静」", "雨まし", "ます"], 0)
    assert narr == [True, False, True, True, True]
    assert stem == [False, False, False, True, False]
    narr, _ = narration_row_flags(["", "」"], 1)
    assert narr == [False, True]


def test_narration_mask_bias_stem_and_oob_ids():
    dec = _decoder()
    bad, post = dec._forbidden_token_ids()
    assert TID["です"] in bad and TID["ます"] in bad
    assert set(post) == {TID["た"], TID["ょう"], TID["ん"]}
    V = len(VOCAB) + 3
    proc = NarrationStyleLogitsMask(TK, N_CTX, 0, list(bad) + [10 ** 6], list(post) + [10 ** 6], -15.0)
    ids = np.array([[0, 0, TID["静"], TID["静"]], [0, 0, TID["「"], TID["静"]], [0, 0, TID["静"], TID["まし"]]], dtype=np.int64)
    sc = _scores(3, V)
    out = proc(ids, sc)
    assert out is sc
    for t in bad:
        assert sc[0, t] == -15.0 and sc[1, t] == 0.0 and sc[2, t] == -15.0
    for t in post:
        assert sc[0, t] == 0.0 and sc[1, t] == 0.0 and sc[2, t] == -15.0
    untouched = [i for i in range(V) if i not in set(bad) | set(post)]
    assert np.all(sc[:, untouched] == 0.0)


# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("s,conn", [
    ("窓が", True), ("雨は", True), ("降って", True), ("なく", True), ("吹く", False), ("降った", False),
    ("ながら", True), ("ので", True), ("", False), ("青い", False),
])
def test_is_connective_end(s, conn):
    assert _is_connective_end(s) is conn


@pytest.mark.parametrize("s,term", [
    ("降った", True), ("青い", True), ("降る", True), ("雨", True), ("カメラ", True), ("窓が", False), ("", False), ("降って", False),
])
def test_is_terminal_end(s, term):
    assert _is_terminal_end(s) is term


def test_boundary_finality_gate():
    # step 3, cap 4 -> t_rel .75 -> scale = clamp((.75-.2)/.4,0,1)*1.6-.6 = 1.0 -> pb = ph_period
    proc = BoundaryBiasLogits(TK, N_CTX, [4, 4], [0, 0], [TID["。"]], [TID["、"]], ph_period=4.0, gate_on=True, gate_val=-12.0)
    ids = _ids("雨はて", "降った")
    sc = _scores(2)
    proc(ids, sc)
    assert sc[0, TID["。"]] == -12.0
    assert sc[1, TID["。"]] == pytest.approx(4.0)
    assert sc[0, TID["、"]] == 0.0 and sc[1, TID["、"]] == 0.0


# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("s,ok", [
    ("やがて窓の外に広がった", False),
    ("窓の外には", False),
    ("雨が降った", True),
    ("空は青い", True),
    ("それは", True),
    ("これも", True),
    ("静寂", True),
    ("静寂。", False),
    ("カメラ", True),
    ("カメラが", True),
    ("", False),
])
def test_case_frame_ok(s, ok):
    assert case_frame_ok(s) is ok


def test_boundary_short_phase_case_frame_gate():
    # step 4 / cap 5 -> scale 1.0 -> pb = ph_period
    kw = dict(period_ids=[TID["。"]], comma_ids=[], ph_period=8.0, noun_req=True, sent_tail0="")
    ids = _ids("広がった", "雨が降る", "あるあい")
    on = BoundaryBiasLogits(TK, N_CTX, [5] * 3, [0] * 3, case_frame_gate=True, **kw)
    sc = _scores(3)
    on(ids, sc)
    # rows without a case frame and < 18 chars additionally get the -6.0 case-frame penalty
    assert [float(sc[r, TID["。"]]) for r in range(3)] == [-6.0, 8.0, -6.0]
    off =BoundaryBiasLogits(TK, N_CTX, [5] * 3, [0] * 3, case_frame_gate=False, **kw)
    sc = _scores(3)
    off(ids, sc)
    # legacy: any kanji / katakana in the sentence counts as a noun
    assert [float(sc[r, TID["。"]]) for r in range(3)] == [8.0, 8.0, 0.0]


def test_boundary_row_bias_respects_case_frame_gate():
    # Shock-regime per-row short bias (+3 on '。', -3 on '、') is withheld for rows without a case frame
    ids = _ids("広がった", "雨が降る")
    kw = dict(period_ids=[TID["。"]], comma_ids=[TID["、"]], noun_req=True, sent_tail0="")
    on = BoundaryBiasLogits(TK, N_CTX, [5, 5], [3.0, 3.0], case_frame_gate=True, **kw)
    sc = _scores(2)
    on(ids, sc)
    assert [float(sc[r, TID["。"]]) for r in range(2)] == [-6.0, 3.0]  # row0: 0.0 (withheld) + -6.0 penalty
    assert [float(sc[r, TID["、"]]) for r in range(2)] == [-3.0, -3.0]
    off = BoundaryBiasLogits(TK, N_CTX, [5, 5], [3.0, 3.0], case_frame_gate=False, **kw)
    sc = _scores(2)
    off(ids, sc)
    assert [float(sc[r, TID["。"]]) for r in range(2)] == [3.0, 3.0]


def test_case_frame_short_sentence_penalty():
    P = TID["。"]
    kw = dict(period_ids=[P], comma_ids=[], noun_req=True)

    def run(tails, sent_tail0="", **extra):
        proc = BoundaryBiasLogits(TK, N_CTX, [5] * len(tails), [0] * len(tails), sent_tail0=sent_tail0, **kw, **extra)
        sc = _scores(len(tails))
        proc(_ids(*tails), sc)
        return [float(sc[r, P]) for r in range(len(tails))]

    # no case frame, < 18 chars, connective (に) and terminal (た) endings alike: -6.0
    assert run(["窓の外に"], case_frame_gate=True) == [-6.0]
    assert run(["やがて広がった"], case_frame_gate=True) == [-6.0]
    # gate off: no penalty
    assert run(["窓の外に"], case_frame_gate=False) == [0.0]
    assert run(["やがて広がった"], case_frame_gate=False) == [0.0]
    # case frame present: no penalty (positive push is covered by the ph_period tests)
    assert run(["雨が降った"], case_frame_gate=True) == [0.0]
    ph = BoundaryBiasLogits(TK, N_CTX, [5], [0], period_ids=[P], comma_ids=[], ph_period=8.0, noun_req=True, case_frame_gate=True)
    sc = _scores(1)
    ph(_ids("雨が降った"), sc)
    assert float(sc[0, P]) == pytest.approx(8.0)
    # sent_tail0 counts toward the length: 14 + 4 = 18 chars -> no penalty; 13 + 4 = 17 -> penalty
    assert run(["窓の外に"], sent_tail0="あ" * 14, case_frame_gate=True) == [0.0]
    assert run(["窓の外に"], sent_tail0="あ" * 13, case_frame_gate=True) == [-6.0]
    # force_close: paragraph end must be able to close -> no penalty (close_boost applies as before)
    assert run(["窓の外に"], case_frame_gate=True, force_close=True, close_boost=2.0) == [2.0]
    # configurable params
    assert run(["窓の外に"], case_frame_gate=True, case_frame_penalty=3.0) == [-3.0]
    assert run(["窓の外に"], case_frame_gate=True, case_frame_min_chars=4) == [0.0]


def test_decoder_case_frame_params_default():
    dec = _decoder()
    assert dec.case_frame_penalty == 6.0 and dec.case_frame_min_chars == 18


# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("ch", list("梦变这们说时"))
def test_count_non_jis_kanji_detects_simplified(ch):
    assert count_non_jis_kanji(ch) == 1
    assert has_non_jis_kanji("夜の" + ch)


def test_count_non_jis_kanji_normal_japanese_is_zero():
    for s in ("々髪嘘", "カタカナのテスト", "ひらがなだけ", "「静かな夜」。窓の外、雨が降る。", "夢と変化と説明と時間"):
        assert count_non_jis_kanji(s) == 0, s
        assert not has_non_jis_kanji(s), s
    assert count_non_jis_kanji("，．") == 2  # full-width comma / full stop still counted


def test_forbidden_token_ids_ban_non_jis_ideograph_tokens():
    vocab = VOCAB + ["梦", "变", "这", "们", "说", "时", "夢", "髪", "々", "夜这", "，"]
    dec = ClauseTwistedSMCDecoder(dim=12, seed=0)
    dec.tokenizer = MockTokenizer(vocab=vocab)
    bad, _ = dec._forbidden_token_ids()
    idx = {s: i for i, s in enumerate(vocab)}
    for s in ("梦", "变", "这", "们", "说", "时", "夜这"):
        assert idx[s] in bad, s
    for s in ("夢", "髪", "々", "静", "雨", "，"):
        assert idx[s] not in bad, s


# ---------------------------------------------------------------------------------------------------------------
def test_kanji_bigrams_and_targets():
    c = kanji_bigrams("静寂が降る静寂")
    assert c[("静", "寂")] == 2 and sum(c.values()) == 2
    assert lex_bigram_targets("静寂が降り静") == {"寂"}
    assert lex_bigram_targets("静寂が降り静", 2) == set()
    assert lex_bigram_targets("静寂と静寂と静", 2) == {"寂"}
    assert lex_bigram_targets("静寂が降り") == set()   # last char not a kanji
    assert lex_bigram_targets("静") == set()           # first occurrence


def test_token_classes():
    content, first_kanji, hira_only = _decoder()._token_classes()
    assert len(content) == len(hira_only) == len(VOCAB)
    assert first_kanji["寂"] == [TID["寂"], TID["寂れ"], TID["寂び"]]
    assert hira_only[TID["た"]] and hira_only[TID["まし"]] and not hira_only[TID["静"]] and not hira_only[TID["。"]]
    assert content[TID["静"]] and not content[TID["の"]]


def _rep_proc(prefix, min_count=1, lex=2.5, cr=1.0):
    content, fk, _ = _decoder()._token_classes()
    return ContentRepLogits(TK, N_CTX, content, fk, prefix, cr, lex, min_count)


def test_lex_decay_applies_to_first_kanji_tokens_only():
    proc = _rep_proc("静寂が降りた。")
    sc = _scores(1)
    proc(_ids("静"), sc)
    hit = {TID["寂"], TID["寂れ"], TID["寂び"]}
    for i in range(sc.shape[1]):
        assert sc[0, i] == (-2.5 if i in hit else 0.0)


def test_lex_decay_first_occurrence_and_min_count():
    sc = _scores(1)
    _rep_proc("雨が降る。").__call__(_ids("静"), sc)
    assert np.all(sc == 0.0)
    sc = _scores(1)
    _rep_proc("静寂が降る。", min_count=2)(_ids("静"), sc)
    assert np.all(sc == 0.0)
    sc = _scores(1)
    _rep_proc("静寂と静寂が降る。", min_count=2)(_ids("静"), sc)
    assert sc[0, TID["寂"]] == -2.5


def test_lex_decay_uses_row_tail():
    proc = _rep_proc("", min_count=2)
    ids = _ids("静寂と静寂と静")[:, :]  # same length single row
    sc = _scores(1)
    proc(ids, sc)
    assert sc[0, TID["寂"]] == -2.5


def test_content_rep_penalty_division_multiplication_and_hiragana_exempt():
    proc = _rep_proc("", lex=0.0, cr=2.0)
    ids = np.array([[TID["山"], TID["窓"], TID["の"], TID["雨"]]], dtype=np.int64)
    sc = _scores(1)
    sc[0, TID["山"]] = 3.0       # present, positive -> /2
    sc[0, TID["窓"]] = -3.0      # present, negative -> *2
    sc[0, TID["の"]] = 3.0       # hiragana: exempt
    sc[0, TID["空"]] = 3.0       # content but absent: unchanged
    sc[0, TID["降"]] = -np.inf   # absent -inf stays
    sc[0, TID["雨"]] = -np.inf   # present -inf stays -inf (no NaN)
    proc(ids, sc)
    assert sc[0, TID["山"]] == 1.5 and sc[0, TID["窓"]] == -6.0
    assert sc[0, TID["の"]] == 3.0 and sc[0, TID["空"]] == 3.0
    assert sc[0, TID["降"]] == -np.inf and sc[0, TID["雨"]] == -np.inf
    assert not np.isnan(sc).any()


# ---------------------------------------------------------------------------------------------------------------
def _ban(stem_guard, ngram_n=0, soft_n=0, hist=None, soft=None, ctx_tail=""):
    _, _, hira = _decoder()._token_classes()
    return ContextNgramBan(
        TK, N_CTX, ngram_n, hist or {}, soft_n, 4.0, soft or {}, hira_only=hira, stem_guard=stem_guard, ctx_tail=ctx_tail,
    )


def test_stem_guard_exempts_hiragana_after_hiragana_hard_ban():
    hist = {(TID["雨"], TID["が"]): {TID["た"], TID["山"], TID["。"]}}
    ids = _ids("雨が")
    sc = _scores(1)
    _ban(True, ngram_n=3, hist=hist)(ids, sc)
    assert sc[0, TID["た"]] == 0.0                       # hiragana-only continuation exempt
    assert sc[0, TID["山"]] == -1e4 and sc[0, TID["。"]] == -1e4
    sc = _scores(1)
    _ban(False, ngram_n=3, hist=hist)(ids, sc)
    assert sc[0, TID["た"]] == -1e4 and sc[0, TID["山"]] == -1e4


def test_stem_guard_exempts_soft_decay_and_keeps_kanji_end_banned():
    soft = {(TID["が"],): {TID["た"], TID["山"]}}
    sc = _scores(1)
    _ban(True, soft_n=2, soft=soft)(_ids("雨が"), sc)
    assert sc[0, TID["た"]] == 0.0 and sc[0, TID["山"]] == -4.0
    # row ends with a kanji: no exemption
    soft2 = {(TID["雨"],): {TID["た"], TID["山"]}}
    sc = _scores(1)
    _ban(True, soft_n=2, soft=soft2)(_ids("が雨"), sc)
    assert sc[0, TID["た"]] == -4.0 and sc[0, TID["山"]] == -4.0


def test_stem_guard_uses_ctx_tail_when_row_tail_empty():
    hist = {(TID["雨"], TID["が"]): {TID["た"]}}
    ids = np.array([[TID["雨"], TID["が"]]], dtype=np.int64)   # n_ctx = 2, empty generated tail
    sc = _scores(1)
    _ban(True, ngram_n=3, hist=hist, ctx_tail="雨が")(ids, sc)
    assert sc[0, TID["た"]] == 0.0
    sc = _scores(1)
    _ban(True, ngram_n=3, hist=hist, ctx_tail="雨")(ids, sc)
    assert sc[0, TID["た"]] == -1e4


def test_ngram_ban_row_ngram_and_oob_ids():
    # n-gram created inside the row itself: (雨 が) -> た is banned the second time
    ids = np.array([[0, 0, TID["雨"], TID["が"], TID["た"], TID["雨"], TID["が"]]], dtype=np.int64)
    sc = _scores(1)
    _ban(False, ngram_n=3, hist={(TID["雨"], TID["が"]): {10 ** 6}})(ids, sc)
    assert sc[0, TID["た"]] == -1e4
    assert (sc == -1e4).sum() == 1


def test_first_ban_only_at_first_step():
    _, _, hira = _decoder()._token_classes()
    proc = ContextNgramBan(TK, N_CTX, 0, {}, 0, 4.0, {}, first_ban=np.array([TID["、"], 10 ** 6]))
    sc = _scores(2)
    proc(np.zeros((2, N_CTX), dtype=np.int64), sc)
    assert np.all(sc[:, TID["、"]] == -1e4) and (sc == -1e4).sum() == 2
    sc = _scores(2)
    proc(np.zeros((2, N_CTX + 1), dtype=np.int64), sc)
    assert np.all(sc == 0.0)
