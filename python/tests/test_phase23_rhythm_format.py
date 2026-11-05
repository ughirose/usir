"""Phase 23: rhythm profiles, adnominal guard, CV band, lex compound exemption, collapse guard, metrics, formatter."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from usir_py.hsmm_longform_director import HSMMLongformDirector
from usir_py.realtime_decoder import (
    AdaptiveBAOABIntegrator,
    BoundaryBiasLogits,
    ClauseTwistedSMCDecoder,
    ContentRepLogits,
)
from usir_py.text_format import format_novel
from usir_py.text_metrics import (
    adnominal_split_suspects,
    comma_per_100,
    max_kanji_repeat_in_sentence,
    mean_sentence_len,
    repeat_2gram_in_sentence_max,
    text_metrics,
)

VOCAB = ["<eos>", "。", "、", "た", "る", "い", "り", "雨", "が", "降", "懐", "中", "時", "計", "古", "を", "歯", "と"]
TID = {s: i for i, s in enumerate(VOCAB)}
N_CTX = 2


class Tok:
    eos_token_id = 0

    def __init__(self, vocab=VOCAB, prefix=""):
        self.vocab = list(vocab)
        self.prefix = prefix

    def __len__(self):
        return len(self.vocab)

    def encode(self, text, add_special_tokens=False):
        return [self.vocab.index(ch) for ch in text]

    def decode(self, ids, skip_special_tokens=True):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return "".join(self.vocab[i] for i in ids if not (skip_special_tokens and i == 0))

    def batch_decode(self, seqs, skip_special_tokens=False):
        if hasattr(seqs, "tolist"):
            seqs = seqs.tolist()
        return [self.prefix + self.decode(s, skip_special_tokens=skip_special_tokens) for s in seqs]


TK = Tok()


def _ids(*tails):
    return np.array([[0, 0] + TK.encode(t) for t in tails], dtype=np.int64)


def _scores(P):
    return np.zeros((P, len(VOCAB) + 2), dtype=np.float32)


def _dec(**kw):
    kw.setdefault("rhythm", True)
    return ClauseTwistedSMCDecoder(dim=12, seed=0, **kw)


# ------------------------------------------------------------------ 1. rhythm profiles
def test_cycle_is_unchanged_default():
    d = _dec()
    assert d.rhythm_profile == "cycle"
    assert d._rhythm_params("") == (8.0, -12.0, 13, True)
    assert d._rhythm_params("あ。") == (-12.0, 4.0, 20, False)
    assert d._rhythm_params("あ。い。") == (-12.0, 4.0, 22, False)
    assert d._rhythm_params("あ。い。" + "あ" * 90) == (10.0, -10.0, 14, False)


def test_steady_holds_short_and_never_pushes():
    d = _dec(rhythm_profile="steady")
    assert d._rhythm_params("あ" * 5)[:2] == (-12.0, 0.0)           # hold below 16
    assert d._rhythm_params("あ" * 15)[:2] == (-12.0, 0.0)
    assert d._rhythm_params("あ" * 16)[:2] == (0.0, 0.0)            # no positive push, no comma push
    assert d._rhythm_params("あ" * 60)[:2] == (0.0, 0.0)
    assert d._rhythm_params("あ" * 90)[:2] == (0.0, 0.0)
    assert d._rhythm_params("あ" * 91)[:2] == (3.0, 0.0)            # mild push only beyond 90
    # sentence index does not matter: previous sentences do not change the mode
    assert d._rhythm_params("あ。い。う。" + "あ" * 20)[:2] == (0.0, 0.0)
    # comma chain guard: -4 once the sentence has >= 3 commas; commas of earlier sentences do not count
    assert d._rhythm_params("あ" * 20 + "、" * 2)[1] == 0.0
    assert d._rhythm_params("あ" * 20 + "、" * 3)[1] == -4.0
    assert d._rhythm_params("、、、。" + "あ" * 20)[1] == 0.0
    for n in (0, 10, 20, 60, 100):
        pb, cb, _, _ = d._rhythm_params("あ" * n + "、" * 4)
        assert cb <= 0.0 and pb <= 3.0


def test_steady_thresholds_configurable_and_profile_per_paragraph():
    d = _dec(rhythm_profile="steady", steady_min_chars=5, steady_max_chars=10, steady_max_commas=1)
    assert d._rhythm_params("あ" * 4)[0] == -12.0
    assert d._rhythm_params("あ" * 5)[0] == 0.0
    assert d._rhythm_params("あ" * 11)[0] == 3.0
    assert d._rhythm_params("あ、")[1] == -4.0
    d.rhythm_profile = "cycle"
    assert d._rhythm_params("")[0] == 8.0
    with pytest.raises(ValueError):
        d.rhythm_profile = "bogus"


def test_pulse_caps_period_push_and_drops_long_comma_push():
    cyc, pul = _dec(), _dec(rhythm_profile="pulse")
    pre_short, pre_med0, pre_med1 = "", "あ。", "あ。い、"
    pre_long_sealed = "あ。い。" + "あ" * 20
    pre_long_mid = "あ。い。" + "あ" * 60
    pre_long_end = "あ。い。" + "あ" * 90
    for pre in (pre_short, pre_med1, pre_long_mid, pre_long_end):
        assert cyc._rhythm_params(pre)[0] > 3.0
        assert pul._rhythm_params(pre)[0] == 3.0
    # seal stays, comma push removed
    assert cyc._rhythm_params(pre_long_sealed)[:2] == (-12.0, 4.0)
    assert pul._rhythm_params(pre_long_sealed)[:2] == (-12.0, 0.0)
    assert pul._rhythm_params(pre_med0)[:2] == (-12.0, 4.0)        # medium-mode comma push is not a long-mode push
    # negative values / caps / noun_req identical
    for pre in (pre_short, pre_med0, pre_med1, pre_long_sealed, pre_long_mid, pre_long_end):
        a, b = cyc._rhythm_params(pre), pul._rhythm_params(pre)
        assert a[2:] == b[2:]
        assert (a[1] < 0) == (b[1] < 0)


def test_rhythm_profile_needs_rhythm_flag():
    assert _dec(rhythm=False, rhythm_profile="steady")._rhythm_params("あ") is None


# ------------------------------------------------------------------ 2. adnominal guard
def _bias(tails, guard, ph=8.0, **kw):
    proc = BoundaryBiasLogits(TK, N_CTX, [4] * len(tails), [0] * len(tails), [TID["。"]], [TID["、"]], ph_period=ph,
                              adnominal_guard=guard, **kw)
    sc = _scores(len(tails))
    proc(_ids(*tails), sc)
    return [float(sc[r, TID["。"]]) for r in range(len(tails))]


def test_adnominal_guard_caps_positive_period_bias():
    # step 3 / cap 4 -> ph_period applied in full
    tails = ["雨が降る", "雨が降た", "雨が降い", "雨が降り"]
    assert _bias(tails, False) == [8.0] * 4
    assert _bias(tails, True) == [2.0, 2.0, 2.0, 8.0]              # り is not an ambiguous ending


def test_adnominal_guard_leaves_negative_and_small_alone_and_respects_length():
    assert _bias(["雨が降る"], True, ph=1.0) == [1.0]
    assert _bias(["雨が降る"], True, ph=-5.0) == [-5.0]
    # gate (-12) on a connective + guard: unchanged negative
    # sentence already >= adnominal_max_chars: not capped
    assert _bias(["雨が降る"], True, adnominal_max_chars=3) == [8.0]
    assert _bias(["雨が降る"], True, adnominal_max_chars=5) == [2.0]
    # sent_tail0 counts toward the sentence length
    assert _bias(["雨が降る"], True, sent_tail0="あ" * 100) == [8.0]


def test_adnominal_guard_caps_the_net_of_all_sources():
    # ph_period 3 + row_bias 3 -> net 6 -> capped at 2; with a -12 gate the net is negative and untouched
    proc = BoundaryBiasLogits(TK, N_CTX, [4], [3.0], [TID["。"]], [TID["、"]], ph_period=3.0, adnominal_guard=True)
    sc = _scores(1)
    proc(_ids("雨が降る"), sc)
    assert float(sc[0, TID["。"]]) == pytest.approx(2.0)


# ------------------------------------------------------------------ 3. CV band
def _cv_weights(**kw):
    d = _dec(cv_gain=10.0, cv_floor=0.0, **kw)
    prefix = "短い。とても長い文章がここにあります。"
    cands = ["次の文。", "もっと長い長い長い長い長い長い長い長い長い長い文。"]
    return np.asarray(d.compute_particle_weights(prefix, cands)["weights"], dtype=float), prefix, cands


def test_cv_band_penalises_above_ceiling_only():
    base, prefix, cands = _cv_weights()
    ceil = 0.05
    banded, _, _ = _cv_weights(cv_ceil=ceil)
    done = [len(x) for x in ["短い。", "とても長い文章がここにあります。"]]
    expect = []
    for c in cands:
        lens = done + [len(c)]
        cv = float(np.std(lens) / np.mean(lens))
        expect.append(-10.0 * max(0.0, cv - ceil))
    assert any(e < 0 for e in expect)
    assert np.allclose(banded - base, expect)
    # a ceiling above any CV changes nothing
    high, _, _ = _cv_weights(cv_ceil=5.0)
    assert np.allclose(high, base)


# ------------------------------------------------------------------ 4. lex compound exemption
def _lex(prefix, tail, exempt):
    proc = ContentRepLogits(TK, N_CTX, [True] * len(VOCAB), {ch: [TID[ch]] for ch in VOCAB if "一" <= ch <= "鿿"},
                            prefix, 1.0, 3.0, 1, compound_exempt=exempt)
    sc = _scores(1)
    proc(_ids(tail), sc)
    return sc[0]


def test_lex_compound_exempt_allows_completing_a_known_word():
    pre = "懐中時計を見た。"
    off = _lex(pre, "懐中時", False)
    on = _lex(pre, "懐中時", True)
    assert off[TID["計"]] == -3.0            # coinage pressure of the old behaviour
    assert on[TID["計"]] == 0.0              # 懐中時 + 計 completes the earlier run 懐中時計


def test_lex_compound_exempt_keeps_penalty_when_not_a_word_prefix():
    # 古時計 occurred; trailing run is 時 -> 時計 is not a prefix of the run 古時計 -> still penalised
    pre = "古時計を見た。"
    assert _lex(pre, "時", True)[TID["計"]] == -3.0
    # trailing run 懐中 + 時 is a prefix of 懐中時計 -> exempt; 懐中 + 計 is not
    pre2 = "懐中時計を見た。懐中時計を見た。"
    assert _lex(pre2, "懐中", True)[TID["時"]] == 0.0


def test_lex_compound_exempt_off_by_default():
    d = _dec()
    assert d.lex_compound_exempt is False and d.collapse_guard is False and d.adnominal_guard is False
    assert d.cv_ceil is None


# ------------------------------------------------------------------ 5. collapse guard
def test_collapse_guard_penalises_repeated_kanji_and_bigrams():
    cands = ["歯と歯が歯を噛んだ。", "歯が鳴った。", "時計と時計を時計で見た。"]
    off = _dec().compute_particle_weights("", cands)["weights"]
    d = _dec(collapse_guard=True)
    on = d.compute_particle_weights("", cands)["weights"]
    diff = np.asarray(on) - np.asarray(off)
    assert np.allclose(diff, [-8.0, 0.0, -8.0])   # 歯x3 ; 時計 x3 (bigram) and 時/計 x3
    assert d.stats["collapse_cands"] == 2
    d2 = _dec(collapse_guard=True, collapse_penalty=3.0)
    assert np.allclose(np.asarray(d2.compute_particle_weights("", cands)["weights"]) - np.asarray(off), [-3.0, 0.0, -3.0])


def test_collapse_guard_counts_the_open_sentence_not_earlier_ones():
    d = _dec(collapse_guard=True)
    off = _dec().compute_particle_weights("歯が歯を。歯と", ["見た。"])["weights"]
    on = d.compute_particle_weights("歯が歯を。歯と", ["見た。"])["weights"]
    assert np.allclose(on, off)                      # earlier sentence has 歯 twice, open sentence once
    off2 = _dec().compute_particle_weights("歯が歯を", ["見る歯。"])["weights"]
    on2 = d.compute_particle_weights("歯が歯を", ["見る歯。"])["weights"]
    assert np.allclose(np.asarray(on2) - np.asarray(off2), [-8.0])


# ------------------------------------------------------------------ 6. metrics
def test_text_metrics_basic():
    t = "雨が降った。風が吹き、雲が流れた。"
    assert mean_sentence_len(t) == pytest.approx((6 + 11) / 2)
    assert comma_per_100(t) == pytest.approx(100.0 / 17)
    assert max_kanji_repeat_in_sentence("歯と歯が歯を噛んだ。雨が降る。") == 3
    assert max_kanji_repeat_in_sentence("歯が歯を。歯と。") == 2         # per sentence, not per text
    assert repeat_2gram_in_sentence_max("懐中時計と懐中時計と懐中時計。") == 3
    assert repeat_2gram_in_sentence_max("雨が降る。") == 0
    assert text_metrics("")["mean_sentence_len"] == 0.0


def test_adnominal_split_suspects_heuristic():
    assert adnominal_split_suspects("工具を残した。多くの工具が並ぶ。") == 1
    assert adnominal_split_suspects("男は泣いた。女は笑った。") == 1     # heuristic false positive: legitimate 女は
    assert adnominal_split_suspects("雨が降った。ああ、晴れた。") == 0
    assert adnominal_split_suspects("静かだ。") == 0


# ------------------------------------------------------------------ 7. formatter
PARS = [
    "朝の光が窓に差した。" * 12 + "「行こう」と彼は言った。" + "霧が晴れていった。" * 8,
    "「待て」「いやだ」夜が明けた。静かだった。",
]


def test_format_novel_dialogue_alone_and_indent():
    out = format_novel(PARS, target_line_chars=60, seed=1)
    blocks = out.split("\n\n")
    assert len(blocks) == 2
    lines = blocks[0].split("\n")
    dlg = [l for l in lines if "「" in l]
    assert dlg == ["「行こう」と彼は言った。"]                     # alone, attached quotation tail, no indent
    for l in lines:
        assert l.startswith("　") != ("「" in l)
    assert "".join(l.lstrip("　") for l in lines) == PARS[0]
    lines2 = blocks[1].split("\n")
    assert lines2[0] == "「待て」" and lines2[1] == "「いやだ」"
    assert lines2[2].startswith("　夜が明けた。")


def test_format_novel_line_lengths_and_determinism():
    long = ["静かな夜だった。" * 60]
    a = format_novel(long, 120, seed=7)
    assert a == format_novel(long, 120, seed=7)
    assert a != format_novel(long, 120, seed=8)
    lines = a.split("\n")
    assert len(lines) > 1
    body = [len(l) - 1 for l in lines[:-1]]
    assert min(body) >= int(0.6 * 120) and max(body) <= int(1.4 * 120) + 8   # <= target + one 8-char sentence
    assert all(l.startswith("　") for l in lines)
    assert format_novel([], 120) == ""
    assert format_novel(["", "あ。"], 120) == "　あ。"


# ------------------------------------------------------------------ 8. runner wiring
def _runner():
    path = Path(__file__).resolve().parents[1] / "colab" / "run_v4_bidirectional_experiment.py"
    spec = importlib.util.spec_from_file_location("run_v4_for_p23", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_runner_chat_template_compat_and_think_strip():
    mod = _runner()

    class NoThink:
        def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=False):
            return "plain"

    class Think:
        def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=False, enable_thinking=True):
            return f"think={enable_thinking}"

    assert mod.apply_chat_template_compat(NoThink(), [], add_generation_prompt=True, tokenize=False) == "plain"
    assert mod.apply_chat_template_compat(Think(), [], add_generation_prompt=True, tokenize=False) == "think=False"
    assert mod.strip_think("<think>\nあれこれ\n</think>\n本文") == "本文"
    assert mod.strip_think("本文") == "本文"


def test_runner_hook_switches_profile_for_climax_and_forwards_flags():
    mod = _runner()
    forced = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    dec = ClauseTwistedSMCDecoder(integrator=AdaptiveBAOABIntegrator(dim=12, seed=0), seed=0, rhythm=True)
    seen = []
    orig = dec.generate_with_clause_twisted_smc

    def spy(*a, **k):
        seen.append((dec.current_well_idx, dec.rhythm_profile))
        return orig(*a, max_clauses=3, **{x: v for x, v in k.items() if x != "max_clauses"})

    dec.generate_with_clause_twisted_smc = spy
    hook = mod._make_hook(0, dec, False, 0.0, "steady", "pulse")
    HSMMLongformDirector(transition_matrix=forced, seed=0).generate_longform(
        target_paragraphs=4, decoder=dec, paragraph_hook=hook, min_clauses_per_paragraph=2)
    assert seen and {w for w, _ in seen} >= {0, 2}
    for w, prof in seen:
        assert prof == ("pulse" if w == 2 else "steady")
    assert mod._make_hook(0, dec, False, 0.0) is None


def test_runner_formatted_text_without_model():
    res = _runner().run_longform_director("静謐な山寺の夜明け", seed=3, target_paragraphs=3, target_line_chars=60)
    assert "formatted_text" in res and "\n" in res["formatted_text"]
    assert "formatted_text" not in _runner().run_longform_director("静謐な山寺の夜明け", seed=3, target_paragraphs=3)


# ------------------------------------------------------------------ 9. sentencepiece-style tokens
def test_token_classes_strip_sentencepiece_marker():
    vocab = ["<eos>", "▁山", "▁の", "▁カメラ", "▁abc", "。", "川"]
    d = _dec()
    d.tokenizer = Tok(vocab)
    content, first_kanji, hira = d._token_classes()
    assert content == [False, True, False, True, False, False, True]
    assert first_kanji == {"山": [1], "川": [6]}
    assert hira == [False, False, True, False, False, False, False]
    # demonstrative ids also see through the marker
    d2 = _dec()
    d2.tokenizer = Tok(["<eos>", "▁その", "この", "山"])
    assert d2._demo_token_ids()["その"] == [1]


# ------------------------------------------------------------------ 4b. content repetition penalty: paragraph scope
def _cr(scope, exempt_ids=None):
    # prompt (first N_CTX ids) holds 懐中 (theme); the paragraph body so far is empty, the generated tail is 雨が
    proc = ContentRepLogits(TK, N_CTX, [True] * len(VOCAB), {}, "", 2.0, 0.0, 1,
                            cr_paragraph_scope=scope, cr_exempt_ids=exempt_ids)
    ids = np.asarray([[TID["懐"], TID["中"], TID["雨"], TID["が"]]])
    sc = np.ones((1, len(VOCAB)), dtype=np.float64)
    proc(ids, sc)
    return sc[0]


def test_content_rep_paragraph_scope_ignores_prompt_tokens():
    legacy = _cr(False)
    scoped = _cr(True)
    assert legacy[TID["懐"]] == 0.5          # legacy: a theme token in the prompt is penalised before first use
    assert scoped[TID["懐"]] == 1.0          # paragraph scope: prompt tokens are not counted
    assert scoped[TID["雨"]] == 0.5          # tokens generated in the paragraph are still penalised


def test_content_rep_exempt_ids_never_penalised():
    scoped = _cr(True, exempt_ids=[TID["雨"]])
    assert scoped[TID["雨"]] == 1.0
