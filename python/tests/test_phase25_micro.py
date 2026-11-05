"""Phase 25: paragraph-end grace closing, simile decay, body-compound protection."""
from __future__ import annotations

import numpy as np

from usir_py.realtime_decoder import (
    ClauseTwistedSMCDecoder,
    SimileDecayLogits,
    body_compounds,
    count_similes,
    last_sentence_end,
    sentence_finished,
)


class Tok:
    def __init__(self, vocab):
        self.vocab = list(vocab)

    def __len__(self):
        return len(self.vocab)

    def decode(self, ids, skip_special_tokens=True):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return "".join(self.vocab[i] for i in ids)

    def batch_decode(self, seqs, skip_special_tokens=False):
        if hasattr(seqs, "tolist"):
            seqs = seqs.tolist()
        return [self.decode(s) for s in seqs]


# ------------------------------------------------------------------ helpers
def test_sentence_finished_and_last_end():
    assert sentence_finished("雨が降った。")
    assert not sentence_finished("時計は、また、")
    assert not sentence_finished("「時計は、また動く。")          # quote still open
    assert sentence_finished("「時計は動く」")
    assert last_sentence_end("雨が降った。風が、") == len("雨が降った。")
    assert last_sentence_end("時計は、また、") == 0


def test_count_similes_and_body_compounds():
    assert count_similes("まるで夢のようだ。まるで、あたかも") == 3
    assert body_compounds("止まった懐中時計を老職人が見た。") == ["懐中時計", "老職人"]


# ------------------------------------------------------------------ simile decay
VOC = ["<eos>", "まるで", "まる", "雨", "が", "。", "あたか"]


def _sim(prefix, tail_ids, hist):
    tk = Tok(VOC)
    proc = SimileDecayLogits(tk, 1, [1, 2, 6], prefix, hist)
    ids = np.asarray([[3] + tail_ids])
    sc = np.zeros((1, len(VOC)))
    proc(ids, sc)
    return sc[0]


def test_simile_no_penalty_before_first_use():
    assert _sim("", [3, 4], 0)[1] == 0.0


def test_simile_paragraph_penalty_after_one_use():
    sc = _sim("", [1, 3, 5], 0)
    assert sc[1] == -8.0 and sc[2] == -8.0 and sc[6] == -8.0 and sc[3] == 0.0


def test_simile_story_penalty_after_two_uses():
    assert _sim("", [1, 3, 5], 1)[1] == -12.0       # 1 in history + 1 in paragraph


def test_simile_token_ids_prefix_rule():
    d = ClauseTwistedSMCDecoder(dim=12, seed=0, simile_decay=True)
    d.tokenizer = Tok(VOC)
    assert d._simile_token_ids() == [1, 2, 6]


# ------------------------------------------------------------------ grace closing
def _fake_steps(d, outcomes):
    """outcomes: list of (clause, eos) returned by successive step_smc_clause calls."""
    it = iter(outcomes)

    def step(prefix, candidate_clauses=None, target_end=None, phi_bridge=None, resample_method=None):
        clause, eos = next(it)
        d.eos_reached = eos
        return {"selected_clause": clause, "next_prefix": prefix + clause, "ess": 1.0, "term1_llm": [0.0],
                "term3_style": [0.0], "steer_gain": 1.0, "greedy_agree": True}
    d.step_smc_clause = step


def test_grace_close_finishes_sentence_after_eos():
    d = ClauseTwistedSMCDecoder(dim=12, seed=0, grace_close=True)
    _fake_steps(d, [("「時計は、", False), ("また、", False), ("", True), ("動く」", False)])
    out = d.generate_with_clause_twisted_smc(prefix="", max_clauses=8, early_stop_on_punct=False)
    assert out["text"] == "「時計は、また、動く」"
    assert d.stats["grace_paragraphs"] == 1


def test_grace_close_cuts_back_when_still_open():
    d = ClauseTwistedSMCDecoder(dim=12, seed=0, grace_close=True, grace_clauses=2)
    _fake_steps(d, [("雨が降った。", False), ("時計は、", False), ("", True), ("また、", False), ("なお、", False)])
    out = d.generate_with_clause_twisted_smc(prefix="", max_clauses=8, early_stop_on_punct=False)
    assert out["text"] == "雨が降った。"
    assert d.stats["grace_cuts"] == 1


def test_grace_close_off_keeps_legacy_behaviour():
    d = ClauseTwistedSMCDecoder(dim=12, seed=0)
    _fake_steps(d, [("時計は、", False), ("", True)])
    out = d.generate_with_clause_twisted_smc(prefix="", max_clauses=8, early_stop_on_punct=False)
    assert out["text"].startswith("時計は")
    assert "grace_paragraphs" not in d.stats
