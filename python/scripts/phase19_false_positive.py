"""Phase 19 false-positive measurement on real literary sentences (corpus_40_benchmark CORPUS, phases II_Critical / III_Void).

Usage: cd python && uv run python scripts/phase19_false_positive.py
"""
from __future__ import annotations

import re
from collections import Counter

from usir_py.corpus_40_benchmark import CORPUS
from usir_py.realtime_decoder import _I_ADJ_STEMS, _FRAGMENT_RE, _NOUN_END_RE, case_frame_ok

MASTERWORK_PHASES = ("II_Critical", "III_Void")


def sentences(text: str):
    return [s.strip() for s in re.split("[。！？!?]", text) if s.strip()]


def measure():
    n_sent = 0
    ku_all = []          # sentences ending in kanji+く
    ku_flag = []         # ... flagged by rule 1 (stem in _I_ADJ_STEMS)
    after_short = 0
    after_short_noun = 0
    after_short_flip = 0
    noun_end_all = 0
    for ch in CORPUS:
        if ch["phase"] not in MASTERWORK_PHASES:
            continue
        ss = sentences(ch["text"])
        for i, s in enumerate(ss):
            n_sent += 1
            if len(s) > 1 and s[-1] == "く" and "一" <= s[-2] <= "鿿":
                ku_all.append(s)
                if _FRAGMENT_RE.search(s + "。") and s[-2] in _I_ADJ_STEMS:
                    ku_flag.append(s)
            if _NOUN_END_RE.search(s):
                noun_end_all += 1
            if i > 0 and len(ss[i - 1]) <= 15:
                after_short += 1
                if _NOUN_END_RE.search(s):
                    after_short_noun += 1
                if case_frame_ok(s) and not case_frame_ok(s, len(ss[i - 1])):
                    after_short_flip += 1
    return dict(n_sent=n_sent, ku_all=ku_all, ku_flag=ku_flag, after_short=after_short,
                after_short_noun=after_short_noun, after_short_flip=after_short_flip, noun_end_all=noun_end_all)


if __name__ == "__main__":
    r = measure()
    print("masterwork chunks:", sum(1 for c in CORPUS if c["phase"] in MASTERWORK_PHASES), "sentences:", r["n_sent"])
    print("kanji+く。 sentences:", len(r["ku_all"]), "flagged by rule 1:", len(r["ku_flag"]))
    for s in r["ku_all"]:
        print("  ", "FLAG" if s in r["ku_flag"] else "pass", s[-20:])
    print("sentences ending in noun run (all):", r["noun_end_all"], f"{r['noun_end_all'] / r['n_sent']:.3f}")
    print("sentences following a <=15-char sentence:", r["after_short"])
    print("  of which end in noun run:", r["after_short_noun"], f"{r['after_short_noun'] / max(r['after_short'], 1):.3f}")
    print("  of which flip case_frame_ok True->False:", r["after_short_flip"], f"{r['after_short_flip'] / max(r['after_short'], 1):.3f}")
    print(Counter(s[-2:] for s in r["ku_all"]))
