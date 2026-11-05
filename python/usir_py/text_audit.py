"""Lightweight text audit for generated prose: hiragana inflection collapse and person-name candidates.

No author / character names are listed here (Zero Target-Leakage): person names are found structurally by the
morphological analyser's POS tag (名詞,固有名詞,人名). The POS tagger over-reports ordinary nouns that happen to be
surnames (e.g. 寺内, 小川), so `name_candidates` are for manual review, not an automatic verdict.
"""
import re
from typing import Any, Dict, List

# kanji + け + か + (な|に|だ): the stems of "-けさ" (noun) and "-かな" (adjectival noun) fused ("静けかな")
_INFLECTION_COLLAPSE = [
    re.compile(r"[一-鿿]けか[なにだ]"),
    re.compile(r"[一-鿿]さか[なにだ]"),
    re.compile(r"([ぁ-ん])\1{3,}"),       # the same hiragana 4+ times in a row
]


def audit_text(text: str) -> Dict[str, Any]:
    anomalies: List[str] = []
    for rx in _INFLECTION_COLLAPSE:
        anomalies.extend(m.group(0) for m in rx.finditer(text))
    names: List[str] = []
    try:
        from janome.tokenizer import Tokenizer
        for tk in Tokenizer().tokenize(text):
            if tk.part_of_speech.startswith("名詞,固有名詞,人名"):
                names.append(tk.surface)
    except Exception:  # janome missing: names cannot be checked
        names = []
    return {"inflection_anomalies": anomalies, "name_candidates": names}
