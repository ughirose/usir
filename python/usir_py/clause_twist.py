"""Clause-local subject-predicate twist (主述ねじれ) detection.

A sentence is split into clauses at clause boundaries; the subject scope is reset
per clause and a subject/predicate selectional-restriction check runs only inside
one clause. No morphological analyzer is used (regex + small closed word classes),
so recall is deliberately limited and every rule errs toward "not a twist".

Word classes (universal lexical classes only, no author / character / work nouns):
  STIMULUS_NOUNS      percept sources (音, 声, 光, 影, 匂い, ...): not sentient perceivers.
  ABSTRACT_NOUNS      abstract / institutional nouns (問題, 理由, 目標, ...): not agents.
  PERCEIVER_VERBS     verbs needing a sentient experiencer/agent (澄ます, 聞く, 考える, ...).
  AGENT_MOTION_VERBS  verbs needing an animate agent (走る, 歩く, 食べる, ...).
Flagged pairs (single clause only):
  STIMULUS  x PERCEIVER_VERBS      例: 風の音が耳を澄ました
  ABSTRACT  x PERCEIVER_VERBS
  ABSTRACT  x AGENT_MOTION_VERBS   例: 問題が走った
Natural-phenomenon subjects (風, 雨, ...) are NOT checked: personification is common.
Not flagged: copular / nominal predicates (…ことだ, …である), quoted speech (「」),
quotative complements (…と考える), passive / causative forms, and any clause whose
last subject marker names an unlisted noun.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

STIMULUS_NOUNS = ("音", "声", "光", "影", "匂い", "におい", "香り", "色")
ABSTRACT_NOUNS = (
    "問題", "課題", "理由", "原因", "目標", "目的", "結果", "結論", "現象", "状況",
    "時間", "社会", "制度", "方法", "計画", "意見", "事実", "傾向", "効果", "時代",
)
# (dictionary form, "g" godan | "i" ichidan)
PERCEIVER_VERBS = (
    ("澄ます", "g"), ("聞く", "g"), ("嗅ぐ", "g"), ("味わう", "g"), ("望む", "g"),
    ("願う", "g"), ("思う", "g"), ("疑う", "g"), ("眺める", "i"), ("見つめる", "i"),
    ("考える", "i"), ("決める", "i"), ("信じる", "i"),
)
AGENT_MOTION_VERBS = (
    ("走る", "g"), ("歩く", "g"), ("食べる", "i"), ("話す", "g"), ("読む", "g"),
    ("書く", "g"), ("泳ぐ", "g"),
)

# godan final kana -> (a, i, e, o, te, ta)
_GODAN = {
    "う": ("わ", "い", "え", "お", "って", "った"), "く": ("か", "き", "け", "こ", "いて", "いた"),
    "ぐ": ("が", "ぎ", "げ", "ご", "いで", "いだ"), "す": ("さ", "し", "せ", "そ", "して", "した"),
    "つ": ("た", "ち", "て", "と", "って", "った"), "ぬ": ("な", "に", "ね", "の", "んで", "んだ"),
    "ぶ": ("ば", "び", "べ", "ぼ", "んで", "んだ"), "む": ("ま", "み", "め", "も", "んで", "んだ"),
    "る": ("ら", "り", "れ", "ろ", "って", "った"),
}
# Auxiliary tail allowed after a verb form (passive られ / causative せ·させ are excluded on purpose).
_AUX = r"(?:ている|ていた|ていない|ていく|てくる|なかった|ませんでした|ました|ません|ましょう|ます|ない|たい|だろう|でしょう|た)?"


def _forms(verb: str, kind: str) -> List[str]:
    """Surface stems of a verb (dictionary, renyo, neg, te/ta ...) without auxiliaries."""
    if kind == "i":
        stem = verb[:-1]
        return [stem + "る", stem, stem + "て"]
    base, fin = verb[:-1], verb[-1]
    a, i, e, o, te, ta = _GODAN[fin]
    return [base + x for x in (fin, a, i, te, ta, o + "う")]


def _verb_tail_re(verbs: Sequence[Tuple[str, str]]) -> "re.Pattern[str]":
    forms = sorted({f for v, k in verbs for f in _forms(v, k)}, key=len, reverse=True)
    return re.compile("(?:" + "|".join(re.escape(f) for f in forms) + ")" + _AUX + r"[ぞよね]?$")


_PERCEIVER_TAIL = _verb_tail_re(PERCEIVER_VERBS)
_MOTION_TAIL = _verb_tail_re(AGENT_MOTION_VERBS)
_STIMULUS_END = re.compile("(?:" + "|".join(map(re.escape, STIMULUS_NOUNS)) + ")$")
_ABSTRACT_END = re.compile("(?:" + "|".join(map(re.escape, ABSTRACT_NOUNS)) + ")$")

# Clause boundaries. Connective が / から / し are only cut after a predicate-final character
# (た る だ す い う な); 、 is cut only when the preceding character is predicate-final as well.
_PRED_END = "たるだすいうな"
_BOUNDARY_RE = re.compile(
    r"(?:けれども|けれど|けども|けど|ながら|のに|ので"
    rf"|(?<=[{_PRED_END}])が|(?<=[{_PRED_END}])から|(?<=[{_PRED_END}])し(?=[、，,]))[、，,]?"
    rf"|(?<=[{_PRED_END}てで])[、，,]"
)
_QUOTE_RE = re.compile(r"[「『（(][^」』）)]*[」』）)]")
_NP_SPLIT_RE = re.compile(r"[、，,をにでとへ\s]")
_TRAIL_RE = re.compile(r"[。！？!?\s、，,]+$")


@dataclass(frozen=True)
class ClauseTwist:
    clause: str
    subject: str     # head noun phrase before が/は
    predicate: str   # matched verb tail
    kind: str        # "stimulus_x_perceiver" | "abstract_x_perceiver" | "abstract_x_motion"


def split_clauses(sentence: str) -> List[str]:
    """Split one sentence into clauses at connective particles and predicate-final 、."""
    text = _QUOTE_RE.sub("○", sentence)
    out, pos = [], 0
    for m in _BOUNDARY_RE.finditer(text):
        piece = text[pos:m.end()]
        if piece.strip():
            out.append(piece)
        pos = m.end()
    tail = text[pos:]
    if tail.strip():
        out.append(tail)
    return out


def _last_subject_np(clause_body: str) -> Tuple[str, int]:
    """Head noun phrase of the last が/は marker in the clause and the marker's index."""
    idx = max(clause_body.rfind("が"), clause_body.rfind("は"))
    if idx <= 0:
        return "", -1
    np_ = _NP_SPLIT_RE.split(clause_body[:idx])[-1]
    return np_.split("の")[-1], idx


def detect_clause_twists(sentence: str) -> List[ClauseTwist]:
    """Return clause-local subject-predicate twists found in one sentence."""
    found: List[ClauseTwist] = []
    for clause in split_clauses(sentence):
        body = _TRAIL_RE.sub("", clause)
        subj, idx = _last_subject_np(body)
        if not subj:
            continue
        rest = body[idx + 1:]
        kind = None
        m = _PERCEIVER_TAIL.search(rest)
        if m:
            if _STIMULUS_END.search(subj):
                kind = "stimulus_x_perceiver"
            elif _ABSTRACT_END.search(subj):
                kind = "abstract_x_perceiver"
        else:
            m = _MOTION_TAIL.search(rest)
            if m and _ABSTRACT_END.search(subj):
                kind = "abstract_x_motion"
        if kind is None:
            continue
        # Quotative / complement use ("…と考える", "…ように思う") belongs to another subject.
        if rest[:m.start()].endswith(("と", "よう", "ように", "ことと")):
            continue
        found.append(ClauseTwist(clause=clause, subject=subj, predicate=m.group(0), kind=kind))
    return found


def count_clause_twists(sentences: Sequence[str]) -> int:
    return sum(len(detect_clause_twists(s)) for s in sentences)
