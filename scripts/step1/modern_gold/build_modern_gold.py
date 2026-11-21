#!/usr/bin/env python
"""Deterministic generator and verifier for Track A Modern Japanese Gold Draft.

Builds 3 postwar Aozora Bunko literary excerpts (1954-1963, 新字新仮名, public domain),
extracts outermost quoted utterances compatible with ``f9_loader.py`` and
``nishiki segment``, and assigns rule-based surface-grammar politeness labels
(``polite`` / ``plain`` / ``ambiguous``) using Janome without LLM subjectivity.

Usage:
    python scripts/step1/modern_gold/build_modern_gold.py
    python scripts/step1/modern_gold/build_modern_gold.py --verify
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import time
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from janome.tokenizer import Tokenizer

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
PYTHON_DIR = REPO_ROOT / "python"

for _p in (str(SCRIPTS_DIR), str(PYTHON_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fetch_aozora import INDEX_URL, USER_AGENT, clean_aozora_text  # noqa: E402
from usir_py.sensor_eval.f9_loader import extract_outer_quote_spans  # noqa: E402

LOCAL_INDEX_CANDIDATES = (
    REPO_ROOT / "data" / "aozora_cache" / "_index.csv",
    REPO_ROOT.parent / "usir" / "data" / "aozora_cache" / "_index.csv",
)

# F9 gold 8 works + explicit prompt exclusion titles (never overlap)
EXCLUDED_WORK_IDS = frozenset(
    {
        "000148_000752",  # 坊っちゃん
        "000035_000301",  # 人間失格
        "000081_000456",  # 銀河鉄道の夜
        "000074_000429",  # 城のある町にて
        "000129_000689",  # 山椒大夫
        "000050_003586",  # 草迷宮
        "001383_056643",  # 少年
        "001095_042618",  # 桜の森の満開の下
    }
)
EXCLUDED_TITLES = frozenset(
    {
        "人間失格",
        "少年",
        "草迷宮",
        "銀河鉄道の夜",
        "蜘蛛の糸",
        "走れメロス",
        "山月記",
        "羅生門",
        "坊っちゃん",
        "城のある町にて",
        "山椒大夫",
        "桜の森の満開の下",
    }
)

CONTEXT_WINDOW_CHARS = 200

_ONLY_PUNCT = re.compile(r"^[。！？\?\!…―—・、,\s\u3000]*$")
_TRAIL_PUNCT = re.compile(r"[。！？\?\!…―—・、,\s\u3000]+$")
_CUTOFF_RE = re.compile(r"(?:……|…|――|—)$")

FINAL_PARTICLES = frozenset(
    {
        "よ",
        "ね",
        "ねえ",
        "ねぇ",
        "な",
        "なあ",
        "なぁ",
        "わ",
        "さ",
        "ぞ",
        "ぜ",
        "か",
        "かい",
        "かね",
        "かしら",
        "かな",
        "とも",
        "っけ",
        "の",
        "や",
        "い",
        "もん",
        "もの",
    }
)


@dataclass(frozen=True)
class WorkSpec:
    work_id: str
    person_id: str
    aozora_work_id: str
    author: str
    title: str
    first_published: str
    author_death_year: int
    narration: str
    narrator: str | None
    pov: str | None
    line_start: int
    line_end_exclusive: int
    selection_rationale: str
    characters: list[dict[str, Any]]
    utterance_gold: list[tuple[int, str, str | None, str]]


WORK_SPECS: tuple[WorkSpec, ...] = (
    WorkSpec(
        work_id="001798_056779",
        person_id="001798",
        aozora_work_id="056779",
        author="梅崎春生",
        title="ボロ家の春秋",
        first_published="「新潮」新潮社、1954（昭和29）年8月",
        author_death_year=1965,
        narration="first",
        narrator="boku",
        pov=None,
        line_start=27,
        line_end_exclusive=56,
        selection_rationale=(
            "1954年発表の戦後一人称口語小説。語り手「僕」と家主「不破」・同居相手「野呂」の"
            "掛け合いが多く、初対面の敬体から口論による常体への口調変化を同一人物内で含む。"
            "30%位置は12発話連続の無地の文会話で話者起点が欠落するため、地の文の手がかりと"
            "書置き引用（not_speech）を含む15%位置（L27..L55）を抜粋した。"
        ),
        characters=[
            {
                "id": "boku",
                "name": "僕",
                "aliases": ["僕", "私", "俺"],
                "gender": "Male",
                "role": "主人公の貧乏画家（一人称語り手）",
            },
            {
                "id": "fuwa",
                "name": "不破数馬",
                "aliases": ["不破", "不破数馬", "不破さん", "不破氏"],
                "gender": "Male",
                "role": "家主（失踪した男）",
            },
            {
                "id": "noro",
                "name": "野呂",
                "aliases": ["野呂", "野呂君", "彼"],
                "gender": "Male",
                "role": "不破から家を買ったと主張する同居相手",
            },
        ],
        utterance_gold=[
            (0, "attributed", "boku", "直前地の文「しかし念のために訊ねてみました。」"),
            (1, "attributed", "fuwa", "直前地の文「すると不破は憐れむような笑い方で僕を見て、」"),
            (
                2,
                "not_speech",
                None,
                "直前・直後「一枚の便箋が置いてあって…そう書いてありました。もちろん不破の筆跡です。」"
                "書置き文面の引用であり口頭発話ではないため not_speech（Orion確認対象）",
            ),
            (3, "attributed", "boku", "直後地の文「と僕は訊ねました。」"),
            (4, "attributed", "noro", "直後地の文「野呂は訝しげに板の間をのぞき込みました。」"),
            (5, "attributed", "noro", "同一行・野呂の動作描写に続く同一話者継続"),
            (6, "attributed", "boku", "直後地の文「僕も不審に思った。」"),
            (7, "attributed", "boku", "同一行・僕の内面描写に続く同一話者継続"),
            (8, "attributed", "noro", "直後地の文「野呂は眉をひそめました。」"),
            (9, "attributed", "noro", "同一行・野呂の動作描写に続く同一話者継続"),
            (10, "attributed", "boku", "U09野呂の発言への直接応答かつ直後U11野呂の反論との二者交替"),
            (
                11,
                "attributed",
                "noro",
                "直後地の文「そして野呂はじろじろと僕を上から下まで眺め廻して、急に言葉がぞんざいになりました。」",
            ),
            (12, "attributed", "noro", "同一行・野呂の描写に続く同一話者継続"),
            (13, "attributed", "boku", "直後地の文「少々僕もむっとして言い返しました。」"),
            (14, "attributed", "boku", "同一行・僕の応答描写に続く同一話者継続"),
            (15, "attributed", "noro", "直後地の文「バカじゃなかろうかというような眼付で、野呂は僕を見た。」"),
            (16, "attributed", "noro", "同一行・野呂の描写に続く同一話者継続"),
            (17, "attributed", "boku", "直後地の文「今度は僕がびっくりして棒立ちになりました。」"),
            (
                18,
                "attributed",
                "noro",
                "U17僕への直接応答および「大八車で運んで来た」内容が庭へ入った野呂の行動と一致",
            ),
            (
                19,
                "attributed",
                "boku",
                "直後地の文「僕は僕の部屋にかけこんで、領収証と契約書を急いで取出し、…」",
            ),
            (20, "attributed", "noro", "直後地の文「彼はがっくりと板の間に腰をおろし、嘆息するように言いました。」"),
            (21, "attributed", "noro", "同一行・野呂（彼）の描写に続く同一話者継続"),
            (22, "attributed", "boku", "直後地の文「と僕もおろおろと訊ねました。」"),
            (23, "attributed", "boku", "同一行・僕の質問描写に続く同一話者継続"),
            (24, "attributed", "noro", "U22–U23僕の質問への直接応答（二者対話）"),
        ],
    ),
    WorkSpec(
        work_id="001875_059897",
        person_id="001875",
        aozora_work_id="059897",
        author="壺井栄",
        title="雑居家族",
        first_published="「毎日新聞」1955（昭和30）年3月25日～8月15日",
        author_death_year=1967,
        narration="third",
        narrator=None,
        pov=None,
        line_start=892,
        line_end_exclusive=937,
        selection_rationale=(
            "1955年新聞連載の戦後三人称現代家庭小説。女性話者（安江・音枝）と男性話者"
            "（兵六・文吉・進）の多人数会話を含み、身内の常体・女性終助詞と対外的な丁寧語が"
            "交錯する。30%起点（L893）に直前導入段落（L892）を補い、進君とのやりとり完結部（L936）"
            "までを抜粋した。"
        ),
        characters=[
            {
                "id": "yasue",
                "name": "安江",
                "aliases": ["安江", "お安さん", "おかあさん", "彼女"],
                "gender": "Female",
                "role": "一家の主婦",
            },
            {
                "id": "heiroku",
                "name": "兵六",
                "aliases": ["兵六さん", "兵六おじさん", "鷲さん", "彼"],
                "gender": "Male",
                "role": "安江の義兄にあたる出入りの男",
            },
            {
                "id": "otoe",
                "name": "音枝",
                "aliases": ["音枝", "音枝ちゃん"],
                "gender": "Female",
                "role": "安江の娘（主婦代わり）",
            },
            {
                "id": "bunkichi",
                "name": "文吉",
                "aliases": ["文吉", "文吉さん", "夫"],
                "gender": "Male",
                "role": "安江の夫",
            },
            {
                "id": "susumu",
                "name": "斎木進",
                "aliases": ["進君", "斎木少年", "当の進君"],
                "gender": "Male",
                "role": "安江の家に居候している少年",
            },
        ],
        utterance_gold=[
            (0, "attributed", "heiroku", "直前地の文「兵六さんはいんぎんに両手をつき、」"),
            (1, "attributed", "heiroku", "直前地の文「むっとしている安江に、兵六さんは立てつづけて、」"),
            (
                2,
                "attributed",
                "yasue",
                "U01「お安さんという人は…」への安江の応答、直後U03兵六の返しとの対話交替",
            ),
            (3, "attributed", "heiroku", "U02への応答および兵六の一人称「わし」・口調"),
            (4, "attributed", "yasue", "直後地の文「恨みがましさも忘れて安江はつい許す気になり、」"),
            (5, "attributed", "yasue", "直前地の文「安江はつい許す気になり、」および呼称「鷲さん」"),
            (
                6,
                "attributed",
                "heiroku",
                "U05への応答「文吉さんに一つとりなしてくださいよ」および直後「兵六さんはせっせとミルクを運んできてくれた」",
            ),
            (7, "attributed", "otoe", "直後地の文「音枝がそういうと、」"),
            (8, "attributed", "bunkichi", "直前地の文「文吉は文吉で、」"),
            (9, "attributed", "yasue", "直後地の文「安江が顔色をかえる。」"),
            (10, "attributed", "bunkichi", "U09への文吉の応答（「お前の姉婿だった人」）"),
            (
                11,
                "attributed",
                "yasue",
                "U10「お前の姉婿」（安江宛て）とU13「あんた」（安江の夫への呼称）に挟まれた夫婦間の連続問答"
                "（音枝同席場面のためOrion確認メモ記載）",
            ),
            (12, "attributed", "bunkichi", "U11への文吉の応答"),
            (13, "attributed", "yasue", "U12への安江の応答（夫への呼称「あんた」）"),
            (14, "attributed", "bunkichi", "直後地の文「にやっとする文吉に、」"),
            (15, "attributed", "yasue", "直前地の文「安江はくってかかる調子で、」"),
            (16, "attributed", "otoe", "直前地の文「音枝は…それをあんばいして安江に告げるのである。」および呼称「おかあさん」"),
            (17, "attributed", "yasue", "U16「おかあさん」呼びかけへの安江の応答"),
            (18, "attributed", "otoe", "U17への音枝の応答（呼称「兵六おじさん」）"),
            (19, "attributed", "yasue", "U18への安江の応答"),
            (20, "attributed", "otoe", "直後地の文「音枝は素直に引きさがるのだが、」"),
            (21, "attributed", "otoe", "直前地の文「やがてまた安江の顔色をうかがうようにして、」および呼称「おかあさん」"),
            (22, "attributed", "yasue", "U21への安江の応答"),
            (
                23,
                "attributed",
                "otoe",
                "U22安江の詰問に対する音枝の無言反応「…………」（非言語沈黙ターンをattributedとするかnot_speechとするかはOrion確認対象）",
            ),
            (24, "attributed", "yasue", "直後地の文「音枝にはぽんぽんいうくせに、…安江は柔和な顔で…」"),
            (25, "attributed", "yasue", "直前地の文「安江は柔和な顔で五百円をわたしながら、」"),
            (
                26,
                "attributed",
                "susumu",
                "U25への応答・一人称「ぼく」および直後「今時の若い人に珍らしく、両手をついての切口上を聞くと、安江は…」",
            ),
            (27, "attributed", "yasue", "直前地の文「安江はつい慰めずにいられなくなり、」"),
        ],
    ),
    WorkSpec(
        work_id="001801_001786",
        person_id="001801",
        aozora_work_id="001786",
        author="山川方夫",
        title="他人の夏",
        first_published="「中学時代　夏休み臨時増刊号第一五巻第六号」1963（昭和38）年8月5日",
        author_death_year=1965,
        narration="third",
        narrator=None,
        pov="shinichi",
        line_start=7,
        line_end_exclusive=48,
        selection_rationale=(
            "1963年発表の戦後三人称視点小説（視点人物: 慎一）。1960年代の標準的現代口語に極めて近く、"
            "敬体の少年「慎一」と常体の都会女性「女」という明確な口調対比、および言いさし・倒置・"
            "句末断片・発話内引用（父の言葉）など現代会話特有の境界ケースを豊富に含む。深夜の海辺"
            "一場面（L7..L47）を丸ごと抜粋した。"
        ),
        characters=[
            {
                "id": "shinichi",
                "name": "慎一",
                "aliases": ["慎一", "彼"],
                "gender": "Male",
                "role": "海辺の町の少年",
            },
            {
                "id": "onna",
                "name": "女",
                "aliases": ["女", "若い女"],
                "gender": "Female",
                "role": "深夜の海を泳いでいた都会の若い女性",
            },
        ],
        utterance_gold=[
            (0, "attributed", "onna", "直後地の文「若い女の声が呼んだ。」"),
            (1, "attributed", "onna", "直後地の文「女は、ひとりごとのようにいった。」"),
            (2, "attributed", "shinichi", "直後地の文「慎一はその声の方角に向いていった。」"),
            (3, "attributed", "onna", "直後地の文「女は答え、笑った。」"),
            (4, "attributed", "shinichi", "直前地の文「慎一はその方向に泳ぎ寄った。」および海の危険を告げる応答"),
            (5, "attributed", "onna", "U04への応答および直後「女は、睨むような目をしていた。」"),
            (6, "attributed", "onna", "直後地の文「女の顔は見えなかった。彼は答えた。」（U07の問いかけ側）"),
            (7, "attributed", "shinichi", "直前地の文「彼は答えた。」"),
            (8, "attributed", "onna", "U07への問いかけ・直後U09「と彼はいった。」との二者交替"),
            (9, "attributed", "shinichi", "直後地の文「と彼はいった。」"),
            (10, "attributed", "shinichi", "同一行・同一話者継続"),
            (
                11,
                "attributed",
                "shinichi",
                "直前地の文「自分でも、なぜこんなことをしゃべりはじめたのか…」慎一の内面を挟んだ同一話者継続",
            ),
            (12, "attributed", "onna", "U11への問いかけ（二者交替）"),
            (13, "attributed", "shinichi", "U12への応答（二者交替・敬体応答）"),
            (14, "attributed", "onna", "直後地の文「女はひきつったような声で笑った。」"),
            (15, "attributed", "onna", "同一行・女の描写に続く同一話者継続"),
            (16, "attributed", "shinichi", "U15への応答および直後「女はだまった。」"),
            (17, "attributed", "shinichi", "直前地の文「ゆっくりとその女のそばをまわりながら、彼はいった。」"),
            (18, "attributed", "onna", "直前地の文「女は反抗的に答えた。」"),
            (19, "attributed", "shinichi", "U18への応答および直後「女は、ヒステリックにいった。」"),
            (20, "attributed", "onna", "直前地の文「女は、ヒステリックにいった。」"),
            (21, "attributed", "shinichi", "直前地の文「あわてて、慎一はいった。」"),
            (
                22,
                "attributed",
                "shinichi",
                "直前地の文「女は無言だった。」聞き手無言を挟んだU21からの父の言葉引用継続",
            ),
            (
                23,
                "attributed",
                "shinichi",
                "直前地の文「女は怒ったような目つきで、海をみつめていた。」・一人称「ぼくの親父も」および直後「彼は岸に顔を向けた。」",
            ),
        ],
    ),
)


def _split_sentences(inner: str) -> list[str]:
    raw_parts = re.findall(r"[^。！？\?\!]+[。！？\?\!]*", inner)
    parts: list[str] = []
    for p in raw_parts:
        stripped = p.strip()
        if not stripped or _ONLY_PUNCT.match(stripped):
            continue
        parts.append(stripped)
    return parts


def classify_sentence_surface(sent: str, tokenizer: Tokenizer) -> tuple[str, str, str]:
    """Rule-based surface-grammar politeness classification for a single sentence."""
    raw = sent.strip()
    if _ONLY_PUNCT.match(raw):
        return "ambiguous", "silence_or_punct_only", "記号・沈黙のみで文末表現なし"

    has_cutoff = bool(_CUTOFF_RE.search(raw))
    core = _TRAIL_PUNCT.sub("", raw)
    if not core:
        return "ambiguous", "punct_only", "記号のみで文末表現なし"

    # Surface borderline rules checked BEFORE morphological classification
    if re.search(r"(?:んです|のです|ます)って$", core):
        return (
            "ambiguous",
            "hearsay_polite_tte",
            "伝聞「〜んですって」は敬体要素と口語伝聞が混在するため保留",
        )
    if re.search(r"でしょ$", core):
        return (
            "ambiguous",
            "contracted_desho",
            "縮約形「でしょ」は常体会話にも現れるため保留",
        )
    if re.search(r"だって$", core):
        return "ambiguous", "quotative_datte", "伝聞・引用止め「〜だって」のため保留"
    if re.search(r"って$", core):
        return (
            "ambiguous",
            "quotative_tte_ellipsis",
            "引用助詞「〜って」止め（後続述語省略）のため保留",
        )
    if has_cutoff:
        return (
            "ambiguous",
            "ellipsis_or_dash_cutoff",
            f"文末が言いさし（……／――）のため保留（末尾: {core[-6:]}）",
        )

    tokens = list(tokenizer.tokenize(core))
    if not tokens:
        return "ambiguous", "empty_tokens", "形態素解析トークンなし"

    end_idx = len(tokens) - 1
    stripped: list[str] = []
    while end_idx >= 0:
        tok = tokens[end_idx]
        pos0, pos1 = tok.part_of_speech.split(",")[:2]
        surf = tok.surface
        if (
            pos0 == "助詞"
            and pos1 in ("終助詞", "副助詞／並立助詞／終助詞", "間投助詞")
            and surf in FINAL_PARTICLES
        ):
            stripped.append(surf)
            end_idx -= 1
            continue
        if (
            surf in ("の", "もの", "もん")
            and pos0 in ("名詞", "助詞")
            and pos1 in ("非自立", "終助詞", "格助詞", "接続助詞")
        ):
            if end_idx > 0:
                prev_pos0 = tokens[end_idx - 1].part_of_speech.split(",")[0]
                if prev_pos0 in ("動詞", "形容詞", "助動詞"):
                    stripped.append(surf)
                    end_idx -= 1
                    continue
        if surf in ("ねえ", "ねぇ", "なあ", "なぁ", "かい", "かね", "かしら", "かな") and pos0 in (
            "助詞",
            "感動詞",
        ):
            stripped.append(surf)
            end_idx -= 1
            continue
        break

    if end_idx < 0:
        return "ambiguous", "particles_only", f"終助詞・感嘆語のみで文末判定不可（{core}）"

    head = tokens[end_idx]
    h_pos = head.part_of_speech.split(",")
    h_pos0, h_pos1 = h_pos[0], h_pos[1]
    h_cform = head.infl_form
    h_base = head.base_form
    h_surf = head.surface

    # Conjunctive particle at end of sentence
    if h_pos0 == "助詞" and h_pos1 == "接続助詞":
        if h_surf in ("から", "けど", "けれど", "けれども") and end_idx > 0:
            p_tok = tokens[end_idx - 1]
            p_base = p_tok.base_form
            p_pos0 = p_tok.part_of_speech.split(",")[0]
            if p_base in ("です", "ます") or (
                end_idx > 1
                and tokens[end_idx - 2].base_form in ("です", "ます")
                and p_base in ("た", "ぬ", "ん")
            ):
                return "polite", "polite_conj_end", f"敬体＋{h_surf}止め（{core[-8:]}）"
            if p_pos0 in ("動詞", "形容詞", "助動詞"):
                return "plain", "plain_conj_end", f"常体＋{h_surf}止め（{core[-8:]}）"
        return (
            "ambiguous",
            "conj_particle_ellipsis",
            f"接続助詞「{h_surf}」止め（後続省略・末尾: {core[-8:]}）",
        )

    # Polite auxiliaries / request / honorific forms in predicate tail window
    tail_window = tokens[max(0, end_idx - 3) : end_idx + 1]
    for tw in tail_window:
        if tw.base_form in ("です", "ます"):
            return "polite", "polite_aux", f"丁寧助動詞「{tw.base_form}」（末尾: {core[-8:]}）"
        if tw.base_form in ("くださる", "下さる") and tw.infl_form.startswith("命令"):
            return "polite", "polite_kudasai", f"丁寧依頼「{tw.surface}」（末尾: {core[-8:]}）"
        if tw.base_form in ("いらっしゃる", "なさる") and tw.infl_form.startswith("命令"):
            return "polite", "polite_honorific_imp", f"敬語命令「{tw.surface}」（末尾: {core[-8:]}）"

    # Plain predicates at head
    if h_pos0 == "助動詞":
        if h_base in (
            "だ",
            "た",
            "ない",
            "ぬ",
            "ん",
            "う",
            "よう",
            "まい",
            "らしい",
            "たい",
            "そうだ",
            "ようだ",
            "みたいだ",
        ):
            if h_cform in (
                "基本形",
                "未然ウ接続",
                "体言接続",
                "命令ｅ",
                "命令ｒｏ",
                "命令ｙｏ",
            ) or (h_surf in ("だろ", "だろう", "だっ", "じゃ", "な", "まい", "ん") and stripped):
                return (
                    "plain",
                    f"plain_aux_{h_base}",
                    f"常体助動詞「{h_surf}({h_base})」（末尾: {core[-8:]}）",
                )
    if h_pos0 == "動詞":
        if h_cform in ("基本形", "命令ｅ", "命令ｒｏ", "命令ｙｏ", "命令ｉ"):
            return (
                "plain",
                f"plain_verb_{h_cform}",
                f"常体動詞「{h_surf}({h_cform})」（末尾: {core[-8:]}）",
            )
    if h_pos0 == "形容詞":
        if h_cform in ("基本形", "ガル接続"):
            return (
                "plain",
                f"plain_adj_{h_cform}",
                f"常体形容詞「{h_surf}({h_cform})」（末尾: {core[-8:]}）",
            )

    stem_surface = "".join(t.surface for t in tokens[: end_idx + 1])
    if re.search(r"(?:呉れ|くれ|なさんな)$", stem_surface):
        return "plain", "plain_colloquial_imp", f"常体口語命令（末尾: {core[-8:]}）"
    if re.search(r"(?:んだ|のだ|た|だ)って$", stem_surface) and stripped:
        return "plain", "plain_hearsay_ndatte_yo", f"常体伝聞＋終助詞（末尾: {core[-8:]}）"

    return (
        "ambiguous",
        f"non_predicate_{h_pos0}_{h_pos1}",
        f"非述語/体言止め/感嘆詞/倒置等（最終語: {h_surf}[{h_pos0},{h_pos1},{h_cform}]）",
    )


def classify_politeness_surface(quote_text: str, tokenizer: Tokenizer) -> dict[str, str]:
    """Classify an outer-quoted utterance by Strict Policy X surface grammar."""
    inner = quote_text[1:-1].strip()
    if _ONLY_PUNCT.match(inner):
        return {
            "politeness": "ambiguous",
            "rule_id": "silence_or_punct_only",
            "reason": "無言記号のみで文末表現なし",
        }

    sents = _split_sentences(inner)
    if not sents:
        return {
            "politeness": "ambiguous",
            "rule_id": "silence_or_punct_only",
            "reason": "無言記号のみで文末表現なし",
        }

    results = [classify_sentence_surface(s, tokenizer) for s in sents]
    labels = [r[0] for r in results]
    last_lbl, last_rule, last_reason = results[-1]
    has_polite = "polite" in labels
    has_plain = "plain" in labels

    if has_polite and has_plain:
        detail = ", ".join(f"{s[-6:]}={r[0]}" for s, r in zip(sents, results))
        return {
            "politeness": "ambiguous",
            "rule_id": "mixed_polite_and_plain",
            "reason": f"同一発話内に敬体・常体文末が混在（{detail}）",
        }
    if last_lbl in ("polite", "plain"):
        return {
            "politeness": last_lbl,
            "rule_id": last_rule,
            "reason": last_reason,
        }
    if has_polite:
        return {
            "politeness": "ambiguous",
            "rule_id": "final_ambiguous_earlier_polite",
            "reason": f"最終文が非終止・省略・断片のため保留（{last_reason}；参考：先行文末は polite のみ）",
        }
    if has_plain:
        return {
            "politeness": "ambiguous",
            "rule_id": "final_ambiguous_earlier_plain",
            "reason": f"最終文が非終止・省略・断片のため保留（{last_reason}；参考：先行文末は plain のみ）",
        }
    return {
        "politeness": "ambiguous",
        "rule_id": last_rule,
        "reason": last_reason,
    }


def _load_index_rows() -> dict[str, dict[str, str]]:
    csv_path = next((p for p in LOCAL_INDEX_CANDIDATES if p.exists()), None)
    if csv_path is not None:
        raw_bytes = csv_path.read_bytes()
    else:
        req = urllib.request.Request(INDEX_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as resp:
            zdata = resp.read()
        with zipfile.ZipFile(io.BytesIO(zdata)) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".csv"))
            raw_bytes = zf.read(name)

    needed = {spec.work_id for spec in WORK_SPECS}
    rows: dict[str, dict[str, str]] = {}
    reader = csv.DictReader(io.StringIO(raw_bytes.decode("utf-8-sig"), newline=""))
    for row in reader:
        wid = f"{row['人物ID']}_{row['作品ID']}"
        if wid not in needed or row.get("役割フラグ") != "著者":
            continue
        if row.get("作品著作権フラグ") != "なし" or row.get("文字遣い種別") != "新字新仮名":
            continue
        rows[wid] = row

    missing = needed - set(rows)
    if missing:
        raise RuntimeError(f"Works not found or not PD 新字新仮名 in index: {sorted(missing)}")
    return rows


def _fetch_cleaned_aozora(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        zbytes = resp.read()
    time.sleep(0.5)
    with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".txt"))
        raw = zf.read(name).decode("cp932", errors="replace")
    return clean_aozora_text(raw)


def _check_non_overlap() -> None:
    for spec in WORK_SPECS:
        if spec.work_id in EXCLUDED_WORK_IDS:
            raise AssertionError(f"Work {spec.work_id} overlaps F9 gold!")
        if spec.title in EXCLUDED_TITLES:
            raise AssertionError(f"Title {spec.title} overlaps excluded works!")

    # Also scan repository manifests if present
    data_dir = REPO_ROOT / "data"
    if data_dir.exists():
        for mf in sorted(data_dir.glob("aozora*manifest.json")):
            entries = json.loads(mf.read_text(encoding="utf-8"))
            if isinstance(entries, list):
                existing_ids = {f"{e.get('author_id', '')}_{e.get('work_id', '')}" for e in entries}
                for spec in WORK_SPECS:
                    if spec.work_id in existing_ids:
                        raise AssertionError(f"Work {spec.work_id} already exists in {mf.name}")


def build_artifacts() -> dict[str, bytes]:
    """Fetch source texts, build gold/manifest/split payloads, and return filename->bytes."""
    _check_non_overlap()
    index_rows = _load_index_rows()
    tokenizer = Tokenizer()

    manifest_works: list[dict[str, Any]] = []
    gold_works: list[dict[str, Any]] = []
    files_out: dict[str, bytes] = {}

    for spec in WORK_SPECS:
        row = index_rows[spec.work_id]
        zip_url = row["テキストファイルURL"]
        card_url = row["図書カードURL"]
        full_text = _fetch_cleaned_aozora(zip_url)
        lines = full_text.split("\n")

        if spec.line_end_exclusive > len(lines):
            raise AssertionError(
                f"{spec.work_id}: expected >= {spec.line_end_exclusive} lines, got {len(lines)}"
            )

        if spec.work_id == "001801_001786":
            assert lines[spec.line_start].startswith("やはり、海は親しかった"), (
                f"Unexpected start line in {spec.work_id}: {lines[spec.line_start]!r}"
            )
            assert lines[spec.line_end_exclusive].strip() == "", "Expected blank separator at L48"
            assert lines[spec.line_end_exclusive + 1].startswith("真赤なスポーツ・カーが"), (
                "Expected scene break after L48"
            )

        excerpt = "\n".join(lines[spec.line_start : spec.line_end_exclusive]) + "\n"
        char_offset_start = sum(len(ln) + 1 for ln in lines[: spec.line_start])
        char_offset_end = char_offset_start + len(excerpt) - 1
        assert full_text[char_offset_start : char_offset_end + 1] == excerpt, (
            f"Offset slice mismatch for {spec.work_id}"
        )

        spans = extract_outer_quote_spans(excerpt)
        if len(spans) != len(spec.utterance_gold):
            raise AssertionError(
                f"{spec.work_id}: extracted {len(spans)} quotes vs {len(spec.utterance_gold)} gold entries"
            )

        roster_ids = {c["id"] for c in spec.characters}
        utterances_out: list[dict[str, Any]] = []

        for idx, ((s_off, e_off), (g_n, g_status, g_spk, g_cue)) in enumerate(
            zip(spans, spec.utterance_gold)
        ):
            if idx != g_n:
                raise AssertionError(f"{spec.work_id}: index mismatch {idx} != {g_n}")
            if g_status == "attributed":
                if g_spk not in roster_ids:
                    raise AssertionError(f"{spec.work_id} U{idx}: unknown speaker {g_spk!r}")
            else:
                if g_spk is not None:
                    raise AssertionError(f"{spec.work_id} U{idx}: non-attributed speaker must be None")

            quote_text = excerpt[s_off:e_off]
            head = quote_text[1:9]
            ctx_before = excerpt[max(0, s_off - CONTEXT_WINDOW_CHARS) : s_off]
            ctx_after = excerpt[e_off : min(len(excerpt), e_off + CONTEXT_WINDOW_CHARS)]

            pol_res = classify_politeness_surface(quote_text, tokenizer)
            if g_status == "not_speech":
                pol_label: str | None = None
                pol_rule = f"not_speech({pol_res['rule_id']})"
                pol_reason = f"[not_speech/書置き引用] {pol_res['reason']}"
            else:
                pol_label = pol_res["politeness"]
                pol_rule = pol_res["rule_id"]
                pol_reason = pol_res["reason"]

            utterances_out.append(
                {
                    "n": idx,
                    "open": s_off,
                    "close": e_off,
                    "head": head,
                    "quote_text": quote_text,
                    "context_before": ctx_before,
                    "context_after": ctx_after,
                    "status": g_status,
                    "speaker": g_spk,
                    "speaker_cue": g_cue,
                    "politeness": pol_label,
                    "politeness_rule_id": pol_rule,
                    "politeness_reason": pol_reason,
                }
            )

        excerpt_bytes = excerpt.encode("utf-8")
        files_out[f"{spec.work_id}.txt"] = excerpt_bytes

        manifest_works.append(
            {
                "id": spec.work_id,
                "author": spec.author,
                "title": spec.title,
                "first_published": spec.first_published,
                "author_death_year": spec.author_death_year,
                "char_type": row["文字遣い種別"],
                "copyright_flag": row["作品著作権フラグ"],
                "card_url": card_url,
                "zip_url": zip_url,
                "source_sha256": hashlib.sha256(full_text.encode("utf-8")).hexdigest(),
                "line_start": spec.line_start,
                "line_end": spec.line_end_exclusive - 1,
                "char_offset_start": char_offset_start,
                "char_offset_end": char_offset_end,
                "excerpt_chars": len(excerpt),
                "outer_quotes_closed": len(spans),
                "excerpt_sha256": hashlib.sha256(excerpt_bytes).hexdigest(),
                "selection_rationale": spec.selection_rationale,
            }
        )

        gold_works.append(
            {
                "id": spec.work_id,
                "author": spec.author,
                "title": spec.title,
                "narration": spec.narration,
                "narrator": spec.narrator,
                "pov": spec.pov,
                "characters": spec.characters,
                "utterances": utterances_out,
            }
        )

    manifest_obj = {
        "rule": (
            "Track A 現代文ラウンド用ゴールド下書き（Orion 確認・凍結前）。\n"
            "選定条件: 青空文庫・著作権なし・新字新仮名・没年1967年以前・戦後発表（1954-1963）・"
            "F9ゴールド8作品および既存マニフェスト対象と完全非重複・会話比率高。\n"
            "口調ラベル下書き: Janome形態素解析＋表層文末文法規則（Strict Policy X）による"
            "決定論的分類（polite / plain / ambiguous）。"
        ),
        "status": "draft_pending_orion_freeze",
        "works": manifest_works,
    }
    manifest_bytes = (json.dumps(manifest_obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    files_out["manifest.json"] = manifest_bytes

    gold_obj = {
        "note": (
            "Track A Modern Japanese Gold Draft (3 postwar Aozora works, 77 outer quotes / 76 attributed speech).\n"
            "Status: DRAFT pending Orion review & freeze. No LLM evaluation has been run.\n"
            "Politeness draft labels (polite / plain / ambiguous) were assigned deterministically from "
            "surface sentence-final grammar via Janome (no model subjectivity; borderline cases forced to ambiguous)."
        ),
        "status": "draft_pending_orion_freeze",
        "politeness_policy": {
            "labels": ["polite", "plain", "ambiguous"],
            "rule": (
                "Strict Policy X: classify each sentence inside outer quotes by surface ending only; "
                "mix of polite & plain -> ambiguous; final sentence non-finite/ellipsis/fragment -> ambiguous "
                "(even if earlier sentences were polite or plain); hearsay 「〜んですって」 / contracted 「でしょ」 / "
                "quotative 「〜って」「〜だって」 / cutoff 「……」「――」 -> ambiguous."
            ),
        },
        "works": gold_works,
    }
    gold_bytes = (json.dumps(gold_obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    files_out["gold.json"] = gold_bytes

    gold_sha256 = hashlib.sha256(gold_bytes).hexdigest()
    split_obj = {
        "split_version": "modern_gold_draft_v1",
        "created_at": "2026-11-20",
        "status": "draft_pending_orion_freeze",
        "source_gold_dir": "scripts/step1/modern_gold",
        "gold_json_sha256": gold_sha256,
        "split_policy": (
            "Draft split containing all 3 postwar modern-Japanese excerpts under 'dev' for loader "
            "and nishiki segment verification prior to Orion review & freeze."
        ),
        "dev": {
            "work_ids": [spec.work_id for spec in WORK_SPECS],
            "summary": _compute_summary(gold_works),
        },
        "holdout": {
            "sealed": True,
            "work_ids": [],
            "summary": {
                "works": 0,
                "attributed_utterances": 0,
                "undetermined_utterances": 0,
                "not_speech_utterances": 0,
                "total_outer_quotes": 0,
            },
        },
    }
    split_bytes = (json.dumps(split_obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    files_out["modern_split.json"] = split_bytes

    return files_out


def _compute_summary(gold_works: list[dict[str, Any]]) -> dict[str, Any]:
    status_cnt: Counter[str] = Counter()
    pol_cnt: Counter[str] = Counter()
    per_work: list[dict[str, Any]] = []
    for w in gold_works:
        ws = Counter(u["status"] for u in w["utterances"])
        wp = Counter(
            u["politeness"] for u in w["utterances"] if u["status"] == "attributed"
        )
        status_cnt.update(ws)
        pol_cnt.update(wp)
        per_work.append(
            {
                "work_id": w["id"],
                "author": w["author"],
                "title": w["title"],
                "narration": w["narration"],
                "total_quotes": len(w["utterances"]),
                "attributed": ws["attributed"],
                "undetermined": ws["undetermined"],
                "not_speech": ws["not_speech"],
                "politeness_attributed": {
                    "polite": wp["polite"],
                    "plain": wp["plain"],
                    "ambiguous": wp["ambiguous"],
                },
            }
        )
    return {
        "works": len(gold_works),
        "first_person_works": sum(1 for w in gold_works if w["narration"] == "first"),
        "third_person_works": sum(1 for w in gold_works if w["narration"] == "third"),
        "attributed_utterances": status_cnt["attributed"],
        "undetermined_utterances": status_cnt["undetermined"],
        "not_speech_utterances": status_cnt["not_speech"],
        "total_outer_quotes": sum(status_cnt.values()),
        "politeness_attributed": {
            "polite": pol_cnt["polite"],
            "plain": pol_cnt["plain"],
            "ambiguous": pol_cnt["ambiguous"],
        },
        "per_work": per_work,
    }


def verify_on_disk(expected_files: dict[str, bytes]) -> None:
    """Verify byte-for-byte reproducibility and all dataset invariants on disk."""
    for fname, exp_bytes in expected_files.items():
        path = HERE / fname
        if not path.exists():
            raise AssertionError(f"Missing generated file: {path}")
        actual_bytes = path.read_bytes()
        if actual_bytes != exp_bytes:
            raise AssertionError(
                f"Reproducibility SHA-256 mismatch for {fname}: "
                f"disk={hashlib.sha256(actual_bytes).hexdigest()} != "
                f"regen={hashlib.sha256(exp_bytes).hexdigest()}"
            )

    gold = json.loads((HERE / "gold.json").read_text(encoding="utf-8"))
    split = json.loads((HERE / "modern_split.json").read_text(encoding="utf-8"))
    actual_gold_sha = hashlib.sha256((HERE / "gold.json").read_bytes()).hexdigest()
    assert split["gold_json_sha256"] == actual_gold_sha, "gold_json_sha256 mismatch in split"

    for w in gold["works"]:
        wid = w["id"]
        assert wid not in EXCLUDED_WORK_IDS, f"Overlap with F9: {wid}"
        assert w["title"] not in EXCLUDED_TITLES, f"Overlap with excluded title: {w['title']}"
        txt = (HERE / f"{wid}.txt").read_text(encoding="utf-8")
        spans = extract_outer_quote_spans(txt)
        utts = w["utterances"]
        assert len(spans) == len(utts), f"{wid}: span count {len(spans)} != {len(utts)}"
        roster = {c["id"] for c in w["characters"]}

        for idx, ((s_off, e_off), u) in enumerate(zip(spans, utts)):
            assert u["n"] == idx
            assert u["open"] == s_off and u["close"] == e_off
            assert u["quote_text"] == txt[s_off:e_off]
            if u["status"] == "attributed":
                assert u["speaker"] in roster, f"{wid} U{idx}: speaker {u['speaker']} not in roster"
                assert u["politeness"] in ("polite", "plain", "ambiguous")
            else:
                assert u["speaker"] is None
                assert u["politeness"] is None
            if u["politeness"] == "ambiguous":
                assert u["politeness_reason"] and u["politeness_reason"].strip(), (
                    f"{wid} U{idx}: missing politeness_reason for ambiguous"
                )
            assert u["speaker_cue"] and u["speaker_cue"].strip()

    summary = _compute_summary(gold["works"])
    assert summary["total_outer_quotes"] == 77, f"Expected 77 total quotes, got {summary}"
    assert summary["attributed_utterances"] == 76
    assert summary["not_speech_utterances"] == 1
    assert summary["undetermined_utterances"] == 0
    assert summary["politeness_attributed"] == {"polite": 17, "plain": 30, "ambiguous": 29}, (
        f"Unexpected politeness counts: {summary['politeness_attributed']}"
    )
    print("VERIFY OK:", json.dumps(summary, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Re-fetch/rebuild in memory and verify SHA-256 and invariants against files on disk.",
    )
    args = parser.parse_args(argv)

    files_out = build_artifacts()
    if args.verify:
        verify_on_disk(files_out)
        return 0

    HERE.mkdir(parents=True, exist_ok=True)
    for fname, data in files_out.items():
        (HERE / fname).write_bytes(data)

    gold = json.loads(files_out["gold.json"].decode("utf-8"))
    summary = _compute_summary(gold["works"])
    print("GENERATED:", json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
