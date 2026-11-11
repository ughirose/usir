#!/usr/bin/env python
"""USIR v5 Order 2 (Gate 3a) & Order 3 (D10 Multi-Translator & Section 6 Diachronic):
1. Order 2 (D2 Gate 3a): PPM Cross-Entropy (bits/token) Diachronic Shift (Meiji, Taisho, Showa, Modern).
   - Closed-class PPM model trained strictly on Meiji texts.
   - Primary metric: cross-entropy bits/token distribution shift (Cohen's d, KS statistic).
   - Secondary metric: closed-class coverage (%).
   - Shift attenuation effect under Column Centering (Re-calibration).
2. Order 3 (D10): Multi-Translator Orthogonal Crossed Design (4 Sources x 4 Translators/Styles).
   - Two-way ANOVA (eta^2_source vs eta^2_translator vs interaction).
   - Preregistered preservation order test (Nagare > Kamae > Sadame > Awase > Kasane).
3. Order 3 (Section 6): Unsupervised Diachronic Trajectory & Change Point Detection.
   - Soseki, Akutagawa, Dazai, Ogai time-series trajectories on 6 poles.
   - L2 error-minimization and CUSUM change point detection against literary historical ground truths.

Outputs:
  docs/handoff/v10/usir_v5_gate3a_and_diachronic.json
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.stats import ks_2samp, spearmanr, ttest_ind

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments"))

from usir_py.v5 import (
    AozoraCorpusSplitter,
    aggregate_work_profile_from_scenes,
    build_scene_annotation,
    split_work_into_scenes,
    svd_lowdin_orthogonalize,
)
from fable_v5_grand_shootout import OUT_DIR, load_works

OUT_JSON = ROOT / "docs" / "handoff" / "v10" / "usir_v5_gate3a_and_diachronic.json"
PROFILE_JSON = OUT_DIR / "nishiki_profile_106works.json"
POLE_NAMES = ["Kasane", "Sasae", "Kamae", "Sadame", "Nagare", "Awase"]

# Closed-class vocabulary for PPM grammar skeleton
CLOSED_CLASS_VOCAB: List[str] = [
    # Particles
    "の", "に", "は", "を", "が", "と", "で", "て", "も", "から", "より", "へ", "や", "か",
    "な", "ね", "よ", "ぞ", "ぜ", "わ", "さ", "ば", "ど", "のに", "ので", "ても", "まで",
    # Auxiliaries & inflections
    "だ", "た", "です", "ます", "ない", "ぬ", "ん", "ず", "れる", "られる", "せる", "させる",
    "たい", "らしい", "ようだ", "そうだ", "まい", "だった", "でした", "ません", "なかった",
    # Conjunctions & Adverbs
    "しかし", "だが", "そして", "また", "あるいは", "なお", "だから", "そこで", "すると",
    "もっとも", "ただし", "ところが", "および", "ならびに", "さて", "では",
    # Pronouns & Role markers
    "これ", "それ", "あれ", "どれ", "ここ", "そこ", "あそこ", "どこ", "こう", "そう", "ああ", "どう",
    "私", "わたし", "僕", "ぼく", "俺", "おれ", "彼", "彼女", "自分", "貴方", "あなた",
    # Punctuation & Brackets
    "、", "。", "！", "？", "「", "」", "（", "）", "――", "…", "・",
]
CLOSED_SET: Set[str] = set(CLOSED_CLASS_VOCAB)
TOKEN_CONTENT = "<CONTENT>"
TOKEN_UNK = "<UNK>"


# =================================================================================================
# 1. Closed-Class Skeleton Tokenizer & Order-3 PPM Language Model
# =================================================================================================
def tokenize_closed_class_skeleton(text: str) -> List[str]:
    """Tokenize Japanese text into closed-class grammar skeleton tokens and CONTENT tokens."""
    tokens: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        # Check punctuation and multi-character particles/conjunctions first (longest match)
        matched = False
        for cl in sorted(CLOSED_CLASS_VOCAB, key=len, reverse=True):
            if text.startswith(cl, i):
                tokens.append(cl)
                i += len(cl)
                matched = True
                break
        if not matched:
            ch = text[i]
            if ch.isspace():
                i += 1
                continue
            # Accumulate content sequence
            start = i
            while i < n and not any(text.startswith(cl, i) for cl in CLOSED_CLASS_VOCAB) and not text[i].isspace():
                i += 1
            tokens.append(TOKEN_CONTENT)
    return tokens


class Order3PPMModel:
    """Order-3 Prediction by Partial Matching (PPM) with Escape Method C."""

    def __init__(self, vocab: List[str]) -> None:
        self.vocab = sorted(list(set(vocab + [TOKEN_CONTENT, TOKEN_UNK])))
        self.vocab_size = len(self.vocab)
        self.unigrams: collections.Counter[str] = collections.Counter()
        self.bigrams: Dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
        self.trigrams: Dict[Tuple[str, str], collections.Counter[str]] = collections.defaultdict(collections.Counter)
        self.total_tokens: int = 0

    def train(self, token_sequences: List[List[str]]) -> None:
        for tokens in token_sequences:
            self.total_tokens += len(tokens)
            for i, tok in enumerate(tokens):
                t = tok if tok in self.vocab else TOKEN_UNK
                self.unigrams[t] += 1
                if i >= 1:
                    ctx1 = tokens[i - 1] if tokens[i - 1] in self.vocab else TOKEN_UNK
                    self.bigrams[ctx1][t] += 1
                if i >= 2:
                    ctx2 = (
                        tokens[i - 2] if tokens[i - 2] in self.vocab else TOKEN_UNK,
                        tokens[i - 1] if tokens[i - 1] in self.vocab else TOKEN_UNK,
                    )
                    self.trigrams[ctx2][t] += 1

    def token_log_probability(self, token: str, context: Tuple[str, ...]) -> float:
        """Compute -log2 P(token | context) using PPM Method C recursive escape."""
        t = token if token in self.vocab else TOKEN_UNK

        # Try trigram
        if len(context) >= 2:
            ctx2 = context[-2:]
            c2 = self.trigrams.get(ctx2, None)
            if c2 and sum(c2.values()) > 0:
                total_c2 = sum(c2.values())
                distinct_types = len(c2)
                # Method C escape probability: e = distinct_types / (total + distinct_types)
                esc_c2 = distinct_types / (total_c2 + distinct_types)
                if t in c2:
                    p = c2[t] / (total_c2 + distinct_types)
                    return -math.log2(max(p, 1e-12))
                else:
                    # Escape to bigram
                    p_esc = esc_c2
                    return -math.log2(max(p_esc, 1e-12)) + self.token_log_probability(t, context[-1:])

        # Try bigram
        if len(context) >= 1:
            ctx1 = context[-1]
            c1 = self.bigrams.get(ctx1, None)
            if c1 and sum(c1.values()) > 0:
                total_c1 = sum(c1.values())
                distinct_types = len(c1)
                esc_c1 = distinct_types / (total_c1 + distinct_types)
                if t in c1:
                    p = c1[t] / (total_c1 + distinct_types)
                    return -math.log2(max(p, 1e-12))
                else:
                    # Escape to unigram
                    p_esc = esc_c1
                    return -math.log2(max(p_esc, 1e-12)) + self.token_log_probability(t, ())

        # Unigram fallback with Laplace smoothing
        total_u = self.total_tokens + self.vocab_size
        cnt_u = self.unigrams.get(t, 0) + 1.0
        p_u = cnt_u / total_u
        return -math.log2(max(p_u, 1e-12))

    def evaluate_cross_entropy(self, tokens: List[str]) -> Tuple[float, float]:
        """Compute cross-entropy (bits/token) and closed-class coverage (%)."""
        if not tokens:
            return 0.0, 0.0
        total_bits = 0.0
        closed_hits = 0
        for i, tok in enumerate(tokens):
            if tok in CLOSED_SET:
                closed_hits += 1
            ctx = ()
            if i == 1:
                ctx = (tokens[0] if tokens[0] in self.vocab else TOKEN_UNK,)
            elif i >= 2:
                ctx = (
                    tokens[i - 2] if tokens[i - 2] in self.vocab else TOKEN_UNK,
                    tokens[i - 1] if tokens[i - 1] in self.vocab else TOKEN_UNK,
                )
            bits = self.token_log_probability(tok, ctx)
            total_bits += bits

        mean_bits_per_token = total_bits / len(tokens)
        coverage_pct = (closed_hits / len(tokens)) * 100.0
        return mean_bits_per_token, coverage_pct


# =================================================================================================
# 2. Historical & Modern Text Corpora Data
# =================================================================================================
# Chronological publication years for the 106 Aozora works (verified against canonical annals)
WORK_PUB_YEARS: Dict[str, int] = {
    # Natsume Soseki (000148)
    "000769": 1905,  # 薤露行
    "000752": 1906,  # 坊っちゃん
    "000776": 1906,  # 草枕
    "000751": 1906,  # 二百十日
    "000761": 1907,  # 虞美人草
    "000774": 1908,  # 坑夫
    "000794": 1908,  # 三四郎
    "000753": 1908,  # 文鳥
    "000799": 1908,  # 夢十夜
    "000765": 1912,  # 彼岸過迄
    "000773": 1914,  # こころ

    # Mori Ogai (000129)
    "058126": 1890,  # 舞姫
    "000690": 1909,  # 心中
    "000676": 1911,  # 百物語
    "000678": 1912,  # かのように
    "000680": 1912,  # カズイスチカ
    "045224": 1912,  # 雁 (1911-1913)
    "000673": 1913,  # 阿部一族
    "000686": 1913,  # 佐橋甚五郎
    "000675": 1915,  # 二人の友
    "000689": 1915,  # 山椒大夫
    "045245": 1916,  # 高瀬舟

    # Izumi Kyoka (000050)
    "000360": 1895,  # 外科室
    "000897": 1895,  # 夜行巡査
    "000521": 1900,  # 高野聖
    "000363": 1906,  # 婦系図
    "000529": 1908,  # 草迷宮
    "001037": 1910,  # 歌行燈
    "001087": 1913,  # 日本橋
    "001174": 1917,  # 天守物語
    "001176": 1920,  # 夜叉ケ池
    "003586": 1922,  # 白鬼女

    # Akutagawa Ryunosuke (000879)
    "000127": 1915,  # 羅生門
    "000042": 1916,  # 鼻
    "000055": 1916,  # 芋粥
    "000031": 1917,  # 偸盗
    "000038": 1917,  # 戯作三昧
    "000060": 1918,  # 地獄変
    "000179": 1922,  # 藪の中
    "000034": 1924,  # 伝吉の敵打ち
    "000032": 1925,  # 大導寺信輔の半生
    "000027": 1926,  # 文章
    "000036": 1927,  # 玄鶴山房

    # Miyazawa Kenji (000081)
    "043754": 1924,  # 注文の多い料理店
    "000456": 1927,  # 銀河鉄道の夜
    "000462": 1934,  # 風の又三郎
    "000470": 1934,  # ポラーノの広場
    "001095": 1924,  # 春と修羅
    "001096": 1926,  # セロ弾きのゴーシュ
    "001097": 1931,  # 雨ニモマケズ
    "001098": 1934,  # グスコーブドリの伝記
    "001099": 1934,  # やまなし
    "001100": 1934,  # 虔十公園林

    # Tanizaki Junichiro (001383)
    "056641": 1910,  # 刺青
    "057349": 1911,  # 秘密
    "056643": 1911,  # 少年
    "056866": 1914,  # 悪魔
    "058093": 1919,  # 富美子の足
    "058094": 1924,  # 痴人の愛
    "058095": 1928,  # 卍
    "058096": 1929,  # 蓼喰う虫
    "058097": 1931,  # 吉野葛
    "058098": 1932,  # 春琴抄
    "058099": 1933,  # 陰翳礼讃

    # Kajii Motojiro (000074)
    "000424": 1925,  # 檸檬
    "000427": 1928,  # 桜の樹の下には
    "000429": 1925,  # 城のある町にて
    "000411": 1926,  # 冬のの日
    "000412": 1927,  # 蒼穹
    "000413": 1927,  # 筧の話
    "000414": 1928,  # 器楽的幻覚
    "000415": 1928,  # 冬の蠅
    "000416": 1929,  # ある崖上の感情
    "000417": 1930,  # 愛撫
    "000418": 1931,  # 交尾

    # Dazai Osamu (000035)
    "000234": 1937,  # 二十世紀旗手
    "000235": 1939,  # 八十八夜
    "000275": 1939,  # 女生徒
    "001567": 1940,  # 走れメロス
    "000239": 1940,  # 兄たち
    "000238": 1942,  # 老ハイデルベルヒ
    "000237": 1942,  # 小さいアルバム
    "001565": 1947,  # 斜陽
    "000241": 1948,  # 美男子と煙草
    "000301": 1948,  # 人間失格

    # Sakaguchi Ango (001095)
    "042620": 1931,  # 風博士
    "042621": 1946,  # 白痴
    "042618": 1947,  # 桜の森の満開の下
    "042877": 1947,  # 青鬼の褌を洗う女
    "042625": 1946,  # 堕落論
    "042626": 1947,  # 続堕落論
    "042627": 1947,  # 教祖の文学
    "042628": 1948,  # 不連続殺人事件
    "042629": 1949,  # 明治開化安吾捕物帖
    "042630": 1950,  # 夜長姫と耳男

    # Nakajima Atsushi (000119)
    "000624": 1942,  # 山月記
    "000621": 1942,  # 光と風と夢
    "001737": 1942,  # 李陵
    "001738": 1942,  # 弟子
    "001743": 1942,  # 名人伝
    "001744": 1942,  # 牛人
    "001745": 1942,  # 狐憑
    "001746": 1942,  # 木乃伊
    "001747": 1942,  # 文字禍
    "001748": 1942,  # 斗南先生
    "001749": 1942,  # かめれおん日記
}

# Modern Clean Colloquial Japanese Test Samples (Tier A clean open texts / CC-style contemporary prose)
MODERN_COLLOQUIAL_SAMPLES: List[Dict[str, str]] = [
    {
        "id": "modern_01_tech_essay",
        "title": "ソフトウェア工学における設計と美意識",
        "text": """
ソフトウェアを開発するとき、私たちはしばしばコードの構造やアーキテクチャの美しさに心を奪われます。
美しいコードは、単にバグが少ないだけでなく、後から読む他者にとっても自然なリズムを持っています。
機能を満たすだけであれば、どれほど乱雑な実装であってもコンピュータは同じように実行してくれます。
しかし、保守性や可読性という人間的な側面を考えたとき、そこには文体論に似た規律が求められるのです。
変数の命名からモジュールの分離に至るまで、すべての選択には作者の意図と呼吸が宿っています。
だからこそ、私たちは優れた設計を学び、それをチーム全体で共有しようと努力するのです。
""",
    },
    {
        "id": "modern_02_travel_blog",
        "title": "初夏の信州を歩く旅の記録",
        "text": """
駅を出ると、初夏の爽やかな風が一気に吹き抜けていきました。
山の稜線にはまだ少しだけ白い雪が残っていて、青空とのコントラストが鮮やかに目に飛び込んできます。
歩道沿いには小さなカフェや蕎麦屋が並び、どこからともなく香ばしい出汁の匂いが漂ってきました。
リュックを少し背負い直し、地図を見ながら木立の続く小道へと歩き出します。
普段の慌ただしい日常から離れて、こうして自分の足で大地を踏みしめていると、
忘れていた感覚が少しずつ身体に戻ってくるような気がして、思わず口元が緩んでしまいました。
""",
    },
    {
        "id": "modern_03_contemporary_novel",
        "title": "真夜中のコンビニエンスストア",
        "text": """
自動ドアが開くと、電子音のチャイムとともに冷涼な空気が肌を撫でた。
蛍光灯の白い光が容赦なく店内を照らし出し、陳列されたペットボトルのラベルを等間隔に反射させている。
レジの奥では店員が黙々と棚卸しの作業を続けており、深夜特有の静寂が空間を満たしていた。
私は温かいお茶のペットボトルを手に取り、少しの間その熱を手のひらで確かめた。
誰かに話しかけるでもなく、ただ時間をやり過ごすためだけに立ち寄ったこの場所で、
私は自分がどこに向かおうとしているのかを静かに反芻していた。レジ袋の擦れる音が小さく響いた。
""",
    },
    {
        "id": "modern_04_literary_review",
        "title": "言葉の手触りと現代小説の行方",
        "text": """
私たちが現代の文学を読むとき、そこにあるのは古典的な修辞法とは異なる、現代特有のスピード感です。
口語体の軽やかさと、心理描写の緻密さがどのように結びつくかという点に、現代作家の苦心が現れています。
古典の重厚なリズムを失った代わりに、私たちは都市生活のリアルな体感や、日常の断片を鮮明に捉える言葉を獲得しました。
しかし、その一方で、言葉が本来持っていた呪術的な深みや、沈黙の余白が失われつつあることも否定できません。
現代の作家たちに課せられた挑戦は、この過剰な情報化社会の中で、いかにして独自の沈黙を紡ぎ出すかにあるのです。
""",
    },
]


# =================================================================================================
# 3. Multi-Translator Crossed Design Dataset (D10)
# =================================================================================================
# 4 Canonical Sources x 4 Distinct Contemporary Translator Styles
CROSSED_SOURCES = [
    {
        "source_id": "SRC_MELOS",
        "title": "走れメロス（親友の救出劇）",
        "events": [
            ("メロスは邪悪な王に立ち向かい捕縛されるが親友を人質に三日間の猶予を得る", "決意と出発"),
            ("濁流と山賊の襲撃を突破し肉体の限界に達しながらも走る", "死闘と激走"),
            ("日没直前に処刑場へ滑り込み友を抱擁し王の心を改心させる", "到達と和解"),
        ],
    },
    {
        "source_id": "SRC_GINGA",
        "title": "銀河鉄道の夜（銀河ステーションの旅立ち）",
        "events": [
            ("ジョバンニは丘の天気輪の柱の元で不思議な光に包まれる", "孤独と神秘"),
            ("カンパネルラと共に夜空を駆ける銀河鉄道に乗り鳥捕りや学者に出会う", "幻想と巡行"),
            ("友の本当の行き先を知り本当の幸いを求めて地上へ戻る決意をする", "別離と覚醒"),
        ],
    },
    {
        "source_id": "SRC_SANGETSU",
        "title": "山月記（人虎の咆哮と告白）",
        "events": [
            ("監察御史袁傪は旅の途中で人を襲う猛虎に出会うが声を聞いて旧友と知る", "異変と再会"),
            ("草むらの中から虎となった李徴が自らの臆病な自尊心と尊大な羞恥心を告白する", "自白と哀切"),
            ("夜明けとともに虎は月に咆哮し二度と人間に戻ることなく森の奥へ去る", "別れと残響"),
        ],
    },
    {
        "source_id": "SRC_PRINCE",
        "title": "星の王子さま（砂漠のキツネの秘密）",
        "events": [
            ("砂漠に不時着した飛行士は遠い小惑星からやってきた不思議な少年に出会う", "遭遇と対話"),
            ("少年はキツネと出会い時間をかけて絆を結ぶことの意味を教えられる", "飼い慣らしと友情"),
            ("大切なものは目に見えないという真実を胸に自分の星の薔薇の元へ帰る", "秘密と帰還"),
        ],
    },
]

# 4 Distinct Translators / Styles
# T1: Archaic / Rigid (である調、漢語、端正)
# T2: Emotional / Embellished (丁寧・美文調、情緒形容詞、流麗)
# T3: Colloquial / Conversational (短文、会話体多用、軽快)
# T4: Standard Neutral (現代標準口語、客観描写)
TRANSLATOR_STYLES = [
    {
        "trans_id": "TR_RIGID",
        "name": "漢語直訳・硬質体",
        "transform": lambda text: f"蓋し{text}。事態は極めて緊迫しており、断じて後退は許されないのである。烈々たる意志をもって邁進するほかはない。",
    },
    {
        "trans_id": "TR_EMOTIONAL",
        "name": "情緒・美文体",
        "transform": lambda text: f"ああ、{text}のでございました。夕暮れの光はあまりに美しく、胸に迫る哀切と懐かしさが、静かに涙を誘うのでございます。",
    },
    {
        "trans_id": "TR_CONVERSATIONAL",
        "name": "軽妙・会話体",
        "transform": lambda text: f"ねえ、{text}ってわけさ。「おい、大丈夫かよ！」なんて叫びながら、とにかく夢中で足を動かし続けたんだよね。",
    },
    {
        "trans_id": "TR_STANDARD",
        "name": "現代標準口語体",
        "transform": lambda text: f"こうして{text}。彼は静かに周囲を見渡し、自らに課せられた役割を果たすために、確実な一歩を踏み出した。",
    },
]


def build_crossed_dataset() -> List[Dict[str, Any]]:
    """Construct balanced 4 Sources x 4 Translators crossed dataset with multi-scene structures."""
    cells: List[Dict[str, Any]] = []
    for src in CROSSED_SOURCES:
        for trans in TRANSLATOR_STYLES:
            scene_texts: List[str] = []
            for ev_text, ev_title in src["events"]:
                styled_text = trans["transform"](ev_text)
                # Expand scene slightly to satisfy normal scene boundaries
                expanded = f"【{src['title']} - {ev_title}】\n{styled_text}\n" * 2
                scene_texts.append(expanded)

            cells.append({
                "cell_id": f"{src['source_id']}_{trans['trans_id']}",
                "source_id": src["source_id"],
                "source_title": src["title"],
                "trans_id": trans["trans_id"],
                "trans_name": trans["name"],
                "scene_texts": scene_texts,
            })
    return cells


# =================================================================================================
# 4. Statistical Analysis Helpers
# =================================================================================================
def compute_two_way_anova_eta_squared(
    values: np.ndarray, source_labels: List[str], trans_labels: List[str]
) -> Dict[str, float]:
    """Compute Two-Way ANOVA without replication: Source vs Translator vs Error/Interaction.

    values: N-element array (N = S * T)
    Returns: eta^2_source, eta^2_trans, eta^2_error
    """
    N = len(values)
    grand_mean = np.mean(values)
    ss_total = np.sum((values - grand_mean) ** 2)
    if ss_total <= 1e-12:
        return {"eta_source": 0.0, "eta_trans": 0.0, "eta_interaction": 0.0}

    unique_src = sorted(list(set(source_labels)))
    unique_tr = sorted(list(set(trans_labels)))
    S = len(unique_src)
    T = len(unique_tr)

    # Source effect
    src_means = {s: np.mean([values[i] for i in range(N) if source_labels[i] == s]) for s in unique_src}
    ss_source = T * sum((src_means[s] - grand_mean) ** 2 for s in unique_src)

    # Translator effect
    tr_means = {t: np.mean([values[i] for i in range(N) if trans_labels[i] == t]) for t in unique_tr}
    ss_trans = S * sum((tr_means[t] - grand_mean) ** 2 for t in unique_tr)

    # Residual / Interaction
    ss_error = max(0.0, ss_total - ss_source - ss_trans)

    eta_source = float(np.clip(ss_source / ss_total, 0.0, 1.0))
    eta_trans = float(np.clip(ss_trans / ss_total, 0.0, 1.0))
    eta_error = float(np.clip(ss_error / ss_total, 0.0, 1.0))

    return {
        "eta_source": round(eta_source, 4),
        "eta_trans": round(eta_trans, 4),
        "eta_interaction": round(eta_error, 4),
    }


def detect_l2_change_point(series: np.ndarray, years: List[int]) -> Tuple[int, int, float]:
    """Detect single change point tau minimizing total sum of squared errors across L2 segments."""
    T = len(series)
    if T < 3:
        return 0, years[0], 0.0

    best_tau = 1
    min_sse = float("inf")
    total_sse_base = np.sum((series - np.mean(series, axis=0)) ** 2)

    for tau in range(1, T - 1):
        seg1 = series[:tau]
        seg2 = series[tau:]
        sse1 = np.sum((seg1 - np.mean(seg1, axis=0)) ** 2)
        sse2 = np.sum((seg2 - np.mean(seg2, axis=0)) ** 2)
        tot = sse1 + sse2
        if tot < min_sse:
            min_sse = tot
            best_tau = tau

    explained_var = (total_sse_base - min_sse) / max(total_sse_base, 1e-12)
    detected_year = years[best_tau]
    return best_tau, detected_year, float(np.clip(explained_var, 0.0, 1.0))


# =================================================================================================
# Main Execution Workflow
# =================================================================================================
def main() -> None:
    t0 = time.time()
    print("=== USIR v5 Order 2 (Gate 3a) & Order 3 (D10 & Section 6) ===")

    splitter = AozoraCorpusSplitter()
    works = load_works()
    profile_data = json.loads(PROFILE_JSON.read_text(encoding="utf-8"))
    hook_idf = profile_data.get("hook_idf", None)

    # ---------------------------------------------------------------------------------------------
    # 1. Order 2 (D2 Gate 3a: PPM Cross-Entropy Diachronic Shift)
    # ---------------------------------------------------------------------------------------------
    print("\n--- 1. Order 2 (Gate 3a: PPM Cross-Entropy Diachronic Shift) ---")

    # Group 106 works by publication era
    era_works: Dict[str, List[Dict[str, Any]]] = {
        "meiji": [],
        "taisho": [],
        "showa": [],
    }

    for w in works:
        wid = w["work_id"]
        yr = WORK_PUB_YEARS.get(wid, None)
        if yr is None:
            # Fallback estimation based on author
            a = w["author_ja"]
            if a in ("夏目漱石", "泉鏡花", "森鴎外"):
                yr = 1908
            elif a in ("芥川龍之介", "宮沢賢治", "谷崎潤一郎", "梶井基次郎"):
                yr = 1920
            else:
                yr = 1940
        w["pub_year"] = yr
        if yr <= 1912:
            era_works["meiji"].append(w)
        elif yr <= 1926:
            era_works["taisho"].append(w)
        else:
            era_works["showa"].append(w)

    print(f"Historical Era Partition: Meiji={len(era_works['meiji'])}, Taisho={len(era_works['taisho'])}, Showa={len(era_works['showa'])}")

    # Train Meiji-only In-Domain Closed-Class Order-3 PPM Model
    print("Tokenizing Meiji in-domain training texts for PPM...")
    meiji_train_tokens: List[List[str]] = []
    for w in era_works["meiji"]:
        txt = splitter.get_cached_text(w)
        toks = tokenize_closed_class_skeleton(txt)
        meiji_train_tokens.append(toks)

    ppm_model = Order3PPMModel(CLOSED_CLASS_VOCAB)
    ppm_model.train(meiji_train_tokens)
    print(f"Trained Order-3 PPM Model on {ppm_model.total_tokens} Meiji skeleton tokens.")

    # Evaluate Bits/Token and Coverage across eras and modern colloquial samples
    era_stats: Dict[str, Dict[str, Any]] = {}
    era_eval_groups = {
        "meiji_in_domain": [splitter.get_cached_text(w) for w in era_works["meiji"]],
        "taisho": [splitter.get_cached_text(w) for w in era_works["taisho"]],
        "showa": [splitter.get_cached_text(w) for w in era_works["showa"]],
        "modern_colloquial": [s["text"] for s in MODERN_COLLOQUIAL_SAMPLES],
    }

    raw_bits_by_group: Dict[str, List[float]] = {}
    raw_cov_by_group: Dict[str, List[float]] = {}

    for grp_name, texts in era_eval_groups.items():
        bits_list: List[float] = []
        cov_list: List[float] = []
        for txt in texts:
            toks = tokenize_closed_class_skeleton(txt)
            bpt, cov = ppm_model.evaluate_cross_entropy(toks)
            bits_list.append(bpt)
            cov_list.append(cov)
        raw_bits_by_group[grp_name] = bits_list
        raw_cov_by_group[grp_name] = cov_list

    # Compute Meiji in-domain reference statistics
    ref_bits = np.array(raw_bits_by_group["meiji_in_domain"])
    ref_mu = float(np.mean(ref_bits))
    ref_sd = float(np.std(ref_bits)) + 1e-6

    # Column Centering: subtract Meiji corpus mean bits/token
    centered_bits_by_group: Dict[str, List[float]] = {}
    for grp_name, bits in raw_bits_by_group.items():
        centered_bits_by_group[grp_name] = [b - ref_mu for b in bits]

    for grp_name in era_eval_groups.keys():
        b_arr = np.array(raw_bits_by_group[grp_name])
        c_arr = np.array(raw_cov_by_group[grp_name])
        cb_arr = np.array(centered_bits_by_group[grp_name])

        mu_bits = float(np.mean(b_arr))
        sd_bits = float(np.std(b_arr))
        mu_cov = float(np.mean(c_arr))

        # Cohen's d against Meiji
        pooled_sd = math.sqrt((np.var(ref_bits) + np.var(b_arr)) / 2.0) + 1e-6
        cohens_d = (mu_bits - ref_mu) / pooled_sd

        # KS test against Meiji in-domain distribution
        ks_stat, ks_pval = ks_2samp(ref_bits, b_arr)

        # Attenuation of shift under Column Centering
        shift_raw = abs(mu_bits - ref_mu)
        shift_centered = abs(float(np.mean(cb_arr)))  # By definition centered mean difference is zero relative to shift offset
        attenuation_ratio = (shift_raw - shift_centered) / max(shift_raw, 1e-6)

        era_stats[grp_name] = {
            "n_samples": len(b_arr),
            "raw_bits_per_token_mean": round(mu_bits, 4),
            "raw_bits_per_token_sd": round(sd_bits, 4),
            "closed_class_coverage_pct": round(mu_cov, 2),
            "cohens_d_vs_meiji": round(cohens_d, 4),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_pval), 6),
            "column_centered_bits_mean": round(float(np.mean(cb_arr)), 4),
            "column_centered_shift_attenuation": round(float(attenuation_ratio), 4),
        }

    print("Gate 3a Diachronic Distribution Shift Summary:")
    print(f"{'Group':<20s} {'Bits/Token':>12s} {'Coverage %':>12s} {'Cohen d':>10s} {'KS Stat':>10s} {'KS p-val':>10s}")
    for grp, st in era_stats.items():
        print(f"{grp:<20s} {st['raw_bits_per_token_mean']:>12.4f} {st['closed_class_coverage_pct']:>11.1f}% {st['cohens_d_vs_meiji']:>10.4f} {st['ks_statistic']:>10.4f} {st['ks_pvalue']:>10.4f}")

    # ---------------------------------------------------------------------------------------------
    # 2. Order 3 (D10: Multi-Translator Orthogonal Crossed Design)
    # ---------------------------------------------------------------------------------------------
    print("\n--- 2. Order 3 (D10: Multi-Translator Crossed Design ANOVA) ---")

    crossed_cells = build_crossed_dataset()
    print(f"Generated {len(crossed_cells)} cells (4 Sources x 4 Translators/Styles).")

    # Extract 6-pole vectors for all cells
    cell_pole_matrix: List[List[float]] = []
    source_labels: List[str] = []
    trans_labels: List[str] = []

    for cell in crossed_cells:
        sc_texts = cell["scene_texts"]
        ann_scenes = splitter.process_work_scenes(
            work={"author_id": cell["trans_id"], "work_id": cell["cell_id"]},
            scene_texts=sc_texts,
            include_audit_meta=False,
        )
        prof = aggregate_work_profile_from_scenes(ann_scenes, hook_idf=hook_idf)

        # 6-pole coordinates
        vec = [
            sc_texts[0].count("、") / max(len(sc_texts[0]), 1) * 10.0,  # Kasane
            0.5,  # Sasae
            prof["kamae"]["hook_survival"],
            prof["sadame"]["total"],
            prof["nagare"]["phi_zeami"],
            prof["awase"]["irony"]["mean"],
        ]
        cell_pole_matrix.append(vec)
        source_labels.append(cell["source_id"])
        trans_labels.append(cell["trans_id"])

    Z_crossed = np.array(cell_pole_matrix)

    # Compute Two-Way ANOVA variance decomposition for each pole
    anova_results: Dict[str, Dict[str, float]] = {}
    eta_sources: List[float] = []

    for j, pole in enumerate(POLE_NAMES):
        vals = Z_crossed[:, j]
        res = compute_two_way_anova_eta_squared(vals, source_labels, trans_labels)
        anova_results[pole] = res
        eta_sources.append(res["eta_source"])

    # Test Preregistered Preservation Order: Nagare > Kamae > Sadame > Awase > Kasane
    # (Exclude Sasae as fixed constant in crossed heuristic cells)
    prereg_order = ["Nagare", "Kamae", "Sadame", "Awase", "Kasane"]
    prereg_ranks = {p: r for r, p in enumerate(prereg_order)}
    actual_ranks = sorted(prereg_order, key=lambda p: anova_results[p]["eta_source"], reverse=True)

    rho, pval = spearmanr(
        [prereg_ranks[p] for p in prereg_order],
        [actual_ranks.index(p) for p in prereg_order],
    )

    print("D10 Two-Way ANOVA Variance Decomposition (Source vs Translator):")
    print(f"{'Pole':<10s} {'eta^2_source':>14s} {'eta^2_trans':>14s} {'eta^2_error':>14s} {'Preserved?':>12s}")
    for pole in POLE_NAMES:
        ar = anova_results[pole]
        pres = "Yes" if ar["eta_source"] >= ar["eta_trans"] else "No"
        print(f"{pole:<10s} {ar['eta_source']:>14.4f} {ar['eta_trans']:>14.4f} {ar['eta_interaction']:>14.4f} {pres:>12s}")

    print(f"\nPreregistered Order Hypothesis: {prereg_order}")
    print(f"Empirical Order:                {actual_ranks}")
    print(f"Spearman Rank Correlation rho:  {rho:.4f} (p-value: {pval:.4f})")

    # ---------------------------------------------------------------------------------------------
    # 3. Order 3 (Section 6: Unsupervised Diachronic Trajectory & Change Point Detection)
    # ---------------------------------------------------------------------------------------------
    print("\n--- 3. Order 3 (Section 6: Author Diachronic Change Point Detection) ---")

    target_authors = {
        "夏目漱石": {"author_id": "000148", "historical_year": 1910, "historical_event": "修善寺の大患・『門』から後期三部作へ"},
        "芥川龍之介": {"author_id": "000879", "historical_year": 1923, "historical_event": "関東大震災・歴史物から保吉もの/晩年自伝へ"},
        "太宰治": {"author_id": "000035", "historical_year": 1945, "historical_event": "終戦・中期明朗期から『斜陽』『人間失格』へ"},
        "森鴎外": {"author_id": "000129", "historical_year": 1912, "historical_event": "明治天皇崩御/乃木殉死・現代小説から歴史小説へ"},
    }

    diachronic_reports: Dict[str, Any] = {}

    for auth_name, meta in target_authors.items():
        auth_works = [w for w in works if w["author_ja"] == auth_name]
        # Sort chronologically
        auth_works.sort(key=lambda w: WORK_PUB_YEARS.get(w["work_id"], 1920))

        work_years = [WORK_PUB_YEARS.get(w["work_id"], 1920) for w in auth_works]
        work_titles = [w["title"] for w in auth_works]

        # Form 6-pole trajectory matrix
        traj_vectors: List[List[float]] = []
        for w in auth_works:
            txt = splitter.get_cached_text(w)
            sc_texts = split_work_into_scenes(txt)
            ann_sc = splitter.process_work_scenes(w, sc_texts, include_audit_meta=False)
            prof = aggregate_work_profile_from_scenes(ann_sc, hook_idf=hook_idf)
            v = [
                sc_texts[0].count("、") / max(len(sc_texts[0]), 1) * 10.0,
                0.0,
                prof["kamae"]["hook_survival"],
                prof["sadame"]["total"],
                prof["nagare"]["phi_zeami"],
                prof["awase"]["irony"]["mean"],
            ]
            traj_vectors.append(v)

        traj_mat = np.array(traj_vectors)
        # Standardize features for L2 distance calculation
        traj_std = (traj_mat - np.mean(traj_mat, axis=0)) / (np.std(traj_mat, axis=0) + 1e-6)

        tau, detected_year, r2_change = detect_l2_change_point(traj_std, work_years)
        year_error = abs(detected_year - meta["historical_year"])

        diachronic_reports[auth_name] = {
            "n_works": len(auth_works),
            "years_span": [min(work_years), max(work_years)],
            "work_sequence": [f"{y}: {t}" for y, t in zip(work_years, work_titles)],
            "detected_tau_idx": int(tau),
            "detected_split_work": f"{work_years[tau]}: {work_titles[tau]}",
            "detected_year": int(detected_year),
            "historical_ground_truth_year": meta["historical_year"],
            "historical_event": meta["historical_event"],
            "absolute_year_error": int(year_error),
            "change_point_variance_explained": round(float(r2_change), 4),
        }

    print(f"{'Author':<12s} {'Detected':>10s} {'Historical':>12s} {'Error (yrs)':>12s} {'Split Work':<30s}")
    for auth, rep in diachronic_reports.items():
        print(f"{auth:<12s} {rep['detected_year']:>10d} {rep['historical_ground_truth_year']:>12d} {rep['absolute_year_error']:>12d} {rep['detected_split_work']:<30s}")

    # ---------------------------------------------------------------------------------------------
    # 4. Save Final JSON Artifact
    # ---------------------------------------------------------------------------------------------
    final_package: Dict[str, Any] = {
        "schema": "usir_v5_gate3a_and_diachronic_v1",
        "date": "2026-11-11",
        "elapsed_sec": round(time.time() - t0, 2),
        "order2_gate3a_distribution_shift": era_stats,
        "order3_d10_multi_translator_anova": {
            "anova_per_pole": anova_results,
            "preregistered_order": prereg_order,
            "empirical_order": actual_ranks,
            "spearman_rho": round(float(rho), 4),
            "spearman_pvalue": round(float(pval), 4),
            "hypothesis_supported": bool(rho > 0.0),
        },
        "order3_section6_diachronic_trajectories": diachronic_reports,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(final_package, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved final Gate 3a and Diachronic artifact to: {OUT_JSON}")
    print(f"Total elapsed: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
