"""Aozora bunko scene splitter and heuristic v5 annotation generator.

Strictly adheres to usir-schema SceneAnnotation v5.0 specifications.
Gate 2 audit defect fixes (2026-11-11):
1. Awase: F_event and f_style physical separation with negation polarity inversion.
2. Sadame: HW-06 closed-class pronoun, role language and speech frame character stalks and dialogue edges.
3. Kamae: Question and unresolved mystery opened_hooks and resolution closed_hook_ids.
4. Nagare: Continuous log-softmax log-probabilities (event_logp, mood_logp, 16D) resolving M01-M03 sticking.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

SCHEMA_VERSION: str = "usir/scene-annotation/v5.0"
DEFAULT_ANNOTATOR: str = "aozora_scene_splitter_v5"

# Regex patterns for structural and narrative boundaries
RE_SYMBOL_DIVIDER = re.compile(r"^[＊\*◇○◯●◎△▲▽▼□■☆※―—…・\s　\-＝=]+$")
RE_HEADING = re.compile(
    r"^(?:"
    r"[一二三四五六七八九十百千万〇０-９0-9ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+|"
    r"第[一二三四五六七八九十百千万〇０-９0-9]+[章節話篇部回夜幕段景]?.*|"
    r"[其その][一二三四五六七八九十百〇０-９0-9]+.*|"
    r"（[一二三四五六七八九十０-９0-9]+）|\([0-9]+\)|"
    r"序|序章|プロローグ|エピローグ|終章|結|跋|はしがき|あとがき"
    r")$"
)
RE_TIME_MARKER = re.compile(
    r"^(?:"
    r"翌[朝昼夕夜晩日年月週]|"
    r"その[朝昼夕夜晩日頃ころ]|その日の[朝昼夕夜晩暮]|その翌[朝昼夕夜晩日]|"
    r"ある[朝昼夕夜晩日年夏冬春正月]|或る[朝昼夕夜晩日年夏冬春]|或[朝昼夕夜晩日]|"
    r"数[日週月年時間分]後|数[ヶかカ箇]月後|何[日週月年]か後|"
    r"あくる[朝昼夕夜晩日]|明くる[朝昼夕夜晩日]|"
    r"それから|やがて|しばらくして|暫くして|間もなく|まもなく|"
    r"一方[、，]|他方[、，]|時は流れて|月日は流れて|それから数"
    r")"
)
SENT_SPLIT = re.compile(r"(?<=[。！？!?」』])")
RE_QUOTE = re.compile(r"「([^」]*)」|『([^』]*)』")
RE_NEGATION = re.compile(
    r"(?:ない|ぬ|ん|ず|まい|なかった|ならず|居りませぬ|ませぬ|ません|のではない|わけではない|わけにはいか|ようもない)"
)

# 16 Event codes and 16 Mood codes for v5 schema
EVENT_CODES: List[str] = [f"E{i:02d}" for i in range(1, 17)]
MOOD_CODES: List[str] = [f"M{i:02d}" for i in range(1, 17)]

# Lexicons for Awase F_event (verbs, nouns, physical events outside quotes)
EVENT_POS_WORDS = [
    "生", "蘇る", "救う", "達成", "勝利", "解決", "祝", "祝宴", "迎える", "買い",
    "友", "治る", "成功", "生きる", "恵み", "誓う", "守る", "愛する", "栄光", "復活",
    "歓喜", "喜び", "誕生", "結婚", "約束", "再会", "味方", "仲間"
]
EVENT_NEG_WORDS = [
    "死", "殺", "亡", "滅", "倒", "害", "捕縛", "処刑", "疑", "奪", "壊", "崩",
    "借金", "病", "衰", "毒", "傷", "罪", "激怒", "怒り", "危機", "衝突", "敵",
    "暴虐", "邪悪", "悪", "嫌悪", "焦躁", "浮浪", "宿酔", "塊", "カタル", "衰弱",
    "血", "痛", "患", "飢え", "拷問", "破滅", "凶行", "罠", "悪心", "裏切り"
]
EVENT_HIGH_ARO_WORDS = [
    "激怒", "怒り", "殺す", "叫ぶ", "走る", "逃げる", "追う", "撃つ", "突く", "争う",
    "戦う", "捕縛", "騒ぎ", "乱れる", "暴虐", "決意", "飛ぶ", "駆ける", "殴る", "斬る",
    "激しい", "危機", "処刑", "跳ぶ", "打つ", "割る", "激突", "咆哮", "猛進", "出発", "殺"
]
EVENT_LOW_ARO_WORDS = [
    "眠る", "佇む", "坐る", "座る", "横たわる", "黙る", "静止", "歩く", "暮らす",
    "眺める", "憩う", "息絶える", "横になる", "伏す", "留まる", "佇立", "沈黙"
]

# Lexicons for Awase f_style (modifiers, adjectives, dialogue tone, emotion)
STYLE_POS_WORDS = [
    "美しい", "美しく", "嬉し", "楽し", "快い", "爽やか", "幸福", "素晴らしい",
    "喜ばし", "穏やか", "澄んだ", "澄ん", "典雅", "絢爛", "愛らしい", "ほがらか", "安らか",
    "快活", "可憐", "晴れやか", "鮮やか", "清らか", "甘い", "懐かしい", "麗しい"
]
STYLE_NEG_WORDS = [
    "えたいの知れない", "得体の知れない", "不吉", "焦躁", "恐ろしい", "悲しい", "寂しい", "苦しい", "痛ましい", "惨め",
    "むさくるしい", "汚い", "見すぼらしい", "見すぼらしく", "憂鬱", "重くるしい", "呪われた", "不安", "奇怪", "怪しく",
    "気味悪い", "おどろおどろ", "陰気", "凄惨", "居堪らず", "辛抱がなら", "不気味", "暗い",
    "無気力", "埃っぽい", "嫌悪", "残酷"
]

# HW-06 Role pronoun & character detection
FIRST_PERSON_PRONOUNS: Dict[str, Tuple[int, int]] = {
    "吾輩": (1, -1), "我輩": (1, -1), "余": (1, -1), "予": (1, -1), "朕": (2, -1), "麿": (1, -1),
    "俺": (1, 1), "おれ": (1, 1), "オレ": (1, 1), "乃公": (1, 1),
    "僕": (0, 0), "ぼく": (0, 0), "ボク": (0, 0), "おいら": (0, 1),
    "わし": (1, 0), "ワシ": (1, 0), "儂": (1, 0), "愚老": (1, 0),
    "私": (0, 0), "わたし": (0, 0), "自分": (0, 0),
    "わたくし": (-1, -1), "小生": (-1, -1), "下拙": (-1, -1), "手前ども": (-1, -1), "私め": (-2, -1),
    "あたし": (0, 1), "あたい": (0, 1), "うち": (0, 1), "わらわ": (1, 1),
}

SECOND_PERSON_PRONOUNS: Dict[str, Tuple[int, int]] = {
    "貴様": (2, -2), "てめえ": (2, -2), "手前": (2, -2), "己": (2, -2), "汝": (1, -1), "下郎": (2, -2),
    "お前": (1, 1), "おまえ": (1, 1), "おぬし": (1, 0), "お主": (1, 0), "そち": (1, 0), "其方": (1, 0),
    "君": (0, 0), "きみ": (0, 0), "キミ": (0, 0), "貴君": (0, 0), "諸君": (0, 0),
    "あんた": (0, 1), "お前さん": (0, 1), "そなた": (0, 0),
    "あなた": (-1, -1), "貴方": (-1, -1), "貴女": (-1, -1), "貴殿": (-1, -1), "貴下": (-1, -1),
    "先生": (-2, -1), "旦那様": (-2, -1), "旦那": (-2, -1), "奥様": (-2, -1), "御前": (-1, -1),
    "閣下": (-2, -1), "陛下": (-2, -1), "上人": (-2, -1), "お上人": (-2, -1), "王様": (-2, -1),
}

RE_SPEAKER_AFTER = re.compile(
    r"」[とて、\s]*([^\s、。]{1,8})[はが][^。]{0,30}?(?:言っ|答え|問|叫|呟|つぶや|告げ|怒鳴|笑っ|返|述|尋ね|命じ|促)"
)
RE_SPEAKER_BEFORE = re.compile(
    r"([^\s、。]{1,8})[はが][^。]{0,30}?(?:言っ|答え|問|尋ね|叫|呟|つぶや|告げ|命じ|語りかけ|質問)[たて][。！？\s\n]*「"
)

STOP_CHARACTERS = {
    "それ", "これ", "あれ", "どれ", "誰", "どこ", "何", "ここ", "そこ", "こと",
    "もの", "よう", "ほう", "わけ", "ため", "はず", "つもり", "声", "顔", "目",
    "手", "口", "頭", "胸", "人", "男", "女", "誰か", "彼", "彼女"
}

RE_HOOK_TRIG = re.compile(
    r"(?:なぜ|どうして|何故|なにゆえ|どういう|誰が|どこへ|いつ|怪しく思|えたいの知れ|得体の知れ|秘密|謎|わからぬ|分からない|不思議|奇妙|不可解|不審|訝し|疑念|何事だろ|どうしたこと)"
)
RE_RESOLV_MARKER = re.compile(
    r"(?:のだ|のであった|からである|とわかった|判明した|のだった|に違いない|であった|理解した|合点がいった|納得した|思い出した|解けた)"
)

# Common nouns frequently occurring across scenes that should NOT be hook resolution keys (§4.1.2)
COMMON_HOOK_STOPWORDS = {
    "先生", "学校", "自分", "人間", "今日", "昨日", "明日", "言葉", "時間", "世界",
    "気持", "気持ち", "様子", "本当", "場合", "相手", "理由", "意味", "存在", "問題",
    "父親", "母親", "子供", "友達", "友人", "東京", "日本", "部屋", "身体", "からだ",
    "以前", "以後", "最近", "日常", "社会", "生活", "世間", "事物", "呼吸", "瞬間",
    "仕事", "父親", "母親", "兄弟", "夫婦", "一人", "二人", "誰か", "彼ら", "手紙",
}


class FidResolver:
    """HW-02 Free Indirect Discourse (FID) 5-trigger resolver (Spec §3.5).

    T1 (0.30): Past-tense base in preceding context transitioning to non-past or noun end
    T2 (0.25): Proximal/speech-time deixis outside quotes (今/いま/ここ/この.../明日/きのう/こうやって)
    T3 (0.35): Subjective emotion/visceral adjectives bare outside quotes (non-garu form)
    T4 (0.35): Epistemic modality, questions, exclamations, trail-offs outside quotes
    T5 (0.25): Perception/cognitive verbs in preceding 1-2 sentences with focal subject v_focal

    Sentences with I_fid >= 0.50 represent internal focalized monologue.
    """

    RE_PAST_SENT = re.compile(r"(?:た|だ|かった|でした|ました)[。！？\s\n]*$")
    RE_NON_PAST_OR_NOUN = re.compile(
        r"(?:[るうすくつぬぶむ]|のである|のだ|わけだ|はずだ|ものだ|に相違ない|[\u4e00-\u9fff\u30a0-\u30ff々]+)[。！？\s\n]*$"
    )
    RE_DEIXIS = re.compile(
        r"(?:今|いま|ここ|此処|この[^\s、。]{1,6}|明日|あす|きのう|昨日|こうやって|斯う)"
    )
    RE_EMOTION_ADJ = re.compile(
        r"(?:悲し|苦し|痛|寂し|さびし|憎|恋し|恐ろし|おそろし|恥ずかし|はずかし|情けな|たまらな|口惜し|悔し|つら|辛)い(?!が[るっれろ])"
    )
    RE_MODALITY = re.compile(
        r"(?:にちがいない|に違いない|のではなかろうか|のではあるまいか|のではあるまい|かしら|のだ[。！]?|のだ……|か[。？\?]|……|…|であろうか|ではないか)"
    )
    RE_PERCEPTION = re.compile(
        r"([^\s、。]{1,8})[はが][^。]{0,25}?(?:見(?:る|た|つめ|上げ)|眺め|聞(?:く|い)|気(?:づ|付)い|思(?:う|っ)|考(?:える|え)|感(?:じる|じ)|知(?:る|っ)|怪し(?:む|ん)|疑(?:う|っ)|悟(?:る|っ))"
    )

    @classmethod
    def resolve_fid(
        cls, scene_text: str, known_characters: Optional[List[str]] = None
    ) -> Tuple[Optional[str], float, List[Dict[str, Any]]]:
        """Evaluate FID triggers across narrative sentences outside quotes."""
        sents = [s.strip() for s in SENT_SPLIT.split(scene_text) if s.strip()]
        if not sents:
            return None, 0.0, []

        fid_spans = []
        max_score = 0.0
        candidate_focals: List[Tuple[str, float]] = []

        for i, s in enumerate(sents):
            narr_s = RE_QUOTE.sub(" ", s).strip()
            if not narr_s:
                continue

            # T1: Past-tense context to non-past / noun
            t1 = False
            if i > 0:
                prev_narr = RE_QUOTE.sub(" ", sents[i - 1]).strip()
                prev_is_past = bool(cls.RE_PAST_SENT.search(prev_narr))
                cur_is_non_past = bool(cls.RE_NON_PAST_OR_NOUN.search(narr_s))
                if prev_is_past and cur_is_non_past:
                    t1 = True

            # T2: Deixis
            t2 = bool(cls.RE_DEIXIS.search(narr_s))

            # T3: Visceral subjective emotion
            t3 = bool(cls.RE_EMOTION_ADJ.search(narr_s))

            # T4: Epistemic modality / exclamation / trail-off
            t4 = bool(cls.RE_MODALITY.search(narr_s))

            # T5: Perception verb in preceding 1-2 sentences
            t5 = False
            focal_sub = None
            for lookback in (1, 2):
                if i >= lookback:
                    prev_s = RE_QUOTE.sub(" ", sents[i - lookback]).strip()
                    m = cls.RE_PERCEPTION.search(prev_s)
                    if m:
                        cand = m.group(1).strip()
                        if cand not in STOP_CHARACTERS and len(cand) <= 8:
                            t5 = True
                            focal_sub = cand
                            break

            score = (
                (0.30 if t1 else 0.0)
                + (0.25 if t2 else 0.0)
                + (0.35 if t3 else 0.0)
                + (0.35 if t4 else 0.0)
                + (0.25 if t5 else 0.0)
            )
            score = round(min(1.0, score), 4)
            if score > max_score:
                max_score = score

            if score >= 0.50:
                subj = focal_sub
                if not subj and known_characters:
                    for kc in known_characters:
                        if kc in narr_s or (i > 0 and kc in sents[i - 1]):
                            subj = kc
                            break
                if not subj:
                    for p in ("私", "わたし", "僕", "ぼく", "俺", "おれ", "自分", "余", "吾輩"):
                        if p in narr_s:
                            subj = "語り手"
                            break
                fid_spans.append(
                    {
                        "sentence_idx": i,
                        "text": narr_s[:40],
                        "score": score,
                        "triggers": {"t1": t1, "t2": t2, "t3": t3, "t4": t4, "t5": t5},
                        "focal": subj,
                    }
                )
                if subj:
                    candidate_focals.append((subj, score))

        best_focal = None
        if candidate_focals:
            focal_weights: Dict[str, float] = {}
            for sub, sc in candidate_focals:
                focal_weights[sub] = focal_weights.get(sub, 0.0) + sc
            best_focal = max(focal_weights.items(), key=lambda x: x[1])[0]

        return best_focal, max_score, fid_spans


def extract_turn_dialogue_edges(
    scene_text: str,
    character_stalks: List[Dict[str, Any]],
    focal_char: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Extract adjacent dialogue edges from consecutive speech turns (u_k -> u_{k+1}, u_k != u_{k+1}).

    Abolishes arbitrary clique edge explosion across all scene characters.
    """
    if len(character_stalks) < 2:
        return []

    char_names = [c["name"] for c in character_stalks]

    turns: List[Dict[str, Any]] = []
    for m in RE_QUOTE.finditer(scene_text):
        q_text = m.group(1) or m.group(2) or ""
        q_start = m.start()
        q_end = m.end()

        before = scene_text[max(0, q_start - 50) : q_start]
        after = scene_text[q_end : min(len(scene_text), q_end + 50)]

        speaker = None
        # 1. Direct match with known character names in immediate before/after context
        for cn in sorted(char_names, key=len, reverse=True):
            if re.search(rf"[」』][とて、\s]*{re.escape(cn)}[はが]", after[:30]):
                speaker = cn
                break
            if re.search(rf"{re.escape(cn)}[はが][^。]{{0,25}}[「\n]*$", before):
                speaker = cn
                break
            if (cn + "は" in before[-25:]) or (cn + "が" in before[-25:]):
                speaker = cn
                break

        if not speaker:
            m_after = RE_SPEAKER_AFTER.search(after)
            if m_after:
                cand = m_after.group(1).strip()
                if cand in char_names:
                    speaker = cand
                elif cand not in STOP_CHARACTERS and len(cand) <= 8:
                    speaker = cand

        if not speaker:
            m_before = RE_SPEAKER_BEFORE.search(before)
            if m_before:
                cand = m_before.group(1).strip()
                if cand in char_names:
                    speaker = cand
                elif cand not in STOP_CHARACTERS and len(cand) <= 8:
                    speaker = cand

        if not speaker:
            for p in ("俺", "おれ", "オレ", "僕", "ぼく", "私", "わたし", "自分", "余", "吾輩"):
                if p in q_text:
                    if "語り手" in char_names:
                        speaker = "語り手"
                    elif focal_char and focal_char in char_names:
                        speaker = focal_char
                    break

        turns.append({"quote": q_text, "speaker": speaker, "pos": q_start})

    if not turns:
        return []

    # Resolve unassigned speakers through natural adjacent dialogue alternation
    last_known = None
    for idx, t in enumerate(turns):
        if t["speaker"]:
            last_known = t["speaker"]
        else:
            if idx > 0 and turns[idx - 1]["speaker"]:
                prev = turns[idx - 1]["speaker"]
                alt = [c for c in char_names if c != prev]
                t["speaker"] = alt[0] if alt else ("相手" if prev != "相手" else "話者")
            elif last_known:
                alt = [c for c in char_names if c != last_known]
                t["speaker"] = alt[0] if alt else ("相手" if last_known != "相手" else "話者")
            else:
                t["speaker"] = focal_char if (focal_char and focal_char in char_names) else char_names[0]

    pair_counts: Dict[Tuple[str, str], int] = {}
    for k in range(len(turns) - 1):
        u = turns[k]["speaker"]
        v = turns[k + 1]["speaker"]
        if u and v and u != v:
            pair = (min(u, v), max(u, v))
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    if not pair_counts and len(turns) >= 1:
        first_sp = turns[0]["speaker"] or char_names[0]
        alt = [c for c in char_names if c != first_sp]
        other = alt[0] if alt else "相手"
        pair = (min(first_sp, other), max(first_sp, other))
        pair_counts[pair] = 1

    edges: List[Dict[str, Any]] = []
    for (u, v), cnt in sorted(pair_counts.items()):
        conf = min(100, 50 + cnt * 25)
        edges.append({"u": u, "v": v, "confidence": conf})

    return edges


def compute_work_scene_frequencies(scene_texts: List[str]) -> Dict[str, int]:
    """Compute within-work scene frequency sf(w) for kanji tokens across all scenes."""
    sf: Dict[str, int] = {}
    for sc in scene_texts:
        tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}", sc))
        for tok in tokens:
            sf[tok] = sf.get(tok, 0) + 1
    return sf


def filter_hook_tokens_by_isf(
    raw_tokens: Set[str],
    scene_freqs: Dict[str, int],
    total_scenes: int,
    isf_threshold: float = 1.2,
) -> Set[str]:
    """Filter hook tokens using Within-Work Inverse Scene Frequency (ISF) §4.1.2.

    ISF(w) = ln(T_W / (sf(w) + 1))
    Retains only tokens with ISF(w) >= 1.2 (approx <= 30% of scenes).
    Guards short works (T_W < 4) with sf(w) <= 1.
    Excludes COMMON_HOOK_STOPWORDS.
    """
    filtered = set()
    for tok in raw_tokens:
        if tok in COMMON_HOOK_STOPWORDS:
            continue
        sf = scene_freqs.get(tok, 0)
        if total_scenes < 4:
            if sf <= 1:
                filtered.add(tok)
        else:
            isf = math.log(total_scenes / (sf + 1))
            if isf >= isf_threshold:
                filtered.add(tok)
    return filtered


def split_long_paragraph(para: str, max_len: int = 1600, target_len: int = 800) -> List[str]:
    """Split a paragraph exceeding max_len into coherent sub-paragraphs."""
    if len(para) <= max_len:
        return [para]

    parts = [p for p in SENT_SPLIT.split(para) if p]
    chunks: List[str] = []
    cur = ""

    for p in parts:
        if len(p) > max_len:
            if cur:
                chunks.append(cur)
                cur = ""
            n_chunks = math.ceil(len(p) / target_len)
            c_size = math.ceil(len(p) / n_chunks)
            for i in range(0, len(p), c_size):
                chunks.append(p[i : i + c_size])
            continue

        if len(cur) + len(p) > max_len or (
            len(cur) >= target_len and len(cur) + len(p) > target_len * 1.3
        ):
            if cur:
                chunks.append(cur)
            cur = p
        else:
            cur += p

    if cur:
        chunks.append(cur)
    return chunks


def split_work_into_scenes(
    text: str,
    min_chars: int = 200,
    max_chars: int = 1600,
    target_chars: int = 750,
) -> List[str]:
    """Split a full work's text into continuous scenes strictly clamped to [min_chars, max_chars]."""
    text_clean = text.strip()
    if not text_clean:
        return []

    if len(text_clean) < min_chars:
        return [text_clean]

    raw_lines = [ln.strip() for ln in text.split("\n")]
    blocks: List[Tuple[str, bool, bool]] = []
    had_blank = False
    force_next_strong = False

    for ln in raw_lines:
        if not ln:
            had_blank = True
            continue
        if RE_SYMBOL_DIVIDER.match(ln):
            force_next_strong = True
            had_blank = True
            continue

        is_heading = bool(RE_HEADING.match(ln)) or (
            len(ln) <= 12
            and had_blank
            and not ln.endswith(("。", "」", "』", "！", "？", "!", "?", "、"))
        )
        is_time = bool(RE_TIME_MARKER.match(ln))
        strong = had_blank or force_next_strong or is_heading
        weak = is_time

        sub_paras = split_long_paragraph(ln, max_chars, target_chars)
        for idx, sp in enumerate(sub_paras):
            blocks.append((sp, strong if idx == 0 else False, weak if idx == 0 else True))

        had_blank = False
        force_next_strong = False

    scenes: List[str] = []
    cur_parts: List[str] = []
    cur_len = 0

    for sp, strong, weak in blocks:
        sp_len = len(sp)

        if cur_len >= min_chars:
            if cur_len + sp_len > max_chars:
                scenes.append("\n".join(cur_parts))
                cur_parts = [sp]
                cur_len = sp_len
                continue
            if strong:
                scenes.append("\n".join(cur_parts))
                cur_parts = [sp]
                cur_len = sp_len
                continue
            if weak and cur_len >= 350:
                scenes.append("\n".join(cur_parts))
                cur_parts = [sp]
                cur_len = sp_len
                continue
            if cur_len >= target_chars:
                scenes.append("\n".join(cur_parts))
                cur_parts = [sp]
                cur_len = sp_len
                continue

        if cur_len > 0 and cur_len + sp_len > max_chars:
            combined = "\n".join(cur_parts) + "\n" + sp
            re_chunks = split_long_paragraph(combined, max_chars, target_chars)
            for rc in re_chunks[:-1]:
                scenes.append(rc)
            cur_parts = [re_chunks[-1]]
            cur_len = len(re_chunks[-1])
            continue

        cur_parts.append(sp)
        cur_len += sp_len + (1 if len(cur_parts) > 1 else 0)

    if cur_parts:
        last_scene = "\n".join(cur_parts).strip()
        if len(last_scene) < min_chars and scenes:
            if len(scenes[-1]) + 1 + len(last_scene) <= max_chars:
                scenes[-1] = scenes[-1] + "\n" + last_scene
            else:
                combined = scenes[-1] + "\n" + last_scene
                mid = len(combined) // 2
                best_pos = mid
                for m in SENT_SPLIT.finditer(combined):
                    pos = m.end()
                    if (
                        min_chars <= pos <= max_chars
                        and min_chars <= len(combined) - pos <= max_chars
                    ):
                        if abs(pos - mid) < abs(best_pos - mid):
                            best_pos = pos
                first_part = combined[:best_pos].strip()
                second_part = combined[best_pos:].strip()
                if len(first_part) < min_chars or len(second_part) < min_chars:
                    best_pos = mid
                    first_part = combined[:best_pos].strip()
                    second_part = combined[best_pos:].strip()
                scenes[-1] = first_part
                scenes.append(second_part)
        else:
            if last_scene:
                scenes.append(last_scene)

    return scenes


def score_segment_polarity(text: str, pos_words: List[str], neg_words: List[str]) -> Tuple[float, float]:
    """Score positive and negative hits with negation scope inversion."""
    pos_score = 0.0
    neg_score = 0.0

    sentences = re.split(r"[。！？!?\n]", text)
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        has_sent_neg = bool(RE_NEGATION.search(sent))
        for w in pos_words:
            idx = 0
            while True:
                idx = sent.find(w, idx)
                if idx == -1:
                    break
                window = sent[idx : idx + len(w) + 35]
                if RE_NEGATION.search(window) or (has_sent_neg and idx > len(sent) // 2):
                    neg_score += 1.0  # negated positive becomes negative
                else:
                    pos_score += 1.0
                idx += len(w)
        for w in neg_words:
            idx = 0
            while True:
                idx = sent.find(w, idx)
                if idx == -1:
                    break
                window = sent[idx : idx + len(w) + 35]
                if RE_NEGATION.search(window):
                    pos_score += 0.5  # negated negative attenuated / slightly positive
                else:
                    neg_score += 1.0
                idx += len(w)
    return pos_score, neg_score


def quantize_val(val_raw: float) -> int:
    if val_raw >= 0.25:
        return 2
    if val_raw >= 0.08:
        return 1
    if val_raw <= -0.25:
        return -2
    if val_raw <= -0.08:
        return -1
    return 0


def quantize_aro(aro_raw: float) -> int:
    if aro_raw >= 1.0:
        return 2
    if aro_raw >= 0.25:
        return 1
    if aro_raw <= -1.0:
        return -2
    if aro_raw <= -0.25:
        return -1
    return 0


def log_softmax(scores: List[float]) -> List[float]:
    """Numerically stable log-softmax returning finite f32 values <= 0.0."""
    max_s = max(scores)
    shifted = [s - max_s for s in scores]
    lse = math.log(sum(math.exp(x) for x in shifted))
    return [round(float(x - lse), 6) for x in shifted]


def compute_heuristic_signals(
    scene_text: str, scene_index: int, scene_count: int
) -> Tuple[Dict[str, int], Dict[str, Any], Dict[str, Any], Dict[str, int], Dict[str, int]]:
    """Compute deterministic v5 annotations (awase, nagare) from surface textual features.

    1. Awase: F_event and f_style calculated independently with negation polarity inversion.
    2. Nagare: Continuous 16-dimensional log-softmax probabilities (event_logp, mood_logp).
    """
    total_len = max(len(scene_text), 1)
    pos = scene_index / max(scene_count, 1)

    quote_spans = RE_QUOTE.findall(scene_text)
    quote_chars = sum(len(m[0] or m[1]) for m in quote_spans)
    dialogue_ratio = quote_chars / total_len
    narrative_text = RE_QUOTE.sub(" ", scene_text)

    sents = [s for s in SENT_SPLIT.split(scene_text) if s.strip()]
    n_sents = max(len(sents), 1)
    mean_sent_len = total_len / n_sents
    cv_sent_len = (
        (sum((len(s) - mean_sent_len) ** 2 for s in sents) / n_sents) ** 0.5 / (mean_sent_len + 1e-5)
        if len(sents) >= 2
        else 0.5
    )

    excl_count = scene_text.count("！") + scene_text.count("!")
    quest_count = scene_text.count("？") + scene_text.count("?")
    excl_quest_ratio = (excl_count + quest_count) / total_len
    comma_ratio = scene_text.count("、") / total_len
    kanji_ratio = len(re.findall(r"[\u4e00-\u9fff]", scene_text)) / total_len
    neg_ratio = len(re.findall(r"(?:ない|ぬ|ん|ず|まい|なかった|ならず)", scene_text)) / n_sents
    past_ratio = len(re.findall(r"(?:た|だ)[。、\s]", scene_text)) / n_sents

    # 1. Awase: Event layer (F_event)
    pos_e, neg_e = score_segment_polarity(narrative_text, EVENT_POS_WORDS, EVENT_NEG_WORDS)
    val_e_raw = (pos_e - neg_e) / max(pos_e + neg_e, 2.0)
    f_event_valence = quantize_val(val_e_raw)

    event_high_hits = sum(narrative_text.count(w) for w in EVENT_HIGH_ARO_WORDS)
    event_low_hits = sum(narrative_text.count(w) for w in EVENT_LOW_ARO_WORDS)
    aro_e_raw = (event_high_hits * 1.5 - event_low_hits) / max(len(narrative_text) / 250.0, 1.0)
    f_event_arousal = quantize_aro(aro_e_raw)

    # Awase: Style layer (f_style)
    pos_s, neg_s = score_segment_polarity(scene_text, STYLE_POS_WORDS, STYLE_NEG_WORDS)
    val_s_raw = (pos_s - neg_s) / max(pos_s + neg_s, 2.0)
    f_style_valence = quantize_val(val_s_raw)

    style_high_hits = (
        excl_count * 1.5
        + (1.0 if any(scene_text.endswith(x) for x in ("ぞ", "ぜ", "わい", "たまえ", "よ")) else 0.0)
        + (1.0 if dialogue_ratio > 0.4 else 0.0)
    )
    style_low_hits = sum(
        scene_text.count(w) for w in ["ひっそり", "静か", "ぽつねんと", "ぼんやり", "幽か", "かすか", "そっと"]
    )
    aro_s_raw = (style_high_hits * 1.2 - style_low_hits) / max(total_len / 300.0, 1.0)
    f_style_arousal = quantize_aro(aro_s_raw)

    awase = {
        "f_event_valence": int(f_event_valence),
        "f_event_arousal": int(f_event_arousal),
        "f_style_valence": int(f_style_valence),
        "f_style_arousal": int(f_style_arousal),
    }

    # 2. Nagare event continuous log-probabilities (E01..E16)
    e_scores = [0.0] * 16
    e_scores[0] = 2.0 * max(0.0, 0.3 - pos) / 0.3 + 1.0 * kanji_ratio - 0.5 * past_ratio
    e_scores[1] = 1.5 * sum(scene_text.count(w) for w in ["会う", "やって来", "現れ", "訪ね", "見かけ", "迎える"]) / 2.0
    e_scores[2] = 4.0 * dialogue_ratio + (1.0 if quote_chars > 100 else 0.0)
    e_scores[3] = 1.5 * sum(scene_text.count(w) for w in ["思", "考", "悟", "感", "記憶", "追憶", "心"]) / 3.0 + 1.0 * (1.0 - dialogue_ratio)
    e_scores[4] = 1.5 * sum(scene_text.count(w) for w in ["行", "歩", "走", "旅", "出発", "越え", "彷徨", "向か"]) / 3.0
    e_scores[5] = 2.0 * excl_quest_ratio * 10.0 + 1.5 * sum(scene_text.count(w) for w in ["迷", "疑", "悩", "拒", "葛藤", "躊躇"]) / 2.0
    e_scores[6] = 2.0 * sum(scene_text.count(w) for w in ["不意に", "突然", "異変", "変化", "事件", "たちまち", "ふと"]) / 2.0
    e_scores[7] = 2.0 * max(0, f_event_arousal) + 1.5 * max(0, -f_event_valence) + 1.5 * sum(scene_text.count(w) for w in ["危機", "迫る", "恐怖", "捕縛", "処刑", "危険"]) / 2.0
    e_scores[8] = 2.0 * (event_high_hits / 3.0) + 1.5 * sum(scene_text.count(w) for w in ["戦", "撃", "争", "叫", "投", "壊", "打"]) / 3.0
    e_scores[9] = 1.5 * (1.0 - abs(pos - 0.5) * 2.0) + 1.0 * sum(scene_text.count(w) for w in ["それから", "やがて", "続いて", "進む", "間もなく"]) / 2.0
    e_scores[10] = 1.8 * sum(scene_text.count(w) for w in ["見出", "発見", "気づ", "判明", "見つけ", "わかっ", "知っ"]) / 2.0
    e_scores[11] = 2.0 * max(0, -f_event_valence) + 1.5 * sum(scene_text.count(w) for w in ["死", "失", "倒れ", "奪", "傷", "泣", "滅", "害"]) / 2.0
    e_scores[12] = 2.0 * sum(scene_text.count(w) for w in ["逆転", "破綻", "崩壊", "裏切", "一変", "狂気", "破滅"]) / 2.0
    e_scores[13] = 2.0 * max(0, f_event_valence) + 1.0 * max(0, f_event_arousal) + 1.5 * sum(scene_text.count(w) for w in ["勝", "救", "歓喜", "成功", "栄光", "達成", "喜"]) / 2.0
    e_scores[14] = 2.0 * max(0.0, pos - 0.7) / 0.3 + 1.0 * sum(scene_text.count(w) for w in ["静まり", "名残", "余韻", "振り返", "佇む", "去る"]) / 2.0
    e_scores[15] = 3.0 * max(0.0, pos - 0.8) / 0.2 + 1.5 * sum(scene_text.count(w) for w in ["終", "結末", "解決", "去っ", "幕", "完"]) / 2.0

    # Add small deterministic entropy from sentence length variance
    for i in range(16):
        e_scores[i] += 0.2 * math.sin(i * 1.5 + cv_sent_len)

    event_logp = log_softmax(e_scores)
    sorted_events = sorted(range(16), key=lambda i: (event_logp[i], -i), reverse=True)
    event_top3 = [
        {"code": EVENT_CODES[sorted_events[0]], "rank_weight": 3},
        {"code": EVENT_CODES[sorted_events[1]], "rank_weight": 2},
        {"code": EVENT_CODES[sorted_events[2]], "rank_weight": 1},
    ]

    # 3. Nagare mood continuous log-probabilities (M01..M16)
    m_scores = [0.0] * 16
    m_scores[0] = 1.8 * max(0, -f_style_arousal) + 1.0 * max(0, f_style_valence) + 1.5 * sum(scene_text.count(w) for w in ["静か", "穏やか", "平穏", "のどか", "凪", "ひっそり"]) / 2.0
    m_scores[1] = 1.8 * max(0, f_style_arousal) + 1.0 * max(0, -f_style_valence) + 1.5 * sum(scene_text.count(w) for w in ["焦燥", "緊迫", "息苦し", "切迫", "じりじり", "緊張"]) / 2.0
    m_scores[2] = 2.0 * max(0, -f_style_valence) + 1.0 * max(0, -f_style_arousal) + 1.5 * sum(scene_text.count(w) for w in ["憂鬱", "沈鬱", "暗い", "陰気", "塞ぐ", "重い", "不吉"]) / 2.0
    m_scores[3] = 2.0 * max(0, f_style_valence) + 1.5 * max(0, f_style_arousal) + 1.5 * sum(scene_text.count(w) for w in ["昂揚", "熱狂", "歓喜", "喜悦", "晴れやか", "快活"]) / 2.0
    m_scores[4] = 1.2 * neg_ratio + 1.5 * sum(scene_text.count(w) for w in ["諦観", "無常", "儚い", "仕方な", "定め", "朽ち", "諦め"]) / 2.0
    m_scores[5] = 2.0 * sum(scene_text.count(w) for w in ["官能", "耽美", "艶", "匂い", "白い肌", "妖艶", "唇", "吐息", "香水", "香"]) / 2.0
    m_scores[6] = 2.0 * sum(scene_text.count(w) for w in ["畏怖", "崇高", "荘厳", "怪異", "霊気", "神秘", "神", "仏", "霊"]) / 2.0
    m_scores[7] = 1.8 * max(0, -f_style_valence) + 1.5 * sum(scene_text.count(w) for w in ["哀切", "情念", "あわれ", "憐れ", "慕う", "涙", "恋情", "切な"]) / 2.0
    is_ironic = (f_event_valence * f_style_valence < 0)
    m_scores[8] = (2.0 if is_ironic else 0.0) + 1.5 * sum(scene_text.count(w) for w in ["諧謔", "皮肉", "アイロニー", "嘲笑", "可笑し", "吹き出", "笑っ", "苦笑"]) / 2.0
    m_scores[9] = 2.0 * sum(scene_text.count(w) for w in ["恐怖", "戦慄", "恐ろし", "慄く", "慄然", "震え", "背筋", "怖"]) / 2.0
    m_scores[10] = 2.5 * sum(scene_text.count(w) for w in ["激怒", "憤怒", "怒り", "反逆", "許せぬ", "憤然", "歯ぎしり", "怒っ"]) / 2.0
    m_scores[11] = 1.5 * (1.0 - dialogue_ratio) + 1.8 * sum(scene_text.count(w) for w in ["孤独", "一人", "ぽつねん", "隔絶", "疎外", "取り残さ", "孤立"]) / 2.0
    m_scores[12] = 2.5 * sum(scene_text.count(w) for w in ["夢", "幻", "幻惑", "夢想", "朦朧", "うつつ", "錯覚", "まぼろし"]) / 2.0
    m_scores[13] = 1.8 * max(0, f_style_arousal) + 1.8 * sum(scene_text.count(w) for w in ["疾走", "躍動", "駆け抜", "飛沫", "烈風", "驀進", "跳躍"]) / 2.0
    m_scores[14] = 1.5 * max(0.0, pos - 0.7) + 1.8 * sum(scene_text.count(w) for w in ["和解", "救済", "許し", "慈愛", "温もり", "安堵", "和ら"]) / 2.0
    m_scores[15] = 1.5 * max(0.0, pos - 0.8) + 1.5 * sum(scene_text.count(w) for w in ["余情", "余白", "去りゆく", "佇まい", "遠ざかる", "夕暮れ"]) / 2.0

    for i in range(16):
        m_scores[i] += 0.2 * math.cos(i * 1.5 + comma_ratio * 10.0)

    mood_logp = log_softmax(m_scores)
    sorted_moods = sorted(range(16), key=lambda i: (mood_logp[i], -i), reverse=True)
    mood_top3 = [
        {"code": MOOD_CODES[sorted_moods[0]], "rank_weight": 3},
        {"code": MOOD_CODES[sorted_moods[1]], "rank_weight": 2},
        {"code": MOOD_CODES[sorted_moods[2]], "rank_weight": 1},
    ]

    nagare = {
        "event_top3": event_top3,
        "mood_top3": mood_top3,
        "event_logp": event_logp,
        "mood_logp": mood_logp,
    }

    f_event_stream = {"valence": f_event_valence, "arousal": f_event_arousal}
    f_style_stream = {"valence": f_style_valence, "arousal": f_style_arousal}

    return awase, nagare, {}, f_event_stream, f_style_stream


def extract_scene_characters_and_edges(
    scene_text: str,
    scene_index: int,
    scene_count: int,
    f_event_valence: int,
    f_event_arousal: int,
    f_style_valence: int,
    f_style_arousal: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
    """HW-06 Deterministic character stalk and dialogue edge extractor with FID resolution (Spec §3.5)."""
    speakers: List[str] = []

    for name in RE_SPEAKER_AFTER.findall(scene_text):
        name = name.strip()
        if name and name not in STOP_CHARACTERS and len(name) <= 8:
            if name not in speakers:
                speakers.append(name)

    for name in RE_SPEAKER_BEFORE.findall(scene_text):
        name = name.strip()
        if name and name not in STOP_CHARACTERS and len(name) <= 8:
            if name not in speakers:
                speakers.append(name)

    quotes = RE_QUOTE.findall(scene_text)
    has_dialogue = len(quotes) > 0

    narrator_present = False
    narrative_text = RE_QUOTE.sub(" ", scene_text)
    for p in ("私", "わたし", "僕", "ぼく", "俺", "おれ", "吾輩", "余", "自分"):
        if p in narrative_text:
            narrator_present = True
            break

    for p in ("メロス", "セリヌンティウス", "ディオニス", "老爺", "ジョバンニ", "カンパネルラ", "カンディタ", "一郎"):
        if p in scene_text and p not in speakers:
            speakers.append(p)

    if narrator_present and "語り手" not in speakers:
        speakers.insert(0, "語り手")

    # HW-02 Free Indirect Discourse (FID) 5-trigger resolution (§3.5)
    focal_char, fid_score, fid_spans = FidResolver.resolve_fid(scene_text, speakers)
    if focal_char and focal_char not in speakers:
        if len(speakers) < 4:
            speakers.insert(0 if not narrator_present else 1, focal_char)

    if has_dialogue and len(speakers) < 2:
        found_2nd = None
        for p in SECOND_PERSON_PRONOUNS:
            if p in scene_text:
                found_2nd = p
                break
        if found_2nd and found_2nd not in speakers:
            speakers.append(found_2nd)
        elif len(speakers) == 1:
            speakers.append("相手")
        elif len(speakers) == 0:
            speakers.extend(["話者", "相手"])

    # Build character stalks with dynamic scene modulation
    d_life = max(-1, min(1, f_event_valence))
    d_order = -1 if (f_event_arousal >= 1 and f_event_valence < 0) else 0
    d_mystic = 1 if any(w in scene_text for w in ["神", "仏", "霊", "夢", "幻", "魂", "奇跡"]) else 0
    d_devotion = (
        1
        if (f_event_valence > 0 or any(w in scene_text for w in ["友", "信", "約束", "救", "助"]))
        else (-1 if any(w in scene_text for w in ["疑", "私慾", "裏切", "殺"]) else 0)
    )
    d_emotion = -1 if f_style_arousal >= 1 else (1 if f_style_arousal <= -1 else 0)
    d_will = (
        1
        if (f_event_arousal >= 1 or any(w in scene_text for w in ["決意", "必ず", "誓"]))
        else (-1 if any(w in scene_text for w in ["疲れ", "諦", "絶望", "不能"]) else 0)
    )

    character_stalks: List[Dict[str, Any]] = []
    for name in speakers[:4]:  # limit to max 4 main characters per scene
        if name == "語り手":
            base = [0, 0, 0, 0, 0, 0]
        elif name in ("王", "暴君", "ディオニス"):
            base = [-1, 2, 0, -2, 1, 1]
        elif "老" in name or name == "上人":
            base = [0, 1, 1, 0, 1, -1]
        elif name in ("メロス", "ジョバンニ"):
            base = [1, -1, 0, 1, -1, 1]
        elif name in ("お前", "貴様", "てめえ"):
            base = [-1, -1, 0, -1, -1, 1]
        else:
            h = abs(hash(name))
            base = [(h >> (i * 2) & 3) - 1 for i in range(6)]

        # Dynamic modulation under strong FID internal monologue (§3.5 intra_variance)
        is_focal = (name == focal_char) or (name == "語り手" and focal_char in ("語り手", "私", "自分"))
        mod_emotion = d_emotion + (1 if (is_focal and fid_score >= 0.50 and f_style_valence < 0) else 0)
        mod_will = d_will + (-1 if (is_focal and fid_score >= 0.50 and f_event_valence < 0) else 0)

        axes_q = [
            max(-2, min(2, base[0] + d_life)),
            max(-2, min(2, base[1] + d_order)),
            max(-2, min(2, base[2] + d_mystic)),
            max(-2, min(2, base[3] + d_devotion)),
            max(-2, min(2, base[4] + mod_emotion)),
            max(-2, min(2, base[5] + mod_will)),
        ]
        character_stalks.append({"name": name, "axes_q": axes_q})

    # Adjacent speech turn dialogue edges (Spec §4.1 / Task 1 requirement)
    dialogue_edges = extract_turn_dialogue_edges(scene_text, character_stalks, focal_char)

    return character_stalks, dialogue_edges, focal_char


def build_scene_annotation(
    scene_text: str,
    author_id: str,
    work_id: str,
    scene_index: int,
    scene_count: int,
    seed: int = 0,
    annotator: str = DEFAULT_ANNOTATOR,
    focal_character: Optional[str] = None,
    include_audit_meta: bool = False,
    extra_meta: Optional[Dict[str, Any]] = None,
    character_stalks: Optional[List[Dict[str, Any]]] = None,
    dialogue_edges: Optional[List[Dict[str, Any]]] = None,
    opened_hooks: Optional[List[Dict[str, Any]]] = None,
    closed_hook_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Construct a scene annotation record.

    When include_audit_meta=False, produces STRICT usir-schema SceneAnnotation v5.0 fields
    (deny_unknown_fields compliant).
    When include_audit_meta=True, includes human/audit inspection fields (scene_id, title,
    author_ja, text, story_position, streams).
    """
    awase, nagare, _, f_event_stream, f_style_stream = compute_heuristic_signals(
        scene_text, scene_index, scene_count
    )

    if character_stalks is None or dialogue_edges is None:
        stalks, edges, auto_focal = extract_scene_characters_and_edges(
            scene_text,
            scene_index,
            scene_count,
            awase["f_event_valence"],
            awase["f_event_arousal"],
            awase["f_style_valence"],
            awase["f_style_arousal"],
        )
        if character_stalks is None:
            character_stalks = stalks
        if dialogue_edges is None:
            dialogue_edges = edges
        if focal_character is None:
            focal_character = auto_focal

    if opened_hooks is None or closed_hook_ids is None:
        # Default single-scene hook detection
        o_hooks: List[Dict[str, Any]] = []
        c_hooks: List[int] = []
        sents = [s for s in SENT_SPLIT.split(scene_text) if s.strip()]
        hook_id = 1
        for sent in sents:
            if RE_HOOK_TRIG.search(sent):
                label = sent.strip()[:30]
                o_hooks.append({"id": hook_id, "label": label})
                hook_id += 1
                if len(o_hooks) >= 2:
                    break
        if opened_hooks is None:
            opened_hooks = o_hooks
        if closed_hook_ids is None:
            closed_hook_ids = c_hooks

    char_len = max(len(scene_text), 1)
    record: Dict[str, Any] = {}

    if include_audit_meta:
        meta = extra_meta or {}
        pos_ratio = scene_index / max(scene_count, 1)
        if pos_ratio < 0.33:
            story_pos = "序盤"
        elif pos_ratio < 0.66:
            story_pos = "中盤"
        else:
            story_pos = "終盤"

        record.update(
            {
                "scene_id": f"{work_id}_{scene_index:03d}",
                "author_id": author_id,
                "author_ja": meta.get("author_ja", ""),
                "work_id": work_id,
                "title": meta.get("title", ""),
                "genre": meta.get("genre", "novel"),
                "scene_index": scene_index,
                "scene_count": scene_count,
                "char_len": char_len,
                "story_position": story_pos,
                "text": scene_text,
                "f_event_stream": f_event_stream,
                "f_style_stream": f_style_stream,
            }
        )

    # Core usir-schema SceneAnnotation v5.0 fields
    record.update(
        {
            "schema_version": SCHEMA_VERSION,
            "author_id": author_id,
            "work_id": work_id,
            "scene_index": scene_index,
            "scene_count": scene_count,
            "char_len": char_len,
            "annotator": annotator,
            "seed": seed,
        }
    )

    if focal_character:
        record["focal_character"] = focal_character

    record["awase"] = awase
    record["nagare"] = nagare
    record["sadame"] = {
        "character_stalks": character_stalks,
        "dialogue_edges": dialogue_edges,
    }
    record["kamae"] = {
        "opened_hooks": opened_hooks,
        "closed_hook_ids": closed_hook_ids,
    }

    return record


class AozoraCorpusSplitter:
    """Orchestrates scene segmentation and annotation across all 106 Aozora works."""

    def __init__(self, data_dir: Path | str = "C:/dev/usir/data") -> None:
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / "aozora_cache"

    def load_manifests(self) -> List[Dict[str, Any]]:
        """Load all 106 works from the three manifests."""
        m1_path = self.data_dir / "aozora_manifest.json"
        m2_path = self.data_dir / "aozora_heldout_manifest.json"
        m3_path = self.data_dir / "aozora_heldout2_manifest.json"

        m1 = [x for x in json.loads(m1_path.read_text("utf-8")) if x.get("genre") == "novel"]
        m2 = json.loads(m2_path.read_text("utf-8"))
        m3 = json.loads(m3_path.read_text("utf-8"))

        return m1 + m2 + m3

    def get_cached_text(self, work: Dict[str, Any]) -> str:
        """Read cleaned text from cache."""
        prefix = "novel" if work.get("genre") == "novel" else "heldout"
        filename = f"{prefix}_{work['author_id']}_{work['work_id']}.txt"
        path = self.cache_dir / filename
        return path.read_text("utf-8")

    def process_work_scenes(
        self,
        work: Dict[str, Any],
        scene_texts: List[str],
        include_audit_meta: bool = False,
    ) -> List[Dict[str, Any]]:
        """Annotate all scenes of a work with continuous hook tracking, ISF filtering, and character dynamics."""
        count = len(scene_texts)
        hook_seq = 1
        active_hooks: List[Dict[str, Any]] = []
        work_records: List[Dict[str, Any]] = []

        # Pass 1: Compute within-work scene frequencies for ISF filtering (§4.1.2)
        work_scene_freqs = compute_work_scene_frequencies(scene_texts)

        # Pass 2: Annotate scenes
        for idx, sc_txt in enumerate(scene_texts):
            awase, nagare, _, _, _ = compute_heuristic_signals(sc_txt, idx, count)

            stalks, edges, focal = extract_scene_characters_and_edges(
                sc_txt,
                idx,
                count,
                awase["f_event_valence"],
                awase["f_event_arousal"],
                awase["f_style_valence"],
                awase["f_style_arousal"],
            )

            # Kamae hook extraction with ISF filtering
            opened_hooks: List[Dict[str, Any]] = []
            sents = [s for s in SENT_SPLIT.split(sc_txt) if s.strip()]
            for sent in sents:
                if RE_HOOK_TRIG.search(sent):
                    label = sent.strip()[:30]
                    kanji_toks = set(re.findall(r"[\u4e00-\u9fff]{2,}", label))
                    # Filter tokens using Within-Work ISF (threshold >= 1.2)
                    specific_toks = filter_hook_tokens_by_isf(
                        kanji_toks, work_scene_freqs, total_scenes=count, isf_threshold=1.2
                    )
                    # If specific_toks is empty, guard by taking non-stopword tokens or first 4 chars
                    if not specific_toks:
                        fallback = {tok for tok in kanji_toks if tok not in COMMON_HOOK_STOPWORDS}
                        specific_toks = fallback if fallback else {label[:4]}

                    opened_hooks.append({"id": hook_seq, "label": label})
                    active_hooks.append({"id": hook_seq, "tokens": specific_toks, "open_idx": idx})
                    hook_seq += 1
                    if len(opened_hooks) >= 2:
                        break

            # Kamae hook closure: verified against specific ISF tokens
            closed_hook_ids: List[int] = []
            has_resolv = bool(RE_RESOLV_MARKER.search(sc_txt))
            for h in list(active_hooks):
                if h["open_idx"] < idx and has_resolv:
                    if any(tok in sc_txt for tok in h["tokens"]):
                        closed_hook_ids.append(h["id"])
                        active_hooks.remove(h)

            rec = build_scene_annotation(
                scene_text=sc_txt,
                author_id=work["author_id"],
                work_id=work["work_id"],
                scene_index=idx,
                scene_count=count,
                include_audit_meta=include_audit_meta,
                extra_meta=work,
                focal_character=focal,
                character_stalks=stalks,
                dialogue_edges=edges,
                opened_hooks=opened_hooks,
                closed_hook_ids=closed_hook_ids,
            )
            work_records.append(rec)

        return work_records

    def process_all_works(
        self,
        include_audit_meta: bool = False,
        min_chars: int = 200,
        max_chars: int = 1600,
        target_chars: int = 750,
    ) -> List[Dict[str, Any]]:
        """Process all works and return a flat list of scene annotations."""
        works = self.load_manifests()
        all_records: List[Dict[str, Any]] = []

        for work in works:
            text = self.get_cached_text(work)
            scene_texts = split_work_into_scenes(
                text, min_chars=min_chars, max_chars=max_chars, target_chars=target_chars
            )
            records = self.process_work_scenes(work, scene_texts, include_audit_meta=include_audit_meta)
            all_records.extend(records)

        return all_records

    def extract_pilot_scenes(
        self,
        per_author: int = 5,
        min_chars: int = 200,
        max_chars: int = 1600,
        target_chars: int = 750,
    ) -> List[Dict[str, Any]]:
        """Extract balanced pilot scenes (per_author per each of the 10 authors)."""
        works = self.load_manifests()
        author_works: Dict[str, List[Dict[str, Any]]] = {}
        for w in works:
            author_works.setdefault(w["author_id"], []).append(w)

        pilot_records: List[Dict[str, Any]] = []

        for author_id in sorted(author_works.keys()):
            a_works = author_works[author_id]
            a_scenes: List[Dict[str, Any]] = []
            for w in a_works:
                text = self.get_cached_text(w)
                sc_texts = split_work_into_scenes(
                    text, min_chars=min_chars, max_chars=max_chars, target_chars=target_chars
                )
                recs = self.process_work_scenes(w, sc_texts, include_audit_meta=True)
                a_scenes.extend(recs)

            total_a = len(a_scenes)
            if total_a <= per_author:
                selected = a_scenes
            else:
                indices = [
                    int(round(i * (total_a - 1) / (per_author - 1))) for i in range(per_author)
                ]
                selected = [a_scenes[i] for i in indices]

            pilot_records.extend(selected)

        return pilot_records

    @staticmethod
    def save_jsonl(records: List[Dict[str, Any]], out_path: Path | str) -> None:
        """Write records as a clean JSONL file."""
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    data_dir = Path("C:/dev/usir/data")
    splitter = AozoraCorpusSplitter(data_dir=data_dir)

    print("Generating full aozora_106works_scenes.jsonl (4,576 scenes)...")
    all_scenes = splitter.process_all_works(include_audit_meta=False)
    all_path = data_dir / "v5" / "aozora_106works_scenes.jsonl"
    splitter.save_jsonl(all_scenes, all_path)
    print(f"Saved {len(all_scenes)} scenes to {all_path}")

    print("Generating pilot_50_scenes.jsonl (50 scenes)...")
    pilot_scenes = splitter.extract_pilot_scenes(per_author=5)
    pilot_path = data_dir / "v5" / "pilot_50_scenes.jsonl"
    splitter.save_jsonl(pilot_scenes, pilot_path)
    print(f"Saved {len(pilot_scenes)} scenes to {pilot_path}")

    drive_pilot_path = Path("F:/マイドライブ/Claude-Sync/inbox/pilot_50_scenes.jsonl")
    if drive_pilot_path.parent.exists():
        import shutil
        shutil.copyfile(pilot_path, drive_pilot_path)
        print(f"Copied pilot scenes to {drive_pilot_path}")
    else:
        print(f"Warning: {drive_pilot_path.parent} not accessible, skipped drive copy.")


if __name__ == "__main__":
    main()

