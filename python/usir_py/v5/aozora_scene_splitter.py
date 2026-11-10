"""Aozora bunko scene splitter and heuristic v5 annotation generator.

Strictly adheres to usir-schema SceneAnnotation v5.0 specifications.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

# 16 Event codes and 16 Mood codes for v5 schema
EVENT_CODES: List[str] = [f"E{i:02d}" for i in range(1, 17)]
MOOD_CODES: List[str] = [f"M{i:02d}" for i in range(1, 17)]

# Lexicons for heuristic signal estimation
POS_WORDS = {
    "喜", "嬉", "楽", "笑", "幸", "光", "愛", "好", "美", "勝",
    "晴", "快", "朗", "安", "穏", "清", "輝", "祝", "咲", "富"
}
NEG_WORDS = {
    "悲", "泣", "苦", "痛", "怒", "恐", "怖", "死", "血", "殺",
    "壊", "暗", "悪", "病", "冷", "寒", "怨", "闇", "衰", "毒"
}
DYNAMIC_WORDS = {
    "走", "飛", "追", "撃", "叫", "急", "跳", "突", "戦", "打",
    "投", "壊", "裂", "火", "嵐", "激", "崩", "迫", "走る", "追う"
}
STATIC_WORDS = {
    "思", "考", "見", "聞", "知", "感", "黙", "佇", "夜", "月",
    "静", "深", "夢", "影", "眠", "座", "歩", "眺", "佇む", "静か"
}


def split_long_paragraph(para: str, max_len: int = 1600, target_len: int = 800) -> List[str]:
    """Split a paragraph exceeding max_len into coherent sub-paragraphs.

    Splits at sentence boundaries where possible; falls back to balanced chunking
    if a single sentence exceeds max_len.
    """
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
            # Balance chunks evenly so the last piece is not too small
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
    """Split a full work's text into continuous scenes strictly clamped to [min_chars, max_chars].

    Detects blank lines, symbol dividers, chapter/section headings, and spatio-temporal markers.
    Handles tail rebalancing and degenerate edge cases gracefully.
    """
    text_clean = text.strip()
    if not text_clean:
        return []

    # If the entire work itself is shorter than min_chars, return it as a single scene
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
            # 1) Adding sp exceeds max_chars
            if cur_len + sp_len > max_chars:
                scenes.append("\n".join(cur_parts))
                cur_parts = [sp]
                cur_len = sp_len
                continue
            # 2) Strong boundary at or above min_chars
            if strong:
                scenes.append("\n".join(cur_parts))
                cur_parts = [sp]
                cur_len = sp_len
                continue
            # 3) Weak boundary at or above 350 chars
            if weak and cur_len >= 350:
                scenes.append("\n".join(cur_parts))
                cur_parts = [sp]
                cur_len = sp_len
                continue
            # 4) Reached target_chars at a paragraph transition
            if cur_len >= target_chars:
                scenes.append("\n".join(cur_parts))
                cur_parts = [sp]
                cur_len = sp_len
                continue

        # If cur_len < min_chars, but cur_len + sp_len > max_chars, re-chunk combined
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
            # Rebalance with preceding scene
            if len(scenes[-1]) + 1 + len(last_scene) <= max_chars:
                scenes[-1] = scenes[-1] + "\n" + last_scene
            else:
                combined = scenes[-1] + "\n" + last_scene
                mid = len(combined) // 2
                best_pos = mid
                # Search for sentence break near mid
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
                # If either part is still < min_chars, fall back to exact midpoint
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


def compute_heuristic_signals(
    scene_text: str, scene_index: int, scene_count: int
) -> Tuple[Dict[str, int], Dict[str, Any], Dict[str, Any], Dict[str, int], Dict[str, int]]:
    """Compute deterministic draft annotations (awase, nagare) from surface textual features.

    Guarantees:
    - awase values in [-2, 2]
    - nagare top3 codes strictly distinct with weights [3, 2, 1]
    """
    total_len = max(len(scene_text), 1)
    pos = scene_index / max(scene_count, 1)

    # Dialogue ratio
    quote_matches = re.findall(r"「([^」]*)」", scene_text)
    quote_chars = sum(len(m) for m in quote_matches)
    quote_ratio = quote_chars / total_len

    # Punctuation counts
    excl_count = scene_text.count("！") + scene_text.count("!")
    quest_count = scene_text.count("？") + scene_text.count("?")

    # Lexicon hits
    pos_hits = sum(scene_text.count(w) for w in POS_WORDS)
    neg_hits = sum(scene_text.count(w) for w in NEG_WORDS)
    dyn_hits = sum(scene_text.count(w) for w in DYNAMIC_WORDS)
    stat_hits = sum(scene_text.count(w) for w in STATIC_WORDS)

    # 1. Awase: event valence (-2..2)
    val_raw = (pos_hits - neg_hits) / max((pos_hits + neg_hits), 2)
    if val_raw > 0.4:
        f_event_valence = 2
    elif val_raw > 0.1:
        f_event_valence = 1
    elif val_raw < -0.4:
        f_event_valence = -2
    elif val_raw < -0.1:
        f_event_valence = -1
    else:
        f_event_valence = 0

    # Event arousal (-2..2)
    aro_raw = (dyn_hits + excl_count * 2 - stat_hits) / max(total_len / 200, 1)
    if aro_raw > 2.0:
        f_event_arousal = 2
    elif aro_raw > 0.6:
        f_event_arousal = 1
    elif aro_raw < -2.0:
        f_event_arousal = -2
    elif aro_raw < -0.6:
        f_event_arousal = -1
    else:
        f_event_arousal = 0

    # Style valence & arousal (derived from sentence cadence and dialogue)
    f_style_valence = max(-2, min(2, f_event_valence))
    if quote_ratio > 0.4 or excl_count > 0:
        f_style_arousal = max(-2, min(2, f_event_arousal + 1))
    elif stat_hits > dyn_hits and quote_ratio < 0.15:
        f_style_arousal = max(-2, min(2, f_event_arousal - 1))
    else:
        f_style_arousal = f_event_arousal

    awase = {
        "f_event_valence": int(f_event_valence),
        "f_event_arousal": int(f_event_arousal),
        "f_style_valence": int(f_style_valence),
        "f_style_arousal": int(f_style_arousal),
    }

    # 2. Nagare event codes (E01..E16)
    event_scores: Dict[str, float] = {code: 0.0 for code in EVENT_CODES}
    if pos < 0.2:
        event_scores["E01"] += 3.0  # Setting / Introduction
        event_scores["E02"] += 2.0  # Appearance
    elif pos > 0.8:
        event_scores["E16"] += 3.5  # Climax / Conclusion
        event_scores["E15"] += 2.0  # Reflection
    else:
        event_scores["E10"] += 1.5  # Progression

    if quote_ratio > 0.35:
        event_scores["E03"] += 4.0  # Dialogue
    if quest_count > 0:
        event_scores["E06"] += 2.0  # Conflict / Dilemma
        event_scores["E11"] += 1.5  # Discovery
    if dyn_hits > 2:
        event_scores["E09"] += 2.5  # Action
        event_scores["E08"] += 2.0  # Crisis
    if stat_hits > 2:
        event_scores["E04"] += 2.5  # Introspection
    if neg_hits > 2:
        event_scores["E12"] += 2.5  # Loss / Distress
    if pos_hits > 2:
        event_scores["E14"] += 2.5  # Joy / Success

    # Sort descending by score, tie-breaking by code name
    sorted_events = sorted(EVENT_CODES, key=lambda c: (event_scores[c], -int(c[1:])), reverse=True)
    event_top3 = [
        {"code": sorted_events[0], "rank_weight": 3},
        {"code": sorted_events[1], "rank_weight": 2},
        {"code": sorted_events[2], "rank_weight": 1},
    ]

    # 3. Nagare mood codes (M01..M16)
    mood_scores: Dict[str, float] = {code: 0.0 for code in MOOD_CODES}
    if f_event_arousal <= -1 and f_event_valence >= 0:
        mood_scores["M01"] += 3.0  # Calm
    elif f_event_arousal >= 1 and f_event_valence <= -1:
        mood_scores["M02"] += 3.5  # Tense
        mood_scores["M10"] += 2.5  # Fearful
    elif f_event_valence <= -1:
        mood_scores["M03"] += 3.0  # Melancholy
        mood_scores["M08"] += 2.0  # Pathos
    elif f_event_valence >= 1 and f_event_arousal >= 1:
        mood_scores["M04"] += 3.5  # Joyful
        mood_scores["M14"] += 2.5  # Dynamic
    else:
        mood_scores["M16"] += 2.0  # Lingering / Reflective

    if pos > 0.85:
        mood_scores["M15"] += 2.5  # Resignation / Peace
    if "夢" in scene_text or "幻" in scene_text:
        mood_scores["M13"] += 3.0  # Dreamy

    sorted_moods = sorted(MOOD_CODES, key=lambda c: (mood_scores[c], -int(c[1:])), reverse=True)
    mood_top3 = [
        {"code": sorted_moods[0], "rank_weight": 3},
        {"code": sorted_moods[1], "rank_weight": 2},
        {"code": sorted_moods[2], "rank_weight": 1},
    ]

    nagare = {
        "event_top3": event_top3,
        "mood_top3": mood_top3,
    }

    f_event_stream = {"valence": f_event_valence, "arousal": f_event_arousal}
    f_style_stream = {"valence": f_style_valence, "arousal": f_style_arousal}

    return awase, nagare, {}, f_event_stream, f_style_stream


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
        "character_stalks": [],
        "dialogue_edges": [],
    }
    record["kamae"] = {
        "opened_hooks": [],
        "closed_hook_ids": [],
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
            count = len(scene_texts)
            for idx, sc_txt in enumerate(scene_texts):
                rec = build_scene_annotation(
                    scene_text=sc_txt,
                    author_id=work["author_id"],
                    work_id=work["work_id"],
                    scene_index=idx,
                    scene_count=count,
                    include_audit_meta=include_audit_meta,
                    extra_meta=work,
                )
                all_records.append(rec)

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
        # Group works by author
        author_works: Dict[str, List[Dict[str, Any]]] = {}
        for w in works:
            author_works.setdefault(w["author_id"], []).append(w)

        pilot_records: List[Dict[str, Any]] = []

        # For each author, sample per_author scenes evenly across progression
        for author_id in sorted(author_works.keys()):
            a_works = author_works[author_id]
            # Collect all scenes for this author with audit meta
            a_scenes: List[Dict[str, Any]] = []
            for w in a_works:
                text = self.get_cached_text(w)
                sc_texts = split_work_into_scenes(
                    text, min_chars=min_chars, max_chars=max_chars, target_chars=target_chars
                )
                count = len(sc_texts)
                for idx, sc_txt in enumerate(sc_texts):
                    rec = build_scene_annotation(
                        scene_text=sc_txt,
                        author_id=w["author_id"],
                        work_id=w["work_id"],
                        scene_index=idx,
                        scene_count=count,
                        include_audit_meta=True,
                        extra_meta=w,
                    )
                    a_scenes.append(rec)

            total_a = len(a_scenes)
            if total_a <= per_author:
                selected = a_scenes
            else:
                # 5 evenly spaced checkpoints: ~5%, ~25%, ~50%, ~75%, ~95%
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
