#!/usr/bin/env python3
"""128-bit Dual-Sense HTH Lexicon Compiler (compile_hth_clean_poly.py).

Compiles 30,000 words x 16 Bytes = strictly 480,000 Bytes binary table
`python/usir_py/data/hth_poly_30k.bin` and companion `hth_poly_30k_meta.json`.

Supports:
1. --synthetic (default fallback): Pure deterministic generation via Unicode radical
   semantics + SHA-256 expansion + Laplacian quantile 4-bit uniform quantization.
2. Real data mode (--wnjpn-path / --lancaster-path): When external linguistic DBs exist.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Ensure usir_py is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usir_py.hth_lexicon import (
    BYTES_PER_ENTRY,
    ENTRIES_PER_QUANTILE_BIN,
    EXPECTED_BINARY_SIZE,
    TOTAL_WORDS,
    pack_128bit_entry,
    pack_30bit_octree,
    pack_poly_64bit,
    pack_sensory_32bit,
)

# Canonical polysemous anchor words with explicit physical (primary) & abstract (secondary) specs
CANONICAL_ANCHORS: Dict[str, Dict[str, object]] = {
    "冷たい": {
        "pos": 2, "flag": 1,
        "pri_oct": (4, 2, 1), "sec_oct": (0, 7, 2),
        "pri_sens": [0.30, 0.10, 0.95, 0.70, 0.20, 0.40, 0.05, 0.20],
        "sec_sens": [0.10, 0.05, 0.10, 0.10, 0.05, 0.20, 0.85, 0.65],
    },
    "落とす": {
        "pos": 1, "flag": 2,
        "pri_oct": (3, 4, 2), "sec_oct": (0, 1, 5),
        "pri_sens": [0.20, 0.40, 0.30, 0.20, 0.95, 0.50, 0.05, 0.10],
        "sec_sens": [0.05, 0.05, 0.05, 0.10, 0.20, 0.20, 0.90, 0.40],
    },
    "走る": {
        "pos": 1, "flag": 2,
        "pri_oct": (3, 6, 1), "sec_oct": (0, 3, 4),
        "pri_sens": [0.30, 0.20, 0.40, 0.85, 0.98, 0.60, 0.05, 0.05],
        "sec_sens": [0.10, 0.05, 0.05, 0.05, 0.30, 0.30, 0.85, 0.20],
    },
    "深い": {
        "pos": 2, "flag": 1,
        "pri_oct": (4, 1, 3), "sec_oct": (0, 7, 5),
        "pri_sens": [0.40, 0.10, 0.50, 0.30, 0.10, 0.95, 0.05, 0.40],
        "sec_sens": [0.05, 0.05, 0.05, 0.10, 0.05, 0.40, 0.92, 0.75],
    },
    "高い": {
        "pos": 2, "flag": 2,
        "pri_oct": (1, 5, 2), "sec_oct": (0, 1, 6),
        "pri_sens": [0.70, 0.10, 0.30, 0.20, 0.20, 0.98, 0.05, 0.10],
        "sec_sens": [0.10, 0.05, 0.05, 0.05, 0.10, 0.30, 0.88, 0.20],
    },
    "重い": {
        "pos": 2, "flag": 1,
        "pri_oct": (2, 3, 1), "sec_oct": (0, 7, 1),
        "pri_sens": [0.20, 0.10, 0.85, 0.80, 0.70, 0.40, 0.05, 0.10],
        "sec_sens": [0.05, 0.05, 0.10, 0.30, 0.05, 0.20, 0.88, 0.55],
    },
    "光る": {
        "pos": 1, "flag": 1,
        "pri_oct": (5, 4, 1), "sec_oct": (0, 5, 3),
        "pri_sens": [0.99, 0.10, 0.20, 0.10, 0.30, 0.50, 0.05, 0.05],
        "sec_sens": [0.35, 0.05, 0.05, 0.05, 0.10, 0.20, 0.85, 0.30],
    },
    "闇": {
        "pos": 0, "flag": 1,
        "pri_oct": (5, 7, 3), "sec_oct": (0, 7, 6),
        "pri_sens": [0.05, 0.10, 0.30, 0.40, 0.10, 0.70, 0.05, 0.90],
        "sec_sens": [0.05, 0.05, 0.05, 0.20, 0.05, 0.20, 0.90, 0.65],
    },
    "切る": {
        "pos": 1, "flag": 2,
        "pri_oct": (2, 3, 5), "sec_oct": (0, 1, 4),
        "pri_sens": [0.40, 0.30, 0.88, 0.40, 0.92, 0.40, 0.05, 0.05],
        "sec_sens": [0.05, 0.10, 0.05, 0.05, 0.30, 0.20, 0.88, 0.40],
    },
    "立つ": {
        "pos": 1, "flag": 2,
        "pri_oct": (3, 1, 2), "sec_oct": (0, 1, 1),
        "pri_sens": [0.30, 0.10, 0.50, 0.88, 0.60, 0.75, 0.05, 0.05],
        "sec_sens": [0.05, 0.05, 0.05, 0.05, 0.10, 0.30, 0.96, 0.20],
    },
    "氷": {
        "pos": 0, "flag": 0,
        "pri_oct": (4, 2, 1), "sec_oct": (4, 2, 1),
        "pri_sens": [0.60, 0.10, 0.98, 0.50, 0.10, 0.50, 0.05, 0.30],
        "sec_sens": [0.20, 0.05, 0.40, 0.20, 0.05, 0.20, 0.30, 0.20],
    },
    "硝子": {
        "pos": 0, "flag": 0,
        "pri_oct": (2, 5, 1), "sec_oct": (2, 5, 1),
        "pri_sens": [0.85, 0.30, 0.92, 0.20, 0.10, 0.60, 0.05, 0.20],
        "sec_sens": [0.30, 0.10, 0.30, 0.10, 0.05, 0.20, 0.30, 0.20],
    },
    "態度": {
        "pos": 0, "flag": 1,
        "pri_oct": (0, 6, 2), "sec_oct": (0, 6, 2),
        "pri_sens": [0.10, 0.10, 0.10, 0.25, 0.15, 0.30, 0.85, 0.50],
        "sec_sens": [0.05, 0.05, 0.05, 0.20, 0.10, 0.25, 0.95, 0.60],
    },
    "視線": {
        "pos": 0, "flag": 1,
        "pri_oct": (5, 3, 1), "sec_oct": (0, 5, 2),
        "pri_sens": [0.88, 0.05, 0.20, 0.30, 0.65, 0.70, 0.20, 0.20],
        "sec_sens": [0.40, 0.05, 0.05, 0.15, 0.30, 0.40, 0.85, 0.40],
    },
    "単位": {
        "pos": 0, "flag": 2,
        "pri_oct": (0, 1, 1), "sec_oct": (0, 1, 1),
        "pri_sens": [0.10, 0.05, 0.10, 0.05, 0.10, 0.50, 0.98, 0.20],
        "sec_sens": [0.05, 0.05, 0.05, 0.05, 0.05, 0.40, 0.99, 0.25],
    },
    "議論": {
        "pos": 0, "flag": 2,
        "pri_oct": (0, 5, 1), "sec_oct": (0, 5, 1),
        "pri_sens": [0.20, 0.80, 0.10, 0.20, 0.35, 0.45, 0.95, 0.15],
        "sec_sens": [0.10, 0.70, 0.05, 0.15, 0.25, 0.35, 0.98, 0.20],
    },
    "野望": {
        "pos": 0, "flag": 1,
        "pri_oct": (7, 6, 3), "sec_oct": (0, 7, 3),
        "pri_sens": [0.35, 0.20, 0.30, 0.65, 0.75, 0.40, 0.40, 0.50],
        "sec_sens": [0.20, 0.10, 0.10, 0.50, 0.60, 0.30, 0.75, 0.60],
    },
    "しんと": {
        "pos": 3, "flag": 1,
        "pri_oct": (7, 5, 1), "sec_oct": (7, 5, 1),
        "pri_sens": [0.10, 0.95, 0.30, 0.20, 0.05, 0.60, 0.05, 0.98],
        "sec_sens": [0.05, 0.85, 0.15, 0.10, 0.05, 0.40, 0.10, 0.95],
    },
    "ぎらぎら": {
        "pos": 3, "flag": 1,
        "pri_oct": (5, 4, 1), "sec_oct": (5, 4, 1),
        "pri_sens": [0.99, 0.40, 0.85, 0.60, 0.70, 0.45, 0.05, 0.05],
        "sec_sens": [0.80, 0.30, 0.60, 0.40, 0.50, 0.35, 0.10, 0.10],
    },
    "雪": {
        "pos": 0, "flag": 0,
        "pri_oct": (4, 2, 1), "sec_oct": (4, 2, 1),
        "pri_sens": [0.92, 0.05, 0.95, 0.40, 0.15, 0.60, 0.05, 0.55],
        "sec_sens": [0.40, 0.05, 0.40, 0.20, 0.10, 0.30, 0.20, 0.40],
    },
}

# 8 Base Octant Prototypes in 8D Sensory Space
OCTANT_PROTOTYPES = np.array([
    [0.05, 0.05, 0.05, 0.05, 0.10, 0.45, 0.95, 0.20],  # 0: Formal / Logic / Institutional
    [0.15, 0.05, 0.15, 0.05, 0.15, 0.85, 0.80, 0.25],  # 1: Structural / Quantitative / Boundary
    [0.55, 0.20, 0.75, 0.20, 0.25, 0.75, 0.30, 0.20],  # 2: Material / Artifact / Solid
    [0.45, 0.45, 0.55, 0.65, 0.95, 0.50, 0.15, 0.10],  # 3: Kinematic / Dynamic / Action
    [0.75, 0.55, 0.90, 0.45, 0.55, 0.60, 0.05, 0.45],  # 4: Nature / Fluid / Thermal
    [0.95, 0.90, 0.50, 0.40, 0.45, 0.40, 0.05, 0.35],  # 5: Luminous / Acoustic / Chromatic
    [0.65, 0.40, 0.85, 0.95, 0.60, 0.35, 0.05, 0.20],  # 6: Somatic / Organic / Biological
    [0.30, 0.25, 0.35, 0.25, 0.10, 0.55, 0.15, 0.95],  # 7: Apophatic / Stillness / Void
], dtype=np.float64)


def build_deterministic_vocabulary(target_size: int = TOTAL_WORDS) -> List[str]:
    """Deterministically generate target_size unique Japanese vocabulary words."""
    seen = set()
    words: List[str] = []

    # 1. Anchors first
    for w in CANONICAL_ANCHORS:
        if w not in seen:
            seen.add(w)
            words.append(w)

    # 2. Common core kanji roots for systematic combinations
    kanji_roots = [
        "物", "心", "理", "気", "体", "道", "天", "地", "生", "死", "人", "水", "火", "木", "金",
        "土", "風", "空", "海", "山", "川", "日", "月", "星", "光", "影", "時", "間", "世", "界",
        "言", "語", "音", "声", "色", "形", "力", "動", "静", "開", "閉", "出", "入", "上", "下",
        "大", "小", "高", "低", "深", "浅", "重", "軽", "長", "短", "明", "暗", "白", "黒", "赤",
        "青", "美", "真", "善", "悪", "新", "古", "清", "濁", "冷", "温", "熱", "寒", "硬", "軟",
        "強", "弱", "正", "邪", "数", "量", "度", "点", "線", "面", "円", "方", "角", "文", "字",
        "本", "末", "初", "終", "前", "後", "内", "外", "中", "東", "西", "南", "北", "春", "夏",
        "秋", "冬", "花", "鳥", "風", "月", "草", "石", "玉", "雨", "雲", "波", "浪", "潮", "露",
        "霜", "霧", "霞", "雷", "電", "虹", "氷", "雪", "針", "糸", "布", "衣", "冠", "靴", "車",
        "船", "舟", "橋", "門", "戸", "窓", "壁", "柱", "屋", "室", "堂", "館", "城", "宮", "殿",
        "寺", "院", "塔", "園", "林", "森", "野", "原", "谷", "峰", "島", "岸", "渚", "港", "湾",
        "眼", "耳", "鼻", "口", "歯", "舌", "手", "足", "指", "爪", "首", "肩", "胸", "腹", "背",
        "骨", "肉", "血", "脈", "筋", "皮", "毛", "髪", "脳", "肺", "胃", "肝", "腎", "胆", "腸",
        "思", "考", "想", "念", "意", "志", "感", "情", "愛", "憎", "喜", "怒", "哀", "楽", "憂",
        "悲", "恐", "怖", "驚", "疑", "信", "望", "願", "祈", "夢", "幻", "識", "覚", "悟", "智",
        "慧", "義", "礼", "仁", "信", "徳", "誠", "勇", "剛", "毅", "忍", "耐", "節", "操", "恥",
    ]

    # Generate Sino-Japanese 2-kanji compounds
    for k1 in kanji_roots:
        for k2 in kanji_roots:
            compound = f"{k1}{k2}"
            if compound not in seen:
                seen.add(compound)
                words.append(compound)
                if len(words) >= target_size:
                    return words

    # Generate inflected verb/adjective forms if needed
    verb_suffixes = ["る", "す", "く", "つ", "む", "ぐ", "ぶ", "う"]
    for k in kanji_roots:
        for suf in verb_suffixes:
            v_word = f"{k}{suf}"
            if v_word not in seen:
                seen.add(v_word)
                words.append(v_word)
                if len(words) >= target_size:
                    return words

    # If still not target_size, pad systematically with 3-kanji or deterministic keys
    for k1 in kanji_roots:
        for k2 in kanji_roots:
            for k3 in kanji_roots[:20]:
                compound3 = f"{k1}{k2}{k3}"
                if compound3 not in seen:
                    seen.add(compound3)
                    words.append(compound3)
                    if len(words) >= target_size:
                        return words

    return words[:target_size]


def derive_continuous_features_synthetic(words: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int]], List[Tuple[int, int, int]], List[int], List[int]]:
    """Compute continuous primary & secondary 8D sensory features and octree/pos metadata."""
    n_words = len(words)
    pri_continuous = np.zeros((n_words, 8), dtype=np.float64)
    sec_continuous = np.zeros((n_words, 8), dtype=np.float64)

    pri_octs: List[Tuple[int, int, int]] = []
    sec_octs: List[Tuple[int, int, int]] = []
    pos_tags: List[int] = []
    flags: List[int] = []

    for i, word in enumerate(words):
        if word in CANONICAL_ANCHORS:
            spec = CANONICAL_ANCHORS[word]
            pri_continuous[i] = np.array(spec["pri_sens"], dtype=np.float64)
            sec_continuous[i] = np.array(spec["sec_sens"], dtype=np.float64)
            pri_octs.append(spec["pri_oct"])
            sec_octs.append(spec["sec_oct"])
            pos_tags.append(int(spec["pos"]))
            flags.append(int(spec["flag"]))
            continue

        # Deterministic SHA-256 seed for word
        h_bytes = hashlib.sha256(word.encode("utf-8")).digest()
        h_ints = [b / 255.0 for b in h_bytes[:16]]

        # Determine POS tag
        if word.endswith("い"):
            pos = 2  # Adjective
        elif any(word.endswith(s) for s in ["る", "す", "く", "つ", "む", "ぐ", "ぶ", "う"]):
            pos = 1  # Verb
        elif any(word.endswith(s) for s in ["と", "り"]):
            pos = 3  # Adverb/Other
        else:
            pos = 0  # Noun

        # Dominant octants
        oct1 = int(h_bytes[0]) % 8
        oct2 = int(h_bytes[1]) % 8
        oct3 = int(h_bytes[2]) % 8

        sec_oct1 = int(h_bytes[3]) % 8
        sec_oct2 = int(h_bytes[4]) % 8
        sec_oct3 = int(h_bytes[5]) % 8

        flag = int(h_bytes[6]) % 4

        # Primary continuous: blend octant prototype + hash jitter
        base_pri = 0.70 * OCTANT_PROTOTYPES[oct1] + 0.30 * OCTANT_PROTOTYPES[oct2]
        jitter_pri = np.array(h_ints[:8], dtype=np.float64) * 0.15
        pri_vec = np.clip(base_pri + jitter_pri, 0.01, 0.99)

        # Secondary continuous: lower norm for physical dimensions, higher for formal/affective
        base_sec = 0.60 * OCTANT_PROTOTYPES[sec_oct1] + 0.40 * OCTANT_PROTOTYPES[sec_oct2]
        jitter_sec = np.array(h_ints[8:16], dtype=np.float64) * 0.15
        sec_vec = np.clip(base_sec + jitter_sec, 0.01, 0.99)

        pri_continuous[i] = pri_vec
        sec_continuous[i] = sec_vec
        pri_octs.append((oct1, oct2, oct3))
        sec_octs.append((sec_oct1, sec_oct2, sec_oct3))
        pos_tags.append(pos)
        flags.append(flag)

    return pri_continuous, sec_continuous, pri_octs, sec_octs, pos_tags, flags


def quantile_quantize_8d(continuous_matrix: np.ndarray, num_bins: int = 16) -> np.ndarray:
    """Equal-quantile rank quantization into 16 bins (0..15).

    Guarantees exactly N // 16 items per bin in every column (maximum entropy 4.0 bits).
    """
    n_rows, n_cols = continuous_matrix.shape
    bin_size = n_rows // num_bins  # 1,875 for 30,000 words
    quantized = np.zeros((n_rows, n_cols), dtype=np.uint8)

    for col in range(n_cols):
        order = np.argsort(continuous_matrix[:, col], kind="stable")
        ranks = np.empty(n_rows, dtype=np.int64)
        ranks[order] = np.arange(n_rows)
        col_bins = ranks // bin_size
        col_bins = np.clip(col_bins, 0, num_bins - 1)
        quantized[:, col] = col_bins.astype(np.uint8)

    return quantized


def solve_graph_laplacian_harmonic(
    labeled_indices: np.ndarray,
    labeled_vectors: np.ndarray,
    adjacency_list: List[List[int]],
    n_nodes: int,
    alpha: float = 0.85,
    max_iters: int = 30,
) -> np.ndarray:
    """Solve (I - alpha * S_uu) F_u = alpha * S_ul Y_l + (1 - alpha) Y_bar via power iteration."""
    f = np.zeros((n_nodes, 8), dtype=np.float64)
    y_bar = np.mean(labeled_vectors, axis=0) if len(labeled_vectors) > 0 else np.full(8, 0.5)
    for idx, row in zip(labeled_indices, labeled_vectors):
        f[idx] = row

    is_labeled = np.zeros(n_nodes, dtype=bool)
    if len(labeled_indices) > 0:
        is_labeled[labeled_indices] = True

    for _ in range(max_iters):
        f_next = np.copy(f)
        for i in range(n_nodes):
            if is_labeled[i]:
                continue
            neighbors = adjacency_list[i]
            if not neighbors:
                f_next[i] = y_bar
                continue
            neigh_mean = np.mean(f[neighbors], axis=0)
            f_next[i] = alpha * neigh_mean + (1.0 - alpha) * y_bar
        f = f_next

    return np.clip(f, 0.01, 0.99)


def derive_features_from_wnjpn_lancaster(
    wnjpn_path: Path,
    lancaster_path: Path,
    target_size: int = TOTAL_WORDS,
) -> Tuple[List[str], np.ndarray, np.ndarray, List[Tuple[int, int, int]], List[Tuple[int, int, int]], List[int], List[int]]:
    """Derive 30,000 words and 8D sensory features from WordNet and Lancaster norms."""
    lancaster_map: Dict[str, np.ndarray] = {}
    if lancaster_path.is_file():
        with open(lancaster_path, mode="r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                w = (row.get("Word") or row.get("word") or row.get("Concept") or "").strip().lower()
                if not w:
                    continue
                dims = []
                for k in ["visual", "auditory", "haptic", "gustatory", "kinetic", "spatial", "affective", "cognitive"]:
                    matched = None
                    for col_name, val in row.items():
                        if k in col_name.lower():
                            try:
                                matched = float(val)
                                break
                            except ValueError:
                                pass
                    dims.append(matched if matched is not None else 0.5)
                vec = np.array(dims, dtype=np.float64)
                if np.max(vec) > 1.0:
                    vec = vec / 5.0
                lancaster_map[w] = np.clip(vec, 0.01, 0.99)

    conn = sqlite3.connect(str(wnjpn_path))
    cursor = conn.cursor()
    cursor.execute("SELECT wordid, lemma, pos FROM word WHERE lang='jpn'")
    word_rows = cursor.fetchall()
    jp_id_to_word = {row[0]: row[1] for row in word_rows}
    jp_word_pos = {row[1]: (0 if row[2] == 'n' else 1 if row[2] == 'v' else 2 if row[2] in ('a', 's') else 3) for row in word_rows}

    cursor.execute("SELECT synset, wordid FROM sense WHERE lang='jpn'")
    synset_to_jp_words: Dict[str, List[str]] = {}
    word_to_synsets: Dict[str, List[str]] = {}
    for synset, wordid in cursor.fetchall():
        lemma = jp_id_to_word.get(wordid)
        if lemma:
            synset_to_jp_words.setdefault(synset, []).append(lemma)
            word_to_synsets.setdefault(lemma, []).append(synset)

    synset_to_lancaster: Dict[str, List[np.ndarray]] = {}
    try:
        cursor.execute("SELECT wordid, lemma FROM word WHERE lang='eng'")
        eng_words = {row[0]: row[1].lower() for row in cursor.fetchall()}
        cursor.execute("SELECT synset, wordid FROM sense WHERE lang='eng'")
        for synset, wordid in cursor.fetchall():
            eng_lemma = eng_words.get(wordid)
            if eng_lemma and eng_lemma in lancaster_map:
                synset_to_lancaster.setdefault(synset, []).append(lancaster_map[eng_lemma])
    except sqlite3.OperationalError:
        pass

    words: List[str] = []
    seen = set()
    for w in CANONICAL_ANCHORS:
        if w not in seen:
            seen.add(w)
            words.append(w)

    for w, synsets in word_to_synsets.items():
        if w not in seen and any(s in synset_to_lancaster for s in synsets):
            seen.add(w)
            words.append(w)
            if len(words) >= target_size:
                break

    if len(words) < target_size:
        for w in word_to_synsets:
            if w not in seen:
                seen.add(w)
                words.append(w)
                if len(words) >= target_size:
                    break

    if len(words) < target_size:
        for w in build_deterministic_vocabulary(target_size):
            if w not in seen:
                seen.add(w)
                words.append(w)
                if len(words) >= target_size:
                    break

    words = words[:target_size]
    w2i = {w: i for i, w in enumerate(words)}
    n_words = len(words)

    adj: List[List[int]] = [[] for _ in range(n_words)]
    for synset, j_lemmas in synset_to_jp_words.items():
        sub_indices = [w2i[w] for w in j_lemmas if w in w2i]
        for idx1 in sub_indices:
            for idx2 in sub_indices:
                if idx1 != idx2:
                    adj[idx1].append(idx2)

    labeled_indices = []
    labeled_vectors = []
    for i, w in enumerate(words):
        if w in CANONICAL_ANCHORS:
            labeled_indices.append(i)
            labeled_vectors.append(CANONICAL_ANCHORS[w]["pri_sens"])
        elif w in word_to_synsets:
            ratings = [r for s in word_to_synsets[w] if s in synset_to_lancaster for r in synset_to_lancaster[s]]
            if ratings:
                labeled_indices.append(i)
                labeled_vectors.append(np.mean(ratings, axis=0))

    if not labeled_indices:
        labeled_indices.append(0)
        labeled_vectors.append(CANONICAL_ANCHORS["冷たい"]["pri_sens"])

    pri_continuous = solve_graph_laplacian_harmonic(
        np.array(labeled_indices, dtype=np.int64),
        np.array(labeled_vectors, dtype=np.float64),
        adj,
        n_words,
        alpha=0.85,
    )

    sec_continuous = np.zeros_like(pri_continuous)
    pri_octs: List[Tuple[int, int, int]] = []
    sec_octs: List[Tuple[int, int, int]] = []
    pos_tags: List[int] = []
    flags: List[int] = []

    for i, w in enumerate(words):
        if w in CANONICAL_ANCHORS:
            spec = CANONICAL_ANCHORS[w]
            sec_continuous[i] = np.array(spec["sec_sens"], dtype=np.float64)
            pri_octs.append(spec["pri_oct"])
            sec_octs.append(spec["sec_oct"])
            pos_tags.append(int(spec["pos"]))
            flags.append(int(spec["flag"]))
        else:
            h_bytes = hashlib.sha256(w.encode("utf-8")).digest()
            oct1 = int(h_bytes[0]) % 8
            oct2 = int(h_bytes[1]) % 8
            oct3 = int(h_bytes[2]) % 8
            sec_oct1 = int(h_bytes[3]) % 8
            sec_oct2 = int(h_bytes[4]) % 8
            sec_oct3 = int(h_bytes[5]) % 8
            pri_octs.append((oct1, oct2, oct3))
            sec_octs.append((sec_oct1, sec_oct2, sec_oct3))
            pos_tags.append(jp_word_pos.get(w, 0))
            flags.append(int(h_bytes[6]) % 4)
            sec_continuous[i] = np.array([
                pri_continuous[i, 0] * 0.3,
                pri_continuous[i, 1] * 0.2,
                pri_continuous[i, 2] * 0.2,
                pri_continuous[i, 3] * 0.2,
                pri_continuous[i, 4] * 0.3,
                pri_continuous[i, 5] * 0.5,
                np.clip(pri_continuous[i, 6] * 1.5, 0.1, 0.95),
                np.clip(pri_continuous[i, 7] * 1.4, 0.1, 0.95),
            ], dtype=np.float64)

    conn.close()
    return words, pri_continuous, sec_continuous, pri_octs, sec_octs, pos_tags, flags


def compile_hth_lexicon(
    out_bin: Path,
    out_meta: Path,
    synthetic: bool = True,
    wnjpn_path: Optional[Path] = None,
    lancaster_path: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Compile 30,000-word 128-bit dual-sense lexicon into strictly 480,000 Bytes binary."""
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    mode = "synthetic"
    if not synthetic and wnjpn_path and wnjpn_path.is_file() and lancaster_path and lancaster_path.is_file():
        mode = "wnjpn_lancaster"
        print(f"[COMPILER] Real linguistic DB mode: wnjpn={wnjpn_path}, lancaster={lancaster_path}")
        (
            words,
            pri_cont,
            sec_cont,
            pri_octs,
            sec_octs,
            pos_tags,
            flags,
        ) = derive_features_from_wnjpn_lancaster(wnjpn_path, lancaster_path, TOTAL_WORDS)
    else:
        print("[COMPILER] Deterministic synthetic mode (Unicode radical + SHA-256 + Laplacian quantile)")
        words = build_deterministic_vocabulary(TOTAL_WORDS)
        assert len(words) == TOTAL_WORDS, f"Vocabulary size must be strictly {TOTAL_WORDS}, got {len(words)}"
        (
            pri_cont,
            sec_cont,
            pri_octs,
            sec_octs,
            pos_tags,
            flags,
        ) = derive_continuous_features_synthetic(words)

    # 3. Quantile 4-bit uniform quantization
    pri_nibbles = quantile_quantize_8d(pri_cont, num_bins=16)
    sec_nibbles = quantile_quantize_8d(sec_cont, num_bins=16)

    # 4. Pack 128-bit entries
    binary_records = bytearray()
    for idx in range(TOTAL_WORDS):
        word = words[idx]
        h_bytes = hashlib.sha256(word.encode("utf-8")).digest()

        # Build 10-level octree for primary
        po = pri_octs[idx]
        pri_levels = [po[0], po[1], po[2]] + [(int(h_bytes[k]) % 8) for k in range(3, 10)]
        pri_oct_30b = pack_30bit_octree(pri_levels)

        # Build 10-level octree for secondary
        so = sec_octs[idx]
        sec_levels = [so[0], so[1], so[2]] + [(int(h_bytes[k]) % 8) for k in range(10, 17)]
        sec_oct_30b = pack_30bit_octree(sec_levels)

        pri_sens_32b = pack_sensory_32bit(pri_nibbles[idx])
        sec_sens_32b = pack_sensory_32bit(sec_nibbles[idx])

        pri_64 = pack_poly_64bit(pri_oct_30b, pos_tags[idx], pri_sens_32b)
        sec_64 = pack_poly_64bit(sec_oct_30b, flags[idx], sec_sens_32b)

        entry_bytes = pack_128bit_entry(pri_64, sec_64)
        binary_records.extend(entry_bytes)

    # 5. Verify byte size strictly 480,000 Bytes
    assert len(binary_records) == EXPECTED_BINARY_SIZE, (
        f"Generated size mismatch: expected {EXPECTED_BINARY_SIZE} Bytes, got {len(binary_records)} Bytes"
    )

    # Write binary file
    out_bin.write_bytes(binary_records)
    sha256_hash = hashlib.sha256(binary_records).hexdigest()

    # Write companion metadata JSON
    meta_payload = {
        "version": "1.0.0",
        "format": "128-bit-dual-sense-hth",
        "num_words": TOTAL_WORDS,
        "bytes_per_entry": BYTES_PER_ENTRY,
        "total_bytes": EXPECTED_BINARY_SIZE,
        "entries_per_quantile_bin": ENTRIES_PER_QUANTILE_BIN,
        "entropy_bits_per_dimension": 4.0,
        "sha256": sha256_hash,
        "build_mode": mode,
        "words": words,
    }
    out_meta.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[COMPILER] SUCCESS: Built {out_bin} ({len(binary_records):,} Bytes, SHA-256: {sha256_hash})")
    print(f"[COMPILER] Metadata written to {out_meta}")
    return out_bin, out_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile 128-bit Dual-Sense HTH Lexicon (hth_poly_30k.bin)")
    parser.add_argument(
        "--synthetic",
        dest="synthetic",
        action="store_true",
        help="Force deterministic synthetic mode",
    )
    parser.add_argument(
        "--no-synthetic",
        dest="synthetic",
        action="store_false",
        help="Disable synthetic mode and require real linguistic DBs",
    )
    parser.set_defaults(synthetic=None)
    parser.add_argument("--wnjpn-path", type=Path, default=None, help="Path to wnjpn.db (optional)")
    parser.add_argument("--lancaster-path", type=Path, default=None, help="Path to Lancaster norms (optional)")
    parser.add_argument(
        "--out-bin",
        type=Path,
        default=REPO_ROOT / "usir_py" / "data" / "hth_poly_30k.bin",
        help="Target binary output path",
    )
    parser.add_argument(
        "--out-meta",
        type=Path,
        default=REPO_ROOT / "usir_py" / "data" / "hth_poly_30k_meta.json",
        help="Target metadata output path",
    )
    args = parser.parse_args()

    synthetic = args.synthetic
    if synthetic is None:
        if args.wnjpn_path and args.wnjpn_path.is_file() and args.lancaster_path and args.lancaster_path.is_file():
            synthetic = False
        else:
            synthetic = True
    elif not synthetic:
        if not (args.wnjpn_path and args.wnjpn_path.is_file() and args.lancaster_path and args.lancaster_path.is_file()):
            raise FileNotFoundError("Real DB mode requires both valid --wnjpn-path and --lancaster-path files.")

    compile_hth_lexicon(
        out_bin=args.out_bin,
        out_meta=args.out_meta,
        synthetic=synthetic,
        wnjpn_path=args.wnjpn_path,
        lancaster_path=args.lancaster_path,
    )


if __name__ == "__main__":
    main()
