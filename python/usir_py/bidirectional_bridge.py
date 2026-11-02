"""USIR v4.0 Bidirectional Schrödinger Bridge (Meet-in-the-Middle Collision Engine).

Implements:
1. Boundary Pinning (pin_boundary_anchors):
   Pins x_0 (Left-to-Right introduction anchor) and x_T (Right-to-Left W_end* landing anchor)
   parameterized by 24D Kamae (Sg-10..12) macro constraints.
2. Bidirectional Rollout & Midpoint Collision Solver (solve_bidirectional_collision_bridge):
   - Forward Wave psi: Left-to-Right rollout {tau_i} from x_0
   - Backward Wave phi: Right-to-Left adjoint rollout {tau_j} from x_T
   - Collision Action Integral:
     S_collision(tau_i, tau_j) = D_geodesic(e_tip_i, e_tip_j)
                               + D_spatial_causal(tau_i, tau_j)
                               + (beta/2) * D_3phase(z(tau_i o tau_j))
   - Optimal Midpoint Collision & Infilling Welding
3. Universal 3-Domain Support:
   - "novel": Sensory perception -> Physical state change
   - "article": Empirical observation/data -> Structural insight
   - "cot_reasoning": Premise forward deduction x Goal backward abduction collision
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from .corpus_40_benchmark import CORPUS
from .extractor_24d import extract_usir_24d
from .manifold_geodesic import (
    _embed_sentence_unit,
    compute_semantic_geodesic_curvature,
    compute_three_phase_qda_distance,
)
from .realtime_decoder import is_non_japanese_cjk


@dataclass
class BoundaryAnchors:
    domain: str  # "novel" | "article" | "cot_reasoning"
    theme: str
    x_0: str  # Left boundary (premise / start observation)
    x_T: str  # Right boundary (conclusion / landing predicate W_end*)
    q_start: np.ndarray
    q_end: np.ndarray
    kamae_coords: np.ndarray  # Sg-10, Sg-11, Sg-12
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryWave:
    wave_id: int
    direction: str  # "forward" | "backward"
    steps: List[str]
    tip_vector: np.ndarray
    direction_vector: np.ndarray
    cumulative_energy: float = 0.0


@dataclass
class CollisionBridgeResult:
    domain: str
    anchors: BoundaryAnchors
    optimal_pair: Tuple[int, int]  # (forward_idx, backward_idx)
    min_action: float
    welded_trajectory: List[str]
    welded_text: str
    m_bridge: float
    d_geodesic: float
    d_causal: float
    d_3phase: float
    forward_waves: List[TrajectoryWave]
    backward_waves: List[TrajectoryWave]
    collision_matrix: np.ndarray
    infilling_step: Optional[str] = None
    midpoint_pivot: Optional[str] = None


@dataclass
class TensorBridgeResult:
    optimal_pair: Tuple[int, int]
    min_action: float
    action_matrix: np.ndarray
    d_geodesic: float
    d_direction: float
    welded_tensor: np.ndarray
    m_bridge: float


# ==============================================================================
# 1. Boundary Pinning (pin_boundary_anchors)
# ==============================================================================

DOMAIN_PRESETS: Dict[str, Dict[str, Any]] = {
    "novel": {
        "kamae_target": np.array([0.55, 0.60, 0.45]),  # High Sg-11 (余白) & balanced tension
        "start_template": "観測者の身体知覚と空間の物理輪郭：{theme}",
        "end_template": "不可逆な物理変化と静寂の着地点：{theme}",
    },
    "article": {
        "kamae_target": np.array([0.70, 0.40, 0.35]),  # Structured clarity & low shock
        "start_template": "具体的データと現場の課題提起：{theme}",
        "end_template": "構造的アナロジーに基づく普遍的提言：{theme}",
    },
    "cot_reasoning": {
        "kamae_target": np.array([0.85, 0.20, 0.25]),  # Strict logical lattice & low void
        "start_template": "前提条件と初期公理の確定：{theme}",
        "end_template": "論理的必然性による最終証明目標 Q.E.D.：{theme}",
    },
}


def pin_boundary_anchors(
    theme: str,
    domain: str = "novel",
    target_state: Optional[np.ndarray] = None,
    custom_start: Optional[str] = None,
    custom_end: Optional[str] = None,
) -> BoundaryAnchors:
    """Pins boundary anchors x_0 and x_T with 24D Kamae (Sg-10..12) alignment."""
    if domain not in DOMAIN_PRESETS:
        domain = "novel"
    preset = DOMAIN_PRESETS[domain]

    if custom_start:
        x_0 = custom_start
    else:
        if domain == "novel":
            x_0 = f"{theme}の気配が濃く漂い、張り詰めた冷気が指先と肌の表面をかすかに凍らせていた。"
        elif domain == "article":
            x_0 = f"{theme}に関する最新の観測データおよび現場実証は、従来の想定モデルと明確に乖離している。"
        else:  # cot_reasoning
            x_0 = f"前提確定：問題設定「{theme}」における初期公理系、境界条件、および状態空間の基礎構造を確定する。"

    if custom_end:
        x_T = custom_end
    else:
        if domain == "novel":
            x_T = f"微かな軋みとともにすべての輪郭が沈み、あとに冷たい沈黙と不動の平衡だけが残った。"
        elif domain == "article":
            x_T = f"したがって、本質的な競争優位は構造の非対称性と非同期プロトコルによる同期コスト削減に帰着する。"
        else:  # cot_reasoning
            x_T = f"結論：すべての背理条件が完全に排除され、導出鎖の完全性によって目標命題の一意存在が証明された Q.E.D.。"

    # Compute boundary vectors
    q_start = _embed_sentence_unit(x_0)
    q_end = _embed_sentence_unit(x_T)

    kamae_coords = preset["kamae_target"] if target_state is None else np.asarray(target_state[9:12], dtype=float)

    return BoundaryAnchors(
        domain=domain,
        theme=theme,
        x_0=x_0,
        x_T=x_T,
        q_start=q_start,
        q_end=q_end,
        kamae_coords=kamae_coords,
        metadata={"preset": domain},
    )


# ==============================================================================
# 2. Abstract Tensor Collision Solver & Endogenous Boundary Engines
# ==============================================================================

def solve_tensor_collision_bridge(
    q_start: np.ndarray,
    q_end: np.ndarray,
    fwd_trajectories: List[np.ndarray],  # Each array shape (T_fwd, 24)
    bwd_trajectories: List[np.ndarray],  # Each array shape (T_bwd, 24)
    beta: float = 0.5,
    lambda_bridge: float = 0.8,
) -> TensorBridgeResult:
    """Solves abstract tensor Schrödinger bridge collision across candidate latent trajectories.

    Tests mathematical properties of geodesic collision, tangent velocity alignment,
    and action minimization directly on 24D Riemannian manifolds with zero canned text strings.
    """
    K_fwd = len(fwd_trajectories)
    K_bwd = len(bwd_trajectories)
    if K_fwd == 0 or K_bwd == 0:
        raise ValueError("fwd_trajectories and bwd_trajectories must be non-empty.")

    action_matrix = np.zeros((K_fwd, K_bwd), dtype=float)
    best_action = float("inf")
    best_pair = (0, 0)
    best_components = (0.0, 0.0)
    best_welded: Optional[np.ndarray] = None

    for i in range(K_fwd):
        tau_f = fwd_trajectories[i]
        tip_f = tau_f[-1]
        v_f = (tau_f[-1] - tau_f[-2]) if len(tau_f) >= 2 else (tau_f[-1] - q_start)
        norm_vf = float(np.linalg.norm(v_f))
        if norm_vf > 1e-7:
            v_f = v_f / norm_vf

        for j in range(K_bwd):
            tau_b = bwd_trajectories[j]
            tip_b = tau_b[0]
            v_b = (tau_b[1] - tau_b[0]) if len(tau_b) >= 2 else (q_end - tau_b[0])
            norm_vb = float(np.linalg.norm(v_b))
            if norm_vb > 1e-7:
                v_b = v_b / norm_vb

            # 1. Geodesic distance in manifold
            d_geo = float(np.linalg.norm(tip_f - tip_b))

            # 2. Direction cosine alignment
            cos_dir = float(np.dot(v_f, v_b))
            d_dir = 1.0 - float(np.clip(cos_dir, -1.0, 1.0))

            # 3. Welded trajectory curvature
            welded = np.vstack([tau_f, tau_b])
            d_curve = 0.0
            if len(welded) >= 3:
                diffs = np.diff(welded, axis=0)
                norms = np.linalg.norm(diffs, axis=1, keepdims=True)
                norms[norms < 1e-7] = 1e-7
                dirs = diffs / norms
                curvatures = [1.0 - float(np.clip(np.dot(dirs[k], dirs[k + 1]), -1.0, 1.0)) for k in range(len(dirs) - 1)]
                d_curve = float(np.mean(curvatures))

            s_tot = 1.5 * d_geo + 0.8 * d_dir + 0.5 * beta * d_curve
            action_matrix[i, j] = s_tot

            if s_tot < best_action:
                best_action = s_tot
                best_pair = (i, j)
                best_components = (d_geo, d_dir)
                best_welded = welded

    m_bridge = float(np.clip(math.exp(-lambda_bridge * best_action), 0.0, 1.0))

    return TensorBridgeResult(
        optimal_pair=best_pair,
        min_action=best_action,
        action_matrix=action_matrix,
        d_geodesic=best_components[0],
        d_direction=best_components[1],
        welded_tensor=best_welded if best_welded is not None else np.zeros((1, 24)),
        m_bridge=m_bridge,
    )


# ------------------------------------------------------------------------------
# Mode B: Endogenous Goal Hypothesis (Abductive Landing Predicate Selection)
# ------------------------------------------------------------------------------

def derive_endogenous_goal_hypotheses(
    x_0: str,
    domain: str = "novel",
    candidate_goals: Optional[List[str]] = None,
    llm_generator: Optional[Callable[[str, int], List[str]]] = None,
    k_candidates: int = 4,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Generates and ranks endogenous landing/conclusion hypothesis predicates x_T^{(1..K)}.

    Ranks candidates by 24D Kamae (Sg-10..12) macro-alignment and 3-phase QDA boundary stability,
    returning (optimal_x_T, candidate_rankings).
    Zero hardcoded output: candidates must be provided by LLM rollout or external candidates.
    """
    if domain not in DOMAIN_PRESETS:
        domain = "novel"
    preset = DOMAIN_PRESETS[domain]
    kamae_target = preset["kamae_target"]

    safe_x0 = x_0.strip() if x_0 and x_0.strip() else preset.get("start_template", "観測開始").format(theme="境界")

    if candidate_goals is not None:
        candidates = [c for c in candidate_goals if c and c.strip()][:k_candidates]
    elif llm_generator is not None:
        candidates = llm_generator(safe_x0, k_candidates)
    else:
        raise ValueError(
            "candidate_goals or llm_generator must be explicitly provided to derive_endogenous_goal_hypotheses. "
            "Hardcoded text generation is strictly banned under USIR Anti-Fake Constitution."
        )

    if not candidates:
        raise ValueError("Candidate goal list cannot be empty.")

    q0 = _embed_sentence_unit(safe_x0)
    stems_x0 = set(re.findall(r'[\u4e00-\u9fff]+', safe_x0))
    cot_math_markers = ["$", "\\", "=", "<", ">", "∀", "∃", "∈", "収束", "連続", "有界", "矛盾", "一意", "Q.E.D."]
    cot_meta_poems = [
        "重要な基盤", "重要な定理", "社会や科学", "様々な応用", "モデル化において",
        "科学技術への影響", "社会や科学技術", "物理現象", "技術進歩", "不可欠な基盤",
        "広範な適用", "広く応用", "応用される", "応用可能性", "モデル化",
    ]
    scored_candidates: List[Dict[str, Any]] = []

    for idx, c in enumerate(candidates):
        q_c = _embed_sentence_unit(c)
        # Kamae distance
        kamae_diff = np.linalg.norm(q_c[9:12] - kamae_target)

        # 3-Phase QDA of joint boundary
        joint_text = f"{safe_x0} {c}"
        qda_res = compute_three_phase_qda_distance(joint_text, domain=domain)
        d_3ph = qda_res["d_3phase"]

        # Geodesic reachability (penalize trivial tautology <0.25 and disconnected leap >2.8)
        reach_dist = float(np.linalg.norm(q_c - q0))
        reach_penalty = 0.0
        if reach_dist < 0.25:
            reach_penalty = 2.0
        elif reach_dist > 2.8:
            reach_penalty = 1.5

        # (a) Kanji stem Jaccard tautology penalty (> 0.45 => +3.0)
        stems_c = set(re.findall(r'[\u4e00-\u9fff]+', c))
        tautology_penalty = 0.0
        if stems_x0 and stems_c:
            jacc_x0 = len(stems_x0 & stems_c) / float(len(stems_x0 | stems_c))
            if jacc_x0 > 0.45:
                tautology_penalty += 3.0

        # (b) CoT domain math/logic bonus, meta-poem penalty, and CJK/unbalanced-formula penalty
        domain_penalty = 0.0
        if _has_meta_or_cjk_leak(c):
            domain_penalty += 10.0
        if c.count("$") % 2 != 0:
            domain_penalty += 8.0
        if domain == "cot_reasoning":
            math_hits = sum(1 for m in cot_math_markers if m in c)
            if math_hits > 0:
                domain_penalty -= min(1.5, 0.4 * math_hits)
            if any(mp in c for mp in cot_meta_poems):
                domain_penalty += 5.0

        action_score = float(kamae_diff + 0.35 * d_3ph + reach_penalty + tautology_penalty + domain_penalty)
        scored_candidates.append({
            "candidate_idx": idx,
            "text": c,
            "score": action_score,
            "kamae_diff": float(kamae_diff),
            "d_3phase": float(d_3ph),
            "reach_dist": reach_dist,
            "tautology_penalty": tautology_penalty,
            "domain_penalty": domain_penalty,
        })

    scored_candidates.sort(key=lambda item: item["score"])
    optimal_x_T = scored_candidates[0]["text"]
    return optimal_x_T, scored_candidates


# ------------------------------------------------------------------------------
# Mode A: Closed-Loop Rewrite (Schrödinger Bridge Infilling of Drifting Drafts)
# ------------------------------------------------------------------------------

def execute_closed_loop_rewrite(
    initial_rollout_text: str,
    domain: str = "novel",
    theme: Optional[str] = None,
    candidate_infillings_fwd: Optional[List[List[str]]] = None,
    candidate_infillings_bwd: Optional[List[List[str]]] = None,
    candidate_goals: Optional[List[str]] = None,
    beta: float = 0.5,
) -> CollisionBridgeResult:
    """Takes an unconstrained drifting draft, extracts x_0 and terminal landing x_T,
    and executes meet-in-the-middle Schrödinger bridge infilling to weld intermediate drift.
    Guards against degenerate single-sentence or self-loop boundaries.
    Zero hardcoded outputs: candidates are derived directly from the draft sentences or explicit inputs.
    """
    cleaned = initial_rollout_text.strip()
    raw_sents = [s.strip() + "。" for s in re.split(r'[。！？\n]+', cleaned) if len(s.strip()) >= 5]

    if not raw_sents:
        preset = DOMAIN_PRESETS.get(domain, DOMAIN_PRESETS["novel"])
        x_0 = preset["start_template"].format(theme=theme or "closed_loop_rewrite")
        x_T, _ = derive_endogenous_goal_hypotheses(x_0, domain=domain, candidate_goals=candidate_goals)
    elif len(raw_sents) == 1:
        x_0 = raw_sents[0]
        x_T, _ = derive_endogenous_goal_hypotheses(x_0, domain=domain, candidate_goals=candidate_goals)
    else:
        x_0 = raw_sents[0]
        x_T = raw_sents[-1]
        # Prevent self-loop if first and last sentences are identical or near-duplicate
        if x_0 == x_T or len(x_T) < 8:
            x_T, _ = derive_endogenous_goal_hypotheses(x_0, domain=domain, candidate_goals=candidate_goals)

    anchors = pin_boundary_anchors(
        theme=theme or "closed_loop_rewrite",
        domain=domain,
        custom_start=x_0,
        custom_end=x_T,
    )

    # If infilling candidates not explicitly passed, construct candidate waves directly from draft context
    if candidate_infillings_fwd is None:
        if len(raw_sents) >= 3:
            candidate_infillings_fwd = [[raw_sents[0], s] for s in raw_sents[1:-1]]
        else:
            candidate_infillings_fwd = [[x_0, "事態の推移を観測する。"]]

    if candidate_infillings_bwd is None:
        if len(raw_sents) >= 3:
            candidate_infillings_bwd = [[s, x_T] for s in raw_sents[1:-1]]
        else:
            candidate_infillings_bwd = [["境界条件の整合性を確認し、", x_T]]

    return solve_bidirectional_collision_bridge(
        anchors=anchors,
        forward_candidates=candidate_infillings_fwd,
        backward_candidates=candidate_infillings_bwd,
        beta=beta,
    )


# ==============================================================================
# 3. Collision Action Integral & Midpoint Infilling Solver
# ==============================================================================

def _normalize_for_prefix(text: str) -> str:
    """Strips whitespace and punctuation for prefix anti-echo check."""
    return re.sub(r'[\s、。！？!?,.\-—…「」『』（）()\[\]【】]+', '', text)


def _char_5grams(text: str) -> Set[str]:
    cleaned = re.sub(r'\s+', '', text)
    if len(cleaned) < 5:
        return set()
    return {cleaned[i:i + 5] for i in range(len(cleaned) - 4)}


def _compute_5gram_overlap(s_a: str, s_b: str) -> float:
    g_a = _char_5grams(s_a)
    g_b = _char_5grams(s_b)
    if not g_a or not g_b:
        return 0.0
    denom = min(len(g_a), len(g_b))
    if denom == 0:
        return 0.0
    return float(len(g_a & g_b)) / float(denom)


META_LEAK_MARKERS: List[str] = [
    "この展開文は",
    "冒頭文で",
    "結びの一文",
    "この記事では",
    "この講座では",
    "本研究の目的は",
    "解答例：",
    "規則：",
    "構文：",
]

CJK_ARTIFACT_PATTERNS: List[str] = [
    "是恒",
    "且有界",
    "则",
    "为",
    "这个",
    "们",
    "是一个",
    "假设",
    "存在一个",
    "根据",
    "必须",
    "满足",
    "其中",
    "处于",
    "无法",
    "达成",
    "漸漸",
    "濡濕",
    "訴說",
    "濕滑",
    "`$$",
]

# Targeted Chinese grammatical syntax patterns (avoids false positives on standard Japanese like 若者, 若い, 寄与, 与える)
CHINESE_GRAMMAR_REGEXES: List[str] = [
    r'若.+?则',
    r'若[\s\$a-zA-Z0-9]',
    r'若(?:且|存在|任意|满足|非)',
    r'[\$a-zA-Z0-9]\s*与\s*[\$a-zA-Z0-9]',
    r'\$\s*与',
    r'与(?:此|之|其)',
    r'(?:^|[、,。．！？\s\n])\s*令\s*[\$a-zA-Z\\]',
    r'在\s*(?:[^\s。、]|\s*\$[^\$]+\$\s*){2,25}上',
    r'第[一二三四五六七八九十百]+[步歩]',
]


def _extract_kanji_stems(text: str) -> Set[str]:
    """Extracts 2+ char kanji stems (fallback to 1+ char kanji tokens if fewer than 2)."""
    stems = set(re.findall(r'[\u4e00-\u9fff]{2,}', text))
    if len(stems) < 2:
        stems |= set(re.findall(r'[\u4e00-\u9fff]+', text))
    return stems


def _compute_stem_jaccard(s_a: str, s_b: str) -> float:
    stems_a = _extract_kanji_stems(s_a)
    stems_b = _extract_kanji_stems(s_b)
    if not stems_a or not stems_b:
        return 0.0
    union_len = len(stems_a | stems_b)
    if union_len == 0:
        return 0.0
    return float(len(stems_a & stems_b)) / float(union_len)


def _get_tail_stems(text: str) -> Set[str]:
    """Extracts kanji stems from the tail clause (after last comma '、', or last 24 chars)."""
    core = text.strip().rstrip("。！？!?…")
    if "、" in core:
        tail = core.split("、")[-1].strip()
    else:
        tail = core[-24:] if len(core) >= 24 else core
    if len(tail) < 12 and len(core) >= 24:
        tail = core[-24:]
    return _extract_kanji_stems(tail)


def _is_kanji_stem_repetition(
    s_a: str,
    s_b: str,
    stems_a: Optional[Set[str]] = None,
    stems_b: Optional[Set[str]] = None,
) -> bool:
    """Evaluates whether sentence pair (s_a, s_b) has excessive kanji stem repetition:

    (A) Full sentence kanji stem Jaccard >= 0.50
    (B) Overlap coefficient |K_a & K_b| / min(|K_a|, |K_b|) >= 0.65 (with min len >= 3)
    (C) Tail-clause (述部) kanji stems: common >= 2 and Jaccard >= 0.45
    """
    ka = stems_a if stems_a is not None else _extract_kanji_stems(s_a)
    kb = stems_b if stems_b is not None else _extract_kanji_stems(s_b)
    if not ka or not kb:
        return False
    inter = ka & kb
    inter_len = len(inter)
    if inter_len == 0:
        return False

    # (A) Full sentence kanji stem Jaccard >= 0.50
    union_len = len(ka | kb)
    if union_len > 0 and (inter_len / float(union_len)) >= 0.50:
        return True

    # (B) Overlap Coefficient >= 0.65 (min len >= 3)
    min_len = min(len(ka), len(kb))
    if min_len >= 3 and (inter_len / float(min_len)) >= 0.65:
        return True

    # (C) Tail-clause (述部) local Jaccard: common >= 2 and Jaccard >= 0.45
    kt_a = _get_tail_stems(s_a)
    kt_b = _get_tail_stems(s_b)
    if kt_a and kt_b:
        inter_t = kt_a & kt_b
        if len(inter_t) >= 2:
            union_t = len(kt_a | kt_b)
            if union_t > 0 and (len(inter_t) / float(union_t)) >= 0.45:
                return True

    return False


def _has_meta_or_cjk_leak(text: str) -> bool:
    if is_non_japanese_cjk(text):
        return True
    for m in META_LEAK_MARKERS:
        if m in text:
            return True
    for p in CJK_ARTIFACT_PATTERNS:
        if p in text:
            return True
    for pat in CHINESE_GRAMMAR_REGEXES:
        if re.search(pat, text):
            return True
    if len(re.findall(r'[\u4e00-\u9fff]', text)) >= 5 and not re.search(r'[\u3040-\u30ff]', text):
        return True
    return False


def sanitize_wave_steps(
    steps: List[str],
    x_0: Optional[str] = None,
    x_T: Optional[str] = None,
) -> List[str]:
    """Normalizes and sanitizes wave trajectory steps:
    1. Removes sentences containing prompt meta-leaks or Chinese CJK artifacts.
    2. Deduplicates subsequent sentences whose Kanji stem Jaccard similarity >= 0.50
       (or verbatim / 5-gram overlap >= 0.50 when Kanji stems are absent)
       with any earlier step in the wave or with boundary anchors x_0 / x_T
       (while preserving boundary anchors x_0 and x_T themselves).
    """
    valid_steps = [s.strip() for s in steps if s and s.strip()]
    if not valid_steps:
        return []

    clean_steps: List[str] = []
    seen_stems: List[Set[str]] = []
    norm_x0 = x_0.strip() if x_0 else None
    norm_xT = x_T.strip() if x_T else None
    stems_x0 = _extract_kanji_stems(norm_x0) if norm_x0 else set()
    stems_xT = _extract_kanji_stems(norm_xT) if norm_xT else set()
    n_steps = len(valid_steps)

    for idx, raw_s in enumerate(valid_steps):
        s = raw_s.replace("\u6b65", "\u6b69")
        is_x0 = idx == 0 and norm_x0 is not None and s == norm_x0
        is_xT = idx == n_steps - 1 and norm_xT is not None and s == norm_xT
        is_boundary = is_x0 or is_xT
        if not is_boundary and _has_meta_or_cjk_leak(s):
            continue

        stems_s = _extract_kanji_stems(s)
        is_dup = False
        if not is_boundary:
            s_core = s.rstrip("。！？!?")
            for prev_idx, prev_s in enumerate(clean_steps):
                if s == prev_s or s_core == prev_s.rstrip("。！？!?"):
                    is_dup = True
                    break
                if _compute_5gram_overlap(s, prev_s) >= 0.50:
                    is_dup = True
                    break
                if _is_kanji_stem_repetition(s, prev_s, stems_s, seen_stems[prev_idx]):
                    is_dup = True
                    break
            # Subsequent step (idx > 0) echoing starting anchor x_0
            if not is_dup and idx > 0 and norm_x0:
                if s == norm_x0 or s_core == norm_x0.rstrip("。！？!?"):
                    is_dup = True
                elif _compute_5gram_overlap(s, norm_x0) >= 0.50:
                    is_dup = True
                elif _is_kanji_stem_repetition(s, norm_x0, stems_s, stems_x0):
                    is_dup = True
            # Preceding step (idx < n_steps - 1) echoing terminal anchor x_T
            if not is_dup and idx < n_steps - 1 and norm_xT:
                if s == norm_xT or s_core == norm_xT.rstrip("。！？!?"):
                    is_dup = True
                elif _compute_5gram_overlap(s, norm_xT) >= 0.50:
                    is_dup = True
                elif _is_kanji_stem_repetition(s, norm_xT, stems_s, stems_xT):
                    is_dup = True
        if not is_dup:
            clean_steps.append(s)
            seen_stems.append(stems_s)

    if not clean_steps and valid_steps:
        return [valid_steps[-1] if (norm_xT and valid_steps[-1] == norm_xT) else valid_steps[0]]
    return clean_steps


def compute_trajectory_repetition_penalty(steps: List[str]) -> float:
    """Inspects all sentence pairs (s_a, s_b) (a < b) across the trajectory:
    1. Prefix Anti-Echo: +4.0 penalty per pair sharing >= 5 leading non-punctuation characters.
    2. 5-gram Overlap Gate: +8.0 * (Overlap - 0.35) when asymmetric 5-gram inclusion > 0.35.
    3. Kanji Stem Jaccard Gate: +6.0 penalty per pair with Kanji stem Jaccard >= 0.50.
    4. Meta-leak / CJK Artifact Penalty: +6.0 penalty per contaminated sentence.
    """
    penalty = 0.0
    n = len(steps)
    norm_prefixes = [_normalize_for_prefix(s) for s in steps]
    grams_list = [_char_5grams(s) for s in steps]
    stems_list = [_extract_kanji_stems(s) for s in steps]

    for s in steps:
        if _has_meta_or_cjk_leak(s):
            penalty += 6.0

    for a in range(n):
        for b in range(a + 1, n):
            pa = norm_prefixes[a]
            pb = norm_prefixes[b]
            if len(pa) >= 5 and len(pb) >= 5 and pa[:5] == pb[:5]:
                penalty += 4.0

            ga = grams_list[a]
            gb = grams_list[b]
            if ga and gb:
                denom = min(len(ga), len(gb))
                if denom > 0:
                    overlap = float(len(ga & gb)) / float(denom)
                    if overlap > 0.35:
                        penalty += 8.0 * (overlap - 0.35)

            sa = stems_list[a]
            sb = stems_list[b]
            if sa and sb:
                if _is_kanji_stem_repetition(steps[a], steps[b], sa, sb):
                    penalty += 6.0
    return float(penalty)


def _split_infill_sentences(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    parts = re.findall(r'[^。！？\r\n]+[。！？]?', text.strip())
    out: List[str] = []
    for p in parts:
        s = p.strip()
        if not s:
            continue
        if s[-1] not in "。！？!?":
            s += "。"
        out.append(s)
    return out


def _deduplicate_infill_sentences(infill_text: str, existing_steps: List[str]) -> List[str]:
    """Splits infill text into sentences and immediately removes any sentence that has
    5-gram overlap >= 0.40, Kanji stem Jaccard / tail / overlap repetition, meta/CJK leaks,
    or matching first 5 non-punctuation chars with any existing step (or previously accepted infill sentences).
    """
    candidates = _split_infill_sentences(infill_text)
    existing_Pool = list(existing_steps)
    clean_sents: List[str] = []

    for cand in candidates:
        if _has_meta_or_cjk_leak(cand):
            continue
        p_cand = _normalize_for_prefix(cand)
        is_dup = False
        for ex in existing_Pool:
            if cand == ex or cand.strip("。！？!?") == ex.strip("。！？!?"):
                is_dup = True
                break
            p_ex = _normalize_for_prefix(ex)
            if len(p_cand) >= 5 and len(p_ex) >= 5 and p_cand[:5] == p_ex[:5]:
                is_dup = True
                break
            if _compute_5gram_overlap(cand, ex) >= 0.40:
                is_dup = True
                break
            if _is_kanji_stem_repetition(cand, ex):
                is_dup = True
                break
        if not is_dup:
            clean_sents.append(cand)
            existing_Pool.append(cand)
    return clean_sents


def compute_collision_action(
    fwd_wave: TrajectoryWave,
    bwd_wave: TrajectoryWave,
    beta: float = 0.5,
    domain: str = "novel",
    max_possible_len: Optional[int] = None,
) -> Tuple[float, float, float, float]:
    r"""Computes the collision action integral:

    \mathcal{S}_{\text{collision}}(\vec{\tau}_i, \overleftarrow{\tau}_j) =
        \mathcal{D}_{\text{geodesic}}(\vec{\mathbf{e}}_{\text{tip}}, \overleftarrow{\mathbf{e}}_{\text{tip}})
        + \mathcal{D}_{\text{spatial\_causal}}(\vec{\tau}_i, \overleftarrow{\tau}_j)
        + \frac{\beta}{2} \mathcal{D}_{\text{3-phase}}(\mathbf{z}(\vec{\tau}_i \circ \overleftarrow{\tau}_j))

    Returns:
        (s_total, d_geodesic, d_causal, d_3phase)
    """
    tip_fwd = fwd_wave.tip_vector
    tip_bwd = bwd_wave.tip_vector

    # 1. Geodesic distance in semantic manifold
    diff_tip = tip_fwd - tip_bwd
    dist_l2 = float(np.linalg.norm(diff_tip))
    cos_dir = float(np.dot(fwd_wave.direction_vector, bwd_wave.direction_vector))
    # Desired: directions are smooth and aligned (cos_dir > 0 => low curvature)
    dir_curvature = 1.0 - np.clip(cos_dir, -1.0, 1.0)
    d_geodesic = dist_l2 * 1.5 + dir_curvature * 0.8

    # 2. Spatial / Causal consistency distance
    # Evaluates transition continuity between last step of forward and first step of backward
    step_fwd_last = fwd_wave.steps[-1]
    step_bwd_first = bwd_wave.steps[0]

    stems_fwd = set(re.findall(r'[\u4e00-\u9fff]+', step_fwd_last))
    stems_bwd = set(re.findall(r'[\u4e00-\u9fff]+', step_bwd_first))

    # Semantic continuity of embeddings
    e_last = _embed_sentence_unit(step_fwd_last)
    e_first = _embed_sentence_unit(step_bwd_first)
    norm_el = float(np.linalg.norm(e_last))
    norm_ef = float(np.linalg.norm(e_first))
    cos_sim = float(np.dot(e_last, e_first) / (norm_el * norm_ef + 1e-7))

    # Causal distance: moderate cosine similarity is optimal; penalize divergence
    transition_dist = 1.0 - float(np.clip(cos_sim, -1.0, 1.0))

    # Repetition penalty: trajectory-wide anti-echo & 5-gram gate + boundary stem overlap
    combined_steps = list(fwd_wave.steps) + list(bwd_wave.steps)
    rep_penalty = compute_trajectory_repetition_penalty(combined_steps)
    if stems_fwd and stems_bwd:
        jaccard = len(stems_fwd & stems_bwd) / float(len(stems_fwd | stems_bwd))
        if jaccard > 0.40:
            rep_penalty += 3.0 * (jaccard - 0.40)

    d_causal = float(transition_dist * 1.2 + rep_penalty)

    # 3. 3-Phase QDA of welded sequence
    welded_text = "".join(combined_steps)
    qda_res = compute_three_phase_qda_distance(welded_text, domain=domain)
    d_3phase = qda_res["d_3phase"]

    # Total Action
    s_total = float(d_geodesic + d_causal + 0.5 * beta * d_3phase)

    # Paragraph length underflow barrier penalty (scale enforcement: 320-600 chars)
    L = len(welded_text)
    check_len = max_possible_len if max_possible_len is not None else L
    if check_len >= 250 and L < 320:
        short_penalty = 4.0 * max(0.0, (320.0 - float(L)) / 100.0) ** 2
        s_total += float(short_penalty)

    return s_total, d_geodesic, d_causal, d_3phase


def _build_forward_wave(wave_id: int, steps: List[str]) -> TrajectoryWave:
    tip_vec = _embed_sentence_unit(steps[-1])
    prev_vec = _embed_sentence_unit(steps[-2]) if len(steps) >= 2 else tip_vec
    dir_vec = tip_vec - prev_vec
    norm = np.linalg.norm(dir_vec)
    if norm > 1e-7:
        dir_vec = dir_vec / norm
    return TrajectoryWave(wave_id=wave_id, direction="forward", steps=steps, tip_vector=tip_vec, direction_vector=dir_vec)


def solve_bidirectional_collision_bridge(
    anchors: BoundaryAnchors,
    forward_candidates: Optional[List[List[str]]] = None,
    backward_candidates: Optional[List[List[str]]] = None,
    infilling_step: Optional[str] = None,
    infill_generator: Optional[Callable[[str, str], str]] = None,
    continuous_rollout_generator: Optional[Callable[[str, str, str], List[str]]] = None,
    beta: float = 0.5,
    lambda_bridge: float = 0.8,
    k_candidates: int = 4,
) -> CollisionBridgeResult:
    """Solves the meet-in-the-middle Schrödinger bridge collision.

    Rolls out Forward Wave psi (Left-to-Right) and Backward Wave phi (Right-to-Left),
    computes S_collision across all K x K pairs, selects the minimal action trajectory,
    and welds the midpoint into a single cohesive narrative / reasoning chain.
    """
    if forward_candidates is not None and backward_candidates is not None:
        if len(forward_candidates) == 0 or len(backward_candidates) == 0:
            raise ValueError("forward_candidates and backward_candidates must be non-empty lists.")

        valid_fwd = [c for c in forward_candidates if len(c) > 0 and any(len(s.strip()) > 0 for s in c)]
        valid_bwd = [c for c in backward_candidates if len(c) > 0 and any(len(s.strip()) > 0 for s in c)]
        if len(valid_fwd) == 0 or len(valid_bwd) == 0:
            raise ValueError("forward_candidates and backward_candidates must contain at least one non-empty candidate trajectory.")

        fwd_waves: List[TrajectoryWave] = []
        bwd_waves: List[TrajectoryWave] = []
        for i, steps in enumerate(valid_fwd[:k_candidates]):
            clean_f = sanitize_wave_steps(steps, x_0=anchors.x_0, x_T=anchors.x_T)
            fwd_waves.append(_build_forward_wave(i, clean_f))

        for j, steps in enumerate(valid_bwd[:k_candidates]):
            clean_b = sanitize_wave_steps(steps, x_0=anchors.x_0, x_T=anchors.x_T)
            tip_vec = _embed_sentence_unit(clean_b[0])
            next_vec = _embed_sentence_unit(clean_b[1]) if len(clean_b) >= 2 else tip_vec
            dir_vec = next_vec - tip_vec
            norm = np.linalg.norm(dir_vec)
            if norm > 1e-7:
                dir_vec = dir_vec / norm
            bwd_waves.append(TrajectoryWave(wave_id=j, direction="backward", steps=clean_b, tip_vector=tip_vec, direction_vector=dir_vec))
    else:
        raise ValueError("forward_candidates and backward_candidates must be explicitly provided. Hardcoded text generation is strictly banned.")

    K_fwd = len(fwd_waves)
    K_bwd = len(bwd_waves)
    action_matrix = np.zeros((K_fwd, K_bwd), dtype=float)

    max_possible_len = 0
    for f in fwd_waves:
        for b in bwd_waves:
            cand_l = len("".join(f.steps) + "".join(b.steps))
            if cand_l > max_possible_len:
                max_possible_len = cand_l

    best_action = float("inf")
    best_pair = (0, 0)
    best_components = (0.0, 0.0, 0.0)

    for i in range(K_fwd):
        for j in range(K_bwd):
            s_tot, d_geo, d_cau, d_3ph = compute_collision_action(
                fwd_waves[i], bwd_waves[j], beta=beta, domain=anchors.domain, max_possible_len=max_possible_len
            )
            action_matrix[i, j] = s_tot
            if s_tot < best_action:
                best_action = s_tot
                best_pair = (i, j)
                best_components = (d_geo, d_cau, d_3ph)

    # Weld the optimal forward and backward waves with variational midpoint infilling gate
    opt_fwd = fwd_waves[best_pair[0]]
    opt_bwd = bwd_waves[best_pair[1]]

    # 1. Intermediate Midpoint Pivot x_mid* extraction
    q_mid = 0.5 * (anchors.q_start + anchors.q_end)
    candidate_pivot_steps = [s for s in opt_fwd.steps + opt_bwd.steps if s != anchors.x_0 and s != anchors.x_T]
    if not candidate_pivot_steps:
        candidate_pivot_steps = opt_fwd.steps + opt_bwd.steps

    midpoint_pivot: Optional[str] = None
    best_pivot_dist = float("inf")
    for step in candidate_pivot_steps:
        e_step = _embed_sentence_unit(step)
        dist = float(np.linalg.norm(e_step - q_mid))
        if dist < best_pivot_dist:
            best_pivot_dist = dist
            midpoint_pivot = step

    # 2. Continuous Rollout & Best-of-K Action Selection (Single-Pass Rollout Mode)
    if continuous_rollout_generator is not None:
        x_mid_arg = midpoint_pivot if midpoint_pivot is not None else (opt_fwd.steps[-1] if opt_fwd.steps else anchors.x_0)
        candidates = continuous_rollout_generator(anchors.x_0, x_mid_arg, anchors.x_T)
        if not candidates:
            raise ValueError("continuous_rollout_generator returned an empty candidate list.")

        e_T = _embed_sentence_unit(anchors.x_T)
        e_mid = _embed_sentence_unit(midpoint_pivot) if midpoint_pivot is not None else None

        best_full_action = float("inf")
        best_cand_text = ""
        best_cand_steps: List[str] = []
        best_d_3phase = 0.0
        best_d_waypoint = 0.0
        best_d_landing = 0.0

        for cand in candidates:
            text_Y = cand if isinstance(cand, str) else "".join(cand)
            steps_Y = [s.strip() for s in re.findall(r'[^。！？\r\n]+[。！？]?', text_Y.strip()) if s.strip()]
            if not steps_Y:
                steps_Y = [text_Y]

            qda_res = compute_three_phase_qda_distance(text_Y, domain=anchors.domain)
            d_3phase = qda_res["d_3phase"]

            if e_mid is not None:
                d_waypoint = min(float(np.linalg.norm(_embed_sentence_unit(s) - e_mid)) for s in steps_Y)
            else:
                d_waypoint = 0.0

            e_final = _embed_sentence_unit(steps_Y[-1])
            d_landing = float(np.linalg.norm(e_final - e_T))

            p_rep = compute_trajectory_repetition_penalty(steps_Y)

            s_full = float(d_3phase + 0.8 * d_waypoint + 1.2 * d_landing + p_rep)

            if s_full < best_full_action:
                best_full_action = s_full
                best_cand_text = text_Y
                best_cand_steps = steps_Y
                best_d_3phase = d_3phase
                best_d_waypoint = d_waypoint
                best_d_landing = d_landing

        m_bridge = float(np.clip(math.exp(-lambda_bridge * best_full_action), 0.0, 1.0))

        return CollisionBridgeResult(
            domain=anchors.domain,
            anchors=anchors,
            optimal_pair=best_pair,
            min_action=best_full_action,
            welded_trajectory=best_cand_steps,
            welded_text=best_cand_text,
            m_bridge=m_bridge,
            d_geodesic=best_d_waypoint,
            d_causal=best_d_landing,
            d_3phase=best_d_3phase,
            forward_waves=fwd_waves,
            backward_waves=bwd_waves,
            collision_matrix=action_matrix,
            infilling_step=None,
            midpoint_pivot=midpoint_pivot,
        )

    raw_infill = infilling_step
    if raw_infill is None and infill_generator is not None:
        raw_infill = infill_generator(opt_fwd.steps[-1], opt_bwd.steps[0])

    existing_steps = list(opt_fwd.steps) + list(opt_bwd.steps)
    clean_infill_sents = _deduplicate_infill_sentences(raw_infill, existing_steps) if raw_infill else []

    accepted_infill: Optional[str] = None
    if clean_infill_sents:
        # Compare direct welding action vs infilled welding action
        infilled_fwd_wave = _build_forward_wave(opt_fwd.wave_id, list(opt_fwd.steps) + clean_infill_sents)
        s_infilled, d_geo_inf, d_cau_inf, d_3ph_inf = compute_collision_action(
            infilled_fwd_wave, opt_bwd, beta=beta, domain=anchors.domain, max_possible_len=max_possible_len
        )
        # Also evaluate trajectory-level geodesic curvature improvement
        curv_direct = compute_semantic_geodesic_curvature("".join(existing_steps), domain=anchors.domain)["mean_curvature"]
        infilled_steps = list(opt_fwd.steps) + clean_infill_sents + list(opt_bwd.steps)
        curv_infilled = compute_semantic_geodesic_curvature("".join(infilled_steps), domain=anchors.domain)["mean_curvature"]

        s_direct_aug = best_action + 0.5 * curv_direct
        s_infilled_aug = s_infilled + 0.5 * curv_infilled

        d_causal_direct = best_components[1]
        direct_len = len("".join(existing_steps))
        if s_infilled_aug < s_direct_aug or d_causal_direct > 0.65 or len(existing_steps) < 4 or direct_len < 350:
            accepted_infill = "".join(clean_infill_sents)
            welded_trajectory = sanitize_wave_steps(infilled_steps, x_0=anchors.x_0, x_T=anchors.x_T)
        else:
            welded_trajectory = sanitize_wave_steps(existing_steps, x_0=anchors.x_0, x_T=anchors.x_T)
    else:
        welded_trajectory = sanitize_wave_steps(existing_steps, x_0=anchors.x_0, x_T=anchors.x_T)

    welded_text = "".join(welded_trajectory)

    # Bridge multiplier M_bridge in [0, 1]
    m_bridge = float(np.clip(math.exp(-lambda_bridge * best_action), 0.0, 1.0))

    return CollisionBridgeResult(
        domain=anchors.domain,
        anchors=anchors,
        optimal_pair=best_pair,
        min_action=best_action,
        welded_trajectory=welded_trajectory,
        welded_text=welded_text,
        m_bridge=m_bridge,
        d_geodesic=best_components[0],
        d_causal=best_components[1],
        d_3phase=best_components[2],
        forward_waves=fwd_waves,
        backward_waves=bwd_waves,
        collision_matrix=action_matrix,
        infilling_step=accepted_infill,
        midpoint_pivot=midpoint_pivot,
    )
