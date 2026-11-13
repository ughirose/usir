"""USIR v5 Stage B-11: F-14 Hybrid Judgment & Harvest Paragraph Noise Floor Alignment.

Replicates the exact F-14 hybrid dimension selection and sigma table lookup
implemented in ``nishiki`` (``crates/nishiki-cli/src/diff_harvester.rs`` and
``crates/kasane-core/src/byor/trajectory.rs``):

1. Trajectory window counting (``window_chars=250``, ``stride_chars=125``)
   with exact character-for-character parity with ``kasane_core::IncrementalAnalyzer::split_sentences``
   and ``kasane_core::TrajectoryAnalyzer::analyze``.
2. Evaluation rule v2 (2026-11-13, B-14; ``is_eight_dim``). Mirrors nishiki
   ``diff_harvester::is_eight_dim`` / ``n_eval_rule_v2``:
   - Both sides ``<= 1`` window -> **8 dimensions** (``both_single_window:8``, excluding
     ``ac_mean_velocity``, ``ac_max_sis``, ``ac_coupling``; ``ac_mean_crest`` stays active).
   - Mixed pair (one side ``<= 1`` window, the other ``>= 2``) -> **8 dimensions**
     (``mixed_window:8``; ac_* enter neither numerator nor denominator). A 7-dim
     diagnostic (dc 7 only, crest excluded) is attached as ``diagnostic_7d_no_crest``
     (diagnostic only, NOT canonical).
   - Both sides ``>= 2`` windows -> **11 dimensions** (``both_multi_window:11``).
   Rationale (B-13 / F-18): 1-window ac_* are exactly 0.0 in 476/476 cases, so 11D on mixed
   pairs yields a dimension-addition artifact ("value - 0"); ac_mean_crest inflates 1.7-2.0x across
   window counts but its contribution is small on Set 1 mixed pairs (7D/8D = 0.942), so the
   canonical dimension set keeps crest. ``is_eight_dim_v1`` keeps the old F-14 rule
   (8D only if both <= 1 window) for before/after comparison; it is not used for evaluation.
   The ac_* sigma population (both sides multi-window pairs only) is unchanged.
3. Length-band lookup by ``L_max = max(len(before), len(after))`` (bands ``S``, ``M``, ``L``)
   and ``ac_multi_window_per_dim_std`` override for the 3 AC velocity/SIS/coupling dimensions
   when evaluating 11D pairs against ``docs/handoff/v10/paragraph_noise_floor_all_216_canonical.json`` (default since F-22; F-17 floor ``paragraph_noise_floor_all_novel.json`` is the historical reference).

GOVERNANCE & NON-CONFLATION GUARD:
    Harvest paragraph noise floor ``sigma`` (``sigma_kind="adjacent_paragraph_diff_sd"``)
    is strictly distinct from Gate 3a within-register scene variance ``sigma_scene_r``
    (24D ``ortho24_cont``). They must never be mixed, compared as equal scales, or substituted.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CANONICAL_F14_NOISE_FLOOR_PATH = REPO_ROOT / "docs" / "handoff" / "v10" / "paragraph_noise_floor_all_216_canonical.json"
# F-17 floor (46 works, SHA 0575B64A...) is retained as history / benchmark reference only.
F17_HISTORICAL_NOISE_FLOOR_PATH = REPO_ROOT / "docs" / "handoff" / "v10" / "paragraph_noise_floor_all_novel.json"
CANONICAL_F14_BOOTSTRAP_CI_PATH = REPO_ROOT / "docs" / "handoff" / "v10" / "ac_sigma_bootstrap_ci.json"

KASANE11_DIM_NAMES: Tuple[str, ...] = (
    "dc_showing",
    "dc_staccato",
    "dc_polyphony",
    "dc_confession",
    "dc_cadence",
    "dc_speculation",
    "semantic_tension",
    "ac_mean_velocity",
    "ac_max_sis",
    "ac_coupling",
    "ac_mean_crest",
)

SHORT_DEAD_DIMS: Tuple[str, ...] = (
    "ac_mean_velocity",
    "ac_max_sis",
    "ac_coupling",
)

KASANE8_ACTIVE_DIM_NAMES: Tuple[str, ...] = tuple(
    dim for dim in KASANE11_DIM_NAMES if dim not in SHORT_DEAD_DIMS
)

KASANE8_ACTIVE_INDICES: Tuple[int, ...] = tuple(
    idx for idx, dim in enumerate(KASANE11_DIM_NAMES) if dim not in SHORT_DEAD_DIMS
)

ALIGN_MIN_SIMILARITY: float = 0.25

_SENTENCE_BOUNDARY_CHARS = frozenset(("\u3002", "\uff01", "\uff1f"))  # 。 ！ ？
_SENTENCE_CLOSER_CHARS = frozenset(
    (
        "\u300d",  # 」
        "\u300f",  # 』
        "\uff09",  # ）
        "\u3015",  # 〕
        "\u201d",  # ”
        "\u2019",  # ’
        "\n",
        "\r",
    )
)

F14_SIGMA_PROVENANCE: Dict[str, Any] = {
    "canonical_path": "docs/handoff/v10/paragraph_noise_floor_all_216_canonical.json",
    "upstream_evidence_path": "C:/dev/nishiki/docs/handoff/evidence/2026-11-14_f21/paragraph_noise_floor_all_216_canonical.json",
    "historical_f17_floor_path": "docs/handoff/v10/paragraph_noise_floor_all_novel.json",
    "bootstrap_ci_path": "docs/handoff/v10/ac_sigma_bootstrap_ci.json",
    "nishiki_commits": {
        "f14_hybrid_and_all_novel_sigma": "48b6eb7",
        "f14_bootstrap_ci_addendum": "fffae58",
    },
    "generator_command": "nishiki gen-paragraph-noise-floor (scope all-216; adopted as canonical in Stage F-21, default switched in F-22)",
    "scope": "all-216",
    "n_works": 216,
    "n_authors": 120,
    "n_pairs_total": 52812,
    "sd_definition": "population SD (ddof=0) of adjacent paragraph differences (p_{t+1} - p_t)",
    "sigma_kind": "adjacent_paragraph_diff_sd",
    "hybrid_rule": {
        "both_single_window": "w_before <= 1 and w_after <= 1 -> n_evaluated=8 (both_single_window:8)",
        "mixed_window": "exactly one side >= 2 -> n_evaluated=8 (mixed_window:8, v2; was 11 under v1)",
        "both_multi_window": "w_before >= 2 and w_after >= 2 -> n_evaluated=11 (both_multi_window:11, ac_sigma_source=multi_window_pairs)",
        "rule_version": "v2",
    },
    "non_conflation_guard": (
        "Harvest adjacent-paragraph noise floor sigma (sigma_kind='adjacent_paragraph_diff_sd' in kasane11_v1/ortho28) "
        "is strictly distinct from Gate 3a within-register scene variance sigma_scene_r (24D ortho24_cont). "
        "Never mix, equate, or substitute these quantities."
    ),
}


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_f14_noise_floor(path: Optional[Path] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load F-14 all-novel paragraph noise floor JSON together with verified provenance."""
    target = Path(path) if path is not None else CANONICAL_F14_NOISE_FLOOR_PATH
    data = json.loads(target.read_text(encoding="utf-8"))
    kasane11 = data.get("kasane11", {})
    source_meta = kasane11.get("source", {})
    prov = dict(F14_SIGMA_PROVENANCE)
    prov["loaded_path"] = str(target)
    prov["sha256"] = compute_sha256(target)
    prov["n_works_in_file"] = source_meta.get("n_works")
    prov["n_pairs_in_file"] = kasane11.get("n_pairs")
    if CANONICAL_F14_BOOTSTRAP_CI_PATH.exists():
        prov["bootstrap_ci_sha256"] = compute_sha256(CANONICAL_F14_BOOTSTRAP_CI_PATH)
    return data, prov


def split_sentences_kasane(text: str) -> List[str]:
    """Split text into sentences identically to ``kasane_core::IncrementalAnalyzer::split_sentences``.

    Note: Whitespace and newlines are preserved intentionally (no ``.strip()``), matching Rust's
    slice boundaries because ``KasaneParser::tokenize_raw`` counts every character inside each slice.
    """
    if not text:
        return []
    slices: List[str] = []
    start = 0
    n = len(text)
    i = 0
    while i < n:
        if text[i] in _SENTENCE_BOUNDARY_CHARS:
            end = i + 1
            while end < n and text[end] in _SENTENCE_CLOSER_CHARS:
                end += 1
            slices.append(text[start:end])
            start = end
            i = end
        else:
            i += 1
    if start < n:
        slices.append(text[start:])
    return slices


def window_count_kasane(text: str, window_chars: int = 250, stride_chars: int = 125) -> int:
    """Return the exact sliding-window count of ``kasane_core::TrajectoryAnalyzer::analyze``."""
    sent_slices = split_sentences_kasane(text)
    if not sent_slices:
        return 0
    target_window = max(int(window_chars), 25)
    target_stride = max(int(stride_chars), 10)
    char_counts = [len(s) for s in sent_slices]
    n_sents = len(sent_slices)

    windows = 0
    start_idx = 0
    while start_idx < n_sents:
        acc_chars = 0
        end_idx = start_idx
        while end_idx < n_sents and (acc_chars < target_window or end_idx == start_idx):
            acc_chars += char_counts[end_idx]
            end_idx += 1
        windows += 1
        if end_idx >= n_sents:
            break
        stepped_chars = 0
        next_start = start_idx
        while next_start < n_sents and (stepped_chars < target_stride or next_start == start_idx):
            stepped_chars += char_counts[next_start]
            next_start += 1
        start_idx = next_start
    return windows


EVAL_RULE_VERSION: str = "v2"


def is_eight_dim(windows: Tuple[int, int]) -> bool:
    """Rule v2: 11D only when both sides have >= 2 windows; [1,1] and mixed pairs are 8D."""
    return not (int(windows[0]) >= 2 and int(windows[1]) >= 2)


def is_eight_dim_v1(windows: Tuple[int, int]) -> bool:
    """Rule v1 (F-14): 8D only when both sides have <= 1 window. Comparison use only."""
    return int(windows[0]) <= 1 and int(windows[1]) <= 1


def n_eval_rule_v2(windows: Tuple[int, int]) -> str:
    """Mirror of Rust ``n_eval_rule_v2``."""
    w0, w1 = int(windows[0]), int(windows[1])
    if w0 <= 1 and w1 <= 1:
        return "both_single_window:8"
    if w0 >= 2 and w1 >= 2:
        return "both_multi_window:11"
    return "mixed_window:8"


def char_bigrams(text: str) -> List[int]:
    """Build sorted non-whitespace character 2-grams identical to Rust ``diff_harvester::bigrams``."""
    cs = [ord(c) for c in text if not c.isspace()]
    if not cs:
        return []
    if len(cs) == 1:
        return [cs[0] << 32]
    return sorted(((cs[i] << 32) | cs[i + 1]) for i in range(len(cs) - 1))


def char_bigram_dice(text_a: str, text_b: str) -> float:
    """Compute non-whitespace character 2-gram Dice coefficient identical to Rust ``diff_harvester::dice``."""
    ba = char_bigrams(text_a)
    bb = char_bigrams(text_b)
    if not ba and not bb:
        return 1.0
    if not ba or not bb:
        return 0.0
    inter = 0
    i = 0
    j = 0
    la = len(ba)
    lb = len(bb)
    while i < la and j < lb:
        va = ba[i]
        vb = bb[j]
        if va == vb:
            inter += 1
            i += 1
            j += 1
        elif va < vb:
            i += 1
        else:
            j += 1
    return (2.0 * inter) / float(la + lb)


def select_length_band(l_max: int, bands: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Select length band matching Rust ``NoiseFloor::band_for(l_max)``."""
    if not bands:
        return None
    first = bands[0]
    if l_max < int(first.get("min_chars", 0)):
        return first
    for band in bands:
        min_c = int(band.get("min_chars", 0))
        max_c = band.get("max_chars")
        if l_max >= min_c and (max_c is None or l_max <= int(max_c)):
            return band
    return None


def _is_valid_sigma(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and val > 0.0


def evaluate_f14_hybrid_pair(
    text_before: str,
    text_after: str,
    noise_floor_data: Dict[str, Any],
    delta_kasane11: Optional[Dict[str, float]] = None,
    use_band: bool = True,
    provenance: Optional[Dict[str, Any]] = None,
    apply_dice_gate: bool = True,
    align_min_similarity: float = ALIGN_MIN_SIMILARITY,
) -> Dict[str, Any]:
    """Evaluate a paragraph pair under F-14/F-15 hybrid judgment with canonical Dice gate.

    When ``apply_dice_gate`` is True (canonical ``harvest-diff`` default), pairs with
    ``char_bigram_dice(text_before, text_after) < align_min_similarity`` (default ``0.25``)
    are recorded as ``kind="unaligned"`` with ``n_evaluated=None`` and excluded from
    noise-floor evaluation.
    """
    dice_sim = char_bigram_dice(text_before, text_after)
    w_before = window_count_kasane(text_before)
    w_after = window_count_kasane(text_after)
    windows = (w_before, w_after)
    l_max = max(len(text_before), len(text_after))
    eight_dim = is_eight_dim(windows)
    mixed_window_pair = (w_before >= 2) != (w_after >= 2)

    bands = noise_floor_data.get("length_bands", {}).get("bands", [])
    band = select_length_band(l_max, bands) if use_band else None
    band_name_raw = str(band["name"]) if band is not None else "overall"

    if apply_dice_gate and dice_sim < align_min_similarity:
        return {
            "kind": "unaligned",
            "eval_rule": EVAL_RULE_VERSION,
            "eval_rule": EVAL_RULE_VERSION,
            "aligned": False,
            "dice_similarity": dice_sim,
            "align_min_similarity": align_min_similarity,
            "band": band_name_raw,
            "l_max_chars": l_max,
            "window_counts": [w_before, w_after],
            "mixed_window_pair": mixed_window_pair,
            "is_eight_dim": eight_dim,
            "n_eval_rule": None,
            "n_evaluated": None,
            "active_dims": [],
            "excluded_structurally_dead_dims": list(KASANE11_DIM_NAMES),
            "ac_sigma_source": "not_applicable",
            "sigma_used": {},
            "sigma_kind": F14_SIGMA_PROVENANCE["sigma_kind"],
            "ratios": None,
            "provenance": provenance or F14_SIGMA_PROVENANCE,
        }

    if band is not None:
        band_name = str(band["name"])
        band_kasane = band["kasane11"]
        base_sigma = dict(band_kasane["per_dim_std"])
        ac_multi_sigma = band_kasane.get("ac_multi_window_per_dim_std") or {}

        if eight_dim:
            excluded = list(SHORT_DEAD_DIMS)
            active_dims = list(KASANE8_ACTIVE_DIM_NAMES)
            n_eval_rule = n_eval_rule_v2(windows)
            ac_sigma_source = "not_applicable"
            merged_sigma = {d: base_sigma.get(d) for d in active_dims}
        else:
            excluded = []
            active_dims = list(KASANE11_DIM_NAMES)
            n_eval_rule = "both_multi_window:11"
            merged_sigma = dict(base_sigma)
            if isinstance(ac_multi_sigma, dict) and ac_multi_sigma:
                for k, v in ac_multi_sigma.items():
                    merged_sigma[k] = v
                ac_sigma_source = "multi_window_pairs"
            else:
                ac_sigma_source = "band_all_pairs"
    else:
        band_name = "overall"
        excluded = []
        active_dims = list(KASANE11_DIM_NAMES)
        n_eval_rule = n_eval_rule_v2(windows)
        ac_sigma_source = "not_applicable"
        merged_sigma = dict(noise_floor_data["kasane11"]["per_dim_std"])

    evaluated_sigma = {d: float(merged_sigma[d]) for d in active_dims if _is_valid_sigma(merged_sigma.get(d))}
    n_evaluated = len(evaluated_sigma)

    ratios_result: Optional[Dict[str, Any]] = None
    if delta_kasane11 is not None:
        per_dim_ratios: Dict[str, Optional[float]] = {}
        valid_vals: List[float] = []
        invalid_dims: List[str] = []
        max_val = float("-inf")
        argmax_dim: Optional[str] = None
        n_above_1 = 0
        for dim in active_dims:
            sigma_val = evaluated_sigma.get(dim)
            if sigma_val is None:
                per_dim_ratios[dim] = None
                invalid_dims.append(dim)
                continue
            r_val = abs(float(delta_kasane11[dim])) / sigma_val
            per_dim_ratios[dim] = r_val
            valid_vals.append(r_val)
            if r_val > 1.0:
                n_above_1 += 1
            if r_val > max_val:
                max_val = r_val
                argmax_dim = dim
        ratios_result = {
            "per_dim": per_dim_ratios,
            "max": max_val if valid_vals else None,
            "argmax": argmax_dim,
            "mean": (sum(valid_vals) / len(valid_vals)) if valid_vals else None,
            "median": statistics.median(valid_vals) if valid_vals else None,
            "n_above_1": n_above_1,
            "n_evaluated": len(valid_vals),
            "invalid_or_missing_sigma": invalid_dims,
        }

    diagnostic_7d: Optional[Dict[str, Any]] = None
    if band is not None and mixed_window_pair and delta_kasane11 is not None:
        # Diagnostic only (not canonical): dc 7 dims (ac_* and ac_mean_crest excluded), band sigma.
        dc7 = [d for d in KASANE11_DIM_NAMES if not d.startswith("ac_")]
        vals7 = [abs(float(delta_kasane11[d])) / float(base_sigma[d]) for d in dc7 if _is_valid_sigma(base_sigma.get(d))]
        diagnostic_7d = {
            "diagnostic_only": True,
            "dims": dc7,
            "n_evaluated": len(vals7),
            "max": max(vals7) if vals7 else None,
            "mean": (sum(vals7) / len(vals7)) if vals7 else None,
            "median": statistics.median(vals7) if vals7 else None,
            "n_above_1": sum(1 for v in vals7 if v > 1.0),
        }

    return {
        "kind": "pair",
        "eval_rule": EVAL_RULE_VERSION,
        "diagnostic_7d_no_crest": diagnostic_7d,
        "aligned": True,
        "dice_similarity": dice_sim,
        "align_min_similarity": align_min_similarity if apply_dice_gate else None,
        "band": band_name,
        "l_max_chars": l_max,
        "window_counts": [w_before, w_after],
        "mixed_window_pair": mixed_window_pair,
        "is_eight_dim": eight_dim,
        "n_eval_rule": n_eval_rule if band is not None else None,
        "n_evaluated": n_evaluated,
        "active_dims": active_dims,
        "excluded_structurally_dead_dims": excluded,
        "ac_sigma_source": ac_sigma_source if band is not None else "overall",
        "sigma_used": evaluated_sigma,
        "sigma_kind": F14_SIGMA_PROVENANCE["sigma_kind"],
        "ratios": ratios_result,
        "provenance": provenance or F14_SIGMA_PROVENANCE,
    }
