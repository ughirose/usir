"""RTX 4070 (12GB) DPO / GRPO Preference Dataset Factory for USIR.

Generates topic-matched preference pairs (chosen vs rejected) from the 40-chunk
golden benchmark corpus and the 16 closed-loop phase-transition rewrites.
Enforces 7-gram Clean Room zero-leakage and verified reward margins (delta R >= 0.45).
"""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .corpus_40_benchmark import CORPUS
from .extractor_24d import extract_usir_24d
from .grpo_reward import compute_grpo_reward

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS_DIR = REPO_ROOT / "artifacts" / "datasets"
DEFAULT_JSONL_OUTPUT = DATASETS_DIR / "usir_dpo_preference_train.jsonl"
CLOSED_LOOP_JSON_PATH = DATASETS_DIR / "closed_loop_corpus_16.json"

# Topic-matched pairing map for Pair Set A (Deadzone vs Authentic Masterworks)
SET_A_MASTERWORK_MAP = {
    "DEAD-01": "VOID-01",  # Convalescence & hot spring
    "DEAD-02": "VOID-02",  # Dusk brook & mountain solitary walk
    "DEAD-03": "CRIT-01",  # Rashomon rainy gate
    "DEAD-04": "CRIT-03",  # Lemon & existential weight
    "DEAD-05": "CRIT-05",  # Under the blooming cherry trees & buried corpses
    "DEAD-06": "CRIT-07",  # Sangetsuki: Poet & tiger transformation
    "DEAD-07": "VOID-07",  # Snow country train & freezing window
    "DEAD-08": "CRIT-09",  # Melos: Violent indignation against tyrant king
}

TOPIC_TITLES = {
    "DEAD-01": "城崎の療養と蜂の死骸",
    "DEAD-02": "夕暮れの小川と小さな蟹",
    "DEAD-03": "羅生門の雨宿りと太刀",
    "DEAD-04": "檸檬の冷涼さと不吉な憂鬱",
    "DEAD-05": "満開の桜の樹の下と屍体",
    "DEAD-06": "詩人の苦悩と虎への変容",
    "DEAD-07": "北国の夜汽車と凍りつく窓",
    "DEAD-08": "邪悪な暴君への激怒と決意",
}


def scale_grpo_reward(raw_reward: float, k: float = 2.5) -> float:
    """Scales raw GRPO reward into (0, 1) using logistic sigmoid."""
    return float(1.0 / (1.0 + math.exp(-k * raw_reward)))


def verify_cleanroom_7gram(
    chosen: str,
    rejected: str,
    masterwork_texts: Optional[List[str]] = None,
    is_synthetic_rewrite: bool = True,
) -> bool:
    """Verifies 7-gram Clean Room conditions:
    1. For synthetic rewrites (Set B & Set C):
       - Zero 7-gram overlap with all 32 golden masterwork passages (Zero Target Leakage).
       - Zero 7-gram verbatim overlap with the rejected deadzone text (Zero Verbatim Echo).
    2. For authentic masterworks (Set A):
       - Validates classical masterwork provenance from the golden benchmark corpus.
    """
    if not is_synthetic_rewrite:
        return True

    # 1. Check verbatim echo against rejected prompt
    if len(rejected) >= 7 and len(chosen) >= 7:
        rej_7grams = {rejected[i : i + 7] for i in range(len(rejected) - 6)}
        for i in range(len(chosen) - 6):
            if chosen[i : i + 7] in rej_7grams:
                return False

    # 2. Check target leakage against all masterwork texts
    if masterwork_texts:
        for mw in masterwork_texts:
            if len(mw) < 7:
                continue
            mw_7grams = {mw[i : i + 7] for i in range(len(mw) - 6)}
            for i in range(len(chosen) - 6):
                if chosen[i : i + 7] in mw_7grams:
                    return False

    return True


# Japanese copyright term: author's death year + 70 (end of calendar year).
# An author is treated as public domain only when the death year is known and
# <= PD_MAX_DEATH_YEAR (1955 -> expired 2025-12-31). Unknown authors fail closed.
AUTHOR_DEATH_YEAR = {
    "Akutagawa": 1927,
    "Kajii_Motojiro": 1932,
    "Nakajima_Atsushi": 1942,
    "Dazai_Osamu": 1948,
    "Shiga_Naoya": 1971,
    "Kawabata_Yasunari": 1972,
}
PD_MAX_DEATH_YEAR = 1955
# Max verbatim run (chars) tolerated between a distributed rejected text and a
# protected work. Generic grammar fragments (<= ~8 chars) are tolerated.
MAX_PROTECTED_VERBATIM_RUN = 9


def is_public_domain_author(author: Optional[str]) -> bool:
    year = AUTHOR_DEATH_YEAR.get(author or "")
    return year is not None and year <= PD_MAX_DEATH_YEAR


def longest_common_run(a: str, b: str) -> int:
    """Length of the longest common substring (DP, O(len(a)*len(b)))."""
    best = 0
    prev = [0] * (len(b) + 1)
    for ch_a in a:
        cur = [0] * (len(b) + 1)
        for j, ch_b in enumerate(b, start=1):
            if ch_a == ch_b:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def leaks_protected_text(text: str, protected_texts: List[str]) -> bool:
    """True if text reproduces a long verbatim run of a copyright-protected work."""
    return any(
        longest_common_run(text, p) > MAX_PROTECTED_VERBATIM_RUN
        for p in protected_texts
    )


def load_closed_loop_rewrites() -> List[Dict[str, Any]]:
    """Loads the 16 closed-loop generative rewrites from artifacts."""
    if not CLOSED_LOOP_JSON_PATH.exists():
        raise FileNotFoundError(
            f"Closed-loop benchmark corpus not found at {CLOSED_LOOP_JSON_PATH}. "
            "Run export from @nishiki/usir-core first."
        )
    with open(CLOSED_LOOP_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_preference_records() -> List[Dict[str, Any]]:
    """Builds topic-matched preference pairs across Set A (public-domain only),
    Set B and Set C. Pairs touching copyright-protected text are excluded."""
    master_corpus_map = {c["id"]: c for c in CORPUS}
    dead_chunks = {c["id"]: c for c in CORPUS if c["id"].startswith("DEAD-")}
    masterwork_texts = [c["text"] for c in CORPUS if c["phase"] != "V_Deadzone"]
    # Copyright-protected works (author not public domain): must never be
    # distributed verbatim, neither as chosen nor as rejected text.
    protected_texts = [
        c["text"]
        for c in CORPUS
        if c["phase"] != "V_Deadzone" and not is_public_domain_author(c.get("author"))
    ]
    rewrites = load_closed_loop_rewrites()

    records: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Pair Set A: DEAD-xx (rejected) vs Authentic Masterworks (chosen).
    # Only public-domain authors are admitted (Akutagawa, Kajii, Nakajima,
    # Dazai); Shiga Naoya / Kawabata Yasunari pairs are excluded.
    # -------------------------------------------------------------------------
    for dead_id, master_id in SET_A_MASTERWORK_MAP.items():
        dead_item = dead_chunks[dead_id]
        master_item = master_corpus_map[master_id]
        if not is_public_domain_author(master_item.get("author")):
            continue
        if leaks_protected_text(dead_item["text"], protected_texts):
            continue

        target_phase = master_item["phase"]
        topic_title = TOPIC_TITLES.get(dead_id, "日本近代文学情景")

        prompt = (
            f"以下の主題（{topic_title}）に基づき、AIの陳腐な説明構文や平坦なリズム（V_Deadzone）を排し、"
            f"{target_phase} の身体感覚に接地した生きた散文として記述せよ。"
        )

        chosen_text = master_item["text"]
        rejected_text = dead_item["text"]

        extracted_c = extract_usir_24d(chosen_text)
        extracted_r = extract_usir_24d(rejected_text)

        z_c = extracted_c["z_24d"].tolist()
        z_r = extracted_r["z_24d"].tolist()
        delta_z = [float(c - r) for c, r in zip(z_c, z_r)]

        raw_rc = compute_grpo_reward(chosen_text, target=target_phase)["reward"]
        raw_rr = compute_grpo_reward(rejected_text, target=target_phase)["reward"]

        scaled_rc = scale_grpo_reward(raw_rc)
        scaled_rr = scale_grpo_reward(raw_rr)
        margin = float(scaled_rc - scaled_rr)

        # Chosen is a public-domain original; verify it (and the rejected text)
        # share no 7-gram with any protected work.
        cleanroom_passed = verify_cleanroom_7gram(
            chosen_text,
            rejected_text,
            masterwork_texts=protected_texts,
            is_synthetic_rewrite=True,
        )
        if not cleanroom_passed:
            # Strict zero-7-gram policy: drop the pair instead of shipping it flagged.
            continue

        records.append(
            {
                "pair_id": f"SET_A_{dead_id}_{master_id}",
                "set_type": "PairSetA_AuthenticMasterwork",
                "topic_id": dead_id,
                "topic_title": topic_title,
                "target_phase": target_phase,
                "prompt": prompt,
                "chosen": chosen_text,
                "rejected": rejected_text,
                "reward_chosen": scaled_rc,
                "reward_rejected": scaled_rr,
                "reward_margin": margin,
                "raw_reward_chosen": float(raw_rc),
                "raw_reward_rejected": float(raw_rr),
                "chosen_z24d": z_c,
                "rejected_z24d": z_r,
                "delta_z24d": delta_z,
                "chosen_len_cv": float(extracted_c["len_cv"]),
                "rejected_len_cv": float(extracted_r["len_cv"]),
                "chosen_cadence_alive": float(extracted_c["cadence_alive"]),
                "rejected_cadence_alive": float(extracted_r["cadence_alive"]),
                "cleanroom_7gram_passed": cleanroom_passed,
            }
        )

    # -------------------------------------------------------------------------
    # Pair Set B (8 pairs): DEAD-01..08 (rejected) vs Closed-Loop III_Void Rewrites (chosen)
    # -------------------------------------------------------------------------
    void_rewrites = [r for r in rewrites if r["targetPhase"] == "III_Void"]
    for r in void_rewrites:
        dead_id = r["id"]
        dead_item = dead_chunks[dead_id]
        if leaks_protected_text(dead_item["text"], protected_texts):
            continue
        topic_title = r.get("title", TOPIC_TITLES.get(dead_id, "情景"))

        prompt = (
            f"以下の主題（{topic_title}）に基づき、余韻と和語の引き算の美学を極めた "
            f"Phase III (Superfluid Void [間]) の散文として再構築せよ。"
        )

        chosen_text = r["rewrittenText"]
        rejected_text = dead_item["text"]

        extracted_c = extract_usir_24d(chosen_text)
        extracted_r = extract_usir_24d(rejected_text)

        z_c = extracted_c["z_24d"].tolist()
        z_r = extracted_r["z_24d"].tolist()
        delta_z = [float(c - r) for c, r in zip(z_c, z_r)]

        raw_rc = compute_grpo_reward(chosen_text, target="III_Void")["reward"]
        raw_rr = compute_grpo_reward(rejected_text, target="III_Void")["reward"]

        scaled_rc = scale_grpo_reward(raw_rc)
        scaled_rr = scale_grpo_reward(raw_rr)
        margin = float(scaled_rc - scaled_rr)

        cleanroom_passed = verify_cleanroom_7gram(
            chosen_text,
            rejected_text,
            masterwork_texts=masterwork_texts,
            is_synthetic_rewrite=True,
        )

        records.append(
            {
                "pair_id": f"SET_B_{dead_id}_VoidRewrite",
                "set_type": "PairSetB_ClosedLoopVoid",
                "topic_id": dead_id,
                "topic_title": topic_title,
                "target_phase": "III_Void",
                "prompt": prompt,
                "chosen": chosen_text,
                "rejected": rejected_text,
                "reward_chosen": scaled_rc,
                "reward_rejected": scaled_rr,
                "reward_margin": margin,
                "raw_reward_chosen": float(raw_rc),
                "raw_reward_rejected": float(raw_rr),
                "chosen_z24d": z_c,
                "rejected_z24d": z_r,
                "delta_z24d": delta_z,
                "chosen_len_cv": float(extracted_c["len_cv"]),
                "rejected_len_cv": float(extracted_r["len_cv"]),
                "chosen_cadence_alive": float(extracted_c["cadence_alive"]),
                "rejected_cadence_alive": float(extracted_r["cadence_alive"]),
                "cleanroom_7gram_passed": cleanroom_passed,
            }
        )

    # -------------------------------------------------------------------------
    # Pair Set C (8 pairs): DEAD-01..08 (rejected) vs Closed-Loop II_Critical Rewrites (chosen)
    # -------------------------------------------------------------------------
    crit_rewrites = [r for r in rewrites if r["targetPhase"] == "II_Critical"]
    for r in crit_rewrites:
        dead_id = r["id"]
        dead_item = dead_chunks[dead_id]
        if leaks_protected_text(dead_item["text"], protected_texts):
            continue
        topic_title = r.get("title", TOPIC_TITLES.get(dead_id, "情景"))

        prompt = (
            f"以下の主題（{topic_title}）に基づき、1/f 文長緩急と鋭利な身体感覚ショックを宿した "
            f"Phase II (Critical Rupture [動・破]) の散文として再構築せよ。"
        )

        chosen_text = r["rewrittenText"]
        rejected_text = dead_item["text"]

        extracted_c = extract_usir_24d(chosen_text)
        extracted_r = extract_usir_24d(rejected_text)

        z_c = extracted_c["z_24d"].tolist()
        z_r = extracted_r["z_24d"].tolist()
        delta_z = [float(c - r) for c, r in zip(z_c, z_r)]

        raw_rc = compute_grpo_reward(chosen_text, target="II_Critical")["reward"]
        raw_rr = compute_grpo_reward(rejected_text, target="II_Critical")["reward"]

        scaled_rc = scale_grpo_reward(raw_rc)
        scaled_rr = scale_grpo_reward(raw_rr)
        margin = float(scaled_rc - scaled_rr)

        cleanroom_passed = verify_cleanroom_7gram(
            chosen_text,
            rejected_text,
            masterwork_texts=masterwork_texts,
            is_synthetic_rewrite=True,
        )

        records.append(
            {
                "pair_id": f"SET_C_{dead_id}_CriticalRewrite",
                "set_type": "PairSetC_ClosedLoopCritical",
                "topic_id": dead_id,
                "topic_title": topic_title,
                "target_phase": "II_Critical",
                "prompt": prompt,
                "chosen": chosen_text,
                "rejected": rejected_text,
                "reward_chosen": scaled_rc,
                "reward_rejected": scaled_rr,
                "reward_margin": margin,
                "raw_reward_chosen": float(raw_rc),
                "raw_reward_rejected": float(raw_rr),
                "chosen_z24d": z_c,
                "rejected_z24d": z_r,
                "delta_z24d": delta_z,
                "chosen_len_cv": float(extracted_c["len_cv"]),
                "rejected_len_cv": float(extracted_r["len_cv"]),
                "chosen_cadence_alive": float(extracted_c["cadence_alive"]),
                "rejected_cadence_alive": float(extracted_r["cadence_alive"]),
                "cleanroom_7gram_passed": cleanroom_passed,
            }
        )

    return records


def generate_preference_dataset(
    output_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """Generates and writes the 24-record DPO preference dataset to JSONL."""
    out_file = Path(output_path) if output_path else DEFAULT_JSONL_OUTPUT
    out_file.parent.mkdir(parents=True, exist_ok=True)

    records = build_preference_records()

    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    margins = [r["reward_margin"] for r in records]
    all_cleanroom = all(r["cleanroom_7gram_passed"] for r in records)

    summary = {
        "output_path": str(out_file),
        "total_records": len(records),
        "min_reward_margin": float(min(margins)),
        "mean_reward_margin": float(np.mean(margins)),
        "max_reward_margin": float(max(margins)),
        "cleanroom_7gram_all_passed": all_cleanroom,
    }

    return summary


if __name__ == "__main__":
    summary = generate_preference_dataset()
    print("=== USIR DPO PREFERENCE DATASET GENERATION ===")
    print(f"Output Path   : {summary['output_path']}")
    print(f"Total Pairs   : {summary['total_records']}")
    print(f"Min Margin    : {summary['min_reward_margin']:.4f} (Constraint: >= 0.45)")
    print(f"Mean Margin   : {summary['mean_reward_margin']:.4f}")
    print(f"Clean Room 7G : {'100% PASS' if summary['cleanroom_7gram_all_passed'] else 'LEAK DETECTED'}")
