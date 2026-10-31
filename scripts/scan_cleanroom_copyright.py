#!/usr/bin/env python3
"""Comprehensive Clean-Room & Japanese Copyright Act Art. 30-4 Audit Scanner for USIR Datasets.

Scans all rows of a DPO/preference dataset (default: artifacts/datasets/usir_dpo_preference_train.jsonl)
against:
  1. 7-gram leakage vs protected benchmark corpus (Zero Target Leakage)
  2. 7-gram verbatim echo between chosen and rejected (Zero Echo)
  3. Banned target keywords (threeRoomGate.ts / .antigravity/rules.md)
  4. Micro-constraint gate rules (microConstraintGate.ts: Tier A clichés, meta Goodhart terms, loanwords)
  5. Japanese Copyright Act (著作権法) Art. 30-4 considerations:
     - Distinguishes authors with active copyright under Japan's 70-year post-mortem rule
       (e.g., Shiga Naoya d. 1971, Kawabata Yasunari d. 1972 protected through 2041/2042)
       from public domain authors (Akutagawa, Kajii, Nakajima, Dazai, etc. d. <= 1967).
     - Assesses non-enjoyment computational training exception (Art. 30-4) vs public
       open-core distribution risk (Art. 21 reproduction / Art. 23 public transmission).

Usage:
    python scripts/scan_cleanroom_copyright.py [--dataset PATH] [--json]
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "artifacts" / "datasets" / "usir_dpo_preference_train.jsonl"

BANNED_TARGET_KEYWORDS = [
    "メロス", "李徴", "島村", "下人", "芥川", "太宰", "中島", "梶井",
    "志賀", "川端", "芭蕉", "漱石", "羅生門", "山月記", "雪国", "檸檬"
]

HARD_AI_CLICHE_FRAMES = [
    "テーマ", "ように思われ", "ように感じられ", "させられたのだった",
    "様々な思いが", "特別な瞬間となった", "改めて深く", "強い後悔と悲しみ",
    "心の均衡を失った", "言いようのない不安を感じていた", "のではないだろうか",
    "と言えるだろう", "のだった。", "思われた。", "感じられた。",
    "傾向が見受けられ", "重要な要素", "多様な側面"
]

META_GOODHART_TERMS = [
    "シンプレクティック", "ハミルトニアン", "マハラノビス",
    "秩序変数", "臨界相", "ヴォイド相", "デッドゾーン"
]

MODERN_LOANWORDS = [
    "猛スピード", "スピード", "システム", "レベル", "ストレス", "タイミング",
    "イメージ", "メリット", "デメリット", "プロセス", "エネルギー", "パーセント",
    "アプローチ", "コミット", "モチベーション", "プロジェクト", "ターゲット",
    "シチュエーション", "プライド", "リアル", "クオリティ", "テンション",
    "リセット", "スペック", "ルーティン", "センス", "モチベ", "コスパ",
    "アピール", "リアリティ"
]

# Japanese Copyright Protection Metadata (as of 2026)
# Under the 2018 TPP11 implementation act, Japan extended copyright from 50 to 70 years post-mortem
# for works whose copyright had not yet expired on December 30, 2018 (i.e. authors who died in 1968 or later).
AUTHOR_COPYRIGHT_STATUS = {
    "Shiga_Naoya": {
        "author": "志賀直哉",
        "death_year": 1971,
        "status": "PROTECTED_COPYRIGHT",
        "expires": "2041-12-31",
        "legal_note": "Died 1971. Copyright was active on Dec 30, 2018; 70-year term applies."
    },
    "Kawabata_Yasunari": {
        "author": "川端康成",
        "death_year": 1972,
        "status": "PROTECTED_COPYRIGHT",
        "expires": "2042-12-31",
        "legal_note": "Died 1972. Copyright was active on Dec 30, 2018; 70-year term applies."
    },
    "Akutagawa": {
        "author": "芥川龍之介",
        "death_year": 1927,
        "status": "PUBLIC_DOMAIN",
        "expires": "1977-12-31",
        "legal_note": "50-year term expired prior to 2018 amendment."
    },
    "Kajii_Motojiro": {
        "author": "梶井基次郎",
        "death_year": 1932,
        "status": "PUBLIC_DOMAIN",
        "expires": "1982-12-31",
        "legal_note": "50-year term expired prior to 2018 amendment."
    },
    "Kajii_Minimal": {
        "author": "梶井基次郎",
        "death_year": 1932,
        "status": "PUBLIC_DOMAIN",
        "expires": "1982-12-31",
        "legal_note": "50-year term expired prior to 2018 amendment."
    },
    "Nakajima_Atsushi": {
        "author": "中島敦",
        "death_year": 1942,
        "status": "PUBLIC_DOMAIN",
        "expires": "1992-12-31",
        "legal_note": "50-year term expired prior to 2018 amendment."
    },
    "Dazai_Osamu": {
        "author": "太宰治",
        "death_year": 1948,
        "status": "PUBLIC_DOMAIN",
        "expires": "1998-12-31",
        "legal_note": "50-year term expired prior to 2018 amendment."
    },
    "Sakaguchi_Ango": {
        "author": "坂口安吾",
        "death_year": 1955,
        "status": "PUBLIC_DOMAIN",
        "expires": "2005-12-31",
        "legal_note": "50-year term expired prior to 2018 amendment."
    },
    "Basho": {
        "author": "松尾芭蕉",
        "death_year": 1694,
        "status": "PUBLIC_DOMAIN",
        "expires": "Historic",
        "legal_note": "Public domain."
    },
    "Soseki": {
        "author": "夏目漱石",
        "death_year": 1916,
        "status": "PUBLIC_DOMAIN",
        "expires": "1966-12-31",
        "legal_note": "50-year term expired prior to 2018 amendment."
    },
    "LLM_Slop": {
        "author": "Synthetic AI",
        "death_year": None,
        "status": "SYNTHETIC_NO_COPYRIGHT",
        "expires": "N/A",
        "legal_note": "Pure synthetic AI slop; no human copyright."
    },
    "Statute": {
        "author": "Public Law",
        "death_year": None,
        "status": "PUBLIC_DOMAIN",
        "expires": "N/A",
        "legal_note": "Japanese Copyright Act Art. 13 (laws/statutes excluded from copyright)."
    },
}


def load_benchmark_corpus():
    """Loads benchmark masterwork corpus from python/usir_py/corpus_40_benchmark.py."""
    usir_py_dir = str(REPO_ROOT / "python" / "usir_py")
    if usir_py_dir not in sys.path:
        sys.path.insert(0, usir_py_dir)
    try:
        import corpus_40_benchmark
        return corpus_40_benchmark.CORPUS
    except Exception as e:
        print(f"Warning: Failed to load benchmark corpus directly: {e}", file=sys.stderr)
        return []


def scan_row(record, idx, corpus_map, masterwork_texts):
    pair_id = record.get("pair_id", f"ROW_{idx}")
    set_type = record.get("set_type", "")
    prompt = record.get("prompt", "")
    chosen = record.get("chosen", "")
    rejected = record.get("rejected", "")
    target_phase = record.get("target_phase", "")

    # 1. 7-gram extraction
    chosen_7grams = {chosen[i:i+7] for i in range(len(chosen) - 6)} if len(chosen) >= 7 else set()
    rej_7grams = {rejected[i:i+7] for i in range(len(rejected) - 6)} if len(rejected) >= 7 else set()

    # 2. Verbatim echo between chosen and rejected
    echo_7grams = sorted(list(chosen_7grams.intersection(rej_7grams)))

    # 3. Masterwork leakage (cross-matching with corpus masterworks)
    masterwork_leakage = defaultdict(list)
    for mw_id, mw_text in masterwork_texts.items():
        if set_type == "PairSetA_AuthenticMasterwork" and pair_id.endswith(mw_id):
            # In Set A, chosen IS the designated masterwork passage
            continue
        if len(mw_text) >= 7:
            for i in range(len(mw_text) - 6):
                gram = mw_text[i:i+7]
                if gram in chosen_7grams:
                    masterwork_leakage[mw_id].append(gram)

    # 4. Banned target keywords (threeRoomGate.ts)
    banned_in_chosen = [w for w in BANNED_TARGET_KEYWORDS if w in chosen]
    banned_in_rejected = [w for w in BANNED_TARGET_KEYWORDS if w in rejected]
    banned_in_prompt = [w for w in BANNED_TARGET_KEYWORDS if w in prompt]

    # 5. Micro-constraint gate checks
    cliche_in_chosen = [f for f in HARD_AI_CLICHE_FRAMES if f in chosen]
    meta_in_chosen = [m for m in META_GOODHART_TERMS if m in chosen]
    loanwords_in_chosen = [lw for lw in MODERN_LOANWORDS if lw in chosen]

    # 6. Japanese Copyright Act Art. 30-4 audit
    copyright_alert = None
    if set_type == "PairSetA_AuthenticMasterwork":
        master_id = pair_id.split("_")[-1]
        mw_entry = corpus_map.get(master_id)
        if mw_entry:
            author_key = mw_entry.get("author")
            c_info = AUTHOR_COPYRIGHT_STATUS.get(author_key, {})
            c_status = c_info.get("status", "UNKNOWN")
            copyright_alert = {
                "master_id": master_id,
                "author": c_info.get("author", author_key),
                "death_year": c_info.get("death_year"),
                "status": c_status,
                "expires": c_info.get("expires"),
                "title": mw_entry.get("title"),
                "is_active_copyright": c_status == "PROTECTED_COPYRIGHT"
            }

    # Determine overall status for this row
    # FAIL criteria: active copyrighted passage distributed in dataset, or verbatim echo between chosen/rejected
    # WARN criteria: public domain masterwork with banned keywords or cross-masterwork overlap
    # PASS: clean-room compliant
    status = "PASS"
    flags = []

    if copyright_alert and copyright_alert["is_active_copyright"]:
        status = "FAIL"
        flags.append(
            f"Active Copyright Infringement Risk (Art. 21/23 JCA): Verbatim reproduction of protected author "
            f"[{copyright_alert['author']}: {copyright_alert['title']}], copyright valid until {copyright_alert['expires']}."
        )

    if echo_7grams:
        status = "FAIL"
        flags.append(f"Verbatim Echo: {len(echo_7grams)} 7-gram overlap(s) between chosen and rejected: {echo_7grams[:3]}")

    if set_type != "PairSetA_AuthenticMasterwork" and masterwork_leakage:
        status = "FAIL"
        flags.append(f"Synthetic Cleanroom Leakage: {len(masterwork_leakage)} masterwork overlap(s): {dict(masterwork_leakage)}")

    if banned_in_chosen:
        if set_type == "PairSetA_AuthenticMasterwork":
            if status != "FAIL":
                status = "WARN"
            flags.append(f"Banned target keywords in authentic chosen text: {banned_in_chosen}")
        else:
            status = "FAIL"
            flags.append(f"Banned target keywords in synthetic chosen text: {banned_in_chosen}")

    if set_type == "PairSetA_AuthenticMasterwork" and masterwork_leakage:
        if status != "FAIL":
            status = "WARN"
        flags.append(f"Cross-masterwork 7-gram clause junction overlap: {list(masterwork_leakage.keys())}")

    if cliche_in_chosen:
        status = "FAIL"
        flags.append(f"Tier A AI Cliché in chosen: {cliche_in_chosen}")

    if meta_in_chosen:
        status = "FAIL"
        flags.append(f"Meta Goodhart term in chosen: {meta_in_chosen}")

    if loanwords_in_chosen:
        status = "FAIL"
        flags.append(f"Modern loanword in chosen: {loanwords_in_chosen}")

    return {
        "row_index": idx,
        "pair_id": pair_id,
        "set_type": set_type,
        "target_phase": target_phase,
        "status": status,
        "flags": flags,
        "echo_7grams_count": len(echo_7grams),
        "banned_in_chosen": banned_in_chosen,
        "copyright_alert": copyright_alert
    }


def main():
    parser = argparse.ArgumentParser(description="USIR Dataset Clean-Room & Copyright Scanner")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to jsonl dataset")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Error: Dataset not found at {args.dataset}", file=sys.stderr)
        sys.exit(1)

    corpus = load_benchmark_corpus()
    corpus_map = {c["id"]: c for c in corpus}
    masterwork_texts = {c["id"]: c["text"] for c in corpus if c.get("phase") != "V_Deadzone"}

    records = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    results = []
    for idx, rec in enumerate(records):
        res = scan_row(rec, idx, corpus_map, masterwork_texts)
        results.append(res)

    pass_count = sum(1 for r in results if r["status"] == "PASS")
    warn_count = sum(1 for r in results if r["status"] == "WARN")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")

    if args.json:
        summary_payload = {
            "dataset": str(args.dataset),
            "total_records": len(results),
            "counts": {"PASS": pass_count, "WARN": warn_count, "FAIL": fail_count},
            "records": results
        }
        print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
        sys.exit(0 if fail_count == 0 else 1)

    print("=" * 80)
    print("USIR DATASET CLEAN-ROOM & COPYRIGHT ACT ART. 30-4 AUDIT REPORT")
    print(f"Target File: {args.dataset}")
    print(f"Total Rows : {len(results)}")
    print(f"Reference Corpus: 40 golden benchmark passages (32 masterwork / spec chunks)")
    print("=" * 80)

    for r in results:
        status_tag = f"[{r['status']}]"
        print(f"Row {r['row_index']:02d} | {status_tag:<6} | {r['pair_id']:<32} | {r['set_type']}")
        for f in r["flags"]:
            print(f"       -> {f}")

    print("\n" + "=" * 80)
    print("SUMMARY BY SUBSET:")
    set_a = [r for r in results if r["set_type"] == "PairSetA_AuthenticMasterwork"]
    set_b = [r for r in results if r["set_type"] == "PairSetB_ClosedLoopVoid"]
    set_c = [r for r in results if r["set_type"] == "PairSetC_ClosedLoopCritical"]

    print(f"Set A (Authentic Masterworks) : PASS={sum(1 for r in set_a if r['status']=='PASS')}, "
          f"WARN={sum(1 for r in set_a if r['status']=='WARN')}, FAIL={sum(1 for r in set_a if r['status']=='FAIL')} (Total: {len(set_a)})")
    print(f"Set B (Closed-Loop Void)      : PASS={sum(1 for r in set_b if r['status']=='PASS')}, "
          f"WARN={sum(1 for r in set_b if r['status']=='WARN')}, FAIL={sum(1 for r in set_b if r['status']=='FAIL')} (Total: {len(set_b)})")
    print(f"Set C (Closed-Loop Critical)  : PASS={sum(1 for r in set_c if r['status']=='PASS')}, "
          f"WARN={sum(1 for r in set_c if r['status']=='WARN')}, FAIL={sum(1 for r in set_c if r['status']=='FAIL')} (Total: {len(set_c)})")
    print("-" * 80)
    print(f"OVERALL: PASS={pass_count}, WARN={warn_count}, FAIL={fail_count} / {len(results)}")
    print("=" * 80)

    if fail_count > 0:
        sys.exit(1)
    elif warn_count > 0:
        sys.exit(0)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
