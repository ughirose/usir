#!/usr/bin/env python3
"""Stage B-22 Task 5: Verify all 20 frozen artifacts match their pre-execution SHA-256 digests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

BASELINE_15_PATH = Path(r"C:\dev\usir\scripts\b17\frozen_sha256_baseline.json")
OUT_PATH = Path(__file__).resolve().parent / "b22_task5_sha256_verify.json"

EXTRA_5_FROZEN: dict[str, str] = {
    r"C:\dev\usir\docs\handoff\v10\paragraph_noise_floor_all_216_canonical.json": (
        "2FC18E6717E4537ACBB522BC1D10F3C434B3C9BB434B19AF3803293EE7C84974"
    ),
    r"C:\dev\nishiki\docs\handoff\evidence\2026-11-14_f21\paragraph_noise_floor_all_216_canonical.json": (
        "2FC18E6717E4537ACBB522BC1D10F3C434B3C9BB434B19AF3803293EE7C84974"
    ),
    r"C:\dev\usir\docs\handoff\v10\paragraph_noise_floor_all_216.json": (
        "07E43FDF11E3C91D641A08FF509AB9DDFE153C9F0DBE2030B3C81DFFF199888D"
    ),
    r"C:\dev\nishiki\docs\handoff\evidence\2026-11-14_b17\paragraph_noise_floor_all_216.json": (
        "07E43FDF11E3C91D641A08FF509AB9DDFE153C9F0DBE2030B3C81DFFF199888D"
    ),
    r"C:\dev\usir\data\v5\aozora_106works_scenes.jsonl": (
        "07ED3F726D5BB948E9B77BB1E35F1AEC99B7897EE8DCAEFF24C7540302DE0828"
    ),
}


def sha256_upper(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    baseline_15: dict[str, str] = json.loads(BASELINE_15_PATH.read_text(encoding="utf-8"))
    all_expected: dict[str, str] = {}
    all_expected.update(baseline_15)
    all_expected.update(EXTRA_5_FROZEN)

    results = []
    all_matched = True
    for raw_path, expected in all_expected.items():
        p = Path(raw_path)
        if not p.exists():
            actual = "MISSING"
            matched = False
        else:
            actual = sha256_upper(p)
            matched = actual == expected.upper()
        if not matched:
            all_matched = False
        results.append(
            {
                "file": raw_path,
                "expected": expected.upper(),
                "actual": actual,
                "matched": matched,
                "size_bytes": p.stat().st_size if p.exists() else None,
            }
        )
        status = "OK" if matched else "MISMATCH"
        print(f"[{status}] {raw_path} -> {actual}")

    payload = {
        "n_files": len(results),
        "all_matched": all_matched,
        "files": results,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nVerified {len(results)} frozen artifacts: all_matched={all_matched}")
    if not all_matched:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
