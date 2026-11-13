#!/usr/bin/env python3
"""Stage B-17 Task 5: 凍結成果物の SHA-256 不変検証スクリプト."""
import hashlib
import json
import sys
from pathlib import Path

BASELINE_PATH = Path(__file__).resolve().parent / "frozen_sha256_baseline.json"


def main() -> None:
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    results = []
    all_ok = True
    for fpath, expected in baseline.items():
        p = Path(fpath)
        actual = hashlib.sha256(p.read_bytes()).hexdigest().upper()
        ok = actual == expected
        if not ok:
            all_ok = False
        results.append({"file": fpath, "expected": expected, "actual": actual, "unchanged": ok})
        print(f"[{'OK' if ok else 'NG'}] {p.name}: {actual}")
    assert all_ok, "One or more frozen artifacts changed!"
    out_path = Path(__file__).resolve().parent / "b17_task5_sha256_verify.json"
    out_path.write_text(json.dumps({"all_unchanged": all_ok, "files": results}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"All {len(results)} frozen artifacts verified unchanged.")


if __name__ == "__main__":
    main()
