#!/usr/bin/env python
"""Phase 34: every number registered in scripts/paper_numbers.json must (1) appear verbatim in its paper file and
(2) equal the value recomputed from a measured result JSON, formatted with the registered format.

Registry entry:
  {"id": "...", "file": "usir_v4_paper.md", "text": "<literal as printed>",
   "source": "docs/handoff/v9/p28c_q3_8b.json", "expr": "<python expression over `d` (the loaded JSON)>",
   "fmt": "{:.2f}"}
`text` must contain fmt.format(eval(expr)).  A source JSON is rejected when it declares itself mock
(provenance.mock_used / metadata.mock_mode true, or a model name containing "Mock").  The registry may also list
"forbidden" literals that must not appear in any paper file (retired mock values).

Exit code 0 when all checks pass.  Usage: python scripts/verify_paper_numbers.py [--registry PATH]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "scripts" / "paper_numbers.json"


def is_mock(d: Any) -> bool:
    if not isinstance(d, dict):
        return False
    prov = d.get("provenance") or {}
    meta = d.get("metadata") or {}
    if prov.get("mock_used") is True or meta.get("mock_mode") is True:
        return True
    for k in ("model", "model_id", "generator"):
        for src in (d, meta):
            v = src.get(k) if isinstance(src, dict) else None
            if isinstance(v, str) and "mock" in v.lower():
                return True
    return False


def check(registry: Dict[str, Any], root: Path = ROOT) -> List[str]:
    errors: List[str] = []
    cache: Dict[str, Any] = {}
    texts: Dict[str, str] = {}
    env = {"math": math, "statistics": statistics, "sum": sum, "len": len, "min": min, "max": max, "abs": abs,
           "round": round, "sorted": sorted}
    for e in registry.get("numbers", []):
        f, src = e["file"], e["source"]
        if f not in texts:
            p = root / f
            if not p.exists():
                errors.append(f"{e['id']}: paper file missing {f}")
                continue
            texts[f] = p.read_text(encoding="utf-8")
        if src not in cache:
            sp = root / src
            if not sp.exists():
                errors.append(f"{e['id']}: source missing {src}")
                continue
            cache[src] = json.loads(sp.read_text(encoding="utf-8"))
        d = cache[src]
        if is_mock(d):
            errors.append(f"{e['id']}: source {src} is mock")
            continue
        try:
            val = eval(e["expr"], {**env, "d": d})  # noqa: S307 - registry is repository-controlled
        except Exception as exc:
            errors.append(f"{e['id']}: expr failed ({exc!r})")
            continue
        s = e.get("fmt", "{}").format(val)
        if s not in e["text"]:
            errors.append(f"{e['id']}: JSON gives {s!r}, registered text is {e['text']!r}")
        if e["text"] not in texts[f]:
            errors.append(f"{e['id']}: text {e['text']!r} not found in {f}")
    for lit in registry.get("forbidden", []):
        for f in registry.get("paper_files", []):
            p = root / f
            if p.exists() and lit in p.read_text(encoding="utf-8"):
                errors.append(f"forbidden literal {lit!r} found in {f}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=str(REGISTRY))
    reg = json.loads(Path(ap.parse_args().registry).read_text(encoding="utf-8"))
    errs = check(reg)
    n = len(reg.get("numbers", []))
    if errs:
        print(f"verify_paper_numbers: {len(errs)} error(s) over {n} registered numbers")
        for x in errs:
            print("  -", x)
        return 1
    print(f"verify_paper_numbers: OK ({n} numbers, {len(reg.get('forbidden', []))} forbidden literals)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
