"""Run `nishiki diagnose` on a work with merged annotations and report per-lens status."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def run_diagnose(nishiki: str, text: Path, characters: Path, annotations: Path) -> dict[str, Any]:
    r = subprocess.run(
        [nishiki, "diagnose", str(text), "--format", "json", "--characters", str(characters), "--annotations", str(annotations)],
        capture_output=True, check=True,
    )
    j = json.loads(r.stdout.decode("utf-8"))
    return {
        "schema": j.get("schema"),
        "lenses": {l["id"]: {"status": l["status"], "reason": l.get("reason")} for l in j["lenses"]},
        "warnings": j.get("warnings", []),
        "raw": j,
    }


def all_ok(res: dict[str, Any]) -> bool:
    return len(res["lenses"]) == 6 and all(v["status"] == "ok" for v in res["lenses"].values())
