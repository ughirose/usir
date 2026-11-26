"""Hand-curated closed location lists (Phase 1.5 Stage 2 Step 1, design item 3).

A work has a location list only if a human wrote `locations/<work_id>.json`. Lists are never generated from measured data.
Without a list, `location` is not part of any key (free-text locations vary too much: Step 0 / 0.5 measurements).

File format (sensor.locations/1):
  {"schema": "sensor.locations/1", "work_id": "...",
   "locations": [{"id": "L1", "label": "下宿", "aliases": ["自分の部屋", ...]}, ...]}
`none` (no location) is always a legitimate value and needs no entry. A non-empty location text that matches no label/alias
maps to `unlisted` (distinct from `none`), so a surface variant missing from `aliases` is visible instead of silently split.
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any

LOCATIONS_DIR = Path(__file__).resolve().parent / "locations"
SCHEMA = "sensor.locations/1"
NONE_ID = "none"
UNLISTED_ID = "unlisted"


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    return "".join(ch for ch in s if unicodedata.category(ch)[0] not in "ZPSC")


def load_locations(work_id: str, directory: Path | None = None) -> dict[str, Any] | None:
    """Return {"ids": [...], "by_text": {normalised label/alias: id}} or None when the work has no list."""
    p = Path(directory or LOCATIONS_DIR) / f"{work_id}.json"
    try:
        j = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(j, dict) or j.get("schema") != SCHEMA or j.get("work_id") != work_id or not j.get("locations"):
        return None
    by_text: dict[str, str] = {}
    for loc in j["locations"]:
        for t in [loc["label"], *loc.get("aliases", [])]:
            by_text.setdefault(_norm(t), loc["id"])
    return {"ids": [loc["id"] for loc in j["locations"]], "by_text": by_text}


def location_id(work_id: str, location: str | None, directory: Path | None = None) -> str | None:
    """None when the work has no list (location excluded from keys); otherwise a list id, `none` or `unlisted`."""
    lst = load_locations(work_id, directory)
    if lst is None:
        return None
    if location is None or not _norm(location):
        return NONE_ID
    return lst["by_text"].get(_norm(location), UNLISTED_ID)
