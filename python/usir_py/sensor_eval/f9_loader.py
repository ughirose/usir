"""F9 Gold dataset loader with Dev/Holdout split isolation and quote-numbering cross-check."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class F9Character:
    """Character entry from F9 gold metadata."""

    id: str
    aliases: list[str]
    gender: str


@dataclass(frozen=True)
class F9UtteranceGold:
    """Gold annotation for a single outermost quoted utterance."""

    n: int
    open_offset: int
    close_offset: int
    head: str
    status: str  # "attributed" | "undetermined" | "not_speech"
    speaker: str | None
    quote_text: str
    gold_politeness: str | None = None


@dataclass(frozen=True)
class F9Work:
    """One F9 literary excerpt with numbered text and gold utterances."""

    work_id: str
    author: str | None
    title: str | None
    narration: str
    narrator: str | None
    pov: str | None
    characters: list[F9Character]
    text: str
    numbered_text: str
    scene_hash: str
    utterances: list[F9UtteranceGold]


@dataclass(frozen=True)
class F9Dataset:
    """Loaded split of the F9 gold corpus."""

    split: str
    gold_dir: str
    gold_json_sha256: str
    split_version: str
    works: list[F9Work]


def extract_outer_quote_spans(text: str) -> list[tuple[int, int]]:
    """Return half-open [start, end) character spans of outermost 「...」/『...』 quotes."""
    spans: list[tuple[int, int]] = []
    depth = 0
    start: int | None = None
    for idx, ch in enumerate(text):
        if ch in "「『":
            if depth == 0:
                start = idx
            depth += 1
        elif ch in "」』" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                spans.append((start, idx + 1))
                start = None
    return spans


def number_outer_quotes(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Insert [U0], [U1], ... markers immediately before each outermost opening quote."""
    spans = extract_outer_quote_spans(text)
    if not spans:
        return text, spans
    parts: list[str] = []
    cursor = 0
    for idx, (start, _end) in enumerate(spans):
        parts.append(text[cursor:start])
        parts.append(f"[U{idx}]")
        cursor = start
    parts.append(text[cursor:])
    return "".join(parts), spans


def _cross_check_nishiki_segment(
    nishiki_bin: Path, txt_path: Path, expected_opens: list[int]
) -> None:
    """Verify that `nishiki segment` produces identical utterance boundaries."""
    proc = subprocess.run(
        [str(nishiki_bin), "segment", str(txt_path), "--scene-split", "none", "--format", "json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"nishiki segment failed for {txt_path} (rc={proc.returncode}): {proc.stderr.strip()}"
        )
    report = json.loads(proc.stdout)
    scenes = report.get("scenes", [])
    if len(scenes) != 1:
        raise AssertionError(
            f"Expected 1 scene from nishiki segment for {txt_path}, got {len(scenes)}"
        )
    seg_starts = [u["local"]["start"] for u in scenes[0]["utterances"]]
    gold_starts = [o + 1 for o in expected_opens]
    if seg_starts != gold_starts:
        raise AssertionError(
            f"Utterance boundary mismatch vs nishiki segment for {txt_path.name}: "
            f"seg={seg_starts[:5]} vs gold={gold_starts[:5]}"
        )


def load_f9_dataset(
    gold_dir: str | Path,
    split_file: str | Path,
    split: str = "dev",
    unlock_holdout: bool = False,
    politeness_gold_path: str | Path | None = None,
    nishiki_bin: str | Path | None = None,
    gold_json_path: str | Path | None = None,
) -> F9Dataset:
    """Load the requested F9 split (`dev` or `holdout`) with strict holdout sealing."""
    if split not in ("dev", "holdout"):
        raise ValueError(f"split must be 'dev' or 'holdout', got {split!r}")

    split_path = Path(split_file)
    split_cfg: dict[str, Any] = json.loads(split_path.read_text(encoding="utf-8"))

    if split == "holdout":
        sealed = bool(split_cfg.get("holdout", {}).get("sealed", True))
        if sealed and not unlock_holdout:
            raise PermissionError(
                "Holdout split is sealed to prevent data leakage. "
                "Do not inspect or evaluate holdout during Step 1 Dev iteration; "
                "pass unlock_holdout=True only for the single final evaluation."
            )

    gold_root = Path(gold_dir)
    resolved_gold_json = (
        Path(gold_json_path) if gold_json_path is not None else (gold_root / "gold.json")
    )
    gold_bytes = resolved_gold_json.read_bytes()
    actual_sha = hashlib.sha256(gold_bytes).hexdigest()
    expected_sha = split_cfg.get("gold_json_sha256")
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(
            f"gold.json SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
        )

    gold_raw = json.loads(gold_bytes.decode("utf-8"))
    manifest_meta: dict[str, dict[str, Any]] = {}
    manifest_path = gold_root / "manifest.json"
    if manifest_path.exists():
        m_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in m_raw.get("works", []):
            manifest_meta[entry["id"]] = entry

    politeness_overlay: dict[str, dict[str, str]] = {}
    if politeness_gold_path is not None:
        politeness_overlay = json.loads(
            Path(politeness_gold_path).read_text(encoding="utf-8")
        )

    target_ids: list[str] = list(split_cfg[split]["work_ids"])
    works_by_id = {w["id"]: w for w in gold_raw["works"]}

    loaded_works: list[F9Work] = []
    for wid in target_ids:
        if wid not in works_by_id:
            raise KeyError(f"Work id {wid!r} from split_file not found in gold.json")
        w_raw = works_by_id[wid]
        txt_path = gold_root / f"{wid}.txt"
        text = txt_path.read_text(encoding="utf-8")
        numbered_text, spans = number_outer_quotes(text)

        raw_utts = w_raw["utterances"]
        if len(spans) != len(raw_utts):
            raise AssertionError(
                f"Work {wid}: extracted {len(spans)} outer quotes vs {len(raw_utts)} gold utterances"
            )
        for idx, (span, u_raw) in enumerate(zip(spans, raw_utts)):
            if u_raw["n"] != idx:
                raise AssertionError(f"Work {wid}: non-sequential gold n={u_raw['n']} at {idx}")
            if span[0] != u_raw["open"]:
                raise AssertionError(
                    f"Work {wid} U{idx}: span start {span[0]} != gold open {u_raw['open']}"
                )

        if nishiki_bin is not None:
            _cross_check_nishiki_segment(
                Path(nishiki_bin), txt_path, [u["open"] for u in raw_utts]
            )

        work_politeness = politeness_overlay.get(wid, {})
        utterances: list[F9UtteranceGold] = []
        for span, u_raw in zip(spans, raw_utts):
            n = int(u_raw["n"])
            pol = u_raw.get("politeness") or work_politeness.get(str(n))
            utterances.append(
                F9UtteranceGold(
                    n=n,
                    open_offset=span[0],
                    close_offset=span[1],
                    head=str(u_raw["head"]),
                    status=str(u_raw["status"]),
                    speaker=u_raw["speaker"],
                    quote_text=text[span[0] : span[1]],
                    gold_politeness=pol,
                )
            )

        chars = [
            F9Character(
                id=str(c["id"]),
                aliases=[str(a) for a in c.get("aliases", [])],
                gender=str(c.get("gender", "Unknown")),
            )
            for c in w_raw["characters"]
        ]
        meta = manifest_meta.get(wid, {})
        loaded_works.append(
            F9Work(
                work_id=wid,
                author=meta.get("author"),
                title=meta.get("title"),
                narration=str(w_raw["narration"]),
                narrator=w_raw.get("narrator"),
                pov=w_raw.get("pov"),
                characters=chars,
                text=text,
                numbered_text=numbered_text,
                scene_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                utterances=utterances,
            )
        )

    return F9Dataset(
        split=split,
        gold_dir=str(gold_root),
        gold_json_sha256=actual_sha,
        split_version=str(split_cfg.get("split_version", "unknown")),
        works=loaded_works,
    )
