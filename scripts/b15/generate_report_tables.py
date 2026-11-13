#!/usr/bin/env python3
"""Generate clean Markdown tables for Stage B-15 report from JSON outputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import build_real_revision_pairs_v1 as b12  # noqa: E402


def main() -> None:
    t1 = json.loads((REPO_ROOT / "scripts/b15/b15_task1_inventory.json").read_text(encoding="utf-8"))
    t2 = json.loads((REPO_ROOT / "scripts/b15/b15_task2_candidates.json").read_text(encoding="utf-8"))
    t3 = json.loads((REPO_ROOT / "scripts/b15/b15_task3_projection.json").read_text(encoding="utf-8"))

    # 1. Task 1 full 60-work table sorted by (genre, author_id, work_id)
    ns1 = sorted(t1["non_set1_106_works"], key=lambda r: (r["genre"], r["author_id"], r["work_id"]))
    lines = []
    lines.append("## Task 1: 60 Works Table")
    lines.append("| # | work_id | 作品名 | 著者 | 時代区分 | 区分 | 文字遣い | 総ペア | M全 | M[≥2,≥2] | L全 | L[≥2,≥2] |")
    lines.append("|---:|---|---|---|---|---|---|---:|---:|---:|---:|---:|")
    for i, r in enumerate(ns1, 1):
        lines.append(
            f"| {i} | `{r['work_id']}` | {r['title']} | {r['author_ja']} | {r['era']} | {r['genre']} | {r['char_type']} | "
            f"{r['n_pairs_total']} | {r['M_all']} | {r['M_both_multi']} | {r['L_all']} | {r['L_both_multi']} |"
        )

    # 2. Task 2 genuinely new candidates (not in B-12 CANDIDATE_WORK_PAIRS)
    b12_wids = set()
    for _, a, b, _, _ in b12.CANDIDATE_WORK_PAIRS:
        b12_wids.add(a)
        b12_wids.add(b)
    new_cross = [
        e for e in t2["cross_edition_same_ortho"]
        if not ({ed["work_id"] for ed in e["editions"]} & b12_wids)
    ]
    lines.append("\n## Task 2: Genuinely New Candidates (not in B-12 43-pair list)")
    lines.append(f"Count: {len(new_cross)}")
    for e in new_cross:
        eds = " / ".join(
            f"{ed['work_id']}:{ed['title']}({ed['char_type']}, 底本:{ed['base_book']}, URL:{ed['card_url']})"
            for ed in e["editions"]
        )
        lines.append(f"- {e['author']}『{e['norm_title']}』[{e['ndc']}]: {eds}")

    out_path = REPO_ROOT / "scripts/b15/b15_tables_preview.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path} ({len(new_cross)} genuinely new Task 2 candidates)")


if __name__ == "__main__":
    main()
