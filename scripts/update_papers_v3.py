#!/usr/bin/env python3
"""Update USIR Camera-Ready Paper manuscripts to v3.0 specification."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path


def load_section(name: str) -> str:
    path = Path("scripts/paper_sections") / name
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def main() -> None:
    paper_v2_path = Path("usir_v2_camera_ready_paper.md")
    paper_v3_path = Path("usir_v3_camera_ready_paper.md")

    with open(paper_v2_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Load section components
    title_abstract = load_section("title_abstract.md")
    sec3_4 = load_section("sec3_4_gaps.md")
    sec4_6 = load_section("sec4_6_table3.md")
    sec4_7 = load_section("sec4_7_table4.md")
    sec5_cleanroom = load_section("sec5_studio_cleanroom.md")
    sec6_conclusion = load_section("sec6_conclusion.md")

    # 2. Replace Title & Abstract (up to '## 1. Introduction')
    intro_match = re.search(r"\n## 1\. Introduction", content)
    if not intro_match:
        raise ValueError("Could not find '## 1. Introduction' in manuscript.")
    body_from_intro = content[intro_match.start() :]

    # 3. Insert Section 3.4 before '## 4. Empirical Evaluation'
    sec4_match = re.search(r"\n## 4\. Empirical Evaluation", body_from_intro)
    if not sec4_match:
        raise ValueError("Could not find '## 4. Empirical Evaluation' in manuscript.")
    part_before_sec4 = body_from_intro[: sec4_match.start()]
    part_from_sec4 = body_from_intro[sec4_match.start() :]

    # Strip existing 3.4 if present to ensure idempotence
    m34 = re.search(r"\n###?\s*3\.4", part_before_sec4)
    if m34:
        part_before_sec4 = part_before_sec4[: m34.start()]

    updated_body_1 = part_before_sec4.rstrip() + "\n\n" + sec3_4 + "\n\n"

    # 4. In part_from_sec4, extract from Section 4 through Section 4.5
    # Strip any existing 4.6 or later sections
    m46 = re.search(r"\n(?:\-\-\-\s*\n+)?(?:###?\s*4\.6|##\s*5\.)", part_from_sec4)
    if not m46:
        raise ValueError("Could not find '### 4.6' or '## 5.' in manuscript.")
    part_sec4_thru_45 = part_from_sec4[: m46.start()].rstrip()

    # Assemble complete paper v3
    full_paper = (
        title_abstract
        + "\n\n"
        + updated_body_1.lstrip()
        + part_sec4_thru_45
        + "\n\n"
        + sec4_6
        + "\n\n"
        + sec4_7
        + "\n\n---\n\n"
        + sec5_cleanroom
        + "\n\n---\n\n"
        + sec6_conclusion
        + "\n"
    )

    # 5. Write to usir_v2_camera_ready_paper.md
    with open(paper_v2_path, "w", encoding="utf-8") as f:
        f.write(full_paper)
    print(f"[SUCCESS] Updated {paper_v2_path}")

    # 6. Copy to usir_v3_camera_ready_paper.md
    shutil.copy2(paper_v2_path, paper_v3_path)
    print(f"[SUCCESS] Copied to {paper_v3_path}")

    # 7. Verification checks
    for p in [paper_v2_path, paper_v3_path]:
        with open(p, "r", encoding="utf-8") as f:
            t = f.read()
        assert "USIR v3.0" in t
        assert "### 3.4 USIR v3.0 Architectural Extensions (Gaps 1–7)" in t
        assert "### 4.4 Table 2:" in t
        assert "### 4.5 Cross-Modal Musical Generalization" in t
        assert "### 4.6 Table 3: Zero-Overhead Style-LoRA & USIR-GRPO Forge" in t
        assert "### 4.7 Table 4: Cross-Linguistic Universality on English Public-Domain Masterworks ($N=400$)" in t
        assert "## 5. USIR Studio v3.0 Atelier & Clean-Room Governance" in t
        assert "## 6. Conclusion" in t
        assert t.count("### 4.4 Table 2:") == 1
        assert t.count("### 4.5 Cross-Modal Musical Generalization") == 1
        assert t.count("### 4.6 Table 3:") == 1
        assert t.count("### 4.7 Table 4:") == 1
        assert t.count(r"\label{tab:unseen_author_table1}") == 1
        assert t.count(r"\label{tab:generation_control_table2}") == 1
        assert t.count(r"\label{tab:lora_grpo_forge_table3}") == 1
        assert t.count(r"\label{tab:english_cross_linguistic_table4}") == 1
        print(f"[VERIFIED] All headers and tables present and unique in {p}")


if __name__ == "__main__":
    main()
