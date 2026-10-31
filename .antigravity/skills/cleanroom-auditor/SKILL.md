---
name: cleanroom-auditor
description: Audit 3-Room Clean Room compliance, 7-gram non-retention gate, and zero target vocabulary leakage.
---

# Clean Room Auditor Skill

Use this skill to audit copyright compliance and feature extractor purity:

## Audit Principles
1. **Zero Target-Leakage**:
   - Verify that no feature extraction code, regex, or dictionary in `sensor/` or `extractor_24d.py` contains author names, character names (e.g. `メロス`, `李徴`, `島村`, `下人`), or specific literary work titles.
   - Only universal morphological, phonological, syntactic, and broad sensory character categories are permitted.
2. **3-Room Clean Room Architecture**:
   - **Dirty Room**: Extracts numerical statistics and moments from raw text, immediately discarding raw text from memory.
   - **Audit Gate**: Scans Tier S (`128B`) and Tier M (`4096B`) binary outputs to verify zero raw text retention and zero contiguous $\ge 7$-character n-gram overlap with the source input.
   - **Clean Room**: Operates solely on verified binary representations for downstream guidance, SLM rewards, and UI visualizers.
3. **Binary Struct Alignment**:
   - Tier S serialized buffer must be exactly 128 bytes (Magic `0x55534952`).
   - Tier M serialized buffer must be exactly 4096 bytes.
