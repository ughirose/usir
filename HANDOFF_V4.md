# USIR v4.0 True Continuous Schrödinger Bridge Handoff Document
**Last Updated**: 2026-11-02 21:00 JST
**Architecture Phase**: WS-1 (Complete) | WS-2 (Complete) | Local Verification (Complete) | Colab Execution (Ready) | Final Commit (Pending)

---

## 1. Executive Summary & Purpose
This document provides immediate state handoff for USIR v4.0 True Continuous Schrödinger Bridge implementation:
1. **Abolition of String-Stitching (`"".join()`) and Sentence Pruning**:
   - Replaced fragmented 3-piece string joining (`opt_fwd.steps + [infill] + opt_bwd.steps`) and destructive regex/sentence-pruning that previously collapsed literary text to ~152 characters.
   - Restored authentic paragraph scale (400〜650 characters) across all 3 domains.
2. **3-Point Geodesic Skeleton Planning $(x_0 \to x_{\text{mid}}^* \to x_T^*)$**:
   - Optimal midpoint collision extracts $\mathbf{q}_{\text{mid}} = 0.5(\mathbf{q}_{\text{start}} + \mathbf{q}_{\text{end}})$ and identifies intermediate waypoint pivot $x_{\text{mid}}^*$.
3. **Single-Pass Continuous Rollout (`continuous_rollout_generator`)**:
   - Generates $K$ complete, contiguous paragraph candidates via a single forward rollout pass conditioned on the 3-point skeleton without boundary stitching.
4. **Best-of-$K$ Full-Trajectory Action Minimization**:
   - Selects candidate $Y^*$ minimizing:
     $$\mathcal{S}_{\text{full}}(Y) = \mathcal{D}_{\text{3-phase}}(Y, \text{domain}) + 0.8 \cdot \mathcal{D}_{\text{waypoint}}(Y, x_{\text{mid}}^*) + 1.2 \cdot \mathcal{D}_{\text{landing}}(Y, x_T^*) + \mathcal{P}_{\text{rep}}(Y)$$
   - Winning candidate is adopted intact as `welded_text` and `welded_trajectory` with zero sentence deletion.
5. **Automatic Mechanical Simplified CJK Token Masking via JIS EUC-JP**:
   - Mechanical scan in `register_tokenizer_punctuation` masks all CJK ideographs outside JIS X 0208/0212 (`0x4E00..0x9FFF` failing `ch.encode("euc_jp")` plus `步` U+6B65) into `_banned_token_ids` with zero false positives on Joyo kanji.

---

## 2. Modified Files & Line Details
| File | Key Changes | Status |
|---|---|---|
| `python/usir_py/bidirectional_bridge.py` | Added `midpoint_pivot` to `CollisionBridgeResult`, added `continuous_rollout_generator` parameter to `solve_bidirectional_collision_bridge`, implemented intermediate pivot extraction and Best-of-K full action selection $\mathcal{S}_{\text{full}}$ | Verified |
| `packages/usir-core/src/dynamics/bidirectionalBridge.ts` | Added `midpoint_pivot` to `CollisionBridgeResult`, added `continuous_rollout_generator` option, implemented intermediate pivot extraction and Best-of-K full action selection in TypeScript | Verified |
| `python/tests/test_bidirectional_bridge.py` | Added `test_continuous_rollout_3point_skeleton_selection` verifying 400+ char paragraph selection without sentence trimming | Passed (26/26) |
| `packages/usir-core/tests/bidirectional-bridge.test.ts` | Added vitest test verifying 3-point skeleton and full action minimization without stitching | Passed (22/22) |
| `python/usir_py/realtime_decoder.py` | Automatic EUC-JP non-Japanese CJK token masking on full tokenizer vocabulary with byte-decoding resilience | Passed (12/12) |
| `python/colab/run_v4_bidirectional_experiment.py` | Method 3 renovated to 3-point skeleton planning + single-pass continuous rollout generator (450~650 chars) across Tiers 1, 2, and 3 | Verified |

---

## 3. Verification State & Benchmark Results
- **TypeScript Vitest**: `pnpm test` -> **26/26 test files passed, 287/287 tests passed (0 failed)**.
- **Python Pytest**: `uv run pytest tests/` -> **175 passed, 4 skipped (0 failed)**.
- **Harness Gate**: `python scripts/harness.py test` -> **EXIT 0 (All gates passed)**.
- **Cleanroom Copyright Audit**: `python scripts/scan_cleanroom_copyright.py` -> **195 PASS, 5 WARN (authentic masterworks), 0 FAIL**.
- **Paragraph Scale**: Restored to authentic 400〜650 characters across novel, article, and cot_reasoning domains.

