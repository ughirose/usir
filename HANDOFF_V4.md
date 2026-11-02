# USIR v4.0 Phase-Split Seam Welding & Domain QDA Handoff Document
**Last Updated**: 2026-11-02 15:32 JST
**Architecture Phase**: WS-1 (Complete) | WS-2 (Complete) | Local Verification (Complete) | Colab Execution (Pending)

---

## 1. Executive Summary & Purpose
This document provides immediate state handoff for USIR v4.0 remediation to ensure seamless continuity if model quota or context is exhausted. It tracks the 4 targeted issues identified from commit `c990e9d`:
1. **Forward/Backward Wave Collision & Kanji Stem Echo Paraphrasing**: Solved via `sanitize_wave_steps` with Kanji stem Jaccard $\ge 0.50$ deduplication and `+6.0` penalty.
2. **3-Phase QDA Domain Separation**: Solved via domain-conditioned QDA profiles with relaxed $\sigma_{\text{crit}}^{\text{high}} \in [1.35, 2.65]$ for `cot_reasoning` and domain parameter passed across all call sites.
3. **Prompt Meta-Word & Chinese (CJK) Leakage**: Solved via banned token identification at the tokenizer level with $-100.0$ logits penalty and `_has_meta_or_cjk_leak` regex filtering.
4. **LogitsProcessor Latency & Hallucination**: Solved via 100% GPU tensor vectorization (`torch.isin`, `torch.mv`) eliminating the Python loop over Top-K, model tier upgrade to `Qwen/Qwen2.5-3B-Instruct`, and phase-split prompts.

---

## 2. Modified Files & Line Details
| File | Key Changes | Status |
|---|---|---|
| `packages/usir-core/src/dynamics/manifoldGeodesic.ts` | Added `DOMAIN_QDA_PROFILES` with relaxed CoT $\sigma_{\text{crit}}^{\text{high}} = 2.65$ | Verified |
| `packages/usir-core/src/dynamics/bidirectionalBridge.ts` | Implemented `sanitizeWaveSteps`, `hasMetaOrCjkLeak`, Kanji stem Jaccard $\ge 0.50$ filter, `domain` prop in `computeThreePhaseQdaDistance` | Verified |
| `packages/usir-core/tests/bidirectional-bridge.test.ts` | Unit tests for `sanitizeWaveSteps`, Jaccard deduplication, and non-Kanji echo gates | Passed (16/16) |
| `packages/usir-core/tests/manifold-geodesic.test.ts` | Unit tests for domain-specific QDA profiles | Passed (10/10) |
| `python/usir_py/manifold_geodesic.py` | Python `DOMAIN_QDA_PROFILES` for novel, article, cot_reasoning | Verified |
| `python/usir_py/bidirectional_bridge.py` | Python `sanitize_wave_steps`, `_has_meta_or_cjk_leak`, Jaccard stem deduplication, passing `domain=anchors.domain` | Verified |
| `python/usir_py/realtime_decoder.py` | Fully vectorized `USIRPyTorchLogitsProcessor.__call__` via GPU tensor operations, CJK & backtick $-100.0$ suppression | Verified |
| `python/usir_py/mcp_server.py` | Passed `domain=domain` to `compute_three_phase_qda_distance` and `compute_multiplicative_grpo_reward` | Verified |
| `python/tests/test_manifold_geodesic.py` | Domain-conditioned QDA tests | Passed |
| `python/tests/test_bidirectional_bridge.py` | `sanitize_wave_steps` Kanji stem Jaccard and CJK filtering tests | Passed |
| `python/tests/test_realtime_decoder.py` | Vectorized processor, CJK masking, and domain runner tests | Passed |
| `python/colab/run_v4_bidirectional_experiment.py` | Upgraded to `Qwen2.5-3B-Instruct`, phase-split forward/backward prompts, setting drift penalty in judge | Verified |

---

## 3. Verification State
- **TypeScript Vitest**: `pnpm test` -> **26/26 test files passed, 281/281 tests passed**.
- **Python Pytest**: `uv run pytest tests/` -> **168 passed, 4 skipped**.
- **Harness Gate**: `python scripts/harness.py test` -> **EXIT 0 (All gates passed)**.
- **Cleanroom Copyright Audit**: `python scripts/scan_cleanroom_copyright.py` -> **195 PASS, 5 WARN (authentic masterworks), 0 FAIL**.
- **Local Lattice Solver**: `python/colab/run_v4_bidirectional_experiment.py` -> **Spearman $\rho(\mathcal{D}_{\text{3-phase}}, \text{Quality}) = -0.7726$ ($p = 4.73 \times 10^{-19}$)**.

---

## 4. Next Step & Resumption Command
To execute the authentic Google Colab T4 benchmark:
```powershell
python scripts/run_colab_benchmark.py
```
Or via Colab MCP server:
```python
colab_execute_file("python/colab/run_v4_bidirectional_experiment.py", accelerator="T4")
```
After artifact generation, inspect `artifacts/blind_evaluation_sheet.md` to confirm 0 CJK leaks and 0 stem repetitions, then commit and push.
