# STATUS: Phase 1.5 Stage 1 (sensor_runner L1/L2 cache)

- ブランチ: feat/phase15-stage1-cache（worktree: C:\dev\usir-phase15）
- ベース: feat/sensor-runner @ b3da0e4
  - 指示は「usir の現在ブランチから worktree」だったが、現在ブランチ phase2/gold-audit（a9fbbec）に scripts/sensor_runner が存在しないため、sensor_runner を含む feat/sensor-runner から切った（逸脱。レポートにも記載）。
- 設計メモ正本: 2026-11-24_usir_pack_v1_task6_phase15_incremental_design.md（ローカル: C:\Users\user\.gemini\antigravity\brain\a615e3a2-...\scratch\pack_v1\）
- Gate: 2026-11-26 Orion 承認（条件1: L2 prior は構造化サマリの Early Cutoff／条件2: L3 は実装しない）

## 進捗
- [x] 棚卸し（現行 cache.py = scene_hash 単体キー + instruction_version 一致のみ）
- [x] worktree 作成
- [ ] ベースライン実測
- [ ] 実装（common/cache/prepare/merge）
- [ ] パリティテスト・効果実測
- [ ] レポート（Drive inbox + docs/handoff/2026-11-26_phase15_stage1_report.md）

## NEXT
ベースライン実測（全量 merge + diagnose の出力保存、所要時間）。
