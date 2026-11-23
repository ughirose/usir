# USIR ドキュメント索引（2026-11-23）

> **注意**: 現在の開発先端は **`feat/fable-v5-shootout`** ブランチにあり、`main`（2026-11-11 `1a30ec2`）は更新停止中です。統合設計・計画正本は Google Drive `05_Nishiki_錦/00_INDEX_AND_SSOT/`、各段階完了報告の正本は `Claude-Sync/inbox/` を参照してください。

---

## 1. 現行ドキュメント・稼働手順（Active）

| ファイル | 内容 | 状態 |
|---|---|---|
| [scripts/sensor_runner/SESSION.md](../scripts/sensor_runner/SESSION.md) | ハイブリッドセンサー・セッションバッチ運用手順。 | **現行運用** |
| [docs/handoff/2026-11-22_sensor_runner.md](handoff/2026-11-22_sensor_runner.md) | `sensor_runner` 設計とキャッシュ・検証仕様。 | **現行仕様** |
| [docs/handoff/2026-11-22_sensor_live_run2_handoff.md](handoff/2026-11-22_sensor_live_run2_handoff.md) & [2026-11-22_v31_usir_fixes.md](handoff/2026-11-22_v31_usir_fixes.md) | 実走 2（『二百十日』40 場面）結果および v3.1/全 null 警告修正の限界記録。 | **現行実測** |
| [docs/handoff/2026-11-20_phase2_stage2_report.md](handoff/2026-11-20_phase2_stage2_report.md) & [2026-11-20_phase2_step1.md](handoff/2026-11-20_phase2_step1.md) | ローカル 14B センサー定量評価（`fact_updates` F1=0.169 等の限界確定データ）。 | **現行ベースライン** |
| [schemas/sensor_v2_phase0_contract.md](../schemas/sensor_v2_phase0_contract.md) | Phase 0 `sensor.v2` 契約。*(注意: Nishiki `/3` の `trait_variations` / `target/category` 追従待ち)*。 | **更新待ち** |
| [docs/handoff/2026-11-08_usir_v4_final_summary.md](handoff/2026-11-08_usir_v4_final_summary.md) | USIR v4（Phase 18–49）の検証結果・否定仮説総括。 | **v4 完了記録** |

---

## 2. テスト・スクリプト参照により位置を固定している旧文書

以下のルート直下ファイルは `scripts/`・`python/`・`packages/` 内のコードまたはテストが直接パス参照しているため、移動せずそのまま保全しています（**v2/v3 論文ファイルには Phase 34 監査による撤回注記あり**）：

- `overnight_campaign_report.md` / `usir_adversarial_stress_test_and_feasibility_audit.md`
- `usir_v2_camera_ready_paper.md` / `usir_v2_camera_ready_paper_ja.md` / `usir_v3_camera_ready_paper_ja.md` / `usir_v4_paper.md`

---

## 3. アーカイブ済み文書（[docs/archive/v2_v4_legacy/](archive/v2_v4_legacy/)）

コード・スクリプトからの参照が 0 件の旧 USIR v2〜v4 引継ぎ資料・ロードマップ・旧撤回論文（計 8 ファイル）を [docs/archive/v2_v4_legacy/](archive/v2_v4_legacy/) へ退避しました。新規設計には使用しないでください。
