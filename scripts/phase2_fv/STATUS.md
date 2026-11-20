# STATUS: sensor fact_updates / values_6d 失敗分析と改善1回

- 開始: 2026-11-20
- ブランチ: phase2/sensor-fact-values-analysis (base: phase2/sensor-full-annotation 7f75069)。既存ブランチへマージしない。
- 対象: 公開テキストのみ（妻の原稿は使わない）
- 計画: 作業1 FP/FN 5分類 + values_6d ズレ分析 → 作業2 fact_updates プロンプト1版再測定 / values_6d 既知情報付き1回再測定 → JSON と報告
- 進捗: [x] ワークツリー作成  [ ] 現状ファイル調査  [ ] 作業1  [ ] 作業2  [ ] 報告
- [x] 作業1-a: fact_updates FP78/FN60 全件手分類 → scripts/phase2_fv/results/fact_error_classification.json（FP: A5 B23 C28 D14 E8 / FN: M45 B5 C7 D3）。ベースラインの件数は stageB_metrics.json(all) と一致（FP78/FN60）。
- 次: 作業1-b values_6d ズレ分析、作業2
- [x] 作業1-b: values_6d ズレ分析 → results/values_mismatch_summary.json（照合対象の金非0軸66: 同符号17 / 予測0 44 / 逆符号5。ゴール ド71人物場面のうち25は予測に人物自体が無い）
- [~] 作業2: 3本を順次実行中（results/run_control, run_fact_v2, run_values_ctx）。control は同一プロンプトの再実行で、ベースラインとの非決定性（前回 modern 8/11 場面同一）を測るための対照。
- NEXT: 3本完了後、measure_stage_b.py で再測定 → results/*_metrics.json → レポート
