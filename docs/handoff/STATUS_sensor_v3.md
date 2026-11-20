# STATUS: sensor v3 改訂・3回測定 (2026-11-20 開始)
- ブランチ: phase2/sensor-v3-revision（土台 phase2/sensor-full-annotation 7f75069）。既存ブランチへのマージはしない。
- 根拠: Claude-Sync/inbox/2026-11-20_phase2_evening_judgment.md
- 対象: 公開テキストのみ（妻の原稿は不使用）。モデル qwen2.5:14b。ゴール: scripts/phase2_stage2/gold/stage_b_gold.json
## 進捗
- [x] ワークツリー作成
- [x] 改訂注釈生成コード（python/usir_py/sensor_eval/scene_annotator_v3.py, scripts/phase2_sensor_v3/run_v3.py）
- [ ] 3回実行（run3.sh をバックグラウンド実行中: run1..run3）
- [ ] 測定（事実P/R、同席・時間・場所、時刻、特性符号一致）
- [ ] レポート F:\マイドライブ\Claude-Sync\inbox\2026-11-20_sensor_v3_report.md
## 次の1手
既存 run_full_annotation.py / measure_stage_b.py / gold の構造を読み、v3 プロンプトを書く。
