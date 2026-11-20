# STATUS: sensor v3 改訂・3回測定 (2026-11-20 開始)
- ブランチ: phase2/sensor-v3-revision（土台 phase2/sensor-full-annotation 7f75069）。既存ブランチへのマージはしない。
- 根拠: Claude-Sync/inbox/2026-11-20_phase2_evening_judgment.md
- 対象: 公開テキストのみ（妻の原稿は不使用）。モデル qwen2.5:14b。ゴール: scripts/phase2_stage2/gold/stage_b_gold.json
## 進捗
- [x] ワークツリー作成
- [ ] 改訂注釈生成コード（時刻ISO化・事実6述語・特性モデル）
- [ ] 3回実行
- [ ] 測定（事実P/R、同席・時間・場所、時刻、特性符号一致）
- [ ] レポート F:\マイドライブ\Claude-Sync\inbox\2026-11-20_sensor_v3_report.md
## 次の1手
既存 run_full_annotation.py / measure_stage_b.py / gold の構造を読み、v3 プロンプトを書く。
