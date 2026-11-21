# STATUS sensor Nagare/Awase fill
- ブランチ: phase2/sensor-nagare-awase（phase2/stage4-fact-sentence-index f5e3b5e から分岐）、worktree C:\dev\usir-nagare-awase
- nishiki 側: phase2/nagare-awase-fill（C:\dev\nishiki-nagare-awase-fill）。既存ブランチへのマージなし
- updated: 2026-11-20
- state: 完了
- done: 凍結構成（qwen2.5:14b, seed 42, temp 0）で場面ごとの追加呼び出し。Nagare 32/35、Awase 35/35 場面を充足（v1 16/29 → 最終）。出来事 top3 の重なり 0.5208（充足 32 場面）、風情・Awase は未測定、logp 未出力
- 成果: scripts/phase2_nagare_awase/（out=最終、out_v1/v2/v3、gold、results_overlap_*.json）
- report: nishiki 側 docs/handoff/2026-11-20_nagare_awase_sensor_report.md
