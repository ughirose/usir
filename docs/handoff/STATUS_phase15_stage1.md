# STATUS: Phase 1.5 Stage 1 (sensor_runner L1/L2 cache)

- ブランチ: feat/phase15-stage1-cache（worktree: C:\dev\usir-phase15）、ベース feat/sensor-runner @ b3da0e4（現在ブランチに sensor_runner が無いため）
- 状態: 実装・テスト・実測・レポート完了。main へのマージは Orion 判断待ち。
- レポート: docs/handoff/2026-11-26_phase15_stage1_report.md
- 検証: sensor_runner 59 テスト合格、パリティ（4ケース + 3作品×5回）全一致、harness 全体の失敗 7 件はベースでも同一。

## NEXT
Orion が report §6 の判断点（ledger_hash の範囲、Early Cutoff の健全性、40場面作品での L3 要否測定）を決める。

## Stage 2 Step 0（2026-11-26 開始、計測のみ・キャッシュ実装は変更しない）
- 指示: Drive inbox の 2026-11-26_phase15_stage2_step0_prompt_v1.md / consult_synthesis_v1.md
- 計測A（繰り返し注釈の揺れ）、計測B（prior 感度）。注釈呼び出し上限 合計300回。
- 結果: scripts/sensor_runner/results/phase15_stage2_step0_*.json、レポート docs/handoff/2026-11-26_phase15_stage2_step0_report.md
- 状態: 計測完了（注釈148件、試行151）、レポート作成済み。59 テスト合格、実装変更なし。main 未マージ。
- NEXT: Orion が Step 0 レポート §4・§5 を読み、Step 1 の方針（location ID 化、facts の型付け、前提変化表示方式）を決める。

## Stage 2 Step 0.5（2026-11-26 開始、計測のみ・キャッシュ実装/本番スキーマ/物差しは変更しない）
- 指示: Drive inbox の 2026-11-26_phase15_stage2_step05_prompt_v1.md
- Q-A 型付き facts の一致率・語彙被覆、Q-B location 候補リスト選択の一致率。20場面×5回、注釈上限150回。
- 結果: scripts/sensor_runner/results/phase15_stage2_step05_*.json、レポート docs/handoff/2026-11-26_phase15_stage2_step05_report.md
- 状態: 計測完了（受理100件）、レポート作成済み。59 テスト合格、実装変更なし。main 未マージ。
- 結果要約: 型付き facts 不一致率 T3 1.00 / T0 0.88（鍵に不可）、location 閉じた語彙一致 0.90（楽観的条件）、prior ハッシュ 0.74→0.835（再計算値）。
- NEXT: Orion が Step 0.5 レポート §3・§4 を読み、Step 1 の方針を決める。
