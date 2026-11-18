# P-4a STATUS（2026-11-18）
- ブランチ: stage/p4a-detector-ref（stage/detector-v4 `8ad54fa` から作成。本流へ未マージ）
- 5時間枠使用量: 着手時 14%
- 状態: 完了
- 結果: 採用 B_anscombe_shrink（事前登録規則）。合算 AUC 0.8803、CI [0.8681, 0.8922]、p_OP 93.0（最小 p 規則）、保持 FPR P 0.0709 / N 0.0693。分岐条件（CI 下限 ≥ 0.85 かつ FPR ≤ 0.10）を満たす
- ゲート: pytest 700 passed / 18 skipped、Vitest 378 passed、v4 再現の差 0.0、凍結は追加のみ
- レポート: docs/handoff/2026-11-18_usir_v5_stage_p4a_report.md（Claude-Sync/inbox と Drive inbox にも配置）
- 未解決: レポート §9（感覚語 TPR 0.278、小標本 m≤8 で FPR>0.10、tail_p の同点感度、longdouble は 64bit）
- NEXT: Orion が結果を判断（P-4b 着手可否、J9 の追加測定の要否）
