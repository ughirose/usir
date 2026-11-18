# DETV4 STATUS（2026-11-18）
- ブランチ: stage/detector-v4（dee3909 から作成）
- 5時間枠使用量: 着手時 6%
- 進捗: T0〜T4 完了。ゲート全通過（pytest 689 passed／18 skipped、開始時基準 678 + 新規 11。Vitest 378。パリティ不一致 0。凍結照合は追加のみ）
- 結果（シナリオ A）: 合算 AUC 0.8603（CI 0.8461〜0.8726）、p_OP=97.0 で保持 FPR 0.0304、OP-B(p=90) で FPR 0.1002
- 未解決: p_OP 規則が上限 97.0 に飽和（検出率平均 0.5106）。q3' の感覚語 AUC 0.6785 < v3 全体率 0.7420。詳細は report §9
- レポート: docs/handoff/2026-11-18_usir_v5_stage_detector_v4_report.md（inbox にも同一内容を配置）
- NEXT: Orion が p_OP 規則と q3' の扱いを判断
