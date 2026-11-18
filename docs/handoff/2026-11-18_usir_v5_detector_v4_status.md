# DETV4 STATUS（2026-11-18）
- ブランチ: stage/detector-v4（dee3909 から作成）
- 5時間枠使用量: 6%（着手時、get_usage）
- 進捗: T0 完了（v3 再現一致）、T1 設計文書コミット済み
- NEXT: T2 実装（signals.py に L_narr 追加 → detector_v4.py → テスト）
- T0: v3 再実行で AUC 0.8690 / FPR_OP-B 0.1023、表層のみ 0.8822、B' 0.5958。detv3_results.json との差は seconds のみ。
- 開始時 pytest 件数: 未計測（T2 後にまとめて記録）
