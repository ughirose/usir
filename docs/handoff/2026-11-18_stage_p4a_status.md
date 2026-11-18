# P-4a STATUS（2026-11-18）
- ブランチ: stage/p4a-detector-ref（stage/detector-v4 `8ad54fa` から作成。本流へはマージしない。判定 J1）
- 5時間枠使用量: 着手時 14%（80% 未満のため着手）
- 基点テスト: `python/tests/test_detector_v3.py` + `test_detector_v4.py` = 17 passed（実測）。全体ベースラインは実測中（下記に追記）
- 進捗: T0 完了（基点確認・ブランチ作成）。実装前
- 事前登録（本プロンプト §2、変更しない）: A/B は合算 AUC の 95% CI 下限が高い方を採用（FPR 制約を満たすこと）。同点（小数第4位）は (A) ポアソン。μ̂ 縮小あり/なしは (A) 内部比較として同規則で先に決める
- 保持 FPR の判定: P（全コーパス p_OP）と N（訓練作品のみの入れ子 p_OP）の両方を報告。分岐条件は両方 ≤ 0.10 を要求（保守側）
- NEXT: scripts/detector_p4/ に signals（L_total 追加）と参照実装を作る
