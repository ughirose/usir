# P-4a STATUS（2026-11-18）
- ブランチ: stage/p4a-detector-ref（stage/detector-v4 `8ad54fa` から作成。本流へはマージしない。判定 J1）
- 5時間枠使用量: 着手時 14%（80% 未満のため着手）
- 基点テスト: `python/tests/test_detector_v3.py` + `test_detector_v4.py` = 17 passed（実測）。全体ベースライン（harness）: pytest 689 passed / 18 skipped、Vitest 378 passed（36 files）、exit 0
- 進捗: signals（L_total 追加、q1-q3 は detv4 npz と完全一致）、参照実装 detector_p4.py、v4 再現（差 0.0）まで完了。v13 計測は事前登録コミット後に実行
- 事前登録（本プロンプト §2、変更しない）: A/B は合算 AUC の 95% CI 下限が高い方を採用（FPR 制約を満たすこと）。同点（小数第4位）は (A) ポアソン。μ̂ 縮小あり/なしは (A) 内部比較として同規則で先に決める
- 保持 FPR の判定: P（全コーパス p_OP）と N（訓練作品のみの入れ子 p_OP）の両方を報告。分岐条件は両方 ≤ 0.10 を要求（保守側）
- 事前登録の補足（計測前に確定）: μ̂ 縮小の同点は縮小なし。B(Anscombe) は期待個数 λ に対する 2√(λ+3/8)−2√(c+3/8) とし、μ̂ は A 内比較で選ばれた方に合わせる。FPR 制約は P/N 両方 ≤ 0.10
- NEXT: measure.py pop → run（detp4_results.json）
