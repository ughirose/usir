# ClauseTwistedSMCDecoder Colab 実機ベンチ修正の引継ぎ (2026-11-04, 記入者: Claude Code)

## 決定事項
- 修正をコミット済み: `4e19fb8`（`realtime_decoder.py`, `build_colab_bench.py`）。push は未実施。
  - 内容: 全角句読点対応、1回バッチ生成（4粒子）、EOS 停止、ログ確率エラー時 -20.0、smcb/prof モード追加。
  - コミット前ゲート: Python 253 passed / 4 skipped、Vitest 337 passed（Claude Code 側で再実行）。
- 「コミットが exit 1 で失敗」の原因: コミット自体は実行されていなかった。Antigravity 履歴 Step 372-377 で exit 1 だったのはコミットではなく、`python scripts/harness.py test | Select-String` のバックグラウンドタスク。出力は 253 passed / 337 passed / `[HARNESS] SUCCESS` で、テスト失敗はない。exit 1 の直接原因は未確認（PowerShell のパイプ/stderr 扱いが疑わしいが、確かめていない）。pre-commit hook は無く、CRLF→LF 警告のみで、コミットは通った。
- Colab 再計測（Qwen2.5-1.5B-Instruct, fp16, L4, 4粒子バッチ, `max_clauses=16` 指定, μ*=II_Critical 重心の先頭12次元）。

| seed | 文字数 | 文数 | 節数 | wall(s) | s/文字 | mean ESS(/4) | incomplete | len_cv | psi_order | phi_void | kappa_kin |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 152 | 5 | 10 | 8.5 | 0.056 | 3.55 | 4 | 0.128 | 0.393 | 0.261 | 0.691 |
| 1 | 168 | 5 | 11 | 9.2 | 0.055 | 3.40 | 4 | 0.242 | 0.834 | 0.255 | 0.340 |
| 2 | 176 | 5 | 12 | 9.0 | 0.051 | 3.21 | 8 | 0.104 | 0.300 | 0.243 | 0.947 |

- logprob_errors は全 seed で 0、重複節率は 0.0、全 seed で EOS により終了。
- コンピュート消費（全数）: Colab L4 で 2 回実行。(1) seed 0 のみ、(2) seed 1,2。各回ともモデル DL+起動を含む。この前に Antigravity が seed 0,1 を L4 で 1 回実行している（履歴 Step 370、結果は未確認）。壁時計の合計は未計測（実行時間は上表の生成時間のみ）。
- seed 0 は seed 1,2 と別プロセス。同一コードだが、seed 0 を 0,1,2 一括で再実行した場合との一致は未確認。

## 限界・既知の課題（正直な評価）
- ESS は 3.21〜3.55 / 4 で、ほぼ選別になっていない（既知課題の再確認）。粒子選別の効果は、この計測からは示せない。
- G_style は単位行列のまま、目標 μ* はコーパス重心のみ（既知課題）。
- 「同一節数」は達成されていない。節数は 10/11/12 でばらつく。EOS で自然終了するため、`max_clauses=16` には届かない。
- 出力は 152〜176 字で、プロンプトの約 600 字には届かない。
- seed 0 の末尾に「静謐的」「燻微光景」など不自然な語がある。外国語混入や語の崩れは、この計測では自動判定していない（目視のみ）。
- phi_void・psi_order・kappa_kin は seed 間のばらつきが大きく（psi_order 0.30〜0.83）、3 seed では有意な傾向は言えない。
- ベースライン（vanilla 生成）と μ* までの距離は、今回計測していない。SMC が vanilla より目標に近いかは未検証。
- 推論時間は約 0.05 s/文字（T4 ではなく L4）。T4 実機での時間は未計測。

## 未完了
- 作業ツリーの整理（ユーザー確認待ち。破棄は取り消せない）:
  - `v4_bidirectional_experiment_*`、`blind_evaluation_sheet.*`、`table2_progress.json`: `git checkout` で戻す案（`--quick` の CPU 結果が GPU 実機結果を上書きしたもの。コミットしない）。
  - 未追跡の `artifacts/colab_run/patch_*.py`（適用済み）と `colab_bench*.py`（生成物 約1.1MB×5）: 削除、または `.gitignore` に追加。
  - `artifacts/usir_5phase_empirical_verification.png`: 出所不明。扱いを決める。
- push（ユーザー確認後）。
- `scripts/harness.py test` が exit 1 を返す原因の調査（テストは通過している）。

## 次の一手
- 上の整理についてユーザー確認を取り、実行する。
- ESS が選別になっていない原因（尤度の温度、リサンプリング条件、G_style）を調べ、vanilla との比較を同条件で測る。
- 「同一節数」が要るなら、EOS を許さず節数を揃える方式（または節数で正規化した指標）を決める。
