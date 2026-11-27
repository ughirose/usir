# Phase 1.5 Stage 2 Step 10 レポート（2026-11-27）

Stage 2 の最終ステップ。(a) Step 9 の内容を `feat/sensor-runner` へ fast-forward で統合、(b) `SENSOR_ANNOTATOR_MODEL_ID` の運用ルールを文書に明記。物差し、出力スキーマ、エポック設計、鍵の仕様、版番号、L1、L3 は変更していない。model_id を unspecified に偽る変更なし。seed の再注釈・差し替えなし（注釈呼び出し 0 回）。main へのマージなし。

## 1. 統合の手順と確認結果
1. 分岐なしの確認: `origin/feat/sensor-runner`（76d1e45）は `feat/phase15-stage2-step9`（8708b5d）の祖先（`git merge-base --is-ancestor` が真）。差分コミットは 3 件（45454ae WIP、b8afa9e 修正、8708b5d レポート）。
2. 統合先端 `8708b5d6f435f2cffd92ba844c7370958b31488c`（detached checkout、作業ツリー clean、環境変数 `SENSOR_ANNOTATOR_MODEL_ID` 未設定）で確認:
   - 全テスト（test_sensor_runner*.py 7 ファイル、既存 6 + step9 の 1）: **90 passed**。
   - パリティ 4 条件（`parity.py`、000035_000301、same/state_change × tail/mid）: すべて annotations bytes・diagnose bytes・cold(L1 off) 一致、未編集再実行 0 call。
3. fast-forward push: `git push origin 8708b5d:refs/heads/feat/sensor-runner`（強制なし）が `76d1e45..8708b5d` として受理された。
4. GitHub API で照合: `feat/sensor-runner` = `feat/phase15-stage2-step9` = `8708b5d6f435f2cffd92ba844c7370958b31488c`（一致）。
5. 運用文書の追加（§2）とレポートを `feat/phase15-stage2-step10` に積み、そのコミットも同じ手順で `feat/sensor-runner` へ fast-forward する（最終先端と照合結果は §4）。なお step10 ブランチは WIP コミットを 76d1e45 から切ったため、8708b5d を取り込むマージコミット（3b716e1、STATUS_phase15_stage1.md の追記衝突を両方残して解消）を含む。

## 2. 運用ルールの明記
- 場所: `scripts/sensor_runner/SESSION.md` の「前提」節の直後に小節「注釈者のモデル ID（`SENSOR_ANNOTATOR_MODEL_ID`）— 必ず守る」を追加（8 行）。
- 内容: 既定値は seed の宣言値（`common.DEFAULT_ANNOTATOR_MODEL_ID`）で未設定でも seed に当たる。seed と別のモデルで注釈する場合は、注釈開始前に実際のモデル ID を必ず設定する（忘れると別モデルの注釈が seed と同じ model_id で記録され、鍵の上で区別できない）。prepare / ingest / merge は同じ値で実行する。`unspecified` など偽の値にしない（実 ID が取れない場合はその旨を値に含める）。seed の再注釈は別作業。
- コミット: `164c8ad26d989126865b3377a8182ccb61794f35`。

## 3. 完了条件の実測（テスト・パリティ実行コミット）
- 統合先端 `8708b5d`: テスト 90 passed、パリティ 4 条件合格（§1-2。環境変数未設定）。
- 運用文書追加後のコミット `164c8ad26d989126865b3377a8182ccb61794f35`（作業ツリー clean、環境変数未設定）: テスト **90 passed**、パリティ 4 条件合格（結果: `results/phase15_stage2_step10_parity_{same,state_change}_{tail,mid}.json`、いずれも 3 項目一致・未編集再実行 0 call）。
- 注釈呼び出し: 0 回。

## 4. 最終先端の照合
- 最終先端 `437535ca437dc55bcd19ed98fd473727f1b995bf`（164c8ad との差分はレポート・STATUS・パリティ結果データのみ、コード・seed・文書本文の変更なし）。
- `git push origin 437535c:refs/heads/feat/sensor-runner`（強制なし）が `8708b5d..437535c` の fast-forward として受理された。
- GitHub API で照合: `feat/sensor-runner` = `feat/phase15-stage2-step10` = `437535ca437dc55bcd19ed98fd473727f1b995bf`（一致）。
- 統合前後の `feat/sensor-runner` 先端: 76d1e45 → 8708b5d → 437535c。

## 5. 変更ファイル
- `scripts/sensor_runner/SESSION.md`（運用ルール追加）
- `docs/handoff/STATUS_phase15_stage1.md`、本レポート
- `scripts/sensor_runner/results/phase15_stage2_step10_parity_*.json`（4 件）
- コード・seed・テストの変更なし（Step 9 までの変更を統合したのみ）。

## 6. 残った問題・判断を仰ぎたい点
1. 注釈者の実モデル ID は未確認のまま（seed の宣言値は `alias=sonnet(exact-model-id-unverified)`）。
2. 運用ルールは文書による規則であり、環境変数の設定忘れを機械的に検出する仕組みはない。検出が必要なら実装変更（例: 既定値のまま新規 ingest する場合に警告を出す）になるため、本ステップでは行っていない。
3. 統合後の `feat/sensor-runner` から main へのマージは未実施（指示どおり）。
