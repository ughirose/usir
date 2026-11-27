# Phase 1.5 Stage 2 Step 11 レポート（2026-11-27）

`feat/sensor-runner`（先端 a2fc742）の内容を `main` へ統合した（usir リポジトリのみ）。物差し、出力スキーマ、エポック設計、鍵の仕様、版番号、L1、L3 は変更していない。model_id を偽る変更なし。seed の再注釈・差し替えなし（注釈呼び出し 0 回）。nishiki リポジトリには触れていない。

## 1. 統合前後の main 先端
- 統合前: `f2454deb`（docs: correct post-Phase-49 GRPO CU estimate …、2026-11-08）。
- 統合後: `a2fc7420d2ae4cd2ad06840ee9bf203784ca2a09`（= `feat/sensor-runner` の先端）。

## 2. 統合方法と競合
- 関係の確認: `git merge-base --is-ancestor origin/main origin/feat/sensor-runner` が真。merge-base は `f2454deb`（main 自身）。main 側のみのコミット 0 件、sensor-runner 側のみのコミット 109 件。
- 方法: **fast-forward**（merge commit なし、履歴の書き換えなし）。`git push origin a2fc742:refs/heads/main`（強制なし）が `f2454de..a2fc742` として受理された。
- 競合: **なし**（fast-forward のため発生しない）。解消作業なし。
- 補足: 確認の途中、1 回目のコマンドで `git rev-parse --short` の引数エラーにより祖先判定が実行されず「分岐」と誤表示した（コマンド連結の不備）。祖先判定を単独で再実行し、上記の結果（祖先、main 側のみ 0 件）を確認した。判定ミスのまま push はしていない。

## 3. 統合後の確認（環境変数 `SENSOR_ANNOTATOR_MODEL_ID` 未設定）
統合先端 = 実行コミット `a2fc7420d2ae4cd2ad06840ee9bf203784ca2a09`（detached checkout、作業ツリー clean、push 前に実行）:
- 全テスト（`python/tests` の test_sensor_runner*.py 7 ファイル）: **90 passed**。
- パリティ 4 条件（`parity.py`、000035_000301、same/state_change × tail/mid）: すべて annotations bytes 一致、diagnose bytes 一致、cold(L1 off) 一致、未編集再実行 0 call・warmup と一致。
- push 後、main の内容は a2fc742 と同一（push したコミットそのもの）であるため、同一ハッシュでのテスト結果をそのまま統合後の main に対する結果とする。
- 注釈呼び出し: 0 回。

## 4. GitHub での確認
- GitHub API: `main` = `feat/sensor-runner` = `a2fc7420d2ae4cd2ad06840ee9bf203784ca2a09`（一致）。
- 比較 API（`f2454deb...main`）: status = ahead、ahead_by = 109、behind_by = 0。
- 完了条件 1（main の先端に sensor-runner の内容が含まれる）を満たす。

## 5. 変更ファイル
- main への変更: `feat/sensor-runner` の 109 コミット分（Phase 1.5 の sensor_runner 一式、seed、テスト、docs/handoff）をそのまま取り込んだ。本ステップでの新規コード変更なし。
- 本ステップのコミット（`feat/phase15-stage2-step11` ブランチのみ、main には入れていない）: `docs/handoff/STATUS_phase15_stage1.md`（Step 11 開始・完了の追記）、本レポート。

## 6. 残った問題・判断を仰ぎたい点
1. main には、本レポートと STATUS 追記は含まれていない（main へ push するのは今回の統合のみ、という指示に従った）。main にも置く場合は別途 fast-forward 等の指示が必要。
2. 環境変数未設定でも seed に当たるのは、`SESSION.md` の運用ルールどおり、既定 model_id が seed の宣言値であるため。別モデルで注釈する場合は設定が必須（Step 10 の文書化済み。機械的な検出はない）。
3. 注釈者の実モデル ID は未確認のまま（alias=sonnet）。
4. main 上の他ブランチ（feat/fable-v5-shootout、docs/f7-diagnosis など）は今回触れていない。
