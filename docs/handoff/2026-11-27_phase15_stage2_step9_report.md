# Phase 1.5 Stage 2 Step 9 レポート（2026-11-27）

Step 8 で判明した問題（新 seed の L2 鍵に model_id が入っており、`SENSOR_ANNOTATOR_MODEL_ID` 未設定だと seed に当たらない）の修正。鍵の仕様、物差し、出力スキーマ、エポック設計、INSTRUCTION_VERSION、L1、L3 は変更していない。model_id を unspecified に偽る変更はしていない。seed の再注釈・差し替えはしていない（注釈呼び出し 0 回）。main へのマージなし。

## 1. 修正内容
- `scripts/sensor_runner/common.py`: 注釈者 model_id の既定値を `unspecified` から、新 seed の注釈者として宣言した値 `DEFAULT_ANNOTATOR_MODEL_ID = "claude-code-subagent:alias=sonnet(exact-model-id-unverified)"` に変更（3 行追加・1 行変更）。環境変数 `SENSOR_ANNOTATOR_MODEL_ID` を設定した場合は従来どおりそちらが優先される（明示指定の上書きは維持）。CLI・prepare・merge・parity はすべて `common.ANNOTATION_MODEL_ID` を参照するため、CLI 側の既定が揃う。
- テスト側: 環境変数を固定する conftest は入れていない。`python/tests` の既存テストは `common` を直接 import するため、CLI と同じ既定値（単一の定義元）がそのまま効く。conftest で別に値を固定すると、既定値の変更を隠してしまうため（回帰テストの目的に反する）。指示書の「テスト側に設定する仕組み」は、この共通既定値で満たしたと判断した。判断が異なる場合は §5 を参照。
- 回帰テスト `python/tests/test_sensor_runner_step9.py`（4 件）:
  1. 既定値が `unspecified` でなく、seed の全 L2 エントリの鍵部品 `model_id` と一致する。
  2. `step2.py` が宣言する MODEL_ID が既定値と同じ。
  3. 環境変数を除いた別プロセスで、seed の cache（一時コピー）に対し 35 場面すべてが hit、pending 0、pack 0 件。
  4. 明示的に `unspecified` を指定すると既定値は上書きされ、hit 0（鍵の仕様が変わっていないことの確認）。
  - 修正前の common.py に戻して実行すると 3 件が失敗することを確認した（4 件目のみ通る）。
  - テストはコミット済み seed を書き換えない（prepare が manifest を書くため、cache を一時ディレクトリへコピーして使う。初版でこれを怠り manifest が書き換わったため修正した。コミット前に検出済みで、コミット済みの seed に影響なし）。
- `scripts/sensor_runner/step9_warm.py`: 環境変数未設定での warm 実行確認用の測定スクリプト（cache を一時コピーして prepare / merge / diagnose）。

## 2. 完了条件の実測（テスト・パリティ実行コミット `b8afa9e3ab09f971834bb65ef99d02830391e732`、作業ツリー clean）
1. テスト（`python/tests` の test_sensor_runner*.py 既存 6 ファイル）:
   - 環境変数未設定: **86 passed**（Step 8 時点では同条件で 80 passed / 6 failed）。
   - 環境変数設定あり（上記の値）: 既存 6 ファイル + 新規 1 ファイルで 90 passed（= 既存 86 + 新規 4）。
   - 新規テスト単体（未設定）: 4 passed。
2. 環境変数未設定の warm 実行（`results/phase15_stage2_step9_warm.json`）: 使用 model_id は既定値、全 7 作品で prepare の pack 0 件（注釈呼び出し 0 回）、L2 は 35 場面すべて hit（adopted 0、miss 0、blocked 0）、merge errors 0、annotations と diagnose の sha256 が Step 8 の cold 実行と 7 作品すべて一致。
3. パリティ 4 条件（環境変数未設定、`parity.py`、000035_000301、same/state_change × tail/mid）: 4 条件すべてで annotations bytes 一致、diagnose bytes 一致、cold(L1 off) 一致、未編集再実行 0 call。結果: `results/phase15_stage2_step9_parity_{same,state_change}_{tail,mid}.json`。
4. 回帰テスト: 上記のとおり追加・合格。

## 3. 変更ファイル
- `scripts/sensor_runner/common.py`（既定値の変更）
- `python/tests/test_sensor_runner_step9.py`（新規）
- `scripts/sensor_runner/step9_warm.py`、`results/phase15_stage2_step9_warm.json`、`results/phase15_stage2_step9_parity_*.json`（4 件）
- `docs/handoff/STATUS_phase15_stage1.md`、本レポート
- seed（`cache/**`）は未変更。

## 4. 統合
本ステップの指示に `feat/sensor-runner` への統合は含まれていないため、作業ブランチ `feat/phase15-stage2-step9`（土台 `feat/sensor-runner` 76d1e45）への push のみ。統合の判断はユーザーに委ねる。

## 5. 残った問題・判断を仰ぎたい点
1. 既定 model_id が「seed の宣言値」になったため、別のモデルで新しく注釈する場合は `SENSOR_ANNOTATOR_MODEL_ID` に実際のモデル ID を必ず設定する必要がある。設定を忘れると、別モデルの注釈が seed と同じ model_id で記録される（鍵の上で区別できない）。運用ルールとして `SESSION.md` への明記が望ましいが、本ステップでは文書は変更していない。
2. 指示書の「テスト側で環境変数を固定する仕組み」は conftest を使わず、共通の既定値で実現した（§1）。conftest 方式を希望する場合は追加できる。
3. 注釈者の実モデル ID は未確認のまま（alias=sonnet）。鍵の model_id にはこの宣言文字列が入っている。
