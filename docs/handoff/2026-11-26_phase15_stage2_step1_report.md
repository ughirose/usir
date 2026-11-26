# Phase 1.5 Stage 2 Step 1 レポート（2026-11-26）

ブランチ: `feat/phase15-stage1-cache`（main へのマージなし）
実装コミット: `eb031f4b5c6fcd24fa6c1ef7f5950a203f377749`（テスト・parity・シナリオはこのコミット時点の作業ツリー、クリーン状態で実行）

## 1. 実装

### 1.1 L2 キー（pack バイト列）
`render.pack_key` = `canon_hash{layer:"L2", key_schema:"pack_bytes/1", instruction_version, model_id, sampling, pack_view}`。
- `pack_view` = pack から位置・メタ項目（`scene_id, scene_index, scene_count, scene_hash, output_path, l2_parts, l2_key`）を除き、`prior` を core（後述）に置換したもの。pack の他の全フィールド（指示ハッシュ、roster、roster_all_ids、ledger_baseline、scene_text、ローカル発話等）は 1 バイトの変更でキーが変わる。
- 成分ハッシュの組合せ方式は廃止。`l2_parts` はデバッグ・parity スタブ用に残存（キーには不使用）。
- `model_id` / `sampling` は環境変数 `SENSOR_ANNOTATOR_MODEL_ID` / `SENSOR_ANNOTATOR_SAMPLING`（既定 `unspecified` / `default`）。
- スキーマ `sensor.cache.l2/2`。

### 1.2 prior_state v2
- core = `status`, `present`（ソート済み集合）, `pov_character`, 任意で `location_id`。
- `location_id` は `locations/<work_id>.json`（手作業キュレーション）がある作品のみ。値は リストID / `"none"`（場所なし。正当な選択）/ `"unlisted"`（リスト外。互いに区別しない）。正規化は NFKC + 空白・記号除去。リストなしの作品は location をキーに含めない。自由記述 location は常にキー外（annotator には表示される）。
- time / trait_state / recent_facts / prev_scene_index はキー外（annotator には表示される）。
- リストは実測データから自動生成しない。本番用リストは同梱していない（`locations/README.md` に形式のみ記載）。

### 1.3 facts と「前提変化あり」記録
- facts はキーに入れない。L2 ラッパーの `parts.prior_facts_digest`（`prior.recent_facts` の canon_hash）を保存し、hit 時に現在の前提と比較。
- 差があれば sensor_runner 側の `cache/stale/<work>.json`（`premise_changed`）に記録。`nishiki.diagnose/3` と sensor.v2 は不変（merge 結果のレコードにフラグなし。テストで確認）。prepare / merge のレポートには `premise_changed` 件数を出す。
- 一覧と再注釈: `cli.py stale [--work ID] [--reannotate [--scene N ...]]`。再注釈は明示指定のみ（`forced` に登録 → 次の prepare がその pack を出力 → ingest で解除）。自動再検証なし。
- 履歴不明のエントリ（Stage 1 移行分）は digest = None で、フラグ対象外。

### 1.4 ledger epoch 凍結
- pack 生成・キー計算・解決は committed ledger（`ledger/<work>.traits.json`）のみ使用。
- ingest の `ledger_additions` は `ledger/<work>.pending.json` に記録（先勝ち、committed は上書きしない）。検証・merge は effective（committed + pending）を使用。
- `cli.py epoch status|commit|reset`。`commit` で pending を一括適用。最大 2 回（`MAX_EPOCHS=2`）。3 回目は `cutoff`（未適用の追加と未解決シーンを報告、終了コード 1）。`reset` で新しい run のカウンタに戻す。

### 1.5 pack の ledger 表示（項目5）
現行 pack を確認: `ledger_baseline` はシーンに可視なキャラ（roster 絞り込み後）のみ。global（instructions）は常にキーに入る。別名・同一性の変更は roster 側の変更として扱われ、可視キャラの別名変更はキーが変わる。pack 形式は変更なし（キー構成のみ変更）。`roster_all_ids` はキーに含まれるため、キャラの追加・削除は全シーンのキーを変える（意図的・要判断、§6）。

## 2. 測定結果（完了条件別）

| 条件 | 結果 |
|---|---|
| (a) 既存59件 | 合格。既存 59 + 新規 13 = **72 passed**（`pytest tests/test_sensor_runner.py tests/test_sensor_runner_phase15.py tests/test_sensor_runner_step1.py`、コミット `eb031f4`、作業ツリーclean、16 s）。既存テストファイルは無改変。 |
| (b) property テスト | 新規 `python/tests/test_sensor_runner_step1.py` 13 件すべて合格（下記） |
| (c) Stage 1 parity | `parity.py` 000035_000301（7シーン）を 同一/state_change × tail/mid の4条件で実行、全条件で annotations bytes・diagnose bytes・cold(L1 off) 一致、未編集再実行 0 call。結果 JSON: `docs/handoff/data/phase15_step1_parity_*.json` |
| (d) 1シーン編集 = 1 call | 同一状態編集: warm 1 call / cold 7 call（tail・mid とも）。state_change 編集: warm 2 call（編集シーン + 次シーンの present/pov が変わるため）/ cold 7 call |
| (d) epoch 凍結 | 下記シナリオ |

### (b) テスト対応
- 隠れキャラの ledger 変更 → キー不変 / 可視キャラ・関係（roster 属性）・別名・global（INSTRUCTIONS 変更）→ キー変更: `test_hidden_character_ledger_change...`, `test_roster_alias_and_attribute_changes...`, `test_global_instruction_model_and_sampling...`
- pack 1 バイト変更 → キー変更（全キー対象フィールドを総当たり。除外項目は不変）: `test_one_byte_change_of_any_keyed_field...`, `test_pack_file_bytes_reproduce_the_stored_key`
- facts のみ変更 → キー不変 + 前提変化セット、明示再注釈で解除: `test_facts_change_keeps_key_and_marks_premise_changed...`, `test_premise_check_does_not_touch_the_diagnose_input_schema`
- location リストなし作品 → 表記変更でキー不変 / リストあり作品 → `location_id` 反映、`none` は正当値、`unlisted`: `test_prior_core_is_present_and_pov_only...`, `test_location_id_is_keyed_only_for_works_with_a_hand_curated_list`
- epoch 凍結・commit・cutoff・先勝ち、Stage 1 エントリ移行: 残りのテスト

### (d) epoch 凍結シナリオ
方法: 4 シーンの合成作品（シーン 2 のみ ghost を `ledger_additions` で新規追加するスタブ annotator、pack の関数として決定的）。「ingest した pack 数 = LLM call 数」。同一スクリプト（scratchpad `epoch_scenario.py`）を Stage 1 コミット `bc6452d`（一時 worktree、実行後削除）と本実装で実行。

| | 最初の注釈ループ（pending=0 まで）の call 数 | prepare 周回 |
|---|---|---|
| Stage 1 (`bc6452d`) | 6（4シーン + ledger 変更で再注釈 2） | 7 |
| Step 1 | 4（全シーン 1 回ずつ） | 5 |
| Step 1 の epoch 1 commit 後 | +1（シーン 2 のみ。ghost が可視のため）、以後 pending=0 | — |

限界: 4 シーンの合成データ。実 LLM は使用していない。実作品での差は未測定（再注釈になるシーン数は ghost 等の追加キャラが可視なシーン数に依存）。

## 3. 旧キー（Stage 1 `sensor.cache.l2/1`）エントリの扱い
- 旧キーを再計算して一致したエントリのみ、新 v2 キーで採用（source `migrated-v1`、`prior_facts_digest=None`）。不一致の旧エントリは hit として読まない（ledger 変更等）。
- seed キャッシュ 35 件を移行済み（`cli.py migrate`: 7 作品 hit=35, miss=0, blocked=0）。v2 ファイル 35 件を新規コミット。旧 v1 ファイル 35 件はそのまま残置（明示 gc まで削除しない）。

## 4. 変更ファイル
- 新規: `scripts/sensor_runner/render.py`, `epoch.py`, `locations.py`, `locations/README.md`, `python/tests/test_sensor_runner_step1.py`, `docs/handoff/data/phase15_step1_parity_{same,state_change}_{tail,mid}.json`, `cache/l2/*/*.json`（v2 移行 35 件）
- 変更: `cache.py`, `resolve.py`（全面書換）, `prepare.py`, `ingest.py`, `merge.py`, `cli.py`, `keys.py`（スキーマ定数のみ）, `common.py`（モデルID/サンプリング定数）
- 無変更: L1 関連、`parity.py`、step0/step05 ツール、既存テスト、`nishiki.diagnose` 側

## 5. 守った制約
閾値・数式・出力スキーマ変更なし。L1 と L3 は未変更。location リストの自動生成なし。main への push / merge なし。

## 6. 残課題・判断が必要な点
1. `roster_all_ids` がキーに入るため、キャラ 1 人の追加・削除で作品の全シーンが再注釈になる。隠れキャラ ledger 変更の分離（項目5）とは別問題。roster のうち可視キャラのみキー化する案は、annotator に見せる情報を変えるため未実施。判断を要する。
2. 位置・メタ項目（scene_index 等）をキーから除外した（Stage 1 の「シーン位置はキーに入れない」不変条件に合わせた）。pack には含まれる。
3. `model_id` 既定が `unspecified`。実運用で環境変数を設定しないとモデル切替がキーに反映されない。
4. 本番 location リストは未作成（人手キュレーション待ち）。リスト無し作品では location 由来の不一致を前提変化検知もしない（free-text location はキー外・digest 対象は facts のみ）。
5. state_change 編集は warm 2 call になる（次シーンの present/pov が core に入るため）。想定内だが、1 シーン編集 = 1 call が成り立つのは編集が present/pov に影響しない場合に限る。
6. epoch 凍結中は pack 生成は committed ledger、検証・merge は effective ledger。このため epoch 1 の間に追加されたキャラの baseline は後続シーンの pack に出ない（epoch 2 で解消）。実 LLM では epoch 2 に再注釈されたシーンの結果が変わりうるが、未測定。
7. 前提変化の検知精度（facts digest は typed でも 0〜12% しか一致しない Step 0/0.5 の知見）: 実データでは「前提変化あり」が多数立つ可能性がある。件数は未測定。
8. 実 LLM を使った Step 1 の end-to-end 測定は未実施（スタブのみ）。
