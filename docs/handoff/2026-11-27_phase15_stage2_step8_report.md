# Phase 1.5 Stage 2 Step 8 レポート（2026-11-27）

seed 7作品・全35場面を、現行の指示文（Step 5 で1行追加した版）と現行の手順で cold 再注釈し、コミット済み seed を差し替えた。物差し、出力スキーマ（nishiki.diagnose/3、sensor.v2）、エポック設計、鍵の仕様、INSTRUCTION_VERSION（strong.v1）、L1、L3 は変更していない。main へのマージなし。実装コードの変更なし。

## 1. 旧 seed の所在（復元方法）
- 差し替え前の seed は、コミット `22b2e26`（feat/sensor-runner の旧先端、本ステップの土台）に、そのまま残っている。`scripts/sensor_runner/cache/l2/**`（35 場面 × 4 ファイル = 140 ファイル）と `cache/manifests/*.json`（7 ファイル）。
- 復元: `git checkout 22b2e26 -- scripts/sensor_runner/cache/l2 scripts/sensor_runner/cache/manifests`（差し替え後のファイルを先に消す必要がある。l2 は旧ファイル名と新ファイル名が異なるため）。
- 旧 seed の注釈者表記は `claude-sonnet-5-5 (session model, manual annotation)`、provenance は `seed:hybrid-full-run@6043212`（Step 6 レポートの指摘どおり、根拠の言葉が無い手作業注釈）。
- 旧 seed の l2 は 1 場面あたり 4 ファイル（l2/1 移行分、l2/2 移行分、l2/3 移行分 ×2）。新 seed は 1 場面 1 ファイル（l2/3、source=ingest）。

## 2. 再注釈
- 方法: 新規コンテキストの Sonnet サブエージェント（model=sonnet を明示）、1 pack = 1 エージェント、作品内は場面順、作品間は並列（ラウンド内最大7件）。ledger はコミット済み seed のものを複製した sandbox（`step2_work/s8a`、git 管理外）で、L2 は空の状態から開始。production の prepare / ingest / merge を `scripts/sensor_runner/step2.py` 経由で使用。
- 注釈呼び出し: 合計 35 回（上限 50）。内訳はラウンド別 7 / 7 / 7 / 6 / 4 / 3 / 1。35 件すべて初回 ACCEPT（試行 1 回、reject 0 件）。
- 注釈 1 件の時間: 平均 16.5 秒、中央値 16.35 秒、最小 12.58 秒、最大 23.16 秒。合計 577.5 秒（各エージェントの start〜end の和。並列実行のため実時間はこれより短い）。
- ledger_additions はどの場面も空。ledger の更新（エポック commit）は発生していない。ledger ファイルは差し替え前後でバイト一致。
- SENSOR_ANNOTATOR_MODEL_ID: `claude-code-subagent:alias=sonnet(exact-model-id-unverified)`（step2.py が既定で設定）。サブエージェントの実モデル ID は取得できず、alias=sonnet 指定までが確認範囲。この値は L2 鍵の `model_id` に入る（後述の注意参照）。

## 3. 差し替え
- 全 35 場面がそろってから実施（途中で止まった場合は差し替えない運用。今回は全件完了）。`cache/l2/<scene_hash>/` の旧 4 ファイルを削除し、新 1 ファイルを置いた。`cache/manifests/*.json` も新注釈者の記録に更新。ledger と l1 は変更なし。
- コミット: `cd02ac0`。

## 4. 差し替え後の確認
- warm 実行（新 seed を複製し、production の prepare / merge / diagnose を実行）: prepare の出力 pack 0 件（呼び出し 0 回）、L2 は 35 場面すべて hit（adopted 0、miss 0、blocked 0）、merge の errors 0。annotations と diagnose の sha256 が cold 実行（再注釈直後の sandbox）と 7 作品すべて一致。作品別の値は `step2_work/s8w/warm/summary.json`（git 管理外）。
- パリティ 4 条件（`parity.py`、000035_000301、same/state_change × tail/mid）: 4 条件すべてで annotations bytes 一致、diagnose bytes 一致、cold(L1 off) 一致、未編集再実行 0 call。結果: `scripts/sensor_runner/results/phase15_stage2_step8_parity_{same,state_change}_{tail,mid}.json`。
- 既存テスト 86 件（`python/tests` の test_sensor_runner*.py 6 ファイル）: `SENSOR_ANNOTATOR_MODEL_ID` を上記の値に設定して 86 passed。実行コミットは `3fc4e79fbff3ec0a2fa8bee20f20a1bfcd504528`（作業ツリー clean）。
- **注意（未解決）**: 同じコミットで環境変数を設定せずに実行すると 80 passed / 6 failed（test_seed_prepare_delta_and_diagnose_all_lenses_ok、test_committed_seed_cache_is_fully_strict_keyed、test_warm_partial_recompute_equals_cold_full_recompute 4 件）。原因: seed の L2 鍵に `model_id`（上記の値）が入っており、環境変数が未設定だと既定値 `unspecified` で鍵を計算するため seed に当たらない。旧 seed は `unspecified` で鍵が作られていた。つまり新 seed を使う実行は、同じ `SENSOR_ANNOTATOR_MODEL_ID` の設定が前提になる。パリティ 4 条件も同様（未設定だと実行時に失敗した）。鍵の仕様は変更禁止のため、コードは変えていない。

## 5. 新旧 seed の比較（参考情報）
`scripts/sensor_runner/step8_compare.py`、結果 `results/phase15_stage2_step8_compare.json`。バイト一致 = annotator / provenance を除く正規化 JSON が完全一致。
- バイト一致場面数: 0/35。
- フィールド別の差異場面数（/35）: present 35、pov_character 0、location 13、facts 35、trait_changes 28、ledger_additions 0、time 19、event_top3 35、mood_top3 35、awase 35。
- 意味項目の差異場面数: present_ids 1、pov 0、location 13、event_codes 24、event_top1 14、mood_codes 23、mood_top1 15、trait_set 22、facts_count 23、awase4 25、relative_order 7。全意味項目一致の場面: 0/35。
- 例（000035_000301 場面 0）: facts 件数 4 → 3、awase の f_arousal 1 → 0、location は旧が短い語、新がその語に修飾が付いた文字列（文字列の不一致）。
- 読み取り: Step 6・7 の結果と整合する（旧 seed は形式が異なり、同じ方式の注釈同士でも揺れるため、差異の原因を指示文・方式・揺れに分けることはこの比較ではできない）。

## 6. 統合
- 事前確認: `origin/feat/sensor-runner`（22b2e26）は作業ブランチの祖先で、分岐なし。
- 統合先端 `673cf19`（cd02ac0・3fc4e79 の後にレポートと STATUS のみ追加、コード・seed の差分なし）で、`SENSOR_ANNOTATOR_MODEL_ID` 設定下に既存 86 件 passed を再確認。パリティ 4 条件は 3fc4e79 で取得済みで、673cf19 との差分は docs のみ。
- `git push origin 673cf19:refs/heads/feat/sensor-runner`（強制なし）が `22b2e26..673cf19` の fast-forward として受理された。
- GitHub API で照合: `feat/sensor-runner` = `feat/phase15-stage2-step8` = `673cf19bd84b9afaf15c36cd74a8b21d45aa4de4`（一致）。
- 注意: 統合後の `feat/sensor-runner` も、環境変数未設定ではテスト 6 件が失敗する（§4、§8-1）。

## 7. 変更ファイル
- `scripts/sensor_runner/cache/l2/**`（旧 140 ファイル削除、新 35 ファイル）、`cache/manifests/*.json`（7 ファイル更新）
- `scripts/sensor_runner/step8_compare.py`、`results/phase15_stage2_step8_compare.json`、`results/phase15_stage2_step8_parity_*.json`（4 ファイル）
- `docs/handoff/STATUS_phase15_stage1.md`、本レポート
- git 管理外: `step2_work/s8a`（sandbox、注釈・タイミング・試行ログ）、`step2_work/s8w`（warm 確認）

## 8. 残った問題・判断を仰ぎたい点
1. 新 seed を使うには `SENSOR_ANNOTATOR_MODEL_ID` を同じ値に設定する必要がある。未設定の既定環境（CI など）ではテスト 6 件が失敗し、実行時も seed に当たらず 35 場面が再注釈対象になる。対処案: (a) テスト・CLI 側で環境変数を固定する、(b) 旧 seed と同じ `unspecified` の model_id で鍵を作り直す（鍵の仕様は変えないが、model_id を偽る形になる）、(c) 現状のまま運用ルールとして明記する。実装変更になるため本ステップでは行っていない。
2. 注釈者の実モデル ID は未確認（alias=sonnet まで）。
3. 新旧 seed の差異の内訳（指示文・方式・揺れ）は未分離。
