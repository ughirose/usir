# Phase 1.5 Stage 2 Step 6 レポート（2026-11-27）: 2つの判断材料の検証実験（計測のみ）

ブランチ: `feat/phase15-stage2-step6`（土台 `feat/sensor-runner` 22b2e26、main へのマージなし）
テスト: 既存 86 件合格（実行コミット `76a71a0636af19517c75cc63697fbb7b75bea57e`、作業ツリー clean）。実装コード（`scripts/sensor_runner/*.py` の既存ファイル）は変更なし。追加は測定スクリプト 2 本（`step6_exp1.py`、`step6_exp2.py`）と結果データのみ。
結果データ: `scripts/sensor_runner/results/phase15_stage2_step6_exp1.json`、`phase15_stage2_step6_exp2.json`

## 0. 注釈者・呼び出し数
- 注釈者: 独立した fresh-context の Sonnet サブエージェント（1 pack = 1 エージェント）。`SENSOR_ANNOTATOR_MODEL_ID = claude-code-subagent:alias=sonnet(exact-model-id-unverified)`。正確なモデル ID は未確認（alias のみ）。
- 呼び出し **11 / 上限 20**（実験 1 のみ。実験 2 は LLM 不使用）。全件初回 ACCEPT。

## 1. 実験 1: 旧指示と新指示で注釈は変わるか
手順: seed 7 作品から `000035_000301`（7 場面、多い方）と `001798_056779`（4 場面、少ない方の一つ）の計 11 場面を選択。作業コピー（`step2_work/s6a`、seed の台帳のみ・L2 空）で新指示文により cold 注釈（場面順に前場面の注釈を前提として 8 ラウンド）。コミット済み seed キャッシュの注釈と場面ごとに突合（メタ項目 `annotator`/`provenance` のみ除外）。seed データは無変更。
**重要な交絡**: seed の旧注釈の注釈者は `claude-sonnet-5-5 (session model, manual annotation)`（別セッション・別手順の手作業注釈）、今回は `claude-code-session:sonnet-subagent`（サブエージェント）。差は「指示文 1 行」だけではなく、注釈者・手順・prior の連鎖も含む。したがって下記の差は指示文の効果として分離できない。

| 項目 | 結果（11 場面） |
|---|---|
| バイト一致（メタ除外） | **0 / 11**（メタ込みでも 0） |
| 差異のあった場面数（フィールド別） | present 11 / pov 0 / location 3 / facts 11 / trait_changes 9 / ledger_additions 0 / time 8 / event_top3 11 / mood_top3 11 / awase 11 |
| 意味的な一致（値の集合・コード） | present の人物 id 集合 10/11、pov 11/11、location 8/11、trait_changes（人物・軸・変動値）2/11、facts の件数 4/11、event コード集合 2/11（先頭 4/11）、mood コード集合 6/11（先頭 8/11）、awase 4 値 5/11、time.relative_order 8/11 |
- present が 11 場面で「差異」なのは、旧注釈の `basis` が null（根拠語句なし）で、新注釈は本文の逐語の根拠語句を持つため（人物 id の集合では 10/11 一致）。facts も同様に旧は `basis` が null。つまり旧 seed は現行の strong.v1 の規則（basis 必須）に完全には沿っていない形式。
- ledger_additions の差異は 0（seed の台帳が全人物を含むため両方空）。**今回の指示文変更の対象（ledger_additions）に関する差異は観測されなかった**。
- 差異の実例（最大 3 件）:
  1. 000035_000301 場面 1 の location: 旧「仙遊館」→ 新「自分の部屋」。
  2. 000035_000301 場面 2 の location: 旧「銀座の或る大カフエ」→ 新 null。
  3. 000035_000301 場面 0 の event_top3: 旧 E14（逃げて、さすがに、いい気持はせず、死ぬ事にしました）/ E09 / E06 → 新 E14（逃げました）/ E02 / E10。
- 結論（この実験から言えること）: 新旧の注釈は項目によらず大きく異なり、バイト一致はない。ただし差の大半は注釈者・手順の違い（旧は basis なし、別セッション）に由来しうるため、**指示文 1 行の効果は分離できていない**。「seed を新指示で作り直す」ことによる内容の変化は、少なくとも event/mood/trait の選択では大きい。限界: 2 作品・11 場面・各 1 回、旧新で注釈者が異なる。

## 2. 実験 2: 先勝ちのベクトルで診断は変わるか
手順（LLM 不使用）: Step 5 結果データの epoch 履歴（採用値と `discarded_repeats`）から、場面 0/1/2 の onna・shinichi のベクトルを抽出:
- onna: 場面 0 `[-1,0,0,-1,1,0]`、場面 1 `[0,0,0,-1,0,0]`、場面 2 `[0,0,0,0,0,0]`
- shinichi: 場面 0 `[1,0,0,1,0,0]`、場面 1 `[0,0,0,1,0,0]`、場面 2 `[0,0,0,0,0,0]`
作品 001801_001786 の既存 3 注釈（Step 5 の epoch 2 後の L2）は固定し、本番の `merge_work` に渡す台帳基準だけを 3 通りに入れ替えて、`nishiki diagnose` を計算（測定スクリプト `step6_exp2.py`。鍵に使う台帳は元のまま）。
| 比較 | annotations | diagnose |
|---|---|---|
| 場面 0 基準 vs 場面 1 基準 | 差異あり（`values_6d` 12 要素、`ledger_hash`） | 差異 2 項目 |
| 場面 0 基準 vs 場面 2 基準 | 差異あり（18 要素） | 差異 2 項目 |
- 場面ごと（`scenes[]` 3 場面）の diagnose 出力は **3 通りで同一**。`anomaly`・`warnings`・他の 5 つの lens も同一。
- 変わったのは lens `sadame` の作品全体スコア 2 項目のみ:
  | 基準 | e_inter | total |
  |---|---|---|
  | 場面 0（現行） | 0.375 | 0.234375 |
  | 場面 1 | 0.166666667 | 0.104166667 |
  | 場面 2 | 0.0 | 0.0 |
- 解釈: 基準ベクトルの選び方（どの場面のものを採るか）で `sadame` の e_inter / total が 0.375 → 0.167 → 0 と変わる（場面 2 のゼロ基準では 0）。他の指標・場面別出力は変わらない。
- 限界: 注釈（trait_changes）は固定で、別の基準を見ていれば注釈者が別の変動値を書いた可能性は反映していない。1 作品・3 人物ベクトル、n=1。

## 3. 変更ファイル
- 新規: `scripts/sensor_runner/step6_exp1.py`、`step6_exp2.py`、`results/phase15_stage2_step6_exp1.json`、`phase15_stage2_step6_exp2.json`、本レポート
- 変更: `docs/handoff/STATUS_phase15_stage1.md`
- 無変更: 実装本体、物差し、出力スキーマ、鍵の仕様、`INSTRUCTION_VERSION`、seed データ

## 4. 残った問題・判断材料
1. 判断 1（seed の作り直し）: 旧 seed は basis が null の項目を含み、現行規則と形式が異なる。新旧差は大きいが、指示文 1 行の効果は未分離。分離するには「旧指示文で同じ 11 場面を同じサブエージェント方式で再注釈」して新旧を比較する（11 call 追加が必要。今回の上限 20 では残り 9 call のため未実施）。また新注釈同士の再実行差（揺れ）も未測定。
2. 判断 2（先勝ち基準）: 基準ベクトルの選択は `sadame` の作品全体スコアを動かす（上記）。場面別出力は不変。どの場面のベクトルを基準にするのが妥当かの基準は、この実験からは決まらない。
3. 本実験は各 1 回・少数場面。

## 5. 守った制約
実装・物差し・出力スキーマ・鍵の仕様・版番号を変更していない。seed データ無破壊（作業コピー上で実施）。main へのマージなし。
