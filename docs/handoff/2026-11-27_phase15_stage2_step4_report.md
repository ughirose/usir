# Phase 1.5 Stage 2 Step 4 レポート（2026-11-27）: 捨てたベクトルの記録、実 LLM 確認（A）、roster_changed 実測（B）、location 再注釈（C）

ブランチ: `feat/phase15-stage2-step4`（土台 `feat/phase15-stage2-step3` caa65c5、main へのマージなし）
実装コミット: `dabfadc98ac2c824e4faf003a8f03590a38a2087`。テストはこのコミットと、データ・測定ツール追加後の `222579c31b47db96e04a5826f7c58eb826a118d4`（いずれも作業ツリー clean、実装コードは同一）で実行。
結果データ: `scripts/sensor_runner/results/phase15_stage2_step4_results.json`、`phase15_stage2_step4_parity_*.json`（4 件）

## 0. 注釈者・呼び出し数
- 注釈者: 独立した fresh-context の Sonnet サブエージェント（1 pack = 1 エージェント、Step 0/2 と同方式）。`SENSOR_ANNOTATOR_MODEL_ID = claude-code-subagent:alias=sonnet(exact-model-id-unverified)`。**正確なモデル ID は未確認**（alias=sonnet のみ。実体の版は不明。Step 2 と同じ識別子）。
- 呼び出し合計 **11 / 上限 30**（A 6, B 1, C 4）。上限到達なし。所要時間は起動〜ingest 完了の壁時計（並列・ツール遅延を含む）。

## 1. (a) 実装: 捨てられた 2 回目のベクトルの記録
- 挙動は変更なし（先勝ち）。pending の人物を、**異なるベクトル**で再追加した場合のみ記録する（同一ベクトルの再追加は日常的に起きるため記録しない）。
- 置き場所: ①pending ファイル `cache/ledger/<work>.pending.json` の `discarded`（epoch 内）②`epoch commit` 時に epoch 履歴 `<work>.epoch.json` の `history[].discarded_repeats` へ移し、commit の戻り値 `discarded_repeats` にも載せる。③`ingest_file` の戻り値 `ledger_additions_discarded`（その ingest で新たに捨てたもの）。
- 記録内容: `{character, scene_hash, vector（捨てたもの）, kept_vector, kept_scene_hash}`。
- 変更: `cache.py`（`pending_discarded`、`_write_pending` と `add_pending`）、`epoch.py`（`commit_epoch`）、`ingest.py`（戻り値に 2 行）。鍵・出力スキーマ・エポック設計・L1/L3 は無変更。
- 回帰テスト: `python/tests/test_sensor_runner_step4.py` 2 件（先勝ちで 1 回目採用、2 回目の別ベクトルが記録に残る／同一ベクトルは記録しない／commit で履歴へ移る／捨てがなければ記録なし）。全件合格。
- **実 LLM では未発火**: 測定 A の注釈者は再追加を行わなかったため、実データでの記録は 0 件（テストのみで確認）。

## 2. テスト・パリティ
- 既存 84 件 + 新規 2 件 = **86 passed**（18.5 s）。コミット `dabfadc…`（および `222579c…`）、作業ツリー clean。
- パリティ（`parity.py` 000035_000301、same/state_change × tail/mid）: 4 条件すべてで annotations bytes・diagnose bytes・cold(L1 off) 一致、未編集再実行 0 call。

## 3. 測定 A: `ep` 条件（001801_001786、committed ledger 空、3 場面、実 LLM）
| 項目 | 結果 |
|---|---|
| epoch 1 の呼び出し | 3（場面 0,1,2 を各 1 回）。試行 5 回（1+2+2） |
| epoch 1 の reject | **2 件**。ただし `baseline already exists` による重複追加の reject は **0 件**（Step 2 では 3 件）。2 件は場面 1・場面 2 の初回試行が `trait_changes[..]: no ledger baseline for 'shinichi'/'onna'`（場面 0 で pending 済みの人物を `ledger_additions` に入れずに `trait_changes` に使った）。2 試行目で ACCEPT |
| epoch 1 commit | [onna, shinichi] 適用、`discarded_repeats` = 0 |
| epoch 2 | 再注釈は全 **3 場面**、呼び出し **3**、全て初回 ACCEPT |
| 収束 | 収束（epoch 2 後 prepare で 0 pack、commit は `converged`）。最終 merge errors 0、全 L2 hit、フラグ 0 |
| 時間 | 注釈 1 件 中央値 15.48 s / 平均 15.87 s / 範囲 12.8–19.44 s（n=6） |

解釈: 修正（Step 3）で「重複追加の reject」は消えたが、同じ設計判断の裏側（凍結 ledger しか見えない注釈者が、pending 済みの人物を追加せず使うと reject）が実 LLM で 3 場面中 2 場面に出た（Step 3 レポート §5-1 の予想どおり）。再試行で解消しており総 call 数には影響しない（試行は 2 回余分）。

## 4. 測定 B: roster_changed の実データ（作品 000081_000456、6 場面）
- 方法: seed 7 作品の cold キャッシュ（Step 2）をサンドボックスにコピー。人物ファイルは作業コピー（`step2_work/s4b/works/`、環境変数 `STEP2_WORKS_DIR` で切替）にのみ追加。追加人物: `megane`（別名「眼鏡」）。本文中の「眼鏡」は場面 3 のみ（作品内で他の場面には出ない）。コミット済み seed データは無変更（`git status` clean で確認）。
- 追加前（warm）: pack 0、6 場面すべて hit、0 call。
- 追加後（warm）: 再注釈 **1 場面（場面 3 のみ = 現れる場面だけ）**、呼び出し 1（初回 ACCEPT、15.57 s）。再注釈されなかった場面 **5**（場面 0,1,2,4,5）。
- フラグ: roster_changed の記録が立った場面 **5**（0,1,2,4,5。再注釈した場面 3 は現在の一覧で保存されるため記録なし）。premise_changed **1**（場面 4）。
- 出力の差: 追加前後の annotations で変化したのは場面 3 のみ。
- premise_changed（場面 4）の分類（目視）: **実質的な差**。保存側の直近事実「ここは百二十万年前は海岸だった」（LEARNED）が、現在側では「くるみは百二十万年ぐらい前のもの」「獣の骨はボスで今の牛の先祖」（LEARNED）に変わった。場面 3 の再注釈で内容が変わった結果。自動再注釈はしない設計のため、場面 4 は古い前提のまま。n=1。
- 限界: 1 作品・1 人物・1 回。追加人物の別名が本文に 1 場面にしか現れない条件。ledger_additions は出なかった（epoch 2 は発生せず）。

## 5. 測定 C: 001875_059897 の location 再注釈（4 場面を各 1 回、実 LLM）
- 方法: cold キャッシュのコピーから当該 4 場面の L2 を削除し、通常手順で 4 call。
- 結果: 呼び出し 4、全て初回 ACCEPT（時間 中央値 15.61 s / 範囲 13.53–18.01 s）。location は**4 場面とも null のまま**（再注釈前: null×4）。変化 0。
- 解釈: Step 3 の結論（本文に場所を表す逐語句が主な場所として無く、null は指示どおり）と整合。再注釈でも揺れなかった（各場面 n=2: Step 2 の 1 回目 + 今回）。

## 6. 変更ファイル
- 変更: `scripts/sensor_runner/cache.py`、`epoch.py`、`ingest.py`、`step2.py`（測定ツール: 環境変数 `STEP2_WORKS_DIR` で作品ファイルの置き場を切替）、`docs/handoff/STATUS_phase15_stage1.md`
- 新規: `python/tests/test_sensor_runner_step4.py`、`scripts/sensor_runner/results/phase15_stage2_step4_*.json`、本レポート
- 無変更: 鍵の仕様、L1、L3、物差し、出力スキーマ（nishiki.diagnose/3、sensor.v2）、エポック設計

## 7. 残った問題・判断を仰ぎたい点
1. 凍結 ledger しか見えない注釈者が pending 済みの人物を追加せずに使うと reject される（A で 3 場面中 2 場面）。再試行で収束するが、プロンプト（instructions）で「見える ledger に無い人物は毎回 ledger_additions に入れる」と明示するか、検証側で pending 人物の `trait_changes` を許すか（エポック設計への影響を要判断）。
2. (a) の記録は実データで未発火。ベクトルが食い違う実例は未観測。
3. B で場面 3 の再注釈により場面 4 の前提が実質変化したが、自動で検知・再注釈されない（premise_changed の記録のみ。設計どおり）。
4. 測定は各 1 回（B・C は n=1 作品／n=4 場面）。注釈者の揺れの大きさは未測定。

## 8. 守った制約
物差し・出力スキーマ・エポック設計・鍵の仕様を変更していない。`premise_changed` の自動分類は追加していない（上記の分類は私の目視で、機能ではない）。L1・L3 無変更。seed データ無破壊。main へのマージなし。
