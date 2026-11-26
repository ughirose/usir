# Phase 1.5 Stage 2 Step 2 レポート（2026-11-26）: 実 LLM による新 L2 鍵・epoch 凍結・フラグの実測

ブランチ: `feat/phase15-stage2-step2`（base `feat/sensor-runner` 33bf69c、main へのマージなし）
結果データ: `scripts/sensor_runner/results/phase15_stage2_step2_results.json`（コミット 58bcc377bbeef2ad41b2041a7bd3235f3fa9deb6）
測定ツール: `scripts/sensor_runner/step2.py`（実装本体は無変更。prepare_work / ingest_file / commit_epoch / merge_work / run_diagnose を呼ぶ駆動のみ）

## 0. 方法と制約
- 注釈者: 独立した fresh-context の Sonnet サブエージェント（1 pack = 1 エージェント、Step 0/0.5 と同方式）。`SENSOR_ANNOTATOR_MODEL_ID = claude-code-subagent:alias=sonnet(exact-model-id-unverified)`。**正確なモデル ID は未確認**（alias=sonnet 指定のみ。実体の版は不明）。
- サンドボックス: seed キャッシュはコピーして使用。コミット済み seed キャッシュは未変更。
- 所要時間はエージェント起動〜ingest 完了の壁時計（並列実行・ツール遅延を含む。LLM 純推論時間ではない）。
- 呼び出し数: 完了 44 + 429 で未出力 1 = 起動 45 / 上限 150。内訳 cold 35, edA 1, edB 2, ep 補助 6（epoch1 3 + epoch2 3）+ 失敗 1。

## 1. 結果
### A. cold（seed 7 作品 / 35 場面）
- LLM 呼び出し **35**（場面数と一致）。全件初回 ACCEPT（試行 35 = 呼び出し 35、reject 0）。
- 注釈 1 件の時間: 中央値 **16.13 s** / 平均 **16.46 s** / 範囲 **13.62–20.54 s**（n=33。2 件は計測欠損、§3-1）。
- 総時間: 7 ラウンド（pack 数 7,7,7,6,4,3,1）。ラウンド内は並列。最初〜最後のタイムスタンプ差 263 s（ラウンド間の私の操作待ちを含み、純処理時間ではない）。
- epoch: ledger_additions は 0 件（seed の ledger が全人物を既に含む）。全作品 `converged`、epoch 0。**メイン実行では epoch 2 は発生せず、収束挙動は未観測**。最終 merge は errors 0、全 L2 hit。
### B. warm
- LLM call **0**（pack 0、全場面 L2 hit）。7 作品の annotations / diagnose の sha256 が A と全て一致（14/14 バイト一致）。
### C. 1 場面編集（000035_000301 場面 3）
| 編集 | 再注釈 call | 内訳 |
|---|---|---|
| (i) present/pov に影響しない（末尾に風の描写） | **1** | 場面 3 のみ。場面 4–6 は hit（Early Cutoff） |
| (ii) 次場面の present/pov に影響（堀木が入ってくる） | **2** | 場面 3、続けて場面 4。場面 5–6 は hit |
### D. フラグ
- 全作品で roster_changed = **0**（実作品で人物一覧を変更していないため。ここは実 LLM 測定の対象外で、合成 stub の Step 1.5 結果のみが根拠）。
- premise_changed: cold = 0（作品別全 0）。edA 1（場面 4）、edB 1（場面 5）。計 2 件で全数分類した。
  - edA 場面 4: **表記ゆれ**。「銀座裏の屋台の鮨や」→「…のお鮨や」、「震えていた心が静まった」→「ツネ子の一言で震えていた心が静まり、安心した」。事実の骨格は同じ。ただし後者は追加語句があり、純粋な言い換えより一段大きい。
  - edB 場面 5: **実質的な差**。保存側 3 件（MOVED_TO 本所の大工さんの二階 / LEARNED 夫が詐欺罪で… / STATE_CHANGED）に対し現在側 2 件（LEARNED ツネ子の身の上 / STATE_CHANGED）。直前場面の再注釈で直近 5 件の窓の内容が変わった。
  - 限界: n=2。表記ゆれ対実質差の比率として一般化できない。
### E. location
- 返却率 **23 / 35 = 65.7%**（作品別: 6/7, 2/5, 5/6, 4/6, 4/4, 2/3, 0/4）。001875_059897 は 4 場面全て null。理由は未調査（推測: 場所が本文に明示されない作品、または固定の場所リストとの対応外。検証していない）。
### 補助: epoch 凍結の観測（run `ep`、001801_001786、ledger を空にして開始）
- epoch 1: 3 call（試行 3,2,1 回。reject 3 件）。commit 1 で [onna, shinichi] を適用。
- epoch 2: 全 3 場面が再注釈対象（cached 0 / pending 3）→ 3 call、全て初回 ACCEPT。commit 後 converged。最終 merge errors 0、全 hit、フラグ 0。
- つまり ledger 追加がある作品では、epoch 2 で当該作品の全場面を再注釈する（本件 100%）。

## 2. テスト
既存 80 件: 合格（`test_sensor_runner*.py` 4 ファイル、21.3 s）。実行時コミット **58bcc377bbeef2ad41b2041a7bd3235f3fa9deb6**（作業ツリー clean、実装コードは無変更）。

## 3. 発見（修正していない）
1. **測定ツールの競合**: `timing.jsonl` への並列 append で 2 件を取りこぼし（cold.001798_056779.s2.r3 の start、s3.r4 の end）。時間統計は n=33/35。attempts.jsonl は無欠損。
2. **epoch 凍結と検証の食い違い**: pack には committed（凍結）ledger が表示される一方、ingest 検証は effective（committed+pending）を使う。注釈者が pending 済みの人物を再度 ledger_additions に入れると `baseline already exists (no overwrite)` で reject される。再現: run `ep`（空 ledger）の epoch 1 で、場面 0 が onna / shinichi を追加 → 場面 1 の pack は凍結状態で両人物が baseline 未登録に見えるため再度追加 → 2 件とも reject（`attempts.jsonl` の `ep.001801_001786.s1.r2`）。実害は再試行（今回 reject 3 件、各 1 call 内）。
3. premise_changed は表記ゆれでも実質差でも立つ。自動再注釈しない設計のため、利用者が分類する必要がある（判別機能なし）。

## 4. 残った問題・判断点
- 実モデル ID が未確認。鍵に入る model_id はこの alias 文字列。実モデルが変わっても鍵が変わらない。
- epoch 2 の収束挙動（ledger_additions が実際に出る作品）は 1 作品 3 場面でしか観測していない。
- roster_changed の実 LLM 測定は未実施。
- 001875 の location null 4/4 の原因調査、2 の食い違いの扱いは Orion Gate の判断を仰ぎたい。

## 5. 守った制約
実装の修正なし、nishiki.diagnose/3・sensor.v2 無変更、seed キャッシュ無破壊、main へのマージ・push なし。
