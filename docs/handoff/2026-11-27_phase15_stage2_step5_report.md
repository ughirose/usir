# Phase 1.5 Stage 2 Step 5 レポート（2026-11-27）: 注釈者への指示の明示と feat/sensor-runner への統合

ブランチ: `feat/phase15-stage2-step5`（土台 `feat/phase15-stage2-step4` 7ce179c）
指示変更コミット: `c4d367fc56060de77144c81a971471fe99a60fe8`。テスト・パリティ実行コミット: `fd9f28e389de88ed6858eb680eda3a9cf79f015f`（作業ツリー clean、実装コードは c4d367f と同一）。統合先端での再確認は §3。
結果データ: `scripts/sensor_runner/results/phase15_stage2_step5_results.json`、`phase15_stage2_step5_parity_*.json`（4 件）

## 1. (a) 注釈者への指示の明示
- 変更: `scripts/sensor_runner/strong.py` の `INSTRUCTIONS`（注釈者に渡る指示文）に 1 行追加。検証側（ingest）・エポック設計・鍵の仕様・出力スキーマは無変更。
- 追加した文言: 「pack の台帳（ledger_baseline）に無い人物は、他の場面で既に追加済みかもしれない場合でも、毎回 ledger_additions に入れること。重複は先勝ちで吸収される。」（指示の文面の「pack の台帳」は pack の `ledger_baseline` と明記した。）
- 副作用の確認: `instructions` の本文は pack に含まれるため新規の L2 鍵に入る（`instruction_hash` も変わる。`INSTRUCTION_VERSION` は据え置き）。コミット済みの seed キャッシュは、コピー上の確認で全 7 作品 35 場面が hit のまま（旧形式 l2/1 エントリの移行は `INSTRUCTION_VERSION` 基準で、文面変更では外れない）。つまり seed の既存注釈は旧指示文で作られたものが、新指示文のもとでも再利用される。新規に注釈する場面は新指示文を使う。**判断を仰ぎたい点（§5-1）**。
- 実 LLM 再実行（Step 2/4 と同条件: 001801_001786、committed ledger 空、3 場面）:
| 項目 | 結果 |
|---|---|
| epoch 1 の呼び出し | 3、全て初回 ACCEPT（試行 3） |
| epoch 1 の reject | **0 件**（Step 4 では 2 件） |
| epoch 1 commit | [onna, shinichi] 適用 |
| epoch 2 | 全 3 場面を再注釈、3 call、全て初回 ACCEPT |
| 収束 | 収束（epoch 2 後の commit は `converged`）。最終 merge errors 0、全 L2 hit、フラグ 0 |
| 呼び出し合計 | 6 / 上限 10。注釈 1 件 中央値 15.64 s / 平均 15.39 s / 範囲 13.65–17.13 s |
- 注釈者: 独立した fresh-context の Sonnet サブエージェント。`SENSOR_ANNOTATOR_MODEL_ID = claude-code-subagent:alias=sonnet(exact-model-id-unverified)`（正確なモデル ID は未確認、alias のみ。Step 2/4 と同じ）。
- Step 4 の (a)（捨てたベクトルの記録）が実データで初めて発火: 注釈者は場面 1・2 でも onna / shinichi を再追加したが、**ベクトルが場面ごとに異なった**ため、先勝ち（場面 0 のもの）で 4 件が `discarded_repeats` に記録された。例: shinichi 場面 0 `[1,0,0,1,0,0]`（採用）／場面 1 `[0,0,0,1,0,0]`／場面 2 `[0,0,0,0,0,0]`（捨て）。onna も同様（採用 `[-1,0,0,-1,1,0]`）。**§5-2**。
- 限界: 1 作品・3 場面・1 回。

## 2. (b) feat/sensor-runner への統合
1. 子孫確認: `git merge-base --is-ancestor origin/feat/sensor-runner HEAD` が成功（分岐なし）。統合前の `feat/sensor-runner` = `33bf69c0bd08cb1f70026d1bed65494785f9fcfa`、`feat/phase15-stage2-step4` = `7ce179c0fb9803a3f8f699b93ca778521a7ea9e2`（33bf69c の子孫）。
2. 統合対象の解釈: 指示は「step4 先端 7ce179c0 までの内容」を fast-forward とし、「統合後に feat/sensor-runner と作業ブランチの先端コミットが一致」を要求している。(a) の指示変更は step5 ブランチにあり、7ce179c0 だけを統合すると (a) が sensor-runner に入らない。このため **step5 ブランチの先端**（7ce179c の子孫、7ce179c までの全内容を含む）を fast-forward で統合した。
3. 確認と push: §3。強制 push なし。

## 3. 統合先端での確認
（統合先端のコミットハッシュ・テスト・push 結果・GitHub 上の先端照合は、push 後に追記する。→ 末尾 §6）

## 4. テスト・パリティ
- 全テスト: 86 passed（`test_sensor_runner*.py` 6 ファイル、18.1 s）。コミット `fd9f28e389de88ed6858eb680eda3a9cf79f015f`。(a) は指示文のみの変更のため新規テストはなし（指示文の挙動は実 LLM 測定で確認）。
- パリティ（`parity.py` 000035_000301、same/state_change × tail/mid）: 4 条件すべてで annotations bytes・diagnose bytes・cold(L1 off) 一致、未編集再実行 0 call。

## 5. 残った問題・判断を仰ぎたい点
1. 指示文の変更が新規 L2 鍵に影響する（`INSTRUCTION_VERSION` 据え置きで、旧 l2/1 由来の seed 注釈は再利用され続ける）。seed を新指示文で再注釈する／`INSTRUCTION_VERSION` を上げて鍵を分離する、という選択はしていない。新旧の指示文の注釈が同一キャッシュに混在する。
2. 先勝ちの採用ベクトルは「その場面での初見の印象」。同一人物に場面ごとに異なるベクトルが付く（上記の実例）。epoch の設計どおりだが、場面 0 の値が常に採用される点が妥当かは未評価。
3. 測定は各 1 回（n=1）。注釈者の揺れは未測定。

## 6. 統合結果（push 後に追記）
