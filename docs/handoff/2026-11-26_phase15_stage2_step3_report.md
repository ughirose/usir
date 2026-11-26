# Phase 1.5 Stage 2 Step 3 レポート（2026-11-27）: epoch 検証の食い違い修正と計測ツールの競合修正

ブランチ: `feat/phase15-stage2-step3`（main へのマージなし）
土台: `feat/phase15-stage2-step2`（先端 5a8de26）。指示の「feat/sensor-runner 先端 33bf69c」は Step 2 ブランチの祖先（`git merge-base --is-ancestor` で確認）。`step2.py` が Step 2 ブランチにのみあり不具合 B の修正対象のため、そこから分岐した。
実装コミット: `713d2fd55ed5b4a0356378f86e3250489e89c78f`（テストはこのコミット・作業ツリー clean で実行）

## 1. 修正内容
### 不具合 A: ingest の検証基準（`scripts/sensor_runner/ingest.py`）
- 変更前: `cache.ledger()`（committed + pending）で `validate_strong` を実行 → pending 済みの人物を再度 `ledger_additions` に入れると `baseline already exists (no overwrite)` で reject。
- 変更後: `cache.ledger_committed()`（pack の描画に使うのと同じ凍結スナップショット）で検証。pending 内の重複は検証を通り、`Cache.add_pending` の既存仕様（先勝ち：既に committed/pending の人物は無視）で吸収される。2 回目の追加のベクトルが異なっても採用されない（上書きなし）。
- 副作用（仕様どおり）: pack に出ていない pending 人物を `ledger_additions` なしで `trait_changes` に使うと「no ledger baseline」で reject される（pack の内容と検証基準の一致）。committed 済みの baseline の上書きは従来どおり reject。
- 変更行数: 実装 1 箇所（ledger 取得の置換＋コメント）。エポック設計（最大 2 エポック、終了時一括反映）・鍵・L1/L3・物差し・出力スキーマは無変更。

### 不具合 B: 計測ログの並列 append（新規 `scripts/sensor_runner/jsonl_append.py`）
- 原因: `open(p, "a")` + `write` は Windows で原子的でない（CRT が seek + write で模擬）。
- 修正: ロックファイル（`O_CREAT|O_EXCL`、再試行、10 s 超の古いロックは除去、Windows の delete-pending による `PermissionError` も再試行）の下で、1 回のバイナリ書き込み + fsync。`step2.py`（`jl_append`）と `step0.py`（timing / attempts）が共通で使用。`step05.py` は独自の append を持たない（step0 の関数を利用）。
- 修正前の欠損の実測: 同条件（8 プロセス × 40 行 = 320 行、5 試行）で旧方式は 42, 14, 16, 12, 21 行を欠損（破損行は 0）。修正後はテストで 320/320（3 回連続実行で全合格）。

## 2. 完了条件の実測
| 条件 | 結果 |
|---|---|
| 回帰テスト追加 | `python/tests/test_sensor_runner_step3.py` 4 件、全件合格。①`ep` 条件（空 ledger・3 場面・場面 0 で 2 人物追加、場面 1 が別ベクトルで再追加）で accept、pending は先勝ち、epoch commit で [hanako, taro] 適用 ②committed の上書きは従来どおり reject ③pending 人物を追加せず trait_changes に使うと reject ④8 プロセス × 40 件の並列書き込みで 320 件欠損なし・ロックファイル残存なし。**修正前コードでの確認**: ①と③は失敗（①は上記 baseline already exists で reject）、②は合格、④は新ヘルパー自体の検査（旧方式の欠損は §1 の別計測）。 |
| 既存 80 件 | 合格。コミット **`713d2fd55ed5b4a0356378f86e3250489e89c78f`**、作業ツリー clean（`test_sensor_runner.py` + `_phase15.py` + `_step1.py` + `_step15.py`、19.5 s）。新規 4 件を加えて 84 件。 |
| パリティ | `parity.py` 000035_000301、same/state_change × tail/mid の 4 条件すべてで annotations bytes・diagnose bytes・cold(L1 off) 一致、未編集再実行 0 call。結果: `scripts/sensor_runner/results/phase15_stage2_step3_parity_*.json`（4 件）。 |
| `ep` 条件の再現（修正前後） | スタブ annotator（`scripts/sensor_runner/step3_repro.py`）: 作品 001801_001786、committed ledger 空、epoch 1 のループ。スタブは pack が見せる空 ledger に従い、全場面で可視人物（onna, shinichi）を `ledger_additions` に入れる。**修正前コード（HEAD~1 の ingest.py）**: 場面 1 で reject（2 理由 `ledger_additions[onna|shinichi]: baseline already exists`）、2 call で停止。**修正後**: 3 call・reject **0**、全場面 accept、commit で [onna, shinichi] 適用。結果: `phase15_stage2_step3_repro_before.json` / `_after.json`。 |
| コミット・push | 実装 713d2fd、データ ceab621（`feat/phase15-stage2-step3`） |

限界: 再現は決定的スタブで、実 LLM の再実行はしていない（指示で「スタブで可」）。スタブは reject 時に停止するため「修正前の call 数 2」は全件の再試行数ではない。

## 3. 小調査: 001875_059897 の location が 4 場面すべて null だった原因
資料: Step 2 cold の L2 エントリ（`location` の返却値）、`nishiki segment` の場面本文。
- 4 場面の返却値は全て `location: null`（annotations 側で落ちたのではなく、注釈者が null を返した。検証の reject もなし＝全件初回 ACCEPT）。
- 指示文（`strong.py`）: 「location: 場面の主な場所を表す本文の逐語の語句。判断できなければ null。」
- 本文の実物:
  - 場面 1（977 字）: 場所を示す語（家・部屋・うち 等）が 0 件。会話と回想要約のみで場面設定の語句がない。
  - 場面 3（247 字）: 「部屋のこと」「ことしうちぐらい」のみ。前者は話題（借りる部屋の相談）、後者は「今年中」の意味で、場所を表す語句ではない。
  - 場面 0（821 字）: 「安江の家に夏樹という赤ん坊がやってきた」「ここの家」。場所語はあるが、赤ん坊を迎えた過去の出来事の記述の中で、この場面の「主な場所」としては示されていない。「帰宅すると」とあるが場所名の逐語句ではない。
  - 場面 2（751 字）: 「ここの家からもうけようとは」「進君のうちね」「部屋代」。いずれも話題の中の語。
- 対比: location が返る作品（例: 「車室の中」「板の間」「銀座裏の、或る屋台のお鮨や」）は、場面の場所を直接述べる句が本文にある。001875 は家庭内の会話中心で、場所が地の文で明示されない。
- 結論: 本文に場所を表す逐語句が（場面 1・3 は全く、場面 0・2 は主な場所として）無く、「判断できなければ null」の指示どおりの出力と読める。パイプラインの不具合を示す証拠は見つからなかった。
- 限界: 4 場面・1 作品・1 回の注釈のみ。注釈者が別の判断（例: 場面 0 に「安江の家」）をする可能性は否定できない（再注釈による揺れは未測定）。調査は読み取りのみで実装変更なし。

## 4. 変更ファイル
- 変更: `scripts/sensor_runner/ingest.py`、`step2.py`、`step0.py`、`docs/handoff/STATUS_phase15_stage1.md`
- 新規: `scripts/sensor_runner/jsonl_append.py`、`step3_repro.py`、`python/tests/test_sensor_runner_step3.py`、`scripts/sensor_runner/results/phase15_stage2_step3_*.json`（repro 2 + parity 4）、本レポート
- 無変更: L1、L3、鍵・pack の内容、物差し、出力スキーマ（nishiki.diagnose/3、sensor.v2）、エポック設計

## 5. 残った問題・判断を仰ぎたい点
1. pack に出ない pending 人物を `trait_changes` で使うと reject（仕様どおりだが、注釈者は凍結 ledger しか見えないため再試行を要する）。同一エポック内で pending を pack に見せる案は「エポック固定の設計」に触れるため実施していない。
2. 先勝ちで 2 回目以降のベクトルは黙って捨てられる（reject も警告もない）。ベクトルが食い違った場合の記録が必要か。
3. 001875 の location は再注釈で変わるか未測定。
4. 修正後の実 LLM 再実行は未実施（スタブのみ）。

## 6. 守った制約
物差し・出力スキーマ・エポック設計・L1/L3 無変更。`premise_changed` の自動分類は追加していない。main への push・マージなし。
