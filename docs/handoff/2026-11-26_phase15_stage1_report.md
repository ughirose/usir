# Phase 1.5 Stage 1 レポート: sensor_runner L1/L2 キャッシュ

- 日付: 2026-11-26
- ブランチ: `feat/phase15-stage1-cache`（worktree `C:\dev\usir-phase15`）。main へのマージはしていない。
- ベース: `feat/sensor-runner` @ b3da0e4。指示は「現在ブランチ（phase2/gold-audit）から切る」だったが、そのブランチには `scripts/sensor_runner` が存在しないため、sensor_runner を含む `feat/sensor-runner` から切った（逸脱）。
- 設計メモ: 2026-11-24_usir_pack_v1_task6_phase15_incremental_design.md（Drive 版ではなく Antigravity brain 内のローカルコピーを読んだ）。
- Gate: 条件1（L2 prior は構造化サマリの Early Cutoff）、条件2（L3 は未実装）に従った。

## 1. 現状の把握（変更前）

- `cache.py` のキーは `scene_hash` のみ（nishiki segment 由来）。有効判定は `instruction_version` 一致だけ。roster・ledger・直前場面の変更は無効化されない。
- sensor_runner は prepare / ingest / merge の3段で、Strong アノテーションは外部セッション（Claude Code）が pack を読んで出力する。**プロセス内で LLM は呼ばれない**。LLM 呼び出し回数は「annotate されて ingest された pack の数」として数えた。実時間は測れない。
- Step 1 のローカル処理は、プロセス内では `merge.py` の発話ブロック構築（`local/<work>.utterances.json` の話者・宛先 + Janome による Policy X の敬体/常体 + 一人称の表層規則）。話者推定そのものを行うコードはこのリポジトリに無く、ローカルファイルは seed（strong モデル出力）の流用。L1 はこの発話ブロックを対象にした。

## 2. 実装

### キー設計（`keys.py`）
- `scene_text_hash` = SHA256("scene_v1\0" + 正規化本文)。BOM 除去、CRLF/CR→LF。場面番号・オフセットは含めない。
- `roster_hash` = characters.json の正規化 JSON ハッシュ（リスト順・全フィールド）。
- `ledger_hash` = traits.json baseline 全体のハッシュ。
- L2 キー = `scene_text_hash` + `instruction_version` + `roster_hash` + `ledger_hash` + `prior_state_hash`。
- L1 キー = `scene_text_hash` + `roster_hash` + `prev_tail_hash` + `step1_algo_version` + `local_utt_hash`。
  - `prev_tail_hash`: 直前場面の末尾2文（C1/C2 lookback 相当）のハッシュ。先頭場面は空窓。
  - `step1_algo_version`: 版ラベル + Policy X が依存するソース（build_annotations.py, build_modern_gold.py）、FIRST_PERSON リスト、janome の版のハッシュ。
  - `local_utt_hash`: 設計メモに無い追加要素。ローカル話者ファイルの該当場面エントリのハッシュ。入力なので、変われば必ずミスにする（ミスが増える方向の追加で、誤再利用は増えない）。

### Early Cutoff の構造化サマリ（Gate 条件1）
`prior_state` = `{location, present（id の昇順集合）, pov_character}`。直前場面を再注釈してもこの3つが同じなら、後続場面は L2 ヒットのまま連鎖が止まる。
**キーに含めないもの**: 直前場面の time・recent_facts・trait_state。pack の `prior` ブロックにはこれらが表示され続けるため、サマリが同じで facts だけ変わった場合、後続場面の注釈は古い facts を見た状態で再利用される。これは条件1の方針の帰結であり、全量再計算との差が出得る点として §6 に記載した。

### 無効化ルール（設計メモ §3 通り、テストで確認）
- 場面 i の本文変更 → i の L1/L2 が無効化。
- 場面 i の末尾変更 → L1 は i+1 の1場面までに波及。i+2 以降は波及しない。
- L2 は prior サマリが変わった場合のみ i+1 へ波及し、サマリが変わらなければ止まる。
- roster 変更 → 全場面の L1/L2 が無効化。
- ledger 変更 → 全場面の L2 が無効化、L1 は維持。
- instruction_version / step1_algo_version の更新 → 対応層のみ無効化。

### 解決順序と pack
- `resolve.py`: 場面を index 順に解決する。場面 i のキーは場面 i-1 の解決済みエントリから決まるため、i-1 が未解決の場面は `blocked`（キー計算不能）。
- `prepare`: `defer_blocked`（CLI の既定は有効、`--emit-blocked` で無効）で blocked 場面の pack を出さない。ライブラリ既定は互換のため無効だが、同じ本文のエントリが既にある blocked 場面は pack を出さない（直前の再注釈後にヒットし得るため）。
- pack に `l2_parts` / `l2_key` を記録。`ingest` は pack 生成時に直前場面が未注釈だった pack を `pack_prior_unresolved` で拒否し、roster が変わっていれば `pack_stale_roster` で拒否する。エントリは `ledger_additions` 適用後の ledger ハッシュのキーで保存する。
- 既存の旧形式エントリ（`cache/strong/<scene_hash>.json`）は、初回の解決時に現行キーで `cache/l2/` に取り込み（adopt）、旧ファイルは削除する。`cli.py migrate` で一括実行できる。コミット済みの 35 エントリは移行済み（内容は変更なし）。
- 整合性: 書き込みは一時ファイル → fsync → `os.replace`（`write_json` 全体に適用。出力バイトは従来と同一）。読み込み時の破損・スキーマ不一致・キー不一致はミス扱い。GC は `cli.py gc` の明示実行のみ（既定は一覧表示、`--apply` で削除）で、prepare/merge/diagnose は削除しない。
- `cache/l1/` は再生成可能なため `.gitignore` に追加。

## 3. 不変条件の確認

- 物差し・出力スキーマは変更していない。`nishiki.diagnose/3`、sensor.v2 の出力形式、閾値・計算式のファイルは未変更。
- Reduce 相当（場面間結合・全体集計）はキャッシュしていない。L1 は場面単位の発話ブロック、L2 は場面単位の注釈のみ。
- キャッシュに保存する値は整数・文字列のみで、浮動小数点の丸めは入れていない（diagnose 出力は従来経路のまま）。

## 4. パリティと効果の実測

### 変更前との一致（回帰）
変更前（b3da0e4）の全量 merge 出力（7 作品 35 場面）と、変更後コードでの出力を `cmp` で比較: 全 14 ファイルがバイト一致。6 レンズはすべて ok。

### 部分再計算の完全一致（`parity.py`、自動テスト `test_warm_partial_recompute_equals_cold_full_recompute`）
作品 000035_000301（7 場面）の場面 3 に1文を挿入（`tail` = 場面末尾、`mid` = 先頭文の後）。ウォーム（元本文でキャッシュ構築後に編集本文を実行）とコールド（空キャッシュで編集本文を全量実行）を比較。

| ケース | 注釈 JSON バイト一致 | diagnose stdout バイト一致 | ウォームの L2 再計算場面 | 同 L1 miss |
|---|---|---|---|---|
| tail / 状態不変 | 一致 | 一致 | [3] | 2（場面3, 4） |
| mid / 状態不変 | 一致 | 一致 | [3] | 1（場面3） |
| tail / 状態変化（present 変更） | 一致 | 一致 | [3, 4]（5 で打ち切り） | 2 |
| mid / 状態変化 | 一致 | 一致 | [3, 4] | 1 |

- 追加の確認: L1 無効のコールド実行とも一致。未編集の再実行は LLM 呼び出し 0 回で元の出力とバイト一致。
- 感度確認（ミューテーション）: キーから prior を外した変異版では、状態変化ケースの注釈 JSON が不一致になり、このハーネスが誤った再利用を検出できることを確認した（状態不変ケースは一致のまま）。
- 注釈は決定論的スタブ（pack の場面番号・本文・prior サマリの純関数）で、実 LLM は使っていない。スタブはシード注釈の複製に `annotator = stub:<prior ハッシュ>` を付けたもの。実 LLM の出力が同じ入力で毎回同じになる保証は無く、このパリティは「キャッシュ層が入力に対して正しく再利用・無効化する」ことを示すもので、LLM 出力の再現性は示していない。

### 効果（中央値、各5回、`results/phase15_stage1_timing.json`）
場面1つを末尾編集。単位は秒（nishiki は debug ビルド `C:/dev/nishiki-hybrid/target/debug/nishiki.exe`）。

| 作品（場面数） | LLM 呼び出し 前→後 | L1 再計算 前→後 | 合計 前→後 | merge 前→後 | diagnose 前→後 |
|---|---|---|---|---|---|
| 000035_000301 (7) | 7 → 1 | 7 → 2 | 0.466 → 0.237 | 0.071 → 0.038 | 0.107 → 0.107 |
| 001383_056643 (6) | 6 → 1 | 6 → 2 | 0.408 → 0.229 | 0.070 → 0.044 | 0.102 → 0.106 |
| 001875_059897 (4) | 4 → 1 | 4 → 2 | 0.313 → 0.204 | 0.074 → 0.056 | 0.072 → 0.069 |

- 「前」は空キャッシュでの全量実行。合計にはスタブの prepare/ingest 時間が含まれ、実 LLM の所要時間は含まない。LLM 呼び出しの削減が実運用の時間削減の大半を占めるが、その実時間は未測定。
- 全 15 回（3作品×5回）でバイト一致。

## 5. L3（Rust nishiki 内部キャッシュ）へ進むべきか

測定した範囲では、L3 を投入する根拠は無い。
- ウォーム実行 0.20〜0.24 秒のうち diagnose は 0.07〜0.11 秒（約 35〜45%）で、segment が 0.03 秒、merge が 0.04〜0.06 秒。diagnose は編集場面が1つでも全場面を処理するため、前後で変わらない。
- ただし測れたのは最大 7 場面の作品のみ。設計メモが想定する 40 場面（000148_000751）は作品レジストリに無く、場面数が増えたときの diagnose の伸びは未測定。nishiki は debug ビルドなので、release ビルドでは絶対値が小さくなる可能性が高い。
- 判断材料としては、40 場面級の作品を登録して release ビルドで diagnose の所要時間を測ってから決めるのが妥当（推測）。

## 6. 残った問題・判断を仰ぎたい点

1. **ledger_hash が作品全体のため、`ledger_additions` で全場面の L2 が無効化される。** 設計メモ通りの仕様だが、新規注釈の途中で新しい人物が ledger に追加されるたびに、その時点までの全注釈が再注釈対象になる。編集ループ（ledger が安定している）では問題にならない。代替案: キーに含める ledger を「その場面の pack に表示された人物（roster 関連）の baseline」に限定する。注釈が見る情報と一致するため安全だが、設計メモからの変更になるので未実装。
2. **Early Cutoff の健全性。** prior サマリ（location・present・pov）が同じなら、直前場面の facts・trait_state が変わっても後続場面の注釈は再利用される。LLM が実際に見る prior（facts・trait_state・time を含む）とキーの粒度がずれている。pack の prior をサマリのみに絞れば厳密になるが、注釈の入力が変わるため未実施。条件1の範囲内での判断を仰ぎたい。
3. **adopt の自動実行。** 旧形式の無刻印エントリは初回解決時に現行 ledger/roster/prior で取り込まれる。従来の「scene_hash だけで信用する」と同じ信頼水準で、取り込み後は厳密キーになる。`Cache(legacy_adopt=False)` で無効化可能（CLI の `migrate` 以外は既定が有効）。
4. **local ファイルの出所。** L1 の対象はプロセス内で計算する発話ブロックのみで、話者推定そのものはこのリポジトリに実装が無い（seed ファイル）。話者推定を実装する際は、`step1_algo_version` の入力に加える必要がある。
5. 設計メモの Drive 版は読めず、Antigravity brain 内のコピー（同名・10061 バイト）を読んだ。内容の差異は未確認。

## 7. 変更ファイル

追加:
- `scripts/sensor_runner/keys.py`、`resolve.py`、`parity.py`
- `python/tests/test_sensor_runner_phase15.py`（23 テスト）
- `scripts/sensor_runner/cache/l2/**`（35 件、旧 `cache/strong/**` からの移行）
- `scripts/sensor_runner/results/phase15_stage1_*.json`（実測結果）
- `docs/handoff/STATUS_phase15_stage1.md`、本レポート

変更:
- `scripts/sensor_runner/cache.py`（L1/L2 ストア、adopt、破損時ミス、gc）
- `prepare.py`（順次解決・defer・pack に l2_parts）、`ingest.py`（厳密キーで保存・stale pack 拒否）、`merge.py`（L1、解決器、`build_utterances` の切り出し）
- `common.py`（`write_json` のアトミック化）、`cli.py`（`migrate` / `gc` / `--emit-blocked`）、`.gitignore`（`cache/l1/`）

削除:
- `scripts/sensor_runner/cache/strong/*.json`（35 件、`cache/l2/` へ移行）

## 8. テスト結果

- sensor_runner: `test_sensor_runner.py`（既存 36）+ `test_sensor_runner_phase15.py`（23）= 59 件すべて合格。既存テストは無修正。
- リポジトリ全体 `python scripts/harness.py test`: 752 passed, 7 failed, 18 skipped（205 秒）。失敗 7 件は変更前のコミット b3da0e4（`C:\dev\usir-sensor-runner`）でも同じ 7 件が失敗する（データファイル不足の FileNotFoundError 等）。今回の変更とは無関係。TS 側は `node_modules` 未導入のためスキップ。
