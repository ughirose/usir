# Phase 1.5 Stage 2 Step 1.5 レポート（2026-11-26）: roster_all_ids の鍵からの分離

ブランチ: `feat/phase15-stage1-cache`（main へのマージなし）
実装コミット: `03219534666a40746d5c731dd084048a4d2d3852`（テスト・パリティはこのコミット時点・作業ツリーclean で実行。パリティ・シナリオは同コミットの内容で、コミット前に実行した結果と同じコードである）

## 1. 判断根拠（要約）
「鍵はAIが見た入力から作る」原則は維持する。ただし作品の全人物ID一覧のうち、その場面に現れない人物のIDは注釈の内容に実質影響しないと判断し、scene_id などメタ項目と同じ扱いで鍵から外した。鍵に入る人物情報は、その場面に現れる人物の分（roster の可視分、ledger_baseline、prior の present）に限る。残るリスク（削除された人物を指す古い注釈など）は無効化ではなくフラグ記録（`roster_changed`）で拾う。facts を鍵に入れずフラグで扱う Step 1 の判断と同じ線引き。

## 2. 実装
- **pack の中身は不変**: `roster_all_ids` は pack に残り、AI にも表示される（`test_pack_still_shows_roster_all_ids...` で確認、並びは人物ファイル順のまま）。
- **鍵の投影**（`render.pack_view`、スキーマ `pack_bytes/2`）: `roster_all_ids` を除外。あわせて `roster`（その場面の可視人物）は id でソートして集合として扱う（人物ファイルの並び順変更で鍵が変わらないようにするため。現行は並び順を鍵に含めていたので正規化を追加）。旧スキーマ `pack_bytes/1` は旧鍵の再計算（移行）用にのみ残した。
- **安全網**: L2 エントリの `parts.roster_all_digest`（人物ID集合のハッシュ）を保存。ヒット時に現在値と異なれば `resolve.Resolved.roster_changed` を立て、`cache/stale/<work>.json` の `roster_changed` に理由 `roster_changed` で記録（`premise_changed` と同じ置き場。stale スキーマに配列を追加）。prepare / merge のレポートに `roster_changed` を追加。自動再注釈はしない。履歴不明（Stage 1 移行分）は digest = None で記録対象外。
- **CLI**: `cli.py stale` が `premise_changed` と `roster_changed` を両方一覧（理由つき）。`--reannotate [--scene N ...]` は両方が対象（明示指定のみ）。prepare の出力に `roster_changed=` 件数を追加。
- **鍵スキーマ版**: エントリ `sensor.cache.l2/3`、鍵 `pack_bytes/2`。
- **旧エントリの扱い**（Step 1 と同じ方式）: ①l2/3 の鍵が一致すればヒット ②旧 `pack_bytes/1` 鍵を再計算し、一致する l2/2 エントリのみ l2/3 に移行（source `migrated-v2`。保存済みの prior_facts_digest / prior_facts は引き継ぎ。旧鍵は roster_all_ids を含むので一致＝人物一覧が注釈時と同じであり、roster_all_digest は現在値） ③Stage 1 の l2/1 も従来どおり旧鍵一致分のみ移行（source `migrated-v1`、両 digest = None）。不一致のエントリは読まない。旧ファイルは削除せず残置（明示 gc まで）。
- **移行件数**: seed キャッシュ 7 作品の `cli.py migrate` で `adopted` = 7+5+6+6+4+3+4 = **35 件、miss 0、blocked 0**。l2/3 `migrated-v2` ファイル 35 件を新規コミット。l2/2 `migrated-v1` 35 件・l2/1 `adopted-legacy` 35 件は残置（合計: 旧ファイルは削除していない）。

## 3. 完了条件の実測

| 条件 | 結果 |
|---|---|
| 既存72件 | 合格。コミット **`03219534666a40746d5c731dd084048a4d2d3852`**、作業ツリーclean。ただし Step 1 の 2 件（下記）は新仕様に合わせて期待値を変更した（§3.1） |
| 新規プロパティテスト | 新規 `python/tests/test_sensor_runner_step15.py` 8 件、全件合格。合計 **80 passed**（`test_sensor_runner.py` + `_phase15.py` + `_step1.py` + `_step15.py`、17 s）。`pytest tests -k sensor_runner` は 80 passed, 2 skipped（skip は `torch` 未導入の別テスト 2 件で本件と無関係） |
| パリティ | `parity.py` 000035_000301、同一/state_change × tail/mid の4条件で annotations bytes・diagnose bytes・cold(L1 off) 一致、未編集再実行 0 call。warm 1 call（同一）/ 2 call（state_change）、cold 7 call。結果: `docs/handoff/data/phase15_step15_parity_*.json` |
| 人物追加シナリオ | §3.2 |

### 3.1 既存72件の扱い
Step 1 のテスト 2 件は、Step 1 の仕様（roster_all_ids が鍵に入る）を直接検査していた。本ステップの確定仕様（鍵から外す）に合わせて期待値を更新した:
- `test_roster_alias_and_attribute_changes_visible_vs_hidden`: 作品に人物を追加（どの場面にも現れない）→ 鍵 `!=` から `==` に変更。
- `test_one_byte_change_of_any_keyed_field...`: 鍵の投影に含まれるフィールド集合から `roster_all_ids` を除外。
これ以外の70件（既存59件の全量 + Step 1 の11件）は無改変で合格。

### 3.2 新規プロパティテスト（要求との対応）
- 現れない人物を追加 → 全場面の鍵不変・全場面ヒット・各場面に roster_changed 記録・LLM 呼び出しなし: `test_adding_a_character_who_appears_in_no_scene...`
- ある場面に現れる人物を追加 → その場面のみ鍵変更（他は不変、記録つきヒット）: `test_adding_a_character_who_appears_in_one_scene...`
- 人物削除（現れない人物）→ 鍵不変・記録あり。現れる人物（hanako）の削除 → 現れる場面の鍵のみ変更: `test_removing_a_character...`
- 並び順のみ変更 → 鍵不変・記録なし（digest は集合）。pack の `roster_all_ids` は新しい順で表示: `test_character_list_order_alone...`
- 明示再注釈 → 当該場面のみ pack 出力、現在の人物一覧を見て再注釈、記録解除: `test_reannotation_of_roster_changed_scenes_is_explicit...`
- 新エントリが `l2/3`・`pack_bytes/2`・roster_all_digest を保存。diagnose 入力（merge 出力）にフラグなし: `test_new_entries_store_the_roster_digest...`
- l2/2 移行（旧鍵一致のみ）: `test_step1_entry_is_adopted_when_its_pack_bytes_1_key_matches_and_not_otherwise`
- pack 内容不変・`roster_all_ids` 表示維持: `test_pack_still_shows_roster_all_ids...`

### 3.3 人物追加シナリオ（効果の確認）
方法: 4 シーンの合成作品（taro, hanako の 2 人。シーン 0〜3 の本文は固定、シーン 2「翌朝、花子は店を失った。…」、シーン 3「朝になった。…」）。決定的スタブ annotator が全シーンを注釈 → 人物ファイルに 1 人追加 → 再度 prepare→ingest ループを回し、「ingest した pack 数 = LLM call 数」を数える。同一スクリプト（scratchpad `roster_scenario.py`）を Step 1 コミット `eb031f4`（一時 worktree、実行後削除）と本実装で実行。

| 追加人物の別名（現れる場面） | Step 1 の再注釈 call 数 | Step 1.5 の再注釈 call 数（場面） | roster_changed 記録（場面） |
|---|---|---|---|
| どの場面にも現れない | 4（全場面） | **0** | 0,1,2,3 |
| 「店」（場面 2 のみ） | 4 | **1**（場面 2） | 0,1,3 |
| 「朝」（場面 2, 3） | 4 | **2**（場面 2, 3） | 0,1 |

限界: 4 場面の合成データ・スタブ注釈。実 LLM は未使用。実作品での再注釈数は追加人物が現れる場面数に依存する（別名が一般語と衝突する場合は多数の場面に現れうる）。再注釈された場面の結果が下流の prior core を変えれば、次の場面も連鎖して再注釈されうる（今回のスタブは prior core を変えない）。

## 4. 変更ファイル
- 変更: `scripts/sensor_runner/keys.py`（l2 スキーマ定数: 現行 `l2/3`、旧 `l2/2` を `L2_SCHEMA_V2` に）, `render.py`（pack_bytes/2、roster ソート、roster_all_digest、旧スキーマの鍵再計算）, `cache.py`（l2/2 移行、stale 状態に roster_changed）, `resolve.py`（roster_changed 検出・記録）, `prepare.py`, `merge.py`（レポート項目）, `epoch.py`（`list_roster_changed`）, `cli.py`（stale 一覧・件数）, `python/tests/test_sensor_runner_step1.py`（期待値 2 箇所）
- 新規: `python/tests/test_sensor_runner_step15.py`, `docs/handoff/data/phase15_step15_parity_*.json`（4件）, `scripts/sensor_runner/cache/l2/*/*.json`（l2/3 移行 35 件）
- 無変更: L1、L3、`parity.py`、出力スキーマ（nishiki.diagnose/3、sensor.v2）、物差し、pack の内容

## 5. 守った制約
物差し・出力スキーマ・pack の内容は変更していない。L1 と L3 には触れていない。main へのマージ・push なし。パリティは新しい鍵で 4 条件とも一致。

## 6. 残った問題・判断を仰ぎたい点
1. `roster` を id でソートして鍵に入れた（並び順の正規化）。AI が見る pack の並びは人物ファイル順のままなので、並び順だけが変わった場合、再利用される注釈は旧順序の pack で作られたものになる。意味への影響は未測定（影響は小さいと推測するが、実測はしていない）。
2. `roster_changed` は「人物一覧の集合が変わった」ことのみ記録し、どの人物が増減したかは現在の一覧（`current_roster_all_ids`）しか持たない（注釈時の一覧は保存していない）。差分表示が必要なら、エントリに ID リストそのものを保存する拡張が要る。
3. 人物の増減が多い作品では全場面に `roster_changed` が立つため、一覧は件数が多くなる（フィルタは `--scene` のみ）。
4. 人物の別名が一般語と衝突すると多数の場面で「現れる」判定になり、再注釈が増える。別名の設計（roster 側）の問題で本ステップの対象外。
5. 実 LLM を使った測定は未実施（スタブのみ）。
