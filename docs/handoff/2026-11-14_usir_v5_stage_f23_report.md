# USIR v5 Stage F-23 レポート：harness の test_u6_guards cwd 相対パス修正

## 結果
- 修正コミット（usir）: `18e53c2`（push 済み、`origin/feat/fable-v5-shootout` と一致を `git ls-remote` で確認）。
- `python scripts/harness.py test`: 終了コード 0。Python 672 passed / 0 failed / 18 skipped（211.99s）、Vitest 36 files・378 tests passed。
- nishiki: 変更なし（コード・コミットとも無し）。

## タスク1：再現
- 切替前コミット `76cac36` を一時 worktree に展開し、ハーネスと同じ cwd（`python/`）で `uv run pytest tests/v5/test_u6_guards.py -q` を実行。
- 結果: 1 failed / 16 passed。失敗は `test_u6_test_vectors_fixture_pass`（`tests\v5\test_u6_guards.py:234`、`Path('docs/handoff/v10/u6_guards_test_vectors.json').exists()` が False）。フィクスチャ自体は `../docs/handoff/v10/` に存在。
- 結論: 失敗は床の切替（F-22 Task 1）より前から存在し、原因は cwd 相対パス。床の切替とは無関係。
- 制約: 検証は該当ファイルのみ（ハーネス全体は切替前では未実行）。ハーネスの cwd が `python/` であることは `scripts/harness.py:24-25` で確認。

## タスク2：修正案の選択
- 選択: 案A（テスト側修正）。
- 内容: `Path("docs/handoff/v10/...")` を `Path(__file__).resolve().parents[3] / "docs/handoff/v10/..."` に変更（`tests/v5` → リポジトリルート）。アサーション（38 vectors、calibrated False 等）は不変。
- 理由: (1) 変更は 1 テスト内の 3 行で影響範囲が最小。(2) 案B はハーネスの cwd を変えるため `python/` 基準のテスト全体（`uv run` の pyproject 解決含む）に影響しうる。(3) 同一ディレクトリの `test_b14_eval_rule_v2.py` は既に `ROOT` 基準で参照しており、同方式に揃う。
- grep で `tests/` 内の他の cwd 相対 `docs/` 参照は無し（本ファイル 233 行のみ）。

## 変更ファイル
- `python/tests/v5/test_u6_guards.py`（3 行追加・1 行削除）
- 追加（検証成果物）: `scripts/f23/f23_verify_frozen_sha256.py`、`scripts/f23/f23_frozen_sha256_verify.json`、`scripts/f23/f23_frozen_sha256_verify_PRE.json`、`docs/handoff/v10/f23_harness_test.log`、`docs/handoff/v10/f23_pytest_v5_root.log`、本レポート

## タスク3：再検証
| 項目 | 結果 |
|---|---|
| `python scripts/harness.py test` | exit 0（Python 672 passed / 18 skipped、Vitest 378 passed） |
| ルートから `pytest python/tests/v5/` | 73 passed（146.26s） |
| `cargo test --workspace` | 省略。nishiki にコード変更が無いため |
| 実推敲ペア v1.1（439件）パリティ | 専用の実行はせず。上記 pytest の通過で確認（テスト内容の変更なし） |

## タスク4：検証ゲート
- 凍結 15 ファイル（`scripts/b17/frozen_sha256_baseline.json`）の SHA-256: 実行前（PRE）・後ともに全一致（`all_frozen_unchanged: true`, n=15）。
- 216候補 SHA `07E43FDF...` 一致。216正本 SHA `2FC18E67...` は usir・nishiki 両コピーで一致。
- σテーブル（候補と正本）一致。マージン比 1.25（0.8〜1.25）不変。pooled 次元なし。
- `ac_max_sis`: `pooled: False`、`pool_frozen: True`、`freeze_reason` は F-19 fallback のまま。常時 BAND_SPECIFIC。
- 制約: PRE の SHA ゲートはテスト修正後・ハーネス実行前に取得。変更対象が `tests/` の 1 ファイルのみで凍結成果物に触れていないため、実質同等と判断（修正前の取得ではない）。

## Gate 3a／3b・Level 3b
- 変更なし。仕様書・討議パック（Google Docs）の更新も行っていない。

## 備考
- 一時 worktree の後始末で、シェルが worktree 内にあったため初回の `git worktree remove` が Permission denied。ディレクトリ削除と `git worktree prune` で解消済み。
