# USIR v5 Stage F-22 レポート：216作品正本ノイズ床への既定参照の切替

日付: 2026-11-14 / 対象: `usir`（`feat/fable-v5-shootout`）, `nishiki`（`feat/fable-hardcore-v5`）

## 0. 結論
- `CANONICAL_F14_NOISE_FLOOR_PATH` を 216正本（SHA `2FC18E67...84974`）へ切替済み。変更コードは usir の2ファイルのみ。nishiki のコード変更はなし。
- 切替後に v1.1 の Python/Rust eval_rule "v2" パリティ不一致 0/439（Python 側は既定パス `load_f14_noise_floor(None)` で読込）。
- 1SD 跨ぎは 14 件、全て 超過→非超過（緩和方向）で F-21 の値と一致。帯別 abs_max>1 件数も F-21 値と一致（S 13、M 75→64、L 23→20、mixed(8D) 51→43）。
- 凍結 15 ファイル＋216正本の SHA は切替前後とも一致。σテーブル・マージン（0.8〜1.25）・`ac_max_sis` 凍結は不変。
- 未達・注意: `python scripts/harness.py test` は終了コード 1（1 件失敗、§2.2）。失敗は床の切替と無関係（cwd 相対パス）と判断しているが、本ステージでは修正していない。

## 1. 切替の実施内容
コミット（usir）:

| 内容 | コミット |
|---|---|
| Task 1 既定参照切替 | `4cf8b5e` |
| Task 2–4 検証成果物（scripts/f22/） | `c24e582` |
| Task 5 レポート | `本レポートを含むコミット（最終報告に記載）` |

変更ファイル:
- `python/usir_py/v5/f14_hybrid_noise_floor.py`
  - `CANONICAL_F14_NOISE_FLOOR_PATH` を `paragraph_noise_floor_all_216_canonical.json` へ変更。
  - `F17_HISTORICAL_NOISE_FLOOR_PATH`（F-17 床、履歴・ベンチマーク参照）を追加。F-17 床ファイル自体は未変更。
  - `F14_SIGMA_PROVENANCE` の静的メタデータ（canonical_path、upstream_evidence_path、generator_command、scope、n_works 46→216、n_authors 10→120、n_pairs_total 10859→52812）を 216正本に合わせて更新。`historical_f17_floor_path` を追加。σ計算・マージン・hybrid ルールには手を入れていない。
  - モジュール docstring の既定床の記述を更新。
- `python/tests/v5/test_b11_gate3a_and_f14_parity.py`（2行）
  - `test_f14_window_count_and_hybrid_rule_mechanics` の `prov["n_works_in_file"]` を 46→216、`prov["n_pairs_in_file"]` を 10859→52812。検証意図（既定ロード結果のメタデータ整合）は不変。
- 追加（検証用）: `scripts/f22/` 一式（§5）。

テスト更新の判断:
- 切替直後の `pytest tests/v5/` 初回（`scripts/f22/pytest_after_switch_1.log`、cwd=`python/`）の失敗は 2 件。床に起因するのは `test_b11_..._mechanics` の 1 件のみで、上記のとおり更新。もう 1 件 `test_u6_guards.py::test_u6_test_vectors_fixture_pass` は cwd 相対パス不在（§2.2）。
- b12/b13/b14 のテストは F-17 床を明示パスで固定しており、既定参照に依存しないため変更不要（F-17 床を「履歴・ベンチマーク参照」として使い続ける）。
- `scripts/build_real_revision_pairs_v1_1.py`、`run_stage_b13/b14_analysis.py` 等も F-17 床を明示パスで参照する履歴再現用であり、凍結成果物の再現性を保つため変更していない。nishiki 側の参照（`crates/nishiki-cli/tests/paragraph_noise_floor_test.rs`、`scripts/f18/common.py`）も明示パスで、既定値ではないため変更なし。

## 2. 切替後の再検証（タスク2）
### 2.1 結果一覧
| # | コマンド | 結果 |
|---|---|---|
| 1 | usir `pytest python/tests/v5/`（リポジトリルートから） | 73 passed / 0 failed（129.4s）`scripts/f22/pytest_after_switch_2.log` |
| 2 | usir `python scripts/harness.py test` | **終了コード 1**。Python 671 passed / 1 failed / 18 skipped（202.2s）、Vitest 36 files・378 tests passed。`scripts/f22/harness_test.log` |
| 3 | nishiki `cargo test --workspace` | 537 passed / 0 failed / 1 ignored、終了コード 0。`scripts/f22/cargo_test_workspace.log` |
| 4 | nishiki `node --test packages/cm6-kasane/test/*.test.mjs` | 8 tests・8 pass・0 fail。`scripts/f22/node_test_cm6.log` |

### 2.2 harness の 1 件失敗
- 失敗: `tests/v5/test_u6_guards.py::test_u6_test_vectors_fixture_pass`（`Missing fixture file: docs\handoff\v10\u6_guards_test_vectors.json`）。
- 原因: テストが `Path("docs/handoff/v10/u6_guards_test_vectors.json")` と cwd 相対で参照（`test_u6_guards.py:233`）。harness は `cwd=python/` で `uv run pytest tests/` を実行するため、ファイルが見つからない。フィクスチャはリポジトリルートの `docs/handoff/v10/` にあり（コミット `bc348cc`）、ルートから実行した §2.1 #1 では通過する。
- 床の切替との関係: このテストは f14 床を参照しない（`f14`・`noise_floor` の grep に一致なし）。ただし切替前コミットで harness を実行して同じ失敗になることは未確認（パス参照の機構に基づく判断）。
- ゼロ・スコープクリープ原則に従い、テスト・harness は変更していない。

### 2.3 cargo の詳細
- ignored 1 件: `crates/kasane-core/tests/audit_17_corpus.rs:46` の `test_generate_17_corpus_audit_data`（`#[ignore = "Heavy 1.06M-char corpus benchmark; ..."]`）。B-17・F-21 と同じ `audit_17_corpus` の `#[ignore]`。`.rs` 内の `#[ignore` は他に検出されなかった（target 除く）。
- `kasane-wasm`: `src/lib.rs` と `tests/web.rs` はともに 0 tests。ワークスペースに含まれ実行はされたが、検証するテストは存在しない。`nishiki-wasm` は 4＋5 tests が通過。
- nishiki のコードは本ステージで未変更（HEAD `7b0714a`）。Rust のテスト結果は切替の影響を受けないため、切替の検証としての情報量は限定的。

### 2.4 node テストの判断
Rust のみ（変更なし）のため省略可だが、指示に従い実行した。8/8 通過。

### 2.5 v1.1 パリティと対照表
- 実行: `scripts/f22/f22_v11_parity_and_floor_table.py`。Rust は `target/f14/release/nishiki.exe`、Python は 216正本を既定パス（`path=None`）で読込。スクリプト冒頭で `CANONICAL_F14_NOISE_FLOOR_PATH` が 216正本であることを assert。
- データセット v1.1: 439 ペア。
- パリティ不一致: F-17 床 0/439、216正本（既定パス）0/439。

## 3. 切替の影響表（タスク3）
### 3.1 abs_max の分布（v1.1, 439 ペア）
| 指標 | F-17 床 | 216正本 | 差 |
|---|---|---|---|
| 中央値 | 0.4069 | 0.3836 | -0.0233（比の中央値 0.9462） |
| 平均 | 0.6243 | 0.5910 | -0.0333 |
| 95%点 | 1.8052 | 1.6995 | -0.1057 |
| 最大値 | 4.2383 | 3.8670 | -0.3713 |
| abs_max>1 のペア数 | 111 | 97 | -14 |
| >1 の次元総数（全ペア） | 171 | 143 | -28 |

- 1SD 跨ぎ（abs_max>1 の判定が変わったペア）: 14 件（超過→非超過 14、非超過→超過 0）。
- 新旧で abs_max が最も変化したペアの変化幅: 0.4931（跨ぎとは限らない）。
- 次元単位の反転数: 跨ぎペア内で 14、全ペアを数えた F-21 定義で 28（F-21 の 28 と一致。差は、abs_max>1 が維持されたペア内で反転した次元を含むか否か）。

### 3.2 帯別（F-17 → 216正本）
| 帯 | n | abs_max 中央値 | abs_max>1 件数 |
|---|---|---|---|
| L | 137 | 0.1489 → 0.1347 | 23 → 20 |
| M | 252 | 0.5985 → 0.5644 | 75 → 64 |
| S | 50 | 0.6342 → 0.6185 | 13 → 13 |
| mixed | 84 | 1.0751 → 1.0056 | 51 → 43 |

F-21 の値（S 13、M 75→64、L 23→20、mixed(8D) 51→43）と一致。S/M/L は排他、mixed（8D 混在窓ペア）は S/M/L と重なる別集計。

### 3.3 1SD 跨ぎ 14 件の内訳
| idx | pair_id | 帯 | 反転した次元 | その次元の abs ratio（F-17 → 216） | ペアの abs_max（F-17 → 216） |
|---|---|---|---|---|---|
| 20 | horoki_first_vs_new_p862_p445 | M/mixed | ac_mean_crest | 1.026 → 0.982 | 1.026 → 0.982 |
| 27 | horoki_first_vs_new_p1320_p1219 | M | semantic_tension | 1.162 → 0.974 | 1.162 → 0.974 |
| 30 | horoki_first_vs_new_p1687_p1514 | M | semantic_tension | 1.166 → 0.977 | 1.166 → 0.977 |
| 49 | oda_ame_first_vs_rev_p9_p58 | L/mixed | dc_speculation | 1.102 → 0.981 | 1.102 → 0.981 |
| 68 | oda_ame_first_vs_rev_p38_p201 | L/mixed | dc_speculation | 1.020 → 0.909 | 1.020 → 0.909 |
| 76 | oda_ame_first_vs_rev_p49_p228 | M | dc_showing | 1.050 → 0.987 | 1.050 → 0.987 |
| 123 | oda_wagamachi_first_vs_rev_p165_p1044 | M | dc_staccato | 1.042 → 0.935 | 1.042 → 0.935 |
| 131 | kishida_doin_draft1_vs_final_p111_p161 | M/mixed | dc_confession | 1.005 → 0.946 | 1.005 → 0.946 |
| 132 | kishida_doin_draft1_vs_final_p181_p225 | M/mixed | dc_cadence | 1.037 → 0.961 | 1.037 → 0.961 |
| 262 | ichiyo_utsusemi_ed_p22_p22 | L/mixed | ac_mean_crest | 1.049 → 0.982 | 1.049 → 0.982 |
| 280 | miyazawa_ginga_ed_p124_p118 | M/mixed | semantic_tension | 1.007 → 0.844 | 1.007 → 0.844 |
| 319 | hisao_mugetsu_ed_p30_p29 | M/mixed | ac_mean_crest | 1.040 → 0.995 | 1.040 → 0.995 |
| 336 | hisao_mugetsu_ed_p146_p149 | M | semantic_tension | 1.064 → 0.892 | 1.064 → 0.940 |
| 371 | kusaka_saigo_ed_p88_p88 | M | dc_staccato | 1.027 → 0.922 | 1.027 → 0.922 |

- 各ペアで反転した次元は 1 つ。F-17 での次元値は 1.005〜1.166（超過幅 0.005〜0.166）、216 での値は 0.844〜0.995。閾値 1 の近傍からの移動が大半。
- hisao_mugetsu_ed_p146_p149 は、反転した次元の値が 0.892 で、ペアの abs_max（0.940）は別次元の値。

### 3.4 留保
- 14 件はすべて 超過→非超過（緩和方向）で、非超過→超過は 0 件。
- 緩和方向であることが、推敲ラベル（推敲ペアか否か）に対して妥当かは未検証。床の変更で推敲検出の感度が下がっている可能性は、この集計からは排除できない。
- 本集計は v1.1 の 439 ペアに対する判定の変化だけを見ている。検出性能（真陽性率等）は評価していない。

## 4. 検証ゲート（タスク4）
### 4.1 凍結 SHA ゲート
- 実行: `scripts/f22/f22_verify_frozen_sha256.py`（基準 `scripts/b17/frozen_sha256_baseline.json`、15 ファイル、F-21 版と同じ）。
- 切替前（`f22_frozen_sha256_verify_PRE.json`）、切替後（`f22_frozen_sha256_verify.json`）ともに `all_frozen_unchanged: True`、`n_frozen: 15`。
- 216候補（SHA `07E43FDF...`）未変更: True。
- 216正本 SHA（usir・nishiki 双方）: `2FC18E6717E4537ACBB522BC1D10F3C434B3C9BB434B19AF3803293EE7C84974`、コピー同一: True。
- F-17 床 SHA（実測）: `0575B64A75A8AF8DD1AD0DB81BDF4D8898C473CE8AC1876C6E7E2F8B61A6F604`（F-17 床・106候補・実推敲ペア v1/v1.1 は凍結基準の対象）。

| 凍結ファイル | 判定 |
|---|---|
| c:\dev\nishiki\docs\handoff\evidence\2026-11-13_f17\paragraph_noise_floor_all_novel.json | 一致 |
| c:\dev\usir\docs\handoff\v10\paragraph_noise_floor_all_novel.json | 一致 |
| c:\dev\nishiki\docs\handoff\evidence\2026-11-13_b16\paragraph_noise_floor_all_106.json | 一致 |
| c:\dev\usir\docs\handoff\v10\paragraph_noise_floor_all_106.json | 一致 |
| c:\dev\usir\data\usir_v5_eval\real_revision_pairs_v1.jsonl | 一致 |
| c:\dev\usir\data\usir_v5_eval\real_revision_pairs_v1_1.jsonl | 一致 |
| c:\dev\usir\docs\handoff\v10\real_revision_pairs_v1_f17_harvest_summary.json | 一致 |
| c:\dev\usir\docs\handoff\v10\real_revision_pairs_v1_eval_rule_v2_summary.json | 一致 |
| c:\dev\usir\docs\handoff\v10\real_revision_pairs_v1_1_eval_rule_v2_summary.json | 一致 |
| c:\dev\usir\docs\handoff\v10\sparsity_audit_106works_canonical.json | 一致 |
| c:\dev\usir\docs\handoff\v10\sparsity_audit_106works_canonical_v2.json | 一致 |
| c:\dev\usir\data\usir_v5_eval\usir_v5_b12_canonical_eval.json | 一致 |
| c:\dev\usir\data\aozora_manifest.json | 一致 |
| c:\dev\usir\data\aozora_heldout_manifest.json | 一致 |
| c:\dev\usir\data\aozora_heldout2_manifest.json | 一致 |

### 4.2 不変項目
- σテーブル: 候補216と正本でバンド別 σ・全体 per_dim_std が同一: True。いずれのファイルも書換えていない。
- マージン: `equivalence_margin_ratio` = 1.25（下限・上限 0.8〜1.25）。変更なし。
- `ac_max_sis`: 正本の `ac_sigma_pool.dims.ac_max_sis` は `pooled: false`、`pool_frozen: true`、`freeze_notes` 3 件（F-21 の注記）を維持。常時 BAND_SPECIFIC。`ac_coupling`・`ac_mean_velocity` も `pooled: false`。`ac_mean_velocity` は再計測していない。nishiki のプール判定コードは未変更。
- Gate 3a／3b・Level 3b: 変更なし。仕様書・討議パック（Google Docs）の更新も行っていない。

## 5. 成果物
`usir/scripts/f22/`:
- `f22_verify_frozen_sha256.py`、`f22_frozen_sha256_verify_PRE.json`、`f22_frozen_sha256_verify.json`、`pre_switch_sha.log`、`post_switch_sha.log`
- `f22_v11_parity_and_floor_table.py`、`f22_v11_parity_and_floor_table.json`、`f22_table.log`
- `pytest_after_switch_1.log`（切替直後・cwd=`python/`・2 failed）、`pytest_after_switch_2.log`（ルートから・73 passed）
- `harness_test.log`、`cargo_test_workspace.log`、`node_test_cm6.log`

## 6. 未検証・限界
- 床の切替が推敲ラベルに対して妥当かは未検証（§3.4）。
- harness の u6 失敗が切替前から存在することは、切替前コミットでの実行では未確認（§2.2）。
- `kasane-wasm` に実行されるテストは 0 件。
- nishiki の未追跡ファイル `scripts/web_auditor.py`・`scripts/start_web_auditor.bat` は本ステージと無関係で、コミットに含めていない。

## 7. 進捗と NEXT
- 進捗: Task 1〜4 完了。Task 5 は本レポートの配置・コミットで完了。
- NEXT: harness の `test_u6_guards` cwd 相対パス（`test_u6_guards.py:233`）を修正するか、harness の cwd を変更するかの判断（本ステージ対象外）。
