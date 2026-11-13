# Stage B-16 計測器較正レポート：106作品統合ノイズ床生成・3次元プール再計測・F-19再ベンチマーク・実推敲ペア v1.1

- **日付**: 2026-11-13
- **担当範囲**: `usir`（`feat/fable-v5-shootout`）／`nishiki`（`feat/fable-hardcore-v5`）
- **前提・制約**: 評価規則 v2（B-14）適用、F-20 凍結（`ac_max_sis` 常時 BAND_SPECIFIC）維持、本番判定（プール可否・凍結解除・velocity 恒常帯別・Gate 3a/3b・Level 3b）の正式変更は行わない。

---

## エグゼクティブ・サマリー（結論）

1. **タスク1（106作品統合ノイズ床）**: `nishiki gen-paragraph-noise-floor --scope all-106` を実装・実行し、Set 1（46）＋ heldout（30）＋ heldout2（30）＝106作品（22,883 隣接ペア、`[≥2,≥2]` M=323 / L=566）から新規候補 JSON（`paragraph_noise_floor_all_106.json`）を生成した。F-17 床（`paragraph_noise_floor_all_novel.json`）は一切上書きしていない。
2. **タスク2（3次元プール判定再計測）**: 106作品の σ 比 CI（パーセンタイル法、マージン `[0.8, 1.25]`）を再計算した結果、
   - **`ac_coupling`**: 点推定 **1.3347**、95% CI **[1.2143, 1.4723]** → **不合格（帯別維持）**。F-18 の検出力 99.3% の見通し通り、帯別維持が統計的に固まった。
   - **`ac_mean_velocity`**: 点推定 **1.3006**（F-17 床の 1.3093 に引き続き **1.25 の外側**）、95% CI **[0.9815, 1.7093]** → **不合格（帯別維持）**。※恒常帯別の確定判断は本ステージでは行わず Orion の判断に委ねる。
   - **`ac_max_sis`**: F-20 凍結により常時 `BAND_SPECIFIC`。参考診断値として点推定 **0.8809**、95% CI **[0.7261, 1.0713]**（`ci_within_equivalence_margin=false`）。
3. **タスク2.5（106作品母集団での F-19 ベンチマーク）**: `S=5000, B=10000, seed=20261113` で 4 手続きを再実行した。
   - 本番実装されている現行手続き **`percentile` では `ac_max_sis` が empirical（T1max 5.7% > 5.5%, cov 92.4% < 92.5%）で G1/G2 とも不合格、t5（cov 92.3%, noncov_upper 5.5%）で G2 不合格となり、凍結解除条件②は未達**。
   - 一方、代替手続き **`boott`（Bootstrap-t）は全 3 シナリオで `ac_max_sis`（および全 3 次元）が G1 & G2 に合格**した（T1max 2.7〜4.2%、cov 92.9〜95.8%）。ただし `t5` の `ac_coupling` 検出力が `-11.1pt`（G3 未達）、empirical `ac_max_sis` 検出力が `28.8% → 12.7%` へ低下するトレードオフがある。凍結解除自体は本ステージでは実施しない。
4. **タスク2.6（F-20 凍結の下流影響）**: `nishiki` ワークスペース全体（`kasane-wasm` 含む）＋ `usir` 全体を grep し、`ac_multi_window_pooled_dims` / `pool_frozen` / `ac_sigma_pool` を 3 次元前提で読むコードは **0 件**であることを確認した。
5. **タスク3（実推敲ペア v1.1）**: B-15 新規候補 11 組から森鴎外『高瀬舟』（+16 件）と樋口一葉『わかれ道』（+8 件）の計 **+24 ペア**を抽出し、`real_revision_pairs_v1_1.jsonl`（**439 ペア**＝v1 415 + 新規 24）を構築した。全 439 ペアで Python/Rust v2 パリティ検証を実施し、**不一致 0 件（最大差 4.441e-16）**を確認した。
6. **未解決点（Phase B 目標の訂正）**: F-18 スクリプト（`scripts/f18/task1_power.py`）の `K* = 219` は寄与ゼロ作品を含む **総作品数 `K` ベース**（`NW=46` 全体復元抽出）である。したがって B-15 レポートの「`K_M ≥ 220 / K_L ≥ 220`」は過剰表記であり、**総作品数ベースで約 210〜220 作品（寄与率 ~50% のとき `K_M ≈ 105〜110 / K_L ≈ 105〜110`）**に訂正する。

---

## タスク1：106作品統合ノイズ床の生成

- **実装**: `nishiki` (`crates/nishiki-cli/src/paragraph_noise.rs`, `main.rs`) に `--scope all-106`（`Scope::All106`）およびベンチマーク用 TSV ダンプ `--dump-paragraphs-tsv` を追加。
- **対象母集団**: `aozora_manifest.json` (`genre == "novel"`, 46作品) ＋ `aozora_heldout_manifest.json` (30作品) ＋ `aozora_heldout2_manifest.json` (30作品) ＝ **106作品**（全 22,883 隣接段落ペア、10著者、除外 0 作品）。
- **σ 生成母集団**: 評価規則 v2 に基づき、両側複数窓 `[≥2,≥2]` ペアのみ（M帯 `n=323` / `K_M=53`、L帯 `n=566` / `K_L=54`）。
- **出力ファイル**:
  - `nishiki/docs/handoff/evidence/2026-11-13_b16/paragraph_noise_floor_all_106.json`
  - `usir/docs/handoff/v10/paragraph_noise_floor_all_106.json`（SHA-256: `39DA6CC8C2E52ACE514B85C9C26FD79EF41DD2320682AFF736C426ABA969526E`）

### 帯別 σ 比較表（F-17 床 vs 106作品統合床）

| 帯 | 次元 | F-17 床 (`K=46`, M=165 / L=225) | 106作品統合 (`K=106`, M=323 / L=566) | 変化率 |
|---|---|---:|---:|---:|
| **M** | `ac_mean_velocity` | 134.29642 | **130.91763** | -2.52% |
| **M** | `ac_max_sis` | 0.12661 | **0.12769** | +0.85% |
| **M** | `ac_coupling` | 1.40787 | **1.37295** | -2.48% |
| **L** | `ac_mean_velocity` | 102.57322 | **100.65751** | -1.87% |
| **L** | `ac_max_sis` | 0.14936 | **0.14496** | -2.94% |
| **L** | `ac_coupling` | 1.06682 | **1.02863** | -3.58% |

---

## タスク2：3次元のプール判定再計測（K=106、パーセンタイル法 95% CI）

同等マージン `[0.8, 1.25]` での作品単位クラスタブートストラップ（B=10,000, seed=20261113）結果：

| 次元 | 点推定 `σ_M / σ_L` | 95% CI（比） | F-17 床時（参考） | 合否 / 判定 |
|---|---:|---|---|---|
| **`ac_coupling`** | **1.3347** | **[1.2143, 1.4723]** | 1.3197 `[1.1386, 1.5170]` | **不合格（帯別維持）** |
| **`ac_mean_velocity`** | **1.3006** | **[0.9815, 1.7093]** | 1.3093 `[0.9225, 1.9087]` | **不合格（帯別維持）** |
| **`ac_max_sis`** *(F-20凍結)* | *(0.8809)* | *([0.7261, 1.0713])* | *(0.8477 `[0.6305, 1.1583]`)* | **常時 `BAND_SPECIFIC`（CI は診断値のみ）** |

- **`ac_coupling` 結論**: B-15 で予測された検出力 **99.3%** の見通し通り、K=106 では CI 幅が縮小（1.1386..1.5170 → 1.2143..1.4723）しつつ点推定 **1.3347**・上限 **1.4723** ともに 1.25 を明確に上回り、**帯別維持が統計的に固まった**。
- **`ac_mean_velocity` 結論**: 点推定 **1.3006** は F-17 床の 1.3093 に引き続き **1.25 の外側**に留まっている（CI 上限 1.7093）。なお、恒常帯別の確定判断は本ステージでは行わない。

---

## タスク2.5：F-19 ベンチマークの 106作品母集団での再実行

- **実行条件**: `nishiki/scripts/f19/task1_ci_procedures.py` を 106作品 TSV（`set106_paragraphs.tsv`, `NW=106`, `K_active=64`, `stat=centered`）で実行（`S=5,000`, `B=10,000`, `seed=20261113`）。
- **合格基準**: G1（m=1.25 TOST 上側過誤率 ≤ 5.0%、MC上限込み ≤ 5.5%）、G2（両側被覆率 92.5〜97.5% かつ片側非被覆 ≤ 5.0%）。
- **出力**: `nishiki/docs/handoff/evidence/2026-11-13_b16/f19_bench_106works.json` / `f19_gate_106works.json`

### 106作品母集団における `ac_max_sis` 4手続き G1/G2 結果（m=1.25）

| 手続き | シナリオ | T1max (TOST) | 被覆率 (up / lo) | G1 (≤5.5%) | G2 | 検出力 (1.25) |
|---|---|---:|---:|:---:|:---:|---:|
| **percentile** *(現行)* | empirical | **5.7%** | 92.4% (5.7% / 1.9%) | **False** | **False** | 28.8% |
| **percentile** *(現行)* | normal | 3.8% | 93.0% (3.8% / 3.2%) | True | True | 98.1% |
| **percentile** *(現行)* | t5 | 5.4% | 92.3% (5.5% / 2.3%) | True | **False** | 59.1% |
| **bca** | empirical | 4.2% | 93.4% (4.2% / 2.4%) | True | True | 28.0% |
| **bca** | normal | 3.6% | 92.9% (3.5% / 3.6%) | True | True | 98.0% |
| **bca** | t5 | 5.3% | 90.5% (5.7% / 3.9%) | True | **False** | 56.2% |
| **boott** | empirical | **2.7%** | 95.8% (2.7% / 1.5%) | **True** | **True** | 12.7% |
| **boott** | normal | **3.3%** | 93.6% (3.1% / 3.3%) | **True** | **True** | 96.9% |
| **boott** | t5 | **4.2%** | 92.9% (4.4% / 2.7%) | **True** | **True** | 46.9% |
| **analytic_t** | empirical | **6.0%** | 92.0% (6.0% / 2.0%) | **False** | **False** | 18.2% |
| **analytic_t** | normal | 3.2% | 93.8% (3.2% / 3.1%) | True | True | 97.7% |
| **analytic_t** | t5 | 4.4% | 93.6% (4.5% / 2.0%) | True | True | 53.8% |

### 参考：`ac_mean_velocity` / `ac_coupling`（empirical のみ抜粋）
- `ac_mean_velocity`: `percentile` (T1max 1.5%, cov 91.7%, G1=True, **G2=False**), `boott` (T1max 0.4%, cov 92.7%, **G1=True, G2=True**)。
- `ac_coupling`: 全 4 手続きとも empirical で **G1=True, G2=True**（T1max 3.0〜3.6%、cov 93.8〜95.6%、検出力 98.1〜99.2%）。ただし `boott` は `t5` の `ac_coupling` 検出力が `47.1%`（percentile `58.2%` 比 `-11.1pt`）となり G3（`-10pt` 以内）をわずかに下回る。

**検証結論**:
- 本番実装されている現行手続き **`percentile` のもとでは、106作品母集団でも `ac_max_sis` は empirical で G1/G2 不合格・t5 で G2 不合格となり、凍結解除条件②は未達**である（`bca`・`analytic_t` もそれぞれ t5・empirical で不合格）。
- 唯一 **`boott` のみが全 3 シナリオで `ac_max_sis`（および全 3 次元）の G1 & G2 に合格**しており、もし手続きを `boott` へ変更するなら解除条件②を満たし得る。本ステージでは凍結の解除は行わず、判断を Orion に委ねる。

---

## タスク2.6：F-20 凍結の下流影響確認

`nishiki` ワークスペース全体（`crates/`、`packages/` — `kasane-wasm` 含む）および `usir` 全体（`python/`、`packages/`、`scripts/`）を対象に `ac_multi_window_pooled_dims`、`pool_frozen`、`ac_sigma_pool`、`pooled_dims` を `git grep` で検索した。
- **結果**: `crates/nishiki-cli/src/paragraph_noise.rs` 内の JSON 生成・単体テスト箇所のみがヒットし、下流（`diff_harvester.rs`、`kasane-wasm`、`usir` Python/TS コード）で `ac_multi_window_pooled_dims` を 3 次元前提または固定長で読むコードは **0 件**であった（`diff_harvester.rs` は `ac_multi_window_per_dim_std` を次元名キーのマップとして読むのみ）。修正の必要なし。

---

## タスク3：実推敲ペア v1.1 の構築

- **実装**: `usir/scripts/build_real_revision_pairs_v1_1.py`
- **データセット**: `usir/data/usir_v5_eval/real_revision_pairs_v1_1.jsonl`（SHA-256: `55D2DDCF9F0F3AFDDC043E48B6D19632EACA71D601FC3AB98CD24341F78CD9F6`）
  - **総件数**: **439 ペア**（v1 415 件［未変更］＋ 新規 **24 件**）
  - **帯・窓内訳（v1.1 全体）**: `S = 50`, `M = 252`, `L = 137`, `mixed = 84`

### 新規候補 11 組の抽出結果と 0 件理由

| # | グループID | 作品・版 | 抽出数 (S/M/L) | 0件の理由・備考 |
|---|---|---|---:|---|
| 1 | `ogai_takasebune_ed` | 森鴎外『高瀬舟』(000692 vs 000691, 旧字旧仮名) | **16** (9/5/2) | 4件は非本文・50字未満除外、16件採用 |
| 2 | `ichiyo_wakaremichi_ed` | 樋口一葉『わかれ道』(000391 vs 056008, 旧字旧仮名) | **8** (1/2/5) | アライン 8 件すべて採用 |
| 3 | `makino_seibugeki_ed` | 牧野信一『西部劇通信』(045218 vs 052796) | **0** | `whole_dice=0.1501`, 文字数 `5859 vs 1193`（別稿・断片でアライン 0） |
| 4 | `kishida_tonari_ed` | 岸田国士『隣の花』(053193 vs 044891) | **0** | `whole_dice=0.0451`, 文字数 `11753 vs 466`（本編戯曲 vs 短記でアライン 0） |
| 5 | `nakahara_samui_ed` | 中原中也『寒い夜の自我像』(051317 vs 051892) | **0** | 詩作品（`365 vs 584` 字、短行中心でアライン 0） |
| 6〜11 | `miyamoto_*` | 宮本百合子『今年こそは』『文芸時評』『無題（一）〜（四）』 | **0** | `whole_dice=0.0652〜0.4200`（別記事・別年次の同名断片でアライン 0） |

- **パリティ検証（v2, F-17 床）**: Python ミラー（`evaluate_f14_hybrid_pair`）vs Rust `nishiki harvest-diff`（v2）を v1.1 全 **439 ペア**（v1 全 415 件含む）で実行し、**不一致 0 件、最大誤差 `4.441e-16`** を確認（`docs/handoff/v10/real_revision_pairs_v1_1_eval_rule_v2_summary.json`）。

---

## タスク4：検証ゲート

### 1. 凍結成果物 SHA-256 不変ゲート（実行前後 100% 一致）

| ファイル | SHA-256 | 状態 |
|---|---|---|
| `nishiki/.../2026-11-13_f17/paragraph_noise_floor_all_novel.json` | `0575B64A75A8AF8DD1AD0DB81BDF4D8898C473CE8AC1876C6E7E2F8B61A6F604` | **不変** |
| `usir/docs/handoff/v10/paragraph_noise_floor_all_novel.json` | `0575B64A75A8AF8DD1AD0DB81BDF4D8898C473CE8AC1876C6E7E2F8B61A6F604` | **不変** |
| `usir/data/usir_v5_eval/real_revision_pairs_v1.jsonl` | `62DA75271F453BD3A8F161D916FDF7B7B87138B773B564B737F90AEE85B6B8E4` | **不変** |
| `usir/docs/handoff/v10/real_revision_pairs_v1_f17_harvest_summary.json` | `6B650B64F143355D862EB3AE47F12A58E425405BA610339404219335A5BD8611` | **不変** |
| `usir/docs/handoff/v10/real_revision_pairs_v1_eval_rule_v2_summary.json` | `57D74A5EDB5C9C118C408CDFF4A37E047CBA49155C9C86102E87637B63BB2F23` | **不変** |
| `usir/docs/handoff/v10/sparsity_audit_106works_canonical.json` | `D16D95B54C89AD7010CADEA6DA4C1A0D553C2188C24B30CBEA6B0E52C8C6E64A` | **不変** |
| `usir/docs/handoff/v10/sparsity_audit_106works_canonical_v2.json` | `0033B8028A051DB57491803AD1242FDD0E9D552EAA208BCE82596E326D8DE2D4` | **不変** |
| `usir/data/usir_v5_eval/usir_v5_b12_canonical_eval.json` | `50B1EFB38B7D15195EA1DA00CD049E51126F5100E90E03497FC7123D0B5C96FC` | **不変** |

### 2. テストスイート実行結果

- **`usir`**: `uv run --project python pytest python/tests/v5/` → **73 passed, 0 failed** (173.27s)
- **`nishiki`**: `cargo test --workspace`（`kasane-wasm` 含む）→ **536 passed, 0 failed, 1 ignored**（`audit_17_corpus::test_generate_17_corpus_audit_data` の `#[ignore]` 1件のみ）

---

## 未解決点の検証結果：F-18 `K*` の定義と Phase B 目標の訂正

`nishiki/scripts/f18/task1_power.py`（行 18, 75）を確認した結果、`K*`（`ac_max_sis` = **219**, `ac_mean_velocity` = **554**）は **`NW = 46`（寄与ゼロ 20 作品を含む Set 1 全作品）からの復元抽出における総作品数 `K`** として定義されている。
- Set 1 では `46` 作品中 `M` 寄与 `22` 作品（47.8%）、`L` 寄与 `24` 作品（52.2%）であり、`K* = 219` は有効寄与作品数に換算すると **`K_M ≈ 105 / K_L ≈ 114`（`K_active ≈ 124`）** に相当する。
- したがって、B-15 レポート（§4.1）の「`K_M ≥ 220 / K_L ≥ 220`」という表現は総作品数 `K*` と帯別有効寄与数を混同したものであり、**Phase B の目標値は総作品数ベースで `K ≈ 210〜220 作品`（M/L 各帯寄与率 ~50% なら `K_M ≈ 105〜110 / K_L ≈ 105〜110`）** に言い直して確定する。
