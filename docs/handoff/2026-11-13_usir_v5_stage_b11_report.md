# USIR v5 Stage B-11 総合検証レポート：Gate 3a 有限標本較正の実装反映と F-14 ハイブリッド判定の参照整合

**作成日**: 2026-11-13  
**ブランチ**: `feat/fable-v5-shootout`  
**前提ベースコミット**: `8b82177` (Stage B-10 完了; 直前 HEAD `7d1be27`)  
**Stage B-11 実装コミットハッシュ**: `7592579` (`7592579812f4212faeb351f7753b2650cf5f39ce`)  
**関連上流（`C:\dev\nishiki` F-14）**: `48b6eb7`（ハイブリッド 8/11D & `all-novel` $\sigma$）・`fffae58`（clippy修正）・`fe7d058`（F-14 報告書）・`fa14019`（518 tests 完了・`ac_*` ブートストラップ CI 追補）  
**規約遵守**:
- 妻原稿（層B-1）：完全不使用
- 有料 API：完全不使用（課金 \$0.00）
- **Gate 3a 判定**：**`SHIFT_QUANTIFIED_NOT_PASSED`** 維持（閾値更新は「判定基準の較正」であり合否判定の変更ではない）
- **Gate 3b 判定**：**`GATE3B_FROZEN`** 維持（凍結継続・着手禁止）
- **Stage B-12**：未設計・自動開始厳禁
- **`C:\dev\nishiki` リポジトリ**：変更ゼロ保証（`git status --short` クリーン）

---

## 1. エグゼクティブサマリー

Stage B-11 では、Stage B-10 で数理的に解明された「有限標本 LOO Student-$t$ インフレ（$1.2444\times$）と R4 相関構造」に基づく **Gate 3a 外れ値判定閾値の有限標本 Monte Carlo 較正の実装反映（タスク1）**、および **`nishiki` F-14 で確定したウィンドウ数ハイブリッド判定（8D/11D）と再生成 $\sigma$ テーブルの Python 側参照整合（タスク2）** を完了した。

### 主要成果概要
1. **タスク1（Gate 3a 閾値の有限標本較正）**:
   - `experiments/gate3a_register_centering_eval.py` において、24D 連続コア空間（`ortho24_cont`）および場面分散 $\sigma^2_{\text{scene}, r}$ 分母を正準とし、B-10 成果物 JSON (`data/usir_v5_eval/usir_v5_b10_gate3a_scene_scale_and_level3.json`) の `finite_sample_mc_quantiles_24d` から分位点を読み込む機構（未格納時は `seed=42`, `5000` 回試行, LOO Student-$t$ インフレ $1.2444\times$ で自動再計算する `get_or_compute_b10_mc_quantiles` / `compute_b10_mc_quantiles`）を実装した。
   - 64件ベンチマーク（`data/usir_v5_eval/tier_c_modern_benchmark_expanded.jsonl`）を再評価した結果、Condition (a)（無中心化・$\sigma^2_{\text{scene}, r}$ 分母）において、R4 近代対照群の LOO 外れ値率は旧漸近 $\chi^2_{0.95}$ ($d=6.0345$) での **25.0%** から、有限標本 MC 較正閾値では **6.25%（独立正規 95% 閾値 $d=6.9267$）／0.00%（独立正規 99% 閾値 $d=7.7032$ および R4 相関正規 95%/99% 閾値 $d=7.6574 / 9.0381$）** へと正常化した。一方、現代散文（R1〜R3 プール $n=48$）は依然として **100.0%（MC 独立正規 95%）／95.83%（MC 独立正規 99% 縮約LOO）** であり、戦前文学規範からの分布シフトは健在であるため **Gate 3a は `SHIFT_QUANTIFIED_NOT_PASSED` を維持** する。
2. **タスク2（F-14 ハイブリッド判定・新 $\sigma$ 参照整合）**:
   - `python/usir_py/v5/f14_hybrid_noise_floor.py` を新設し、Rust（`kasane_core::IncrementalAnalyzer::split_sentences` / `TrajectoryAnalyzer::analyze` および `nishiki diff_harvester.rs`）と完全同一の文字スライス境界（改行 `\n`・閉じ括弧 `」` 保持）とハイブリッド次元ルール（`w_before <= 1 and w_after <= 1` $\to$ 8次元 `both_single_window:8`、片側でも $\ge 2$ 窓 $\to$ 11次元 `any_multi_window:11`）を実装した。
   - F-14 再生成版 $\sigma$ テーブル（`docs/handoff/v10/paragraph_noise_floor_all_novel.json`、SHA-256: `00C39814...`）とブートストラップ CI（`docs/handoff/v10/ac_sigma_bootstrap_ci.json`、SHA-256: `FDFEF952...`）を取り込み、11D 判定時に Band M/L の `ac_multi_window_per_dim_std` を参照し、完全な由来（provenance）と非混同ガードを記録した。
   - **Python vs Rust `n_evaluated` 整合検証**:
     - **Set 1（青空文庫 46作品・10,859 隣接段落ペア）**: Band S/M/L すべてで窓構成（`both_single`, `mixed_one_side_multi`, `both_multi`）と `n_evaluated` の不一致 **0 / 10,859件（0.00%）**。
     - **Set 2（U-6 改稿 86ペア）**: `nishiki harvest-diff` 実バイナリとの対比で、段落アライナが整合した **77 / 77ペア（Dice $\ge 0.25$）について `window_counts` / `band` / `n_evaluated` / `ac_sigma_source` の不一致 0件**。残り 9ペア（`A_29, A_35, B_19, B_20, B_22, C_19, C_20, C_21, C_22`）は Dice $< 0.25$ により Rust CLI の段落アライナが `kind="unaligned"` として `n_evaluated` 対象外とする一方、Python の事前対応済みペア評価では 8D（6件） / 11D（3件）として全件評価することを乖離要因として明示報告する。
     - **Set 3（`STYLE_POLISH` 28ペア $\times$ 3粒度 = 84評価）**: `span_only` (22/6), `local_context` (7/21), `whole_scene` (0/28) の全84評価で Rust CLI との不一致 **0件**。
3. **回帰テスト合格**:
   - 新設した `python/tests/v5/test_b11_gate3a_and_f14_parity.py`（5テスト）を含む `pytest python/tests/v5/ -v` **全 57 件が 100% PASSED**（133.62s）。

---

## 2. 着手前確認ゲートと規約遵守ステータス

| 確認項目 | 確認結果・実測値 | 状態 |
|---|---|---|
| Stage B-10 ベースコミット | `8b82177`（直前 HEAD: `7d1be27` HW-32 スクリプト整備） | 確認済み |
| `nishiki` F-14 ワークスペーステスト | 518 passed, 0 failed（`fa14019`） | 完了済み |
| `ac_*` $\sigma$ ブートストラップ CI | Band M (`n=165`, 22作品), Band L (`n=225`, 24作品)（`ac_sigma_bootstrap_ci.json`） | 確認済み |
| 妻原稿（層B-1）・有料 API | 完全不使用（\$0.00） | 遵守 |
| Gate 3a / Gate 3b / B-12 制約 | `SHIFT_QUANTIFIED_NOT_PASSED` / `GATE3B_FROZEN` / B-12 未設計・禁止 | 遵守 |

---

## 3. タスク1：Gate 3a 有限標本較正の実装反映と 64件ベンチマーク再評価

### 3.1 実装内容と Monte Carlo 帰無分布分位点
- **正準空間・分母**: `ortho24_cont`（24次元連続コア空間）における対角場面分散 $\sigma^2_{\text{scene}, r}$。
- **有限標本較正の実装**: `experiments/gate3a_register_centering_eval.py` に `compute_b10_mc_quantiles()` および `get_or_compute_b10_mc_quantiles()` を実装した。`data/usir_v5_eval/usir_v5_b10_gate3a_scene_scale_and_level3.json` 内の `finite_sample_mc_quantiles_24d` を優先参照し、欠落時は B-10 §4 同一条件（`seed=42`, `n_trials=5000`, $n_r=16, m=15, \nu=14$, 理論 LOO Student-$t$ インフレ $(16/15)(14/12) = 1.244444\times$, 計 `80,000` LOO 試行）で再計算する。
- **新旧判定閾値一覧（24D `ortho24_cont`、距離 $d = \sqrt{D^2}$）**:

| 閾値種別 | 分類 | $D^2$ 閾値 | $d$ 閾値 | 備考（真の帰無分布での超過率） |
|---|---|---|---|---|
| **旧漸近 $\chi^2_{0.95}$** ($df=24$) | 漸近基準（未較正） | 36.4150 | **6.0345** | LOO ($n_r=16$) 帰無分布での実質超過率 **22.29%〜24.51%** |
| 旧漸近 $\chi^2_{0.99}$ ($df=24$) | 漸近基準（未較正） | 42.9798 | 6.5559 | 参照用漸近 99% |
| 旧漸近 $\chi^2_{0.9973}$ ($3\sigma$, $df=24$) | 漸近基準（未較正） | 47.7634 | 6.9111 | 参照用漸近 $3\sigma$ |
| 青空 106作品 Atlas $3\sigma$ | 対照基準 | — | 7.2641 | 戦前作品全体基準（参考） |
| **新 MC 独立正規 95%** (`mc_indep_p95`) | **有限標本較正基準** | **47.9796** | **6.9267** | 5,000試行実測 $\mathbb{E}[D^2]=29.8831$（実測インフレ $1.2451\times$） |
| **新 MC 独立正規 99%** (`mc_indep_p99`) | **有限標本較正基準** | **59.3388** | **7.7032** | 独立 24D LOO 帰無分布 99% 分位点 |
| **新 MC R4 相関正規 95%** (`mc_r4corr_p95`) | **有限標本較正基準** | **58.6355** | **7.6574** | R4 23有効次元相関構造を含む帰無分布 95% 分位点 |
| **新 MC R4 相関正規 99%** (`mc_r4corr_p99`) | **有限標本較正基準** | **81.6877** | **9.0381** | R4 相関帰無分布 99% 分位点 |

*(参考: 31D `ortho_v2_31d` 独立正規 MC 分位点は 95%=`7.6774`, 99%=`8.4392`, 実測インフレ `1.2446x`)*

### 3.2 64件ベンチマーク（`tier_c_modern_benchmark_expanded.jsonl`）新旧閾値対比結果

#### 表 3-1: Condition (a)（無中心化・$\sigma^2_{\text{scene}, r}$ 分母、24D `ortho24_cont`）
| 分散モード | コホート ($n$) | 平均 $d_M$ | 最大 $d_{\max}$ | 旧 $\chi^2_{0.95}$ | 旧 $\chi^2_{0.99}$ | **新 MC indep 95%** | **新 MC indep 99%** | **新 MC R4corr 95%** | **新 MC R4corr 99%** |
|---|---|---|---|---|---|---|---|---|---|
| **`in_sample`** | **R4 (16)** | 4.8108 | 6.4790 | 6.25% | 0.00% | **0.00%** | **0.00%** | **0.00%** | **0.00%** |
| | R1 (16) | 10.0285 | 11.6329 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 81.25% |
| | R2 (16) | 201.3124 | 201.4285 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| | R3 (16) | 283.9419 | 284.0867 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| | **R1〜R3 プール (48)** | 165.0943 | 284.0867 | 100.00% | 100.00% | **100.00%** | **100.00%** | **100.00%** | **93.75%** |
| **`shrinkage_loo`** | **R4 (16)** | 5.3354 | 7.5838 | **25.00%** | 6.25% | **6.25%** | **0.00%** | **0.00%** | **0.00%** |
| | R1 (16) | 9.1969 | 10.6735 | 100.00% | 100.00% | 100.00% | 87.50% | 87.50% | 62.50% |
| | R2 (16) | 14.7871 | 15.6527 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| | R3 (16) | 20.2337 | 22.1490 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| | **R1〜R3 プール (48)** | 14.7393 | 22.1490 | 100.00% | 100.00% | **100.00%** | **95.83%** | **95.83%** | **87.50%** |
| **`unregularized_loo`** | **R4 (16)** | 5.1874 | 7.2194 | **25.00%** | 6.25% | **6.25%** | **0.00%** | **0.00%** | **0.00%** |
| | R1 (16) | 10.1270 | 11.8681 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 75.00% |
| | R2 (16) | 201.3101 | 201.4500 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| | R3 (16) | 287.9511 | 347.9940 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| | **R1〜R3 プール (48)** | 166.4627 | 347.9940 | 100.00% | 100.00% | **100.00%** | **100.00%** | **100.00%** | **91.67%** |

#### 表 3-2: Condition (b)（Scheme 3 レジスター内 LOO 中心化・診断専用、24D `ortho24_cont`）
*注記: Condition (b) の収束は摂取正規化（intake-normalization）診断であり、Gate 3a 通過証拠には数えない。*

| 分散モード | コホート ($n$) | 平均 $d_M$ | 最大 $d_{\max}$ | 旧 $\chi^2_{0.95}$ | 旧 $\chi^2_{0.99}$ | **新 MC indep 95%** | **新 MC indep 99%** | **新 MC R4corr 95%** | **新 MC R4corr 99%** |
|---|---|---|---|---|---|---|---|---|---|
| **`in_sample`** | **R4 (16)** | 4.8781 | 6.3396 | 12.50% | 0.00% | **0.00%** | **0.00%** | **0.00%** | **0.00%** |
| | **R1〜R3 プール (48)** | 4.6986 | 6.5090 | 6.25% | 0.00% | **0.00%** | **0.00%** | **0.00%** | **0.00%** |
| **`shrinkage_loo`** | **R4 (16)** | 5.4499 | 7.5460 | **31.25%** | 12.50% | **6.25%** | **0.00%** | **0.00%** | **0.00%** |
| | R1 (16) | 4.7468 | 5.8782 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| | R2 (16) | 4.5664 | 6.4051 | 6.25% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| | R3 (16) | 4.0976 | 6.0690 | 6.25% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| | **R1〜R3 プール (48)** | 4.4703 | 6.4051 | 4.17% | 0.00% | **0.00%** | **0.00%** | **0.00%** | **0.00%** |
| **`unregularized_loo`** | **R4 (16)** | 5.2987 | 7.1924 | 12.50% | 12.50% | **6.25%** | **0.00%** | **0.00%** | **0.00%** |
| | R1 (16) | 5.1986 | 6.4527 | 18.75% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| | R2 (16) | 5.2339 | 7.3014 | 25.00% | 6.25% | 6.25% | 0.00% | 0.00% | 0.00% |
| | R3 (16) | 14.6691 | 166.7421 | 12.50% | 12.50% | 12.50% | 6.25% | 6.25% | 6.25% |
| | **R1〜R3 プール (48)** | 8.3672 | 166.7421 | 18.75% | 6.25% | **6.25%** | **2.08%** | **2.08%** | **2.08%** |

### 3.3 ガバナンス上の整理（規約厳守）
1. **閾値変更の意味**: 本更新は有限標本 LOO 効果に対する「判定基準の較正（calibration）」であり、現代散文（R1〜R3）の戦前規範からの分布シフトは Condition (a) で 87.5%〜100.0% 維持されているため、**Gate 3a の公式判定は `SHIFT_QUANTIFIED_NOT_PASSED` を維持** する。
2. **Scheme 2 の扱い**: **参考値のまま** とし、自己レジスター平均を減算した差分を他分布（青空文庫アトラス）の分散で除算する同義反復的補正（`同義反復的補正（self-register mean subtraction divided by another distribution's sigma）`）である旨の注記を維持する。
3. **Scheme 3 の扱い**: Condition (b) における全指標 0.00%（`shrinkage_loo` 現代プール）への収束は、摂取時レジスター条件付き正規化が設計どおり機能することを示す **診断値** に留まり、Gate 3a 通過証拠には一切算入しない。

---

## 4. タスク2：F-14 ハイブリッド判定・新 $\sigma$ 参照整合と Python/Rust 一致検証

### 4.1 参照 $\sigma$ テーブルと由来（Provenance）
`python/usir_py/v5/f14_hybrid_noise_floor.py` は以下の実測テーブルを参照し、実行時 JSON 出力に SHA-256・生成コマンド・標本母集団の由来を記録する：

- **正準ノイズ床テーブル**: `docs/handoff/v10/paragraph_noise_floor_all_novel.json`
  - 上流元: `C:/dev/nishiki/docs/handoff/evidence/2026-11-13_f14/paragraph_noise_floor_all_novel.json`（`nishiki` コミット `48b6eb7`）
  - SHA-256: `00c398144abe13288451a6c297438e884574eef42e0b221ee2369f302db03435`
  - 母集団・定義: `--scope all-novel`（46 作品, 10 作家, **10,859 隣接段落ペア**、両段落 $\ge 30$ 字の差分 `p_{t+1} - p_t` の母集団 SD `ddof=0`、`sigma_kind = "adjacent_paragraph_diff_sd"`）
  - コード注記: `f14_hybrid_noise_floor.py` 内定数 `F14_SIGMA_PROVENANCE` には実装コミット `48b6eb7` とクリップ補正 `fffae58` を記載（ブートストラップ CI 文書追補コミットは上流 `fa14019`）。
- **ブートストラップ 95% CI 検証ファイル**: `docs/handoff/v10/ac_sigma_bootstrap_ci.json`
  - 上流元: `C:/dev/nishiki/docs/handoff/evidence/2026-11-13_f14/ac_sigma_bootstrap_ci.json`（`nishiki` コミット `fa14019`）
  - SHA-256: `fdfef9522de47bdb44c05a6ad2491875e048cc37e29d48d312e176bebe80a964`

#### 表 4-1: 長さ帯別 `ac_multi_window_per_dim_std` と 95% ブートストラップ CI
| 長さ帯 ($L_{\max}$) | 全ペア数 | 両側複窓数 | `ac_mean_velocity` (95% CI) | `ac_max_sis` (95% CI) | `ac_coupling` (95% CI) | `ac_mean_crest` 全ペア SD |
|---|---|---|---|---|---|---|
| **S** ($\le 250$) | 8,258 | 0 | `null`（構造的ゼロ・8D 除外） | `null`（構造的ゼロ・8D 除外） | `null`（構造的ゼロ・8D 除外） | 0.3640 |
| **M** ($251\text{–}500$) | 1,952 | 165 (22作品) | **134.2964** `[101.195, 168.298]` | **0.1266** `[0.1083, 0.1449]` | **1.4079** `[1.2940, 1.5011]` | 0.5379 |
| **L** ($\ge 501$) | 649 | 225 (24作品) | **102.5732** `[86.374, 118.518]` | **0.1494** `[0.1327, 0.1652]` | **1.0668** `[0.9885, 1.1400]` | 0.7740 |

### 4.2 非混同ガード（Gate 3a $\sigma_{\text{scene}, r}$ vs Harvest ノイズ床 $\sigma$）
両者は名前が似ていても**空間・単位・分母対象・統計的役割が完全に別物**であるため、同一視・流用を厳禁とする：

| 区分 | Gate 3a 場面分散 ($\sigma^2_{\text{scene}, r}$) | Harvest 段落ノイズ床 ($\sigma$) |
|---|---|---|
| **特徴空間** | `ortho24_cont` (24D) / `ortho_v2_31d` (31D) | `kasane11_v1` (11D / 8D 活性) |
| **計算単位・母集団** | 同一レジスター短文場面（400〜800字, $n_r=16$）の場面間分散（不偏 `ddof=1`） | 青空文庫 46作品の**同一作品内・隣接段落差分** $p_{t+1} - p_t$（10,859ペア、母 SD `ddof=0`） |
| **役割** | 現代散文と戦前規範との**レジスター分布シフト検定** | 改稿前後の $|\Delta| / \sigma$ **段落自然揺らぎスケール参照** |
| **Python 実装上の隔離** | `experiments/gate3a_register_centering_eval.py` | `python/usir_py/v5/f14_hybrid_noise_floor.py` (`sigma_kind="adjacent_paragraph_diff_sd"`) |

### 4.3 ハイブリッド判定ルールと Python / Rust `n_evaluated` 一致検証

`experiments/scale_separated_level3.py::evaluate_f14_hybrid_alignment_and_parity()` により、ウィンドウ分割（`window_chars=250, stride_chars=125`）・文境界分割（`split_sentences_kasane`、Rust と同じく `.\strip()` を行わず `\n` と閉じ括弧をスライス内に保持）・ハイブリッド判定ルール（`w_before <= 1 and w_after <= 1` $\to$ `n_evaluated=8` [`both_single_window:8`]、片側でも $\ge 2$ 窓 $\to$ `n_evaluated=11` [`any_multi_window:11`]）を 3 つの実ペア集合で検証した。

#### 1) Set 1: 青空文庫 46作品・10,859 隣接段落ペア（対 `paragraph_noise_floor_all_novel.json`）
| 帯域 ($L_{\max}$) | ペア総数 | 両側1窓 (`both_single` $\to$ **8D**) | 片側複窓 (`mixed` $\to$ **11D**) | 両側複窓 (`both_multi` $\to$ **11D**) | 8D 計 / 11D 計 | Rust F-14 との不一致 |
|---|---|---|---|---|---|---|
| **Band S** ($\le 250$) | 8,258 | 8,258 | 0 | 0 | **8,258** / **0** | **0** |
| **Band M** ($251\text{–}500$) | 1,952 | 621 | 1,166 | 165 | **621** / **1,331** | **0** |
| **Band L** ($\ge 501$) | 649 | 0 | 424 | 225 | **0** / **649** | **0** |
| **合計** | **10,859** | **8,879** | **1,590** | **390** | **8,879** / **1,980** | **0 / 10,859 (0.00%)** |

*(注目点: Band M 1,952件のうち 621件が `both_single_window:8` に切り替わり、1,331件が `any_multi_window:11` で `ac_multi_window_per_dim_std` を適用する。)*

#### 2) Set 2: U-6 改稿データセット 86ペア（対 `C:\dev\nishiki\target\f14\release\nishiki.exe harvest-diff`）
| クラス ($n$) | Python 全86ペア評価 | 帯域分布 (S/M/L) | `ac_sigma_source` | Rust CLI アライン済みペアでの一致 |
|---|---|---|---|---|
| `Class_A_micro_polish` (42) | `n_eval=8`: **42**, `n_eval=11`: 0 | S:42 / M:0 / L:0 | `not_applicable`: 42 | アライン済み 40/42 で **0 不一致** |
| `Class_B_macro_revision` (22) | `n_eval=8`: 0, `n_eval=11`: **22** | S:0 / M:4 / L:18 | `multi_window_pairs`: 22 | アライン済み 19/22 で **0 不一致** |
| `Class_C_content_rewrite` (22) | `n_eval=8`: **22**, `n_eval=11`: 0 | S:22 / M:0 / L:0 | `not_applicable`: 22 | アライン済み 18/22 で **0 不一致** |
| **合計 (86)** | **8D: 64 / 11D: 22** | — | — | **77 / 77 アライン済み一致（0 不一致）** |

> **CLI パリティ上の構造的乖離報告（9 / 86ペア）**:  
> `nishiki harvest-diff` CLI の段落アライメント段階（`align_min_similarity = 0.25`、文字 2-gram Dice 係数）において、大幅書換えまたは短文置換を含む 9ペア（`A_29, A_35, B_19, B_20, B_22, C_19, C_20, C_21, C_22`）は Dice $< 0.25$ となり、`harvest-diff` が `kind: "unaligned"`（`n_evaluated` フィールド非出力）として弾く。一方、Python 側の事前対応済みペア評価（`u6_calibration_and_level3_dataset.jsonl`）はアライン済みデータセットを入力とするため、当該 9ペアにもルールを適用し A/C 6件を **8D**、B 3件を **11D** と評価する。アライナを通過した **77/77ペア** については、`window_counts`、`band`、`n_evaluated`、`ac_sigma_source` のすべてが 100% 完全一致した。

#### 3) Set 3: `STYLE_POLISH` 28ペア $\times$ 3粒度 = 84評価（対 `nishiki.exe harvest-diff`）
| 粒度 | 評価数 | `n_eval=8` | `n_eval=11` | 帯域内訳 (S / M / L) | Rust CLI との不一致 |
|---|---|---|---|---|---|
| `span_only` | 28 | **22** | **6** | S:17 / M:8 (うち5件が8D, 3件が11D) / L:3 | **0 / 28** |
| `local_context` | 28 | **7** | **21** | S:6 / M:8 (うち1件が8D, 7件が11D) / L:14 | **0 / 28** |
| `whole_scene` | 28 | **0** | **28** | S:0 / M:2 / L:26 | **0 / 28** |

---

## 5. 冷徹な自己監査（死角・誤爆・取りこぼし・理論と物理の乖離）

1. **誤爆（False Positive）の検証と残存境界**:
   - **Gate 3a 側**: MC 較正により、R4 自己対照群における LOO 外れ値誤爆は $\chi^2_{0.95}$ の **25.00%（4/16件）** から **6.25%（1/16件、独立正規 95%）／0.00%（R4 相関 95%・両モデル 99%）** に抑制された。ただし、標本サイズ $n_r=16$ では 1件の極端値が 6.25% に直結するため、独立正規 95% 下でも確率的変動 $\pm 6.25\%$ を含む。
   - **F-14 側**: Set 1（10,859ペア）および Set 3（84評価）では全件一致したが、`harvest-diff` 単発ペア比較時に前後の空行正規化を行わないと複数段落分割が走る恐れがあり、Python テストハーネス（`_run_rust_harvest_pair`）では `SENTINEL_UNCHANGED_PARAGRAPH` 結合で単一段落比較を強制している。
2. **取りこぼし（False Negative）の検証**:
   - **Gate 3a 側**: R4 相関正規 99% 閾値（$d = 9.0381$）を R1 現代小説に適用した場合、`shrinkage_loo` で R1 の **37.50%（6/16件、検出率 62.50%）** が閾値内に取り込まれる（`in_sample` でも R1 で 18.75% 取りこぼし）。R4 相関 99% 閾値は保守的すぎるため、分布シフト監視の一次警報としては `mc_indep_p95` ($d=6.9267$、R1 検出率 100.0%) または `mc_r4corr_p95` ($d=7.6574$、R1 `shrinkage_loo` 検出率 87.50%・R2/R3 100.0%) との併用が必要である。
   - **F-14 側**: 前述のとおり、Dice $< 0.25$ の 9/86 改稿ペアは `nishiki harvest-diff` 段落アライナ段階で `unaligned` 扱いとなり `n_evaluated` が算出されない。さらに、改行 `\n` を含む段落で `strip()` を行うと Rust `IncrementalAnalyzer::split_sentences` の文字長判定（250/500字境界、1窓/2窓境界）と最大数文字のズレを生じるリスクがある（`split_sentences_kasane` では `.strip()` 禁止として解消済み）。
3. **理論と物理的事実の乖離**:
   - **B-11 に伴う Python 特徴抽出器の仕様スコープ**: `experiments/scale_separated_level3.py` 内の歴史的 H1/H2 評価における `extract_kasane8_active` は既存配列先頭 8要素スライスであり（B-9/B-10 確定値）、Stage B-11 で追加した `evaluate_f14_hybrid_alignment_and_parity()` はこれと切り離して、F-14 の「`ac_mean_velocity` / `ac_max_sis` / `ac_coupling` の 3次元を除外し `ac_mean_crest` を残す 8次元定義（`KASANE8_ACTIVE_DIM_NAMES`）」とウィンドウ数ルールを検証・参照整合する設計とした。将来の共通特徴量統合時（B-12 以降の別タスク、今回は着手禁止）に `KASANE8_ACTIVE_DIM_NAMES` とのインデックス統一を確認する必要がある。

---

## 6. テスト結果（`pytest python/tests/v5/ -v` 実出力）

全 57 件の単体・統合テストが 100% 合格した（失敗 0件）。

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.2.0, pluggy-1.7.0 -- C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe
cachedir: .pytest_cache
rootdir: C:\dev\usir\python
configfile: pyproject.toml
plugins: anyio-4.15.1
collecting ... collected 57 items

python\tests\v5\test_b11_gate3a_and_f14_parity.py::test_b11_gate_status_and_non_conflation_invariants PASSED [  1%]
python\tests\v5\test_b11_gate3a_and_f14_parity.py::test_b10_mc_quantiles_exact_reproducibility PASSED [  3%]
python\tests\v5\test_b11_gate3a_and_f14_parity.py::test_b11_old_vs_new_thresholds_on_64_benchmark PASSED [  5%]
python\tests\v5\test_b11_gate3a_and_f14_parity.py::test_f14_window_count_and_hybrid_rule_mechanics PASSED [  7%]
python\tests\v5\test_b11_gate3a_and_f14_parity.py::test_b11_f14_parity_artifact_and_zero_discrepancies PASSED [  8%]
python\tests\v5\test_gate3a_scene_scale.py::test_gate_status_constants PASSED [ 10%]
python\tests\v5\test_gate3a_scene_scale.py::test_scene_variance_calculations_and_rank_deficiency PASSED [ 12%]
python\tests\v5\test_gate3a_scene_scale.py::test_condition_a_and_b_behavior PASSED [ 14%]
python\tests\v5\test_gate3a_scene_scale.py::test_b10_json_artifact_schema_and_values PASSED [ 15%]
python\tests\v5\test_modern_text_normalizer.py::test_normalize_modern_punctuation PASSED [ 17%]
python\tests\v5\test_modern_text_normalizer.py::test_web_novel_rubi_removal PASSED [ 19%]
python\tests\v5\test_modern_text_normalizer.py::test_dialogue_dominant_layout_and_nan_guards PASSED [ 21%]
python\tests\v5\test_modern_text_normalizer.py::test_cleanroom_irreversible_profile_invariant PASSED [ 22%]
python\tests\v5\test_modern_text_normalizer.py::test_mask_entities_to_black_square PASSED [ 24%]
python\tests\v5\test_narrative_pacing.py::test_hw22_metaphor_leap PASSED [ 26%]
python\tests\v5\test_narrative_pacing.py::test_hw24_cliffhanger PASSED   [ 28%]
python\tests\v5\test_narrative_pacing.py::test_hw25_uid_ratio PASSED     [ 29%]
python\tests\v5\test_narrative_pacing.py::test_extract_all PASSED        [ 31%]
python\tests\v5\test_ortho_v2.py::test_content_char_detection PASSED     [ 33%]
python\tests\v5\test_ortho_v2.py::test_content_char_count PASSED         [ 35%]
python\tests\v5\test_ortho_v2.py::test_v2_m1_non_comma_invariance PASSED [ 36%]
python\tests\v5\test_scene_splitter.py::test_all_106_works_present PASSED [ 38%]
python\tests\v5\test_scene_splitter.py::test_scene_continuity_per_work PASSED [ 40%]
python\tests\v5\test_scene_splitter.py::test_scene_length_bounds PASSED  [ 42%]
python\tests\v5\test_scene_splitter.py::test_usir_schema_strict_compliance PASSED [ 43%]
python\tests\v5\test_scene_splitter.py::test_degenerate_and_boundary_inputs PASSED [ 45%]
python\tests\v5\test_scene_splitter.py::test_pilot_50_scenes PASSED      [ 47%]
python\tests\v5\test_scene_splitter.py::test_fix1_melos_and_lemon_awase PASSED [ 49%]
python\tests\v5\test_scene_splitter.py::test_fix2_sadame_characters_and_edges PASSED [ 50%]
python\tests\v5\test_scene_splitter.py::test_fix3_kamae_opened_and_closed_hooks PASSED [ 52%]
python\tests\v5\test_scene_splitter.py::test_fix4_nagare_continuous_logp_and_mood_diversity PASSED [ 54%]
python\tests\v5\test_scene_splitter.py::test_fid_5_triggers_and_adjacent_dialogue_edges PASSED [ 56%]
python\tests\v5\test_scene_splitter.py::test_within_work_isf_hook_filtering PASSED [ 57%]
python\tests\v5\test_scene_splitter.py::test_pilot_50_split_is_schema_conformant_and_lossless PASSED [ 59%]
python\tests\v5\test_six_pole_atlas.py::test_narrative_rg_normalizer_features PASSED [ 61%]
python\tests\v5\test_six_pole_atlas.py::test_normalized_hsic_properties PASSED [ 63%]
python\tests\v5\test_six_pole_atlas.py::test_svd_lowdin_orthogonalization PASSED [ 64%]
python\tests\v5\test_six_pole_atlas.py::test_goldilocks_void_finder_logic PASSED [ 66%]
python\tests\v5\test_six_pole_atlas.py::test_order0_and_order1_calibration_artifacts PASSED [ 68%]
python\tests\v5\test_six_pole_atlas.py::test_order2_and_order3_diachronic_artifacts PASSED [ 70%]
python\tests\v5\test_u6_guards.py::test_numeral_parser PASSED            [ 71%]
python\tests\v5\test_u6_guards.py::test_numeral_guard_identical_value_style_polish PASSED [ 73%]
python\tests\v5\test_u6_guards.py::test_numeral_guard_value_change_content_rewrite PASSED [ 75%]
python\tests\v5\test_u6_guards.py::test_negation_guard_polarity_flip_content_rewrite PASSED [ 77%]
python\tests\v5\test_u6_guards.py::test_negation_guard_double_negation_exemption PASSED [ 78%]
python\tests\v5\test_u6_guards.py::test_negation_guard_lexical_prefix_ignored PASSED [ 80%]
python\tests\v5\test_u6_guards.py::test_numeral_closed_class_idiom_exclusions PASSED [ 82%]
python\tests\v5\test_u6_guards.py::test_numeral_closed_class_man_ga_ichi_exclusion PASSED [ 84%]
python\tests\v5\test_u6_guards.py::test_numeral_adverbial_juuppun_exclusion PASSED [ 85%]
python\tests\v5\test_u6_guards.py::test_negation_adverb_mazu_exclusion PASSED [ 87%]
python\tests\v5\test_u6_guards.py::test_negation_conjecture_modality_exclusion PASSED [ 89%]
python\tests\v5\test_u6_guards.py::test_negation_adjective_stems_exclusion PASSED [ 91%]
python\tests\v5\test_u6_guards.py::test_numeral_placeholder_orthographic_protection PASSED [ 92%]
python\tests\v5\test_u6_guards.py::test_numeral_fraction_ratio_jubun_no_ichi PASSED [ 94%]
python\tests\v5\test_u6_guards.py::test_numeral_single_unit_kanji_hyaku PASSED [ 96%]
python\tests\v5\test_u6_guards.py::test_u6_test_vectors_fixture_pass PASSED [ 98%]
python\tests\v5\test_u6_guards.py::test_u6_profiles PASSED               [100%]

======================= 57 passed in 133.62s (0:02:13) ========================
```

---

## 7. 生成・変更ファイルおよび SHA-256 台帳

1. **F-14 ハイブリッド判定・由来管理モジュール（新規）**:
   - `python/usir_py/v5/f14_hybrid_noise_floor.py` (SHA-256: `D335324E7C538F799584C12357946FDE9DD9DBCF224C1FC0AE70308E407117C8`)
   - `python/usir_py/v5/__init__.py`（F-14 API エクスポート追加）
2. **F-14 参照成果物（上流 `nishiki` F-14 からの複写固定）**:
   - `docs/handoff/v10/paragraph_noise_floor_all_novel.json` (SHA-256: `00C398144ABE13288451A6C297438E884574EEF42E0B221EE2369F302DB03435`)
   - `docs/handoff/v10/ac_sigma_bootstrap_ci.json` (SHA-256: `FDFEF9522DE47BDB44C05A6AD2491875E048CC37E29D48D312E176BEBE80A964`)
3. **実験・評価スクリプト更新**:
   - `experiments/gate3a_register_centering_eval.py`（`compute_b10_mc_quantiles`, `get_or_compute_b10_mc_quantiles`, `_eval_space` 内 MC 閾値評価, `build_b11_threshold_calibration_comparison`, B-11 JSON 出力）
   - `experiments/scale_separated_level3.py`（`evaluate_f14_hybrid_alignment_and_parity()`）
4. **成果物 JSON**:
   - `data/usir_v5_eval/usir_v5_b10_gate3a_scene_scale_and_level3.json` (SHA-256: `2E0833D5C300273F73222C9EA3B039EC7358006849E9FF4957E4D9105C991C60`)
   - `data/usir_v5_eval/usir_v5_b11_gate3a_calibrated_and_f14_parity.json` (SHA-256: `929BBFE9E7340BCF4F648145403E9C9F82EF9D2C78E7101E09FFE7C417CA2D84`)
5. **単体テスト（新規）**:
   - `python/tests/v5/test_b11_gate3a_and_f14_parity.py`（5 テスト全件合格）
6. **総合検証レポートおよび同期先**:
   - `Claude-Sync/inbox/2026-11-13_usir_v5_stage_b11_report.md`
   - `docs/handoff/2026-11-13_usir_v5_stage_b11_report.md`
   - `F:\マイドライブ\Claude-Sync\inbox\2026-11-13_usir_v5_stage_b11_report.md`
7. **引継ぎ台帳**:
   - `HANDOFF_usir.md`（Stage B-11 完了記録追記）
