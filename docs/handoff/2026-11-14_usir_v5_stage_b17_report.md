# Stage B-17 計測器較正レポート：216作品コーパス拡張・拡張ノイズ床生成・coupling再計測・F-19再検証

- **日付**: 2026-11-14
- **担当範囲**: `usir`（`feat/fable-v5-shootout`）／`nishiki`（`feat/fable-hardcore-v5`）
- **前提・制約**: 評価規則 v2（B-14）適用、標準手続き `percentile` 維持、`ac_mean_velocity` 恒常帯別（再計測対象外）、`ac_max_sis` F-20 凍結維持。本番判定（床の採否・プール可否・凍結解除・Gate 3a/3b・Level 3b）の正式変更は行わない。

---

## エグゼクティブ・サマリー（結論）

1. **タスク1（コーパス拡張 K=106 → 216）**: 青空文庫 `_index.csv` の NDC 913（新字新仮名、`K913` 除外、著作権なし、著者没年 `1868..=1967`）から、現行 10 著者を除外した新規著者プールより **110作品（110新規著者・各1作品）** を決定論的無作為抽出（`seed=20261114`）＋長さ基準（`6,000 <= char_count <= 200,000`、`n_paragraphs >= 30`、`n_pairs_total >= 25`）で選定し、`aozora_heldout3_manifest.json` を追加した。総規模は **K = 216作品、著者 120名**（10名 → 120名）となった。
2. **タスク2（拡張ノイズ床候補の生成）**: `nishiki gen-paragraph-noise-floor --scope all-216` を実装・実行し、216作品（53,028 段落、52,812 隣接ペア、`[≥2,≥2]` M=456 / L=861、`K_active=121`, `K_M=97`, `K_L=93`, 除外 0 作品）から候補 JSON（`paragraph_noise_floor_all_216.json`）を生成した。F-17 床・106作品候補は不変。
3. **タスク3（再計測：`ac_coupling` のみ判定対象）**:
   - **`ac_coupling`**: 点推定 **1.3361**、パーセンタイル法 95% CI **[1.2313, 1.4514]**（対数空間 `[0.2081, 0.3725]`）→ マージン `[0.8, 1.25]` 外側のため **不合格（帯別維持）**。
   - **`ac_max_sis`** *(F-20 凍結中・参考診断値のみ)*: 点推定 **0.9114**、95% CI **[0.7830, 1.0567]**（凍結中のため合否の結論は出さない）。
   - **`ac_mean_velocity`**: 恒常帯別でクローズ済みのため再計測せず。
4. **タスク4（F-19 ベンチマーク 216作品母集団での再実行）**: `S=5,000, B=10,000, seed=20261113, --stat centered` で再実行した結果、現行手続き **`percentile` の `ac_max_sis` が全 3 シナリオ（empirical / normal / t5）で G1（≤5.5%）かつ G2（92.5〜97.5% & 片側≤5.0%）に合格**した（empirical: T1max `3.8%`, cov `93.7%` / normal: T1max `4.0%`, cov `93.4%` / t5: T1max `4.9%`, cov `92.6%`）。よって **F-20 凍結解除条件②を満たした**（条件① `K >= 100` も充足。ただし本ステージでは凍結解除を実施しない）。
5. **タスク5（検証ゲート）**: 凍結成果物 15 ファイルの SHA-256 実行前後完全一致を確認。`usir` `pytest python/tests/v5/` **73 passed, 0 failed**、`nishiki` `cargo test --workspace`（`kasane-wasm` 含む）**537 passed, 0 failed, 1 ignored**。

---

## タスク1：追加作品の選定とマニフェスト構築

- **選定スクリプト**: `usir/scripts/b17/task1_select_heldout3.py`
- **新規マニフェスト**: `usir/data/aozora_heldout3_manifest.json`（110作品、既存マニフェスト未変更）
- **選定リスト**: `usir/scripts/b17/b17_task1_selection_list.json`
- **選定基準と結果**:
  - プール条件: `作品著作権フラグ == "なし"`, `文字遣い種別 == "新字新仮名"`, `役割フラグ == "著者"`, `分類番号` が `913`（`K913` 除外）, `没年月日` `1868..=1967`, 既存 106作品および既存 10著者を除外。
  - 著者配分・無作為性: `seed = 20261114` で著者順・作品順をシャッフルし、ラウンドロビンで 1著者あたり最大 2作品（結果として **110名の新規著者から各 1作品ずつ**）を選定。
  - 長さ・品質フィルタ: `6,000 <= char_count <= 200,000`、`n_paragraphs >= 30`、`n_pairs_total >= 25`（30字以上段落）。条件不適合等で 74 作品をスキップし、110 作品を採用。
  - コーパス全体規模: `Set 1 (46) + heldout (30) + heldout2 (30) + heldout3 (110)` = **K = 216 作品**、総著者数 **120 名**（10名 → 120名）。

---

## タスク2：拡張ノイズ床候補の生成

- **実装**: `nishiki` (`crates/nishiki-cli/src/paragraph_noise.rs`) に `Scope::All216`（`--scope all-216`）および `select_all_216` を追加。単体テスト `b17_select_all_216_includes_heldout3_prefix` を追加。
- **出力ファイル**:
  - `nishiki/docs/handoff/evidence/2026-11-14_b17/paragraph_noise_floor_all_216.json`
  - `usir/docs/handoff/v10/paragraph_noise_floor_all_216.json`（SHA-256: `07E43FDF11E3C91D641A08FF509AB9DDFE153C9F0DBE2030B3C81DFFF199888D`）
- **母集団メタデータ比較（106作品候補 vs 216作品候補）**:

| 指標 | 106作品候補 | 216作品候補 |
|---|---:|---:|
| 総作品数 `K` | 106 | **216** |
| 有効寄与作品数 `K_active` | 64 | **121** |
| 帯別寄与作品数 `K_M` / `K_L` | 53 / 54 | **97 / 93** |
| 著者数 | 10 | **120** |
| 総段落数 / 総隣接ペア数 | 22,989 / 22,883 | **53,028 / 52,812** |
| `M` 全ペア / `[≥2,≥2]` ペア | 3,676 / 323 | **6,770 / 456** |
| `L` 全ペア / `[≥2,≥2]` ペア | 1,553 / 566 | **2,681 / 861** |

### 帯別 σ 比較表（106作品候補 vs 216作品候補、評価規則 v2）

| 帯 | 次元 | 106作品候補 (`M=323 / L=566`) | 216作品候補 (`M=456 / L=861`) | 変化率 |
|---|---|---:|---:|---:|
| **M** | `ac_mean_velocity` | 130.91763 | **134.60689** | +2.82% |
| **M** | `ac_max_sis` | 0.12769 | **0.13877** | +8.68% |
| **M** | `ac_coupling` | 1.37295 | **1.37602** | +0.22% |
| **L** | `ac_mean_velocity` | 100.65751 | **98.73687** | -1.91% |
| **L** | `ac_max_sis` | 0.14496 | **0.15225** | +5.03% |
| **L** | `ac_coupling` | 1.02863 | **1.02989** | +0.12% |

---

## タスク3：再計測（`ac_coupling` のみ判定対象）

同等マージン `[0.8, 1.25]` での作品単位クラスタブートストラップ（パーセンタイル法 95% CI、`B=10,000`、`seed=20261625`、`resamples_dropped=0`）結果：

| 次元 | 点推定 `σ_M / σ_L` | 95% CI（比） | 106作品時（参考） | 合否 / 扱い |
|---|---:|---|---|---|
| **`ac_coupling`** *(判定対象)* | **1.3361** | **[1.2313, 1.4514]** | 1.3347 `[1.2143, 1.4723]` | **不合格（帯別維持）** |
| **`ac_max_sis`** *(F-20凍結・診断値)* | *(0.9114)* | *([0.7830, 1.0567])* | *(0.8809 `[0.7261, 1.0713]`)* | **常時 `BAND_SPECIFIC`（合否結論なし）** |
| **`ac_mean_velocity`** | — | — | — | **再計測せず（恒常帯別でクローズ済み）** |

- `ac_coupling` は K=216 でも CI 下限 `1.2313`（対数空間 `[0.2081, 0.3725]`）かつ点推定 `1.3361` であり、`[0.8, 1.25]` 内に収まらず、帯別維持がさらに強固になった（CI 幅比も縮小）。

---

## タスク4：F-19 ベンチマークの 216作品母集団での再実行

- **実行条件**: `nishiki/scripts/f19/task1_ci_procedures.py` を 216作品 TSV（`set216_paragraphs.tsv`, `NW=216`, `K_active=121`, `stat=centered`）で実行（`S=5,000`, `B=10,000`, `seed=20261113`, 十分統計量による閉形式計算）。
- **合格基準**: G1（m=1.25 TOST 上側過誤率 ≤ 5.0%、MC 上限込み ≤ 5.5%）、G2（両側被覆率 92.5〜97.5% かつ片側非被覆 ≤ 5.0%）。
- **出力**: `nishiki/docs/handoff/evidence/2026-11-14_b17/f19_bench_216works.json` / `f19_gate_216works.json`

### 主対象：現行手続き `percentile` における `ac_max_sis` の G1/G2 結果（m=1.25）

| 手続き | シナリオ | T1max (TOST) | 被覆率 (up / lo) | G1 (≤5.5%) | G2 | 検出力 (1.25) |
|---|---|---:|---:|:---:|:---:|---:|
| **percentile** *(現行)* | **empirical** | **3.8%** | **93.7%** (3.8% / 2.4%) | **True** | **True** | **68.1%** |
| **percentile** *(現行)* | **normal** | **4.0%** | **93.4%** (4.0% / 2.7%) | **True** | **True** | **99.8%** |
| **percentile** *(現行)* | **t5** | **4.9%** | **92.6%** (5.0% / 2.4%) | **True** | **True** | **77.4%** |

### 参考：他手続きにおける `ac_max_sis`（G1/G2）

| 手続き | シナリオ | T1max (TOST) | 被覆率 (up / lo) | G1 (≤5.5%) | G2 | 検出力 (1.25) |
|---|---|---:|---:|:---:|:---:|---:|
| `bca` | empirical / normal / t5 | 2.7% / 3.5% / 5.1% | 94.6% / 93.3% / **90.6%** (up 5.4%) | True / True / True | True / True / **False** | 70.3% / 99.7% / 73.7% |
| `boott` | empirical / normal / t5 | 2.1% / 3.2% / 4.2% | 95.9% / 93.9% / 92.7% | True / True / True | True / True / True | 61.2% / 99.7% / 68.1% |
| `analytic_t` | empirical / normal / t5 | 3.4% / 3.3% / 4.0% | 94.0% / 94.1% / 93.7% | True / True / True | True / True / True | 63.3% / 99.8% / 75.1% |

### 参考：`ac_coupling`（empirical / normal / t5）
- `percentile`: empirical (T1max 4.0%, cov 93.6%, G1=True, G2=True, 検出力 99.9%) / normal (T1max 3.4%, cov 94.1%, G1=True, G2=True, 検出力 99.9%) / t5 (T1max 4.8%, cov 92.9%, G1=True, G2=True, 検出力 77.9%)。

**検証結論**:
- 現行手続き **`percentile` の `ac_max_sis` は empirical / normal / t5 の全 3 シナリオで G1 かつ G2 に合格し、F-20 凍結解除条件②（「拡張母集団での F-19 ベンチマーク再実行で現行手続き（percentile）が G1／G2 に合格」）を満たした**（条件① `K >= 100` も充足）。本ステージでは凍結解除の正式変更は行わない。

---

## 死角・限界・留意事項（客観監査）

1. **t5 シナリオでの余裕の小ささ**: `percentile` の `ac_max_sis` は `t5` シナリオで `T1max = 4.9%`（ハード上限 `5.0%` まで `0.1pt`）、`cov = 92.6%`（G2 下限 `92.5%` まで `+0.1pt`）、`noncov_upper = 5.0%`（上限境界）であり、S=5,000 のモンテカルロ誤差（SE ≈ 0.3%）を考慮すると重い裾分布下では依然として限界値に近い。
2. **診断値 `ac_max_sis` CI の下限**: 216作品時点での `ac_max_sis` σ比診断値は `0.9114`、`95% CI [0.7830, 1.0567]` であり、仮に凍結解除後に再評価したとしても下限 `0.7830` が `0.80` を下回っている（M帯 σ `0.13877` < L帯 σ `0.15225`）。
3. **NDC 913 自動抽出の境界例**: `_index.csv` の `分類番号` と `役割フラグ == "著者"` に基づく無作為抽出において、翻訳（ワイルドオスカー『絵姿』、李孝石『蕎麦の花の頃』）や速記・口述系（三遊亭円朝『菊模様皿山奇談』）が少数含まれている（いずれも青空文庫 NDC 913 新字新仮名の公開テキスト）。

---

## タスク5：検証ゲート

### 1. 凍結成果物 SHA-256 不変ゲート（`usir/scripts/b17/verify_frozen_sha256.py` にて assert）

全 15 ファイルが実行前後で 100% 一致（`all_unchanged: true`）：
- `nishiki/.../2026-11-13_f17/paragraph_noise_floor_all_novel.json`: `0575B64A75A8AF8DD1AD0DB81BDF4D8898C473CE8AC1876C6E7E2F8B61A6F604`（不変）
- `usir/docs/handoff/v10/paragraph_noise_floor_all_novel.json`: `0575B64A75A8AF8DD1AD0DB81BDF4D8898C473CE8AC1876C6E7E2F8B61A6F604`（不変）
- `nishiki/.../2026-11-13_b16/paragraph_noise_floor_all_106.json`: `39DA6CC8C2E52ACE514B85C9C26FD79EF41DD2320682AFF736C426ABA969526E`（不変）
- `usir/docs/handoff/v10/paragraph_noise_floor_all_106.json`: `39DA6CC8C2E52ACE514B85C9C26FD79EF41DD2320682AFF736C426ABA969526E`（不変）
- `usir/data/usir_v5_eval/real_revision_pairs_v1.jsonl`: `62DA75271F453BD3A8F161D916FDF7B7B87138B773B564B737F90AEE85B6B8E4`（不変）
- `usir/data/usir_v5_eval/real_revision_pairs_v1_1.jsonl`: `55D2DDCF9F0F3AFDDC043E48B6D19632EACA71D601FC3AB98CD24341F78CD9F6`（不変）
- `usir/docs/handoff/v10/real_revision_pairs_v1_f17_harvest_summary.json`: `6B650B64F143355D862EB3AE47F12A58E425405BA610339404219335A5BD8611`（不変）
- `usir/docs/handoff/v10/real_revision_pairs_v1_eval_rule_v2_summary.json`: `57D74A5EDB5C9C118C408CDFF4A37E047CBA49155C9C86102E87637B63BB2F23`（不変）
- `usir/docs/handoff/v10/real_revision_pairs_v1_1_eval_rule_v2_summary.json`: `2CC554D11D9367084A8FC6407514E4ED023B6D2071B17640C4877AF861838D7D`（不変）
- `usir/docs/handoff/v10/sparsity_audit_106works_canonical.json`: `D16D95B54C89AD7010CADEA6DA4C1A0D553C2188C24B30CBEA6B0E52C8C6E64A`（不変）
- `usir/docs/handoff/v10/sparsity_audit_106works_canonical_v2.json`: `0033B8028A051DB57491803AD1242FDD0E9D552EAA208BCE82596E326D8DE2D4`（不変）
- `usir/data/usir_v5_eval/usir_v5_b12_canonical_eval.json`: `50B1EFB38B7D15195EA1DA00CD049E51126F5100E90E03497FC7123D0B5C96FC`（不変）
- 既存 3 マニフェスト（`aozora_manifest.json` / `aozora_heldout_manifest.json` / `aozora_heldout2_manifest.json`）: すべて不変。

### 2. テストスイート実行結果

- **`usir`**: `uv run --project python pytest python/tests/v5/` → **73 passed, 0 failed** (128.28s)
- **`nishiki`**: `cargo test --workspace`（`kasane-wasm` 含む）→ **537 passed, 0 failed, 1 ignored**（`audit_17_corpus::test_generate_17_corpus_audit_data` の `#[ignore]` 1件のみ）
