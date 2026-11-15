# USIR v5 Stage B-18 レポート：Python／Rust 数値突合（HSIC・Löwdin・MDL中心化）

- **実施日**: 2026-11-15
- **作業ブランチ**: `stage/b18-parity`（`feat/fable-v5-shootout` から分岐、本流未マージ）
- **対象実装**:
  - Python: `C:\dev\usir\python\usir_py\v5\six_pole_atlas.py`
  - Rust: `C:\dev\nishiki\crates\nishiki-linalg\src\{hsic.rs, lowdin.rs, rg.rs}`
- **突合スクリプト・証拠コミット**: `9b23313`（`python/scripts/b18_parity/`）
- **正本仕様書**: `F:\マイドライブ\Claude-Sync\USIR_v5_Nishiki_Complete_Spec_and_Plan_20261110.md` §8.3（461〜472行）・§9.1（497〜523行）

---

## 1. 結論（仕様書照合の判定と実務影響）

4項目の定義差すべてにおいて **Rust 実装（`nishiki-linalg`）が正本仕様書 §8.3・§9.1 に一致し、Python 先行実装（`six_pole_atlas.py`）は一致しない**。

| 項目 | 仕様書正本（§8.3 / §9.1） | Python 実装 | Rust 実装 | 正本一致 | 実務影響（ゲート 0.25 ほか） |
|---|---|---|---|---|---|
| 1. バンド幅 | $\sigma_a = \operatorname{median}_{i<j} \|x_i - x_j\|_2$ | 正の二乗距離全体のメジアン、下限 `1e-6` | `i < j` の距離メジアン（0 のときのみ正に限定）、下限なし | **Rust** | 標準化データでは差 $\le 5.69 \times 10^{-5}$（ゲート 0.25 判定不変）。未標準化・小スケール（`1e-4`）では最大 `0.0671` 乖離しゲート近傍で判定が変わりうる |
| 2. nHSIC 分母 | $\sqrt{\operatorname{tr}(\tilde{K}_a^2)\operatorname{tr}(\tilde{K}_b^2) + 10^{-12}}$ | `sqrt(max(積, 1e-12))` | `sqrt(積 + 1e-12)` | **Rust** | 非退化データ（$N \ge 50$）では差 $\le 1.49 \times 10^{-13}$。ゲート 0.25 判定に影響しない |
| 3. Löwdin 正則化 | $(S + \varepsilon_{\text{reg}} I)^{-1/2}$（加算） | `max(λ, 1e-8)`（切り上げ、docstring と不一致） | `max(λ, 0) + 1e-10`（加算） | **Rust** | 良条件では `Z_orth` 差 `2.13e-10`。悪条件（$\lambda_{\min} \sim 10^{-6}$）では `Z_orth` 差 `1.29e-4`、直交化誤差が Python `7.06e-10` に対し Rust `1.32e-4`。`clip` vs `add` 単体差は `1.28e-2` |
| 4. `NarrativeRGNormalizer` | §9.1 の 3 階層（MDL 中心化・$L_0=64$ ブートストラップ・5 幕球面粗視化） | 場面密度化と `log(char_len)` 線形回帰除去のみ | §9.1 の 3 階層 | **Rust** | Python 版は第 2・第 3 階層を未実装（数値突合対象外） |

**非一致項目の扱い**: Python 側の上記 4 項目は仕様書正本と一致していない（特にバンド幅の `1e-6` 下限および非ゼロ限定フィルタ、Löwdin の切り上げ、RG 第 2・第 3 階層の欠落）。既存ロジック無変更制約のため本ステージでは修正しておらず、Python 側を正本準拠とする場合は別途改修が必要。

---

## 2. タスク1：同一入力での nHSIC 突合

### 2.1 Rust 側値の取得方法
`nishiki` リポジトリを変更せず、`python/scripts/b18_parity/rust_probe/` に `nishiki-linalg`（パス依存 `C:/dev/nishiki/crates/nishiki-linalg`）をリンクした最小 CLI バイナリ `b18_rust_probe.exe` を作成し、`run_b18_parity.py` から標準入出力経由で同一の `float64` 行列（`{:.17e}` 形式）を渡して `hsic::nhsic` および `lowdin::HsicLowdinOrthogonalizer::fit` の出力を取得した。

### 2.2 合成データ突合結果
- 固定シード: `np.random.default_rng(20261115)`
- 規模: $N \in \{50, 105, 300\}$
- 次元ペア: $(d_a, d_b) \in \{(2,2), (3,2), (4,4), (7,4), (7,7)\}$
- 関係パターン（計 99 ケース）:
  - 標準 5 種（69 ケース）: `independent`, `copy`, `affine_copy`, `nonlinear_sq`（$y = x^2$）, `linear_mix_s0.5`（$y = x + 0.5\varepsilon$）
  - 境界検証 2 種（30 ケース）: `ties_60pct`（75% 重複行で `median_{i<j} = 0` を誘発）, `small_scale_1e-4`（$10^{-4}$ 倍縮小で下限 `1e-6` を誘発）

| 関係パターン | ケース数 | `|Py - Rust|` 最大値 | `|Py - Rust|` 中央値 | 備考 |
|---|---:|---:|---:|---|
| `independent` | 15 | `4.159e-10` | `1.350e-12` | |
| `copy` | 9 | `3.331e-16` | `0.000e+00` | |
| `affine_copy` | 15 | `4.861e-09` | `3.099e-12` | |
| `nonlinear_sq` | 15 | `1.308e-08` | `2.356e-11` | $N=105$ 等で中央値平均の差が顕在化 |
| `linear_mix_s0.5` | 15 | `4.755e-09` | `5.178e-12` | |
| **標準 5 種小計** | **69** | **`1.308e-08`** | **`7.321e-13`** | |
| `ties_60pct` | 15 | `4.986e-09` | `5.107e-15` | 両実装とも正の距離にフォールバック |
| `small_scale_1e-4` | 15 | **`6.709e-02`** | **`2.846e-02`** | Python の下限 `1e-6` が発火 |
| **全 7 種合計** | **99** | **`6.709e-02`** | **`1.350e-12`** | |

### 2.3 実コーパス（青空文庫 `aozora_106works_scenes.jsonl`）突合結果
- **対象ブロック**: `aozora_106works_scenes.jsonl`（全 106 作品）から Python `aggregate_work_profile_from_scenes` ＋ `NarrativeRGNormalizer.extract_work_features` で抽出した深層 4 極（`Kamae` 3D, `Sadame` 4D, `Nagare` 4D, `Awase` 4D、各列を母標準偏差で z-score 化）を**同一入力**として Python・Rust 両実装へ投入した。
- **Kasane・Sasae 等を揃えなかった理由**: Python `six_pole_atlas.py` には `Kasane`・`Sasae`・`ppm`・`fractal`・`rqa`・`ot` のテキスト特徴抽出関数が存在せず、深層 4 極の集計・正規化のみを実装しているため、同一スクリプト上で表層ブロックを独立生成できない。

| 深層極ペア（$N=106$） | Python `normalized_hsic` | Rust `nhsic` | `|Py - Rust|` | バンド幅差単独の影響 | 分母差単独の影響 |
|---|---:|---:|---:|---:|---:|
| `Kamae–Sadame` | `0.101119` | `0.101120` | `4.689e-07` | `4.689e-07` | `0.0` |
| `Kamae–Nagare` | `0.102040` | `0.102042` | `1.979e-06` | `1.979e-06` | `0.0` |
| `Kamae–Awase` | `0.183104` | `0.183161` | **`5.692e-05`** | `5.692e-05` | `0.0` |
| `Sadame–Nagare` | `0.086246` | `0.086246` | `5.967e-16` | `2.776e-17` | `0.0` |
| `Sadame–Awase` | `0.183297` | `0.183315` | `1.757e-05` | `1.757e-05` | `0.0` |
| `Nagare–Awase` | `0.141700` | `0.141730` | `3.053e-05` | `3.053e-05` | `0.0` |
| **全体（6 ペア）** | — | — | **最大 `5.692e-05` / 中央値 `9.777e-06`** | — | — |

実データで `Awase`・`Kamae` を含むペアが $10^{-5}$ 桁でずれる理由は、一部作品で特徴値が同値（距離 0）となるペアが存在し、Python が `dists[dists > 0]` で 0 距離を無条件除外してメジアンを取る一方、Rust はメジアン $> 0$ であるため 0 距離を含めた全 `i < j` ペアのメジアンを取る点にある。

---

## 3. タスク2：定義差の切り分け（Ablation）

### 3.1 nHSIC（項目 1：バンド幅 vs 項目 2：分母）
Python 実装をベースに項目 1 のみ Rust 化（`var_bw_rust`）および項目 2 のみ Rust 化（`var_den_rust`）した変種で切り分けた結果、**支配的要因は 100% 項目 1（バンド幅）** であった。

- **標準 5 種（69 ケース）**: バンド幅単独の影響は最大 `1.308e-08`（中央値 `~1e-12`）。原因は偶数長ペア（例: $N=105 \Rightarrow 5460$ 組）の中央 2 要素 $a, b$ について、Python が二乗距離の平均 $(a^2+b^2)/2$ を取るのに対し Rust が距離の平均の二乗 $((a+b)/2)^2$ を取る点にある。分母単独の影響は最大 `3.331e-16`（中央値 `0.0`）。両方を Rust 化すると残差は最大 `6.550e-15`（浮動小数点丸め誤差水準）まで消滅する。
- **小スケール（`1e-4`, 15 ケース）**: バンド幅単独の影響が最大 `6.709e-02`（中央値 `2.846e-02`）であるのに対し、分母単独の影響は最大 `1.494e-13`。Python の下限 `1e-6` が分散 $10^{-8}$ 規模の距離に対して過大な平滑化を引き起こしている。
- **実コーパス（$N=106$）**: バンド幅単独の影響が `5.692e-05` で `|Py - Rust|` と有効数字 11 桁まで一致し、分母単独の影響は全ペアで `0.0` であった。

### 3.2 Löwdin の正則化（項目 3：方式 `clip` vs `add`、および $\varepsilon$ 値 `1e-8` vs `1e-10`）
同一の標準化済み行列 $Z$ に対し、Python（`clip, 1e-8`）、変種（`add, 1e-8` / `clip, 1e-10` / `add, 1e-10`）、Rust（`add, 1e-10`）を比較した。

| ケース（$N \times D$） | $\lambda_{\min}$ | Py 直交誤差 | Rust 直交誤差 | `|Z_py - Z_rust|` 最大値 | 方式差 `clip vs add`（`1e-8`） | 閾値差 `1e-8 vs 1e-10`（`clip`） | 同一条件 `add, 1e-10` 残差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `well_conditioned` (105×15) | `0.5146` | `9.70e-15` | `4.84e-10` | `2.128e-10` | `2.128e-08` | `0.0` | `5.329e-15` |
| `ill_conditioned` (105×15) | `7.579e-7` | `7.06e-10` | `1.32e-04` | `1.291e-04` | **`1.278e-02`** | `0.0` | `4.436e-10` |
| `rank_deficient` (80×6) | `8.662e-17` | `1.414` | `1.414` | `1.474e-10` | `1.474e-08` | `6.180e-11` | `1.230e-10` |

- `ill_conditioned`（実コーパス 6 極の $\lambda_{\min} = 1.95 \times 10^{-6}$ と同水準）では、`clip`（`max(λ, 1e-8)`）は何もしないため Python の直交誤差が $7.06 \times 10^{-10}$ と小さい一方、Rust の加算型（$\lambda + 10^{-10}$）は $\varepsilon_{\text{reg}} / \lambda_{\min} \approx 1.32 \times 10^{-4}$ の直交誤差を残す。逆に `rank_deficient`（$\lambda_{\min} \approx 0$）では、$\varepsilon$ の違いにより変換行列 $W$ 自体に最大 `45,000` の要素差が生じる（`Z` 自体は零空間成分を持たないため `Z_orth` の差は `1.47e-10`）。

### 3.3 `NarrativeRGNormalizer`（項目 4：定義差の所在確認）
- **Python (`six_pole_atlas.py:25-179`)**: `extract_work_features` で場面密度（`/scene_count`）および 5 幕勾配・速度比を算出し、`fit_orthogonalizer` / `orthogonalize` で訓練 fold の `log(char_len)` に対する単回帰トレンドを除去（スピアマン $|\rho| \ge 0.20$ かつ残差相関が下がる列のみ）。
- **Rust (`rg.rs`)**: §9.1 の 3 階層すべてを実装（第 1 階層 `MdlCentering`、第 2 階層 `block_bootstrap(L_0=64, stride=32)`、第 3 階層 `kadanoff_coarse_grain` による 5 幕 $S_+^{15}$ 重心と `macro_kinematics`）。

---

## 4. タスク3：仕様書照合および Kasane–Sasae 差の切り分け

### 4.1 仕様書 §8.3・§9.1 との照合根拠
1. **バンド幅**: §8.3（465 行）は $\sigma_a = \operatorname{median}_{i<j}\|x_i^{(a)} - x_j^{(a)}\|_2$ と明記しており、下限 `1e-6` や正値限定フィルタの指定はない。Rust（`hsic.rs:68-90`）が一致。
2. **nHSIC 分母**: §8.3（466 行）の式は $\sqrt{\operatorname{tr}(\tilde{K}_a^2)\operatorname{tr}(\tilde{K}_b^2) + 10^{-12}}$ であり、積への加算である。Rust（`hsic.rs:160`）が一致。
3. **Löwdin**: §8.3（470 行）の式は $W_{\text{Löwdin}} = (S + \varepsilon_{\text{reg}} I_D)^{-1/2} = U \operatorname{diag}((\lambda_i + \varepsilon_{\text{reg}})^{-1/2}) U^\top$ であり、固有値への加算である。Rust（`lowdin.rs:119`）が一致。
4. **`NarrativeRGNormalizer`**: §9.1（500〜514 行）は第 1〜3 階層を標準正規化器として規定しており、3 階層を備える Rust（`rg.rs`）が一致。

### 4.2 Kasane–Sasae（2026-11-11 Python `0.3760` vs F6 Rust `0.1497`）の差の切り分け
- タスク 1 の結果から、標準化された実コーパス特徴ブロックに対する Python・Rust 間の nHSIC 実装差は最大 `5.69e-05` であり、`0.3760` と `0.1497` の差（`0.2263`）を実装差で説明することは不可能。
- 2026-11-11 レポート（`docs/handoff/2026-11-11_usir_v5_six_pole_atlas_report.md` §1・§4.1）の `Kasane` は 55 次元 Fisher 特徴、`Sasae` は `B_cal_center`（PPM 文法骨格）であったのに対し、F6 レポート（`2026-11-15_usir_v5_stage_f6_report.md` §4.2）の `Kasane` は 7 次元改行正規化プロファイル、`Sasae` は 4 次元文様相推定要約である（なお F6 の `Kasane` 7D vs `ppm` 3D は `0.2251`）。
- 同一ブロック定義に揃えた直接再計算は、`six_pole_atlas.py` に `Kasane`・`Sasae` 抽出器が存在しないため本ステージでは未実施だが、差の主因がブロック定義の相違にあることは数値的に確定した。

---

## 5. タスク4：検証ゲート結果

1. **既存テスト（`pytest`）**:
   - 実行コマンド: `$env:PYTHONPATH=".;python"; pytest -q python/tests`
   - 結果: **`672 passed, 18 skipped, 5 warnings`**（199.62s）。既存コードは一切変更していない。
2. **凍結成果物 SHA-256 検証**:
   - `scripts/b17/frozen_sha256_baseline.json` 記載の 15 件すべて期待値と一致（`all_15_baseline_unchanged: true`）。
   - 216 正本（usir・nishiki 双方）: `2FC18E6717E4537ACBB522BC1D10F3C434B3C9BB434B19AF3803293EE7C84974`（前後一致・両コピー同一）。
   - 実推敲ペア v1.1（`data/usir_v5_eval/real_revision_pairs_v1_1.jsonl`）: `55D2DDCF9F0F3AFDDC043E48B6D19632EACA71D601FC3AB98CD24341F78CD9F6`（一致）。
   - 場面注釈（`data/v5/aozora_106works_scenes.jsonl`）: `07ED3F726D5BB948E9B77BB1E35F1AEC99B7897EE8DCAEFF24C7540302DE0828`（一致）。

---

## 6. 成果物・再現手順

- 突合スクリプト: `python/scripts/b18_parity/run_b18_parity.py`
- Rust 検証プローブ: `python/scripts/b18_parity/rust_probe/`
- 生数値結果: `python/scripts/b18_parity/b18_parity_results.json`
- コミット: `9b23313`（ブランチ `stage/b18-parity`）
