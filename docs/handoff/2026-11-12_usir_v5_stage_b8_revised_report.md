# USIR v5 Stage B-8 改訂・総合検証報告書 (2026-11-12)

**作業ディレクトリ**: `C:\dev\usir`  
**ブランチ**: `feat/fable-v5-shootout`  
**実施日**: 2026-11-12  
**行動規範遵守**: 全プロジェクト共通行動規範（自己内省・冷徹な客観監査 / 耳障りの良い成功報告の排除 / 死角・限界・誤爆の自己露呈 / 実環境・生データによるストレステスト）

---

## 1. エグゼクティブサマリー

本セッション（Stage B-8）では、Stage B-7 において残されていた以下の課題を一気通貫で解決し、冷徹な客観実測とデータ駆動による検証を完遂した：

1. **タスク1：U6「万が一」閉鎖系除外同期と 38 件テストベクトル確定**
   - 「万が一」「万一」を数詞抽出から完全除外（`extract_numerals("万が一の事態") == []`）。
   - `extract_numbers = extract_numerals` のエイリアス互換性を保証。
   - `docs/handoff/v10/u6_guards_test_vectors.json` を 38 件に拡張・再出力し、`F:\マイドライブ\Claude-Sync\inbox\` に同期完了。
2. **タスク2：Gate 3a 現代散文マルチレジスター分布シフトの実測**
   - 4 レジスター（R1 現代口語小説, R2 現代エッセイ・評論, R3 現代技術・ビジネス散文, R4 青空文庫口語ベースライン）各 8 サンプル（全 32 サンプル、408〜590 文字）を収集し、Tier-C クリーンルーム規格（生テキスト完全非保持）で `data/usir_v5_eval/tier_c_modern_benchmark.jsonl` を生成。
   - R4 基準での分布シフト（PPM 交差エントロピー、閉クラス被覆率、ortho_v2 31D Cohen's d / Mahalanobis 距離）を算出。
   - 列中心化の前後でレジスター間分散 $\eta^2_{\text{register}}$（Raw: 0.4554 $\to$ Standardized: 0.4429）および距離構造を計測。
   - **客観的死角開示**: 初期ヒューリスティック「閉クラス被覆率 85% 以上」は過大であり、自然日本語散文の実測値は 55.6%〜63.2% であることを突き止め、未較正初期値として記録。
3. **タスク3：改稿ペア拡充・U6 ROC 較正・二階層 Level 3b 残差検定**
   - 人手由来一次テキストに基づく 86 ペア（Class A ミクロ推敲 42 件, Class B マクロ改稿 22 件, Class C 内容変更 22 件）のデータセット `data/usir_v5_eval/u6_calibration_and_level3_dataset.jsonl` を構築（LLM 合成ペア完全不使用）。
   - U6 Guards の層化 2 分割 ROC 較正を実施。初期値 (min_jaccard=0.55, max_char_change=0.25) の過剰拒絶（低再現率）を特定し、最適較正パラメータ (0.35, 0.45) で F1=0.7429（楽観バイアス -0.0586）を達成。
   - 二階層 Level 3b 残差検定（H1: Class A ortho28, ortho28+kasane11 / H2: Class B 6pole, 6pole+HW-25）において両側 Wilcoxon 検定を実施。コーパス平均方向を除去した直交残差空間では、いずれの空間もランダム摂動や他作家方向と有意差を持たない（$p > 0.20$）事実を冷徹に開示。
   - **重大な構造解明**: B-7 で報告された Kamae の $\eta^2_{\text{source}}=1.00$（完全分離）について、短文におけるフック開閉判定の離散固定化（0 または 1）とトランスレーター定型句が無影響であることに起因する「構成的アーティファクト」であることをコードから突き止め、成果としての引用を撤回。

---

## 2. タスク1：U6「万が一」閉鎖系除外同期とテストベクトル確定

### 2.1 実装内容
- `python/usir_py/v5/u6_guards.py`:
  - `_IDIOM_PREFIX_EXCLUSIONS` および `_NUMERAL_EXCLUSIONS` に「万が一」を追加。
  - スパン境界判定において `text[start - 2:end] == "万が一"` の判定を追加。
  - 互換性エイリアス `extract_numbers = extract_numerals` を追加。
- `python/tests/v5/test_u6_guards.py`:
  - `test_numeral_closed_class_man_ga_ichi_exclusion` を新設し、万が一の数詞非抽出、推敲判定（STYLE_POLISH）、数詞変更時のリジェクト（CONTENT_REWRITE）を検証。
  - `test_u6_test_vectors_fixture_pass` を 38 件対応に更新。

### 2.2 テストベクトル確定とファイル整合性
- **テストベクトルファイル**: `docs/handoff/v10/u6_guards_test_vectors.json`
  - ベクトル数: 38 件（1〜38, pass_count: 38）
  - SHA-256: `4AC04322811EB30776260F8EC2DEA378FF962A5C44180823218973DBAA3F3EF7`
  - 同期先: `F:\マイドライブ\Claude-Sync\inbox\u6_guards_test_vectors.json`（同一ハッシュ確認済み）
- **単体テスト結果**: `pytest python/tests/v5/`: **46 passed** (test_u6_guards 16 件全 PASS)

---

## 3. タスク2：Gate 3a 現代散文マルチレジスター分布シフトの実測

### 3.1 ベンチマークデータセット仕様
- **ファイル**: `data/usir_v5_eval/tier_c_modern_benchmark.jsonl` (32 件)
- **SHA-256**: `45F5F5D45895F591D075F432DE5AA16F67CA511E0FCE1486741301084E0642B1`
- **クリーンルーム規約**: 著作権法第30条の4に基づき、生テキストはメモリ内でのみ処理され、JSONL には一切出力されない。
- **レジスター構成**:
  - R1: Modern Colloquial Fiction（現代口語小説, 8 件, 平均 439 字）
  - R2: Modern Essay & Criticism（現代エッセイ・評論, 8 件, 平均 425 字）
  - R3: Modern Technical & Business Prose（現代技術・ビジネス散文, 8 件, 平均 429 字）
  - R4: Aozora Bunko Colloquial Baseline（青空文庫口語ベースライン, 8 件, 平均 559 字）

### 3.2 実測数値と分布シフト解析
解析スクリプト: `experiments/gate3a_modern_shift_analysis.py`  
成果物 JSON: `data/usir_v5_eval/usir_v5_gate3a_modern_shift_report.json` (SHA-256: `4FD446ADD2C30B964F36A82D735F01BF7B8D4CC3B20660284A80BD77A6583E57`)

| レジスター | PPM bits/token (95% CI) | R4 基準 $\Delta$ (Cohen's d) | 閉クラス被覆率 (95% CI) | ortho_v2 31D 平均 \|d\| | 対角 Mahalanobis 距離 (95% CI) |
|---|---|---|---|---|---|
| **R1 (現代小説)** | 4.0672 [3.97, 4.17] | -0.6531 ($d = -3.19$) | 58.88% [57.98%, 59.73%] | 0.8740 | 6.64 [6.07, 7.21] |
| **R2 (評論)** | 4.1087 [3.99, 4.22] | -0.6116 ($d = -2.87$) | 56.74% [56.00%, 57.51%] | 1.1578 | 7.97 [7.49, 8.44] |
| **R3 (技術・ビジネス)** | 4.0636 [3.95, 4.17] | -0.6567 ($d = -3.07$) | 55.64% [54.91%, 56.32%] | 1.3533 | 9.06 [8.59, 9.47] |
| **R4 (青空口語基底)** | 4.7203 [4.54, 4.86] | 0.0000 ($d = 0.00$) | 63.22% [61.64%, 64.71%] | 0.0000 | 4.95 [4.26, 5.58] |

### 3.3 列中心化（Column Centering）と分散構造
- **ortho_v2 (31D)**:
  - $\eta^2_{\text{raw}} = 0.4554$
  - $\eta^2_{\text{column\_centered}} = 0.4554$（列平均減算は群間/全分散比を変えないため不変）
  - $\eta^2_{\text{standardized}} = 0.4429$（列分散正規化後）
- **統合 33D 特徴量（PPM + 被覆率 + ortho 31D）**:
  - $\eta^2_{\text{raw}} = 0.6054$（スケールの大きい被覆率 % に支配された見かけの値）
  - $\eta^2_{\text{standardized}} = 0.4632$（全次元標準化後の真のレジスター間分散比）
- **外れ値率（Outlier Rate）の実測**:
  - R4 ベースラインの $3\sigma$ 閾値（7.97）基準: 現代文サンプルの外れ値率 = **62.5%** [95% CI: 41.7%, 83.3%]
  - $\chi^2(df=31)$ 95% 点（6.71）基準: 現代文サンプルの外れ値率 = **83.3%** [95% CI: 66.7%, 95.8%]

### 3.4 監査所見：未較正初期値の開示
- **閉クラス被覆率 $\ge 85\%$ の判定**:
  - **事実**: 現代散文・青空文庫のいずれのレジスターにおいても閉クラス被覆率は 55%〜63% にとどまり、85% に達するものは存在しない。
  - **原因**: 85% という初期閾値は過度に楽観的な先入観（Heuristic Prior）であり、文法スケルトン単語の客観物理量と乖離していた。
  - **提言**: 本番環境の足切り閾値は $52.0\%$（R3 最小値を下回る安全域）に再較正すべきである。
- **外れ値率 $< 10\%$ の判定**:
  - R4（近代文学）を基準とした場合、現代ビジネス散文や批評文は語彙・表記分布が大きくシフトするため、マハラノビス空間上で 62.5% が外れ値領域に入る。現代文前処理器（約物正規化）を適用しても、文体レジスター自体の分布シフトは実在する物理量として残存する。

---

## 4. タスク3：改稿ペア拡充・U6 ROC 較正・二階層 Level 3b 検証

### 4.1 拡充データセット仕様
- **ファイル**: `data/usir_v5_eval/u6_calibration_and_level3_dataset.jsonl` (計 86 ペア)
- **SHA-256**: `93C689E44D68DFBB711DDA62F5F4A99E3551FC82789549DE4DFCC0FCE1C28A4D`
- **内訳**:
  - **Class A（ミクロ推敲）**: 42 ペア（太宰、漱石、芥川、賢治、梶井、中島、鴎外、露伴の短文 1〜3 文、語尾・助詞・接続詞・表記推敲、命題不変）
  - **Class B（マクロ改稿）**: 22 ペア（D10 多翻訳 4 ソース × 4 スタイル 16 ペア ＋ 宮沢賢治『銀河鉄道の夜』異稿等実在異稿 6 ペア、400〜1500 字）
  - **Class C（内容変更）**: 22 ペア（固有名詞置換 6 件、数値変更 6 件、否定反転 6 件、場面差し替え 4 件）
  - **非循環性保証**: LLM 合成ペアは一切使用せず、すべて人手による一次テキストおよび定型文体変換・摂動で構築。

### 4.2 U6 Guards ROC 較正結果
評価スクリプト: `experiments/scale_separated_level3.py`  
層化 2 分割（選択用 43 件 / 評価用 43 件、シード 42、クラス比率維持）  
グリッドサーチ: min_jaccard $\in [0.35, 0.75]$ (0.05 刻み) $\times$ max_char_change $\in [0.15, 0.45]$ (0.05 刻み)

#### Scope 1: Class A (ミクロ推敲) vs Class C (内容変更)
- **現行初期値 (min_jaccard=0.55, max_char_change=0.25)**:
  - 選択用セット: Precision = 0.6250, **Recall = 0.2381**, F1 = 0.3448, FPR = 0.2727
  - 評価用セット: Precision = 0.9000, **Recall = 0.4286**, F1 = 0.5806, FPR = 0.0909
  - **死角**: 閾値が厳しすぎるため、正当なミクロ推敲の半分以上を CONTENT_REWRITE と誤判定して取りこぼしていた（偽陰性過多）。
- **較正後最適値 (min_jaccard=0.35, max_char_change=0.45)**:
  - 選択用セット: Precision = 0.7647, Recall = 0.6190, F1 = 0.6842, FPR = 0.3636
  - 評価用セット: Precision = **0.9286**, Recall = **0.6190**, **F1 = 0.7429**, FPR = **0.0909**
  - **楽観バイアス**: $-0.0586$（評価用セットの方が F1 が高く、過学習なし）

#### Scope 2: Class A + Class B (全改稿) vs Class C (内容変更)
- **現行初期値 (0.55, 0.25)**:
  - 評価用セット: Precision = 0.9000, **Recall = 0.2812**, F1 = 0.4286, FPR = 0.0909
- **較正後最適値 (0.35, 0.45)**:
  - 評価用セット: Precision = **0.9286**, Recall = **0.4062**, **F1 = 0.5652**, FPR = **0.0909**
  - **死角と限界の露呈**: マクロ改稿（Class B）を含めると文字数変化や語彙入替が大きくなるため、表面 Jaccard ガード単独では Recall が 40.6% に低下する。U6 表層ガードはミクロ推敲用であり、マクロ改稿の判定には深層意味・統語計器の併用が不可欠である。

### 4.3 二階層 Level 3b 残差検定実測 (H1 & H2)
成果物 JSON: `data/usir_v5_eval/usir_v5_stage_b8_scale_separation_and_gate3a.json` (SHA-256: `B6441FE5B8EF1787BC89847BD11F64661D6A07D3A9DCAF770F008DE9CE915DC9`)

Gram-Schmidt 直交化によりコーパス平均方向 $u_{\text{corp}}$ を除去した本人残差ベクトル $u_{\text{spec}}$ に対し、両側 Wilcoxon 検定を実施：

#### H1: Class A（ミクロ推敲, n=42）
- 準平行（$\rho_{\perp} < 0.10$）件数: **0 件**（中央値 $\rho_{\perp} = 0.9696$, $\theta = 77.02^\circ$）
- **ortho28**:
  - vs 等方ランダム: $W = 413.0, p = 0.6380$, 平均 $cos_{\text{spec}} = +0.0211$（**非有意**）
  - vs 他作家平均残差: $W = 226.0, p = 0.2280$, 平均 $cos_{\text{spec}} = +0.0352$（**非有意**）
- **ortho28 + kasane11**:
  - vs 等方ランダム: $W = 373.0, p = 0.3330$, 平均 $cos_{\text{spec}} = +0.0285$（**非有意**）
  - vs 他作家平均残差: $W = 226.0, p = 0.2280$, 平均 $cos_{\text{spec}} = +0.0353$（**非有意**）

#### H2: Class B（マクロ改稿, n=22）
- 準平行（$\rho_{\perp} < 0.10$）件数: **0 件**（中央値 $\rho_{\perp} = 0.7636$, $\theta = 130.20^\circ$）
- **6pole**:
  - vs 等方ランダム: $W = 104.0, p = 0.4826$, 平均 $cos_{\text{spec}} = +0.0671$（**非有意**）
  - vs 他スタイル平均残差: $W = 5.0, p = 0.6250$, 平均 $cos_{\text{spec}} = -0.0060$（**非有意**）
- **6pole + HW-25**:
  - vs 等方ランダム: $W = 96.0, p = 0.3369$, 平均 $cos_{\text{spec}} = -0.0693$（**非有意**）
  - vs 他スタイル平均残差: $W = 5.0, p = 0.6250$, 平均 $cos_{\text{spec}} = -0.0049$（**非有意**）

#### 客観的科学結論
拡充された実テキストペアにおいて、コーパス規範回帰方向（$u_{\text{corp}}$）を直交除去した後の純残差成分は、等方ランダム摂動および他者方向と統計的に区別がつかない（$p > 0.20$）。  
すなわち、推敲における文体変化は「文語共通規範コーパス平均への回帰」が主たる成分であり、コーパス平均と完全に直交する超局所空間に個体作家固有の引き込み方向が存在するという仮説（H1, H2）は、本実測データにおいては支持されなかった。

### 4.4 Kamae $\eta^2_{\text{source}}=1.00$（完全分離）の構造解明と撤回
- **再計算結果**:
  - D10 Crossed Dataset (16 セル) において、Kamae の値は：
    - SRC_MELOS: 4 セルとも `0.0000`
    - SRC_GINGA: 4 セルとも `0.9999999999995` ($\approx 1.0$)
    - SRC_SANGETSU: 4 セルとも `0.0000`
    - SRC_PRINCE: 4 セルとも `0.9999999999998` ($\approx 1.0$)
  - 分散分析結果: $\eta^2_{\text{source}} = 1.0000$, $\eta^2_{\text{trans}} = 0.0000$, $\eta^2_{\text{interaction}} = 0.0000$
- **コード精査による発生メカニズムの特定**:
  - `python/usir_py/v5/six_pole_atlas.py` (L375):
    `hook_survival = unresolved_unweighted / (open_unweighted + eps)`
  - 短い場面テキストにおいて、構文フック検出が 0 個（未発火 $\to 0.0$）または 1 個（未回収 $\to 1.0$）の 2 値しか取らない。
  - D10 の 4 つのトランスレータースタイルが付与する前置き・結びの定型文は、フック構文に全く触れないため、スタイル間で値が 1 ビットも変動しなかった。
  - その結果、$SS_{\text{trans}} = 0$, $SS_{\text{error}} = 0$ となり、計算式上 $\eta^2_{\text{source}} = 1.0000$ という人為的極限値が生じていた。
- **決定**:
  - 本数値は「原作意味の完全保存」を証明する実証的成果ではなく、短文離散フック判定による「構成的アーティファクト」であると断定し、成果としての引用を**正式に撤回**する。

---

## 5. 成果物一覧と整合性台帳

| 成果物ファイル | 役割 | レコード数 | SHA-256 |
|---|---|---|---|
| `docs/handoff/v10/u6_guards_test_vectors.json` | U6 ガードテストベクトル | 38 件 | `4AC04322811EB30776260F8EC2DEA378FF962A5C44180823218973DBAA3F3EF7` |
| `data/usir_v5_eval/tier_c_modern_benchmark.jsonl` | Gate 3a 現代散文ベンチマーク | 32 件 | `45F5F5D45895F591D075F432DE5AA16F67CA511E0FCE1486741301084E0642B1` |
| `data/usir_v5_eval/usir_v5_gate3a_modern_shift_report.json` | Gate 3a 分布シフト解析レポート | - | `4FD446ADD2C30B964F36A82D735F01BF7B8D4CC3B20660284A80BD77A6583E57` |
| `data/usir_v5_eval/u6_calibration_and_level3_dataset.jsonl` | U6 較正・Level 3b データセット | 86 件 | `93C689E44D68DFBB711DDA62F5F4A99E3551FC82789549DE4DFCC0FCE1C28A4D` |
| `data/usir_v5_eval/usir_v5_stage_b8_scale_separation_and_gate3a.json` | Stage B-8 総合検証成果物 JSON | - | `B6441FE5B8EF1787BC89847BD11F64661D6A07D3A9DCAF770F008DE9CE915DC9` |
| `docs/handoff/2026-11-12_usir_v5_stage_b8_revised_report.md` | 本総合報告書 | - | 作成完了 |

---

## 6. コミット情報
- コミットハッシュ: `8713c34` (`8713c34a3710bd41c11489dfe94434df1c1b41c1`)
- コミット対象ブランチ: `feat/fable-v5-shootout`
- 検証コマンド: `pytest python/tests/v5/` (46 passed)
