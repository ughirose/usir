# USIR v5 Stage B-13 完了報告書

**作成日**: 2026-11-13  
**対象リポジトリ**: `C:\dev\usir`（ブランチ `feat/fable-v5-shootout`）  
**起点 HEAD (F-17 同期)**: `dfa787d`  
**Stage B-13 実装コミットハッシュ**: `978347facf82d5277d59743885767fe9c51bd2f8`（`978347f`）  
**参照ノイズ床 (F-17) SHA-256**: `0575b64a75a8af8dd1ad0db81bdf4d8898c473ce8ac1876c6e7e2f8b61a6f604`  

---

## 0. ガバナンス・前提検証チェック

| 項目 | 本ステージの状態 | 検証根拠 |
| :--- | :--- | :--- |
| Gate 3a 判定 | `SHIFT_QUANTIFIED_NOT_PASSED`（不変） | `test_b13_governance_gates_unchanged` |
| Gate 3b 判定 | `GATE3B_FROZEN`（不変） | `test_b13_governance_gates_unchanged` |
| 妻の未発表原稿（層B-1） | **一切不使用（0 件）** | 青空文庫公開著作権なし版（415件・106作品）のみ使用 |
| 有料 API | **一切不使用** | ローカル Python 3.12 + Rust `nishiki.exe` で完結 |
| Stage B-14 設計・自動開始 | **未実施（禁止事項遵守）** | 設計文書・起動コード・予約実行なし |
| 凍結成果物（B-8〜B-11） | **不変確認済み** | `pytest` 内 SHA-256 検証 PASS |

### 前提ファイル検証（SHA-256 および F-17 帯別実効値）
- `docs/handoff/v10/paragraph_noise_floor_all_novel.json`（F-17 床）
  - **SHA-256**: `0575b64a75a8af8dd1ad0db81bdf4d8898c473ce8ac1876c6e7e2f8b61a6f604`
  - **Band M 実効値 (`ac_multi_window_per_dim_std`)**: `ac_mean_velocity = 134.296419` (`134.30`), `ac_max_sis = 0.126615` (`0.1266`), `ac_coupling = 1.407868` (`1.4079`)
  - **Band L 実効値 (`ac_multi_window_per_dim_std`)**: `ac_mean_velocity = 102.573217` (`102.57`), `ac_max_sis = 0.149357` (`0.1494`), `ac_coupling = 1.066818` (`1.0668`)
- `data/usir_v5_eval/real_revision_pairs_v1.jsonl`（415件）
  - **SHA-256**: `62da75271f453bd3a8f161d916fdf7b7b87138b773b564b737f90aee85b6b8e4`

---

## 1. タスク1：実推敲ペア v1（415件）の F-17 床での再計測

### 1.1 パリティ・検証ゲート・数理的変化予測の検算
- **成果物**: `docs/handoff/v10/real_revision_pairs_v1_f17_harvest_summary.json`（SHA-256: `6b650b64f143355d862eb3ae47f12a58e425405ba610339404219335a5bd8611`）
- **Python vs Rust (`nishiki.exe harvest-diff`) パリティ不一致**: **0 件 / 415 件**
- **検証ゲート（`n_evaluated = 8` の 196 ペア）**: Band S 40件・Band M 両単窓 106件・Band L 両単窓 50件の計 196 ペアにおいて、F-15 床と F-17 床の `noise_floor_ratio_kasane11` 出力は**全 196 件でビット単位完全一致（差分 0 件）**。
- **数理的変化予測（`n_evaluated = 11` の 219 ペア、非ゼロ差分のペアで比率を実測）**:
  - `ac_mean_velocity`（M 帯, $n=131$）: 実測倍率 `0.87159706`（-12.84%、予測 $\times 0.8716$ と一致）
  - `ac_mean_velocity`（L 帯, $n=72$）: 実測倍率 `1.14115915`（+14.12%、予測 $\times 1.1412$ と一致）
  - `ac_max_sis`（M 帯, $n=129$）: 実測倍率 `1.10836765`（+10.84%、注: 4桁丸め $0.1403/0.1266 = 1.1082$ に対して未丸め真値比 $0.14033569/0.12661474 = 1.10836765$ と完全一致）
  - `ac_max_sis`（L 帯, $n=66$）: 実測倍率 `0.93959904`（-6.04%、注: 4桁丸め $0.1403/0.1494 = 0.9391$ に対して未丸め真値比 $0.14033569/0.14935699 = 0.93959904$ と完全一致）
  - `ac_coupling`（M/L 帯, $n=174$）: 実測倍率 `1.00000000`（不変）

### 1.2 1SD 閾値（$|\Delta|/\sigma \ge 1.0$）を跨いで判定が変化したペア件数
全 219 ペアにおいて厳密に `1.0` に一致する境界値は `0` 件（`> 1.0` と `>= 1.0` は同値）であった。
- **`ac_mean_velocity`**: 計 **8 件**（すべて M 帯、`>= 1.0` $\to$ `< 1.0` へ低下 8 件、上昇 0 件；うち混在ペア 7 件・非混在 1 件）
  - `ginga_initial_vs_final_p14_p330` (`1.1370 -> 0.9910`), `horoki_first_vs_new_p2788_p2295` (`1.0989 -> 0.9578`), `oda_ame_first_vs_rev_p6_p31` (`1.0991 -> 0.9580`), `oda_wagamachi_first_vs_rev_p40_p187` (`1.0109 -> 0.8811`), `origuchi_shisha_ed_p285_p282` (`1.0834 -> 0.9443`, 非混在), `hisao_mugetsu_ed_p25_p25` (`1.1241 -> 0.9797`), `hisao_mugetsu_ed_p57_p58` (`1.0433 -> 0.9094`), `kusaka_saigo_ed_p80_p76` (`1.0972 -> 0.9563`)
- **`ac_max_sis`**: 計 **2 件**（すべて M 帯、`< 1.0` $\to$ `>= 1.0` へ上昇 2 件、低下 0 件；うち混在ペア 2 件）
  - `oda_ame_first_vs_rev_p15_p84` (`0.9202 -> 1.0199`), `oda_ame_first_vs_rev_p16_p89` (`0.9573 -> 1.0611`)

### 1.3 サブセット別 F-15 床 vs F-17 床 対照表（記述統計・格付け主張なし）
B-12 の `band_L_both_multi_window`（111件）を `band_L_both_single_window`（50件, 8D）と真の `band_L_both_multi_window`（61件, 11D）に分離して対比した。

| サブセット (`n`) | `abs_median` 中央値 (F15 $\to$ F17 / $\Delta$) | `abs_mean` 中央値 (F15 $\to$ F17 / $\Delta$) | `abs_max` 中央値 (F15 $\to$ F17 / $\Delta$) | `frac_above_1sd` 平均 (F15 $\to$ F17 / $\Delta$) | `frac_above_2sd` 平均 (F15 $\to$ F17 / $\Delta$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **overall** (`415`) | 0.0235 $\to$ 0.0235 (`+0.0000`) | 0.1252 $\to$ 0.1251 (`-0.0001`) | 0.5015 $\to$ 0.5027 (`+0.0012`) | 0.0633 $\to$ 0.0620 (`-0.0013`) | 0.0109 $\to$ 0.0107 (`-0.0002`) |
| **band_S** (`40`, 8D) | 0.0692 $\to$ 0.0692 (`+0.0000`) | 0.2564 $\to$ 0.2564 (`+0.0000`) | 0.8256 $\to$ 0.8256 (`+0.0000`) | 0.0656 $\to$ 0.0656 (`+0.0000`) | 0.0063 $\to$ 0.0063 (`+0.0000`) |
| **band_M** (`245`) | 0.0988 $\to$ 0.0982 (`-0.0006`) | 0.2016 $\to$ 0.2016 (`+0.0000`) | 0.6616 $\to$ 0.6819 (`+0.0203`) | 0.0762 $\to$ 0.0739 (`-0.0023`) | 0.0134 $\to$ 0.0119 (`-0.0015`) |
| **band_L** (`130`) | 0.0036 $\to$ 0.0036 (`+0.0000`) | 0.0311 $\to$ 0.0306 (`-0.0005`) | 0.1444 $\to$ 0.1481 (`+0.0037`) | 0.0385 $\to$ 0.0385 (`+0.0000`) | 0.0077 $\to$ 0.0098 (`+0.0021`) |
| **mixed_window_pairs** (`84`, 11D) | 0.4732 $\to$ 0.4732 (`+0.0000`) | 0.5875 $\to$ 0.5835 (`-0.0040`) | 1.6715 $\to$ 1.5714 (`-0.1001`) | 0.1937 $\to$ 0.1883 (`-0.0054`) | 0.0411 $\to$ 0.0400 (`-0.0011`) |
| **non_mixed_both_single** (`196`, 8D) | 0.0042 $\to$ 0.0042 (`+0.0000`) | 0.0458 $\to$ 0.0458 (`+0.0000`) | 0.2739 $\to$ 0.2739 (`+0.0000`) | 0.0325 $\to$ 0.0325 (`+0.0000`) | 0.0032 $\to$ 0.0032 (`+0.0000`) |
| **non_mixed_both_multi** (`135`, 11D) | 0.0077 $\to$ 0.0077 (`+0.0000`) | 0.0625 $\to$ 0.0629 (`+0.0004`) | 0.3248 $\to$ 0.3165 (`-0.0083`) | 0.0269 $\to$ 0.0263 (`-0.0006`) | 0.0034 $\to$ 0.0034 (`+0.0000`) |
| **band_M_n_eval_8** (`106`, 8D) | 0.0190 $\to$ 0.0190 (`+0.0000`) | 0.0919 $\to$ 0.0919 (`+0.0000`) | 0.4198 $\to$ 0.4198 (`+0.0000`) | 0.0354 $\to$ 0.0354 (`+0.0000`) | 0.0035 $\to$ 0.0035 (`+0.0000`) |
| **band_M_n_eval_11** (`139`, 11D) | 0.1768 $\to$ 0.1768 (`+0.0000`) | 0.3473 $\to$ 0.3413 (`-0.0060`) | 1.2042 $\to$ 1.0819 (`-0.1223`) | 0.1073 $\to$ 0.1033 (`-0.0040`) | 0.0209 $\to$ 0.0183 (`-0.0026`) |
| **band_L_mixed_window** (`19`, 11D) | 0.3725 $\to$ 0.3725 (`+0.0000`) | 0.5363 $\to$ 0.5536 (`+0.0173`) | 1.6901 $\to$ 1.9286 (`+0.2385`) | 0.1962 $\to$ 0.1962 (`+0.0000`) | 0.0526 $\to$ 0.0670 (`+0.0144`) |
| **band_L_both_single_window** (`50`, 8D) | 0.0008 $\to$ 0.0008 (`+0.0000`) | 0.0106 $\to$ 0.0106 (`+0.0000`) | 0.0615 $\to$ 0.0615 (`+0.0000`) | 0.0000 $\to$ 0.0000 (`+0.0000`) | 0.0000 $\to$ 0.0000 (`+0.0000`) |
| **band_L_both_multi_window** (`61`, 11D) | 0.0153 $\to$ 0.0158 (`+0.0005`) | 0.0625 $\to$ 0.0619 (`-0.0006`) | 0.2649 $\to$ 0.2649 (`+0.0000`) | 0.0209 $\to$ 0.0209 (`+0.0000`) | 0.0000 $\to$ 0.0000 (`+0.0000`) |

---

## 2. タスク2：混在ペアのアーティファクト分離（3段階アブレーション 11D $\to$ 8D $\to$ 7D）

### 2.1 1窓側の `ac_*` 値の実測検証
全 415 ペアの 830 側面のうち、窓数 `w <= 1` であった **476 側面** の生特徴量（`features_before` / `features_after`）を実測した結果：
- `ac_mean_velocity`: 非ゼロ **0 / 476 件**（全件厳密に `0.0`）
- `ac_max_sis`: 非ゼロ **0 / 476 件**（全件厳密に `0.0`）
- `ac_coupling`: 非ゼロ **0 / 476 件**（全件厳密に `0.0`）
- `ac_mean_crest`: 非ゼロ **469 / 476 件**（1窓単独でも非ゼロ値を持つ）

### 2.2 3段階アブレーション（11D $\to$ 8D $\to$ 7D）全体集計
*注: Band S（40件、すべて両単窓）は `ac_*` 3次元のノイズ床 $\sigma = 0.0$ のため、11D 評価は Band M/L のみ（`overall`: $n=375$, `non_mixed_both_single`: $n=156$）で算出。*

| サブセット (`n`) | 次元集合 | 評価数 | `abs_median` 中央値 | `abs_mean` 中央値 | `abs_max` 中央値 (平均) | `frac_above_1sd` 平均 | 最頻 `argmax` 次元 (件数) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **overall** (`415`) | 11D | 375 | 0.0097 | 0.1023 | 0.4646 (0.7686) | 0.0589 | `ac_mean_velocity` (70) |
| | 8D | 415 | 0.0127 | 0.0910 | 0.3687 (0.5756) | 0.0422 | `dc_cadence` (101) |
| | 7D | 415 | 0.0082 | 0.0880 | 0.3284 (0.5163) | 0.0358 | `dc_cadence` (112) |
| **mixed_window_pairs** (`84`) | **11D** | 84 | **0.4732** | **0.5835** | **1.5714** (1.7830) | **0.1883** | `ac_mean_velocity` (52/84, 61.9%) |
| | **8D** | 84 | **0.3301** | **0.4264** | **1.0751** (1.1638) | **0.1190** | `ac_mean_crest` (30/84, 35.7%) |
| | **7D** | 84 | **0.2918** | **0.3747** | **0.9331** (0.9957) | **0.0935** | `dc_showing` (18/84) |
| **non_mixed_both_multi** (`135`) | 11D | 135 | 0.0077 | 0.0629 | 0.3165 (0.5510) | 0.0263 | `ac_max_sis` (27) |
| | 8D | 135 | 0.0054 | 0.0434 | 0.2011 (0.3335) | 0.0083 | `dc_cadence` (37) |
| | 7D | 135 | 0.0026 | 0.0348 | 0.1629 (0.2936) | 0.0085 | `dc_cadence` (44) |
| **non_mixed_both_single** (`196`) | 11D | 156 | 0.0006 | 0.0268 | 0.2325 (0.4107) | 0.0175 | `dc_cadence` (50) |
| | 8D | 196 | 0.0042 | 0.0458 | 0.2739 (0.4902) | 0.0325 | `dc_cadence` (54) |
| | 7D | 196 | 0.0045 | 0.0520 | 0.2739 (0.4642) | 0.0299 | `dc_cadence` (54) |

### 2.3 層別分析①：窓数遷移パターン別
| 遷移パターン (`n`) | 11D `abs_max` 中央値 / `frac_1sd` | 8D `abs_max` 中央値 / `frac_1sd` | 7D `abs_max` 中央値 / `frac_1sd` |
| :--- | :--- | :--- | :--- |
| 混在・隣接1窓跨ぎ `1↔2` (`62`) | 1.5714 / 0.1804 | 1.0381 / 0.1129 | 0.9131 / 0.0876 |
| 混在・多窓ジャンプ `1↔3+` (`22`) | 1.5725 / 0.2107 | 1.2170 / 0.1364 | 1.0598 / 0.1104 |
| 非混在複窓・窓数不変 (`110`) | 0.2299 / 0.0182 | 0.1245 / 0.0023 | 0.1075 / 0.0013 |
| 非混在複窓・窓数変化 (`25`) | 0.9109 / 0.0618 | 0.6392 / 0.0350 | 0.5356 / 0.0400 |

### 2.4 層別分析②：文字数変化率 ($\Delta L / L_{\max}$) $\times$ 2-gram Dice による改稿量統制
混在 84 件のうち **63 件（75.0%）** が $\Delta L / L_{\max} \ge 30\%$（大幅長変化）に集中していた。この改稿量交絡を統制した同一セル内で比較した結果は以下の通りである。

**微修正帯（$\Delta L / L_{\max} < 15\%$ かつ $\text{Dice} \ge 0.85$）での直接比較**:

| 群 (`n`) | 11D `abs_max` 中央値 / `frac_1sd` | 8D `abs_max` 中央値 / `frac_1sd` | 7D `abs_max` 中央値 / `frac_1sd` |
| :--- | :--- | :--- | :--- |
| **混在ペア** (`9`) | **2.3513** / **0.1414** | **0.3285** / **0.0278** | **0.1644** / **0.0000** |
| **非混在全体** (`227`) | 0.1266 / 0.0038 *(eval=213)* | **0.0970** / **0.0011** | **0.0919** / **0.0006** |
| — うち非混在複窓 (`104`) | 0.1964 / 0.0079 | 0.1075 / 0.0000 | 0.0902 / 0.0000 |
| — うち非混在単窓 (`123`) | 0.0947 / 0.0000 *(eval=109)* | 0.0947 / 0.0020 | 0.0947 / 0.0012 |

さらに最微細セル **$\Delta L / L_{\max} < 5\%$ かつ $\text{Dice} \ge 0.85$**（計 232 件）でも、混在（$n=8$）は **11D `2.4099` $\to$ 8D `0.2842`（-88.2%） $\to$ 7D `0.1600`** と激減し、非混在複窓（$n=103$）は 11D `0.1918` $\to$ 8D `0.1071` $\to$ 7D `0.0885` であった。

### 2.5 切り分け判定・結論および改訂提案
- **判定**: 混在ペアの高比率は単一要因ではなく、以下の複合構造である：
  1. **次元追加アーティファクト（ケース1機構）**: 微修正帯（$\Delta L / L_{\max} < 15\%, \text{Dice} \ge 0.85$）において、11D の `abs_max` 中央値 `2.3513` が 8D 化で `0.3285` へ -86.0% 激減する（1窓側 `ac_*` 3次元が 476/476 で厳密に `0.0` であるため「値 − 0」が発生）。
  2. **窓境界ジャンプ（ケース2機構）**: `ac_mean_crest` により 8D $\to$ 7D で微修正混在が `0.3285 -> 0.1644` へさらに半減し、混在84件の 8D でも `argmax` の 30/84 (35.7%) を占める。
  3. **改稿量の交絡（ケース3機構）**: 混在84件全体の 8D (`1.0751`)・7D (`0.9331`) が非混在より高いまま残るのは、混在ペアの 75.0% が $\Delta L / L_{\max} \ge 30\%$ に偏っているためである（非混在でも $\Delta L / L_{\max} \ge 30\%, \text{Dice} < 0.60$ なら 8D `1.162` / 7D `1.033` と同等に高い）。
- **結論（「混在ペアの比率を純粋な文体変化として読めるか」）**: **読めない（No）**。11D 評価は微修正でも `2.35` 超の偽スパイクを生成し、全体では大幅改稿の交絡と分離不能である。
- **Orion 判断用の評価規則改訂案（未実装・提案のみ）**:
  1. `ac_*` 3次元（`ac_mean_velocity`, `ac_max_sis`, `ac_coupling`）は「**両側とも複数窓 (`min(w_before, w_after) >= 2`)**」のときのみ分母に入れ、片側 1 窓の混在ペアは **`n_evaluated = 8` を正本とする**。
  2. 次ステージの分離方法として、① $\Delta L / L_{\max}$ および Dice による改稿量層別化、② 混在ペアへの 8D/7D 併用（`ac_mean_crest` の窓境界ジャンプ分離）を提案する。

---

## 3. タスク3：旧仮名 $\to$ 新仮名34件の感度分析

| コホート (`n`, 作品組数) | 帯内訳 (S/M/L, 混在) | F-17 `band_M` `abs_max` / `f1_mean` | F-17 `band_L` `abs_max` / `f1_mean` | 混在ペア 11D $\to$ 8D $\to$ 7D `abs_max` (`f1_mean`) | 微修正帯 混在 11D $\to$ 8D $\to$ 7D `abs_max` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **全体** (`415`, 25組) | 40 / 245 / 130 (84) | 0.6819 / 0.0739 | 0.1481 / 0.0385 | 1.5714 (0.1883) $\to$ 1.0751 (0.1190) $\to$ 0.9331 (0.0935) | (`n=9`) 2.3513 $\to$ 0.3285 $\to$ 0.1644 |
| **同一正書法** (`381`, 24組) | 20 / 235 / 126 (77) | 0.6745 / 0.0742 | 0.1361 / 0.0339 | 1.6566 (0.1913) $\to$ 1.0494 (0.1201) $\to$ 0.9343 (0.0946) | (`n=9`) 2.3513 $\to$ 0.3285 $\to$ 0.1644 |
| **旧 $\to$ 新仮名** (`34`, 1組) | 20 / 10 / 4 (7) | 1.0878 / 0.0670 | 1.9358 / 0.1818 | 1.3907 (0.1558) $\to$ 1.1943 (0.1071) $\to$ 0.8942 (0.0816) | (`n=0`) — |
| **同一正書法コア** (`267`, 22組, $\Delta L/L_{\max} \le 0.30$) | 8 / 155 / 104 (20) | 0.2896 / 0.0262 | 0.0923 / 0.0122 | **1.6326** (0.1318) $\to$ **0.7794** (0.0375) $\to$ **0.5926** (0.0143) | (`n=9`) 2.3513 $\to$ 0.3285 $\to$ 0.1644 |

- **判断材料のまとめ**:
  - 旧仮名 $\to$ 新仮名 34件（`ginga_initial_vs_final`）は L 帯が 4件のみで `abs_max` 中央値 `1.9358` と異質に高いが、M/L 帯全体では `same_orthography == True`（381件）と全体（415件）のノイズ比水準はほぼ同一である。
  - 大幅長変化（$\Delta L / L_{\max} > 0.30$）を除いたコアセット（267件）でも、混在ペア（$n=20$）の 11D $\to$ 8D $\to$ 7D スパイク消長（`1.6326 -> 0.7794 -> 0.5926`, `frac_above_1sd` `0.1318 -> 0.0375 -> 0.0143`）は明確に再現された。
  - したがって、正書法変換ノイズを除外した **`same_orthography == True`（381件）を今後の Primary データセットに据える**根拠は十分である（34件は歴史的仮名遣い異同の Auxiliary として分離保持）。

---

## 4. タスク4：`sparsity-audit` の 106 作品ファイルでの正式実行と正本記録

- **正本入力**: `C:/dev/usir/data/v5/aozora_106works_scenes.jsonl`（106 作品・4,576 場面、`pilot_50` 不使用）
- **実行コマンド**: `C:\dev\nishiki\target\f14\release\nishiki.exe sparsity-audit --scenes C:/dev/usir/data/v5/aozora_106works_scenes.jsonl --skip-main --out C:/dev/usir/docs/handoff/v10/sparsity_audit_106works_canonical.json`
- **正本出力**: `docs/handoff/v10/sparsity_audit_106works_canonical.json`
  - **LF SHA-256**: `d16d95b54c89ad7010cadea6da4c1a0d553c2188c24b30cbea6b0e52c8c6e64a`
  - **CRLF SHA-256**: `efee120d503c929fd363f81a305cf8845a3c661c271faeea5867dcbb5f9c93a7`
- **F-16 参考実行との一致確認**: `C:\dev\nishiki\docs\handoff\evidence\2026-11-13_f16\sparsity_106works_ref.json` と**バイト単位で完全一致**（`profile_had_errors: false`, `n_scenes_read: 4576`, `n_works: 106`）。

### 4.1 場面単位 8 次元（連続 5、mode_heavy 1、zero_heavy 2）
| 次元名 | 分類 (`class`) | `n` | `n_distinct` | `zero_rate` | `mode_value` | `mode_rate` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `nagare.event_accel` | `continuous` | 4,364 | 4,364 | 0.0000 | 0.0695 | 0.0002 |
| `nagare.event_speed` | `continuous` | 4,470 | 4,470 | 0.0000 | 0.0190 | 0.0002 |
| `nagare.mood_accel` | `continuous` | 4,364 | 4,364 | 0.0000 | 0.0288 | 0.0002 |
| `nagare.mood_speed` | `continuous` | 4,470 | 4,470 | 0.0000 | 0.0018 | 0.0002 |
| `sadame.e_inter` | `continuous` | 4,576 | 134 | 0.3446 | 0.0000 | 0.3446 |
| `sadame.edges_eff` | `mode_heavy` | 4,576 | 7 | 0.3446 | 1.0000 | 0.6226 |
| `awase.detachment` | `zero_heavy` | 4,576 | 6 | 0.8302 | 0.0000 | 0.8302 |
| `awase.irony` | `zero_heavy` | 4,576 | 4 | 0.8979 | 0.0000 | 0.8979 |

### 4.2 作品単位 16 次元（連続 13、discrete 3）
| 次元名 | 分類 (`class`) | `n` | `n_distinct` | `zero_rate` | `mode_value` | `mode_rate` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `kamae.hook_survival` | `continuous` | 106 | 86 | 0.0094 | 0.9000 | 0.0377 |
| `kamae.hook_survival_idf` | `continuous` | 106 | 106 | 0.0094 | 0.0000 | 0.0094 |
| `kamae.opened_hooks` | `continuous` | 106 | 50 | 0.0000 | 7.0000 | 0.0660 |
| `sadame.e_inter` | `continuous` | 106 | 100 | 0.0189 | 0.0000 | 0.0189 |
| `sadame.e_intra` | `continuous` | 106 | 106 | 0.0094 | 0.0000 | 0.0094 |
| `sadame.edges_eff` | `continuous` | 106 | 47 | 0.0189 | 7.0000 | 0.0660 |
| `sadame.total` | `continuous` | 106 | 106 | 0.0000 | 0.2168 | 0.0094 |
| `nagare.phi_zeami` | `continuous` | 106 | 106 | 0.0000 | 0.1988 | 0.0094 |
| `nagare.event_still_fraction` | `continuous` | 106 | 62 | 0.2358 | 0.0000 | 0.2358 |
| `nagare.mood_still_fraction` | `continuous` | 106 | 68 | 0.1792 | 0.0000 | 0.1792 |
| `awase.irony_mean` | `continuous` | 106 | 58 | 0.1604 | 0.0000 | 0.1604 |
| `awase.detachment_mean` | `continuous` | 106 | 78 | 0.0849 | 0.0000 | 0.0849 |
| `awase.detachment_max` | `continuous` | 106 | 5 | 0.0849 | 1.0000 | 0.4340 |
| `nagare.event_zero_speed_fraction` | `discrete` | 106 | 1 | 1.0000 | 0.0000 | 1.0000 |
| `nagare.mood_zero_speed_fraction` | `discrete` | 106 | 1 | 1.0000 | 0.0000 | 1.0000 |
| `awase.irony_max` | `discrete` | 106 | 3 | 0.1604 | 1.0000 | 0.7925 |

---

## 5. `pytest python/tests/v5/` フル実行結果（実出力）

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.2.0, pluggy-1.7.0 -- C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe
cachedir: .pytest_cache
rootdir: C:\dev\usir\python
configfile: pyproject.toml
plugins: anyio-4.15.1
collecting ... collected 68 items

python\tests\v5\test_b11_gate3a_and_f14_parity.py::test_b11_gate_status_and_non_conflation_invariants PASSED [  1%]
python\tests\v5\test_b11_gate3a_and_f14_parity.py::test_b10_mc_quantiles_exact_reproducibility PASSED [  2%]
python\tests\v5\test_b11_gate3a_and_f14_parity.py::test_b11_old_vs_new_thresholds_on_64_benchmark PASSED [  4%]
python\tests\v5\test_b11_gate3a_and_f14_parity.py::test_f14_window_count_and_hybrid_rule_mechanics PASSED [  5%]
python\tests\v5\test_b11_gate3a_and_f14_parity.py::test_b11_f14_parity_artifact_and_zero_discrepancies PASSED [  7%]
python\tests\v5\test_b12_dice_kasane8_and_v1_dataset.py::test_task1_dice_formula_and_u6_unaligned_gate PASSED [  8%]
python\tests\v5\test_b12_dice_kasane8_and_v1_dataset.py::test_task2_gate3a_explicit_primary_auxiliary_alarm_thresholds PASSED [ 10%]
python\tests\v5\test_b12_dice_kasane8_and_v1_dataset.py::test_task3_kasane8_index_unification_and_h1_h2_recalc PASSED [ 11%]
python\tests\v5\test_b12_dice_kasane8_and_v1_dataset.py::test_frozen_b8_to_b11_artifacts_unmodified PASSED [ 13%]
python\tests\v5\test_b12_dice_kasane8_and_v1_dataset.py::test_task4_real_revision_pairs_v1_dataset_and_f15_harvest_diff PASSED [ 14%]
python\tests\v5\test_b13_f17_ablation_and_sparsity.py::test_b13_f17_noise_floor_sha256_and_per_band_ac_values PASSED [ 16%]
python\tests\v5\test_b13_f17_ablation_and_sparsity.py::test_b13_task1_f17_summary_parity_and_verification_gate PASSED [ 17%]
python\tests\v5\test_b13_f17_ablation_and_sparsity.py::test_b13_task2_ablation_and_stratification PASSED [ 19%]
python\tests\v5\test_b13_f17_ablation_and_sparsity.py::test_b13_task3_orthography_and_core_cohorts PASSED [ 20%]
python\tests\v5\test_b13_f17_ablation_and_sparsity.py::test_b13_task4_sparsity_audit_106works_canonical PASSED [ 22%]
python\tests\v5\test_b13_f17_ablation_and_sparsity.py::test_b13_governance_gates_unchanged PASSED [ 23%]
python\tests\v5\test_gate3a_scene_scale.py::test_gate_status_constants PASSED [ 25%]
python\tests\v5\test_gate3a_scene_scale.py::test_scene_variance_calculations_and_rank_deficiency PASSED [ 26%]
python\tests\v5\test_gate3a_scene_scale.py::test_condition_a_and_b_behavior PASSED [ 27%]
python\tests\v5\test_gate3a_scene_scale.py::test_b10_json_artifact_schema_and_values PASSED [ 29%]
python\tests\v5\test_modern_text_normalizer.py::test_normalize_modern_punctuation PASSED [ 30%]
python\tests\v5\test_modern_text_normalizer.py::test_web_novel_rubi_removal PASSED [ 32%]
python\tests\v5\test_modern_text_normalizer.py::test_dialogue_dominant_layout_and_nan_guards PASSED [ 33%]
python\tests\v5\test_modern_text_normalizer.py::test_cleanroom_irreversible_profile_invariant PASSED [ 35%]
python\tests\v5\test_modern_text_normalizer.py::test_mask_entities_to_black_square PASSED [ 36%]
python\tests\v5\test_narrative_pacing.py::test_hw22_metaphor_leap PASSED [ 38%]
python\tests\v5\test_narrative_pacing.py::test_hw24_cliffhanger PASSED   [ 39%]
python\tests\v5\test_narrative_pacing.py::test_hw25_uid_ratio PASSED     [ 41%]
python\tests\v5\test_narrative_pacing.py::test_extract_all PASSED        [ 42%]
python\tests\v5\test_ortho_v2.py::test_content_char_detection PASSED     [ 44%]
python\tests\v5\test_ortho_v2.py::test_content_char_count PASSED         [ 45%]
python\tests\v5\test_ortho_v2.py::test_v2_m1_non_comma_invariance PASSED [ 47%]
python\tests\v5\test_scene_splitter.py::test_all_106_works_present PASSED [ 48%]
python\tests\v5\test_scene_splitter.py::test_scene_continuity_per_work PASSED [ 50%]
python\tests\v5\test_scene_splitter.py::test_scene_length_bounds PASSED  [ 51%]
python\tests\v5\test_scene_splitter.py::test_usir_schema_strict_compliance PASSED [ 52%]
python\tests\v5\test_scene_splitter.py::test_degenerate_and_boundary_inputs PASSED [ 54%]
python\tests\v5\test_scene_splitter.py::test_pilot_50_scenes PASSED      [ 55%]
python\tests\v5\test_scene_splitter.py::test_fix1_melos_and_lemon_awase PASSED [ 57%]
python\tests\v5\test_scene_splitter.py::test_fix2_sadame_characters_and_edges PASSED [ 58%]
python\tests\v5\test_scene_splitter.py::test_fix3_kamae_opened_and_closed_hooks PASSED [ 60%]
python\tests\v5\test_scene_splitter.py::test_fix4_nagare_continuous_logp_and_mood_diversity PASSED [ 61%]
python\tests\v5\test_scene_splitter.py::test_fid_5_triggers_and_adjacent_dialogue_edges PASSED [ 63%]
python\tests\v5\test_scene_splitter.py::test_within_work_isf_hook_filtering PASSED [ 64%]
python\tests\v5\test_scene_splitter.py::test_pilot_50_split_is_schema_conformant_and_lossless PASSED [ 66%]
python\tests\v5\test_six_pole_atlas.py::test_narrative_rg_normalizer_features PASSED [ 67%]
python\tests\v5\test_six_pole_atlas.py::test_normalized_hsic_properties PASSED [ 69%]
python\tests\v5\test_six_pole_atlas.py::test_svd_lowdin_orthogonalization PASSED [ 70%]
python\tests\v5\test_six_pole_atlas.py::test_goldilocks_void_finder_logic PASSED [ 72%]
python\tests\v5\test_six_pole_atlas.py::test_order0_and_order1_calibration_artifacts PASSED [ 73%]
python\tests\v5\test_six_pole_atlas.py::test_order2_and_order3_diachronic_artifacts PASSED [ 75%]
python\tests\v5\test_u6_guards.py::test_numeral_parser PASSED            [ 76%]
python\tests\v5\test_u6_guards.py::test_numeral_guard_identical_value_style_polish PASSED [ 77%]
python\tests\v5\test_u6_guards.py::test_numeral_guard_value_change_content_rewrite PASSED [ 79%]
python\tests\v5\test_u6_guards.py::test_negation_guard_polarity_flip_content_rewrite PASSED [ 80%]
python\tests\v5\test_u6_guards.py::test_negation_guard_double_negation_exemption PASSED [ 82%]
python\tests\v5\test_u6_guards.py::test_negation_guard_lexical_prefix_ignored PASSED [ 83%]
python\tests\v5\test_u6_guards.py::test_numeral_closed_class_idiom_exclusions PASSED [ 85%]
python\tests\v5\test_u6_guards.py::test_numeral_closed_class_man_ga_ichi_exclusion PASSED [ 86%]
python\tests\v5\test_u6_guards.py::test_numeral_adverbial_juuppun_exclusion PASSED [ 88%]
python\tests\v5\test_u6_guards.py::test_negation_adverb_mazu_exclusion PASSED [ 89%]
python\tests\v5\test_u6_guards.py::test_negation_conjecture_modality_exclusion PASSED [ 91%]
python\tests\v5\test_u6_guards.py::test_negation_adjective_stems_exclusion PASSED [ 92%]
python\tests\v5\test_u6_guards.py::test_numeral_placeholder_orthographic_protection PASSED [ 94%]
python\tests\v5\test_u6_guards.py::test_numeral_fraction_ratio_jubun_no_ichi PASSED [ 95%]
python\tests\v5\test_u6_guards.py::test_numeral_single_unit_kanji_hyaku PASSED [ 97%]
python\tests\v5\test_u6_guards.py::test_u6_test_vectors_fixture_pass PASSED [ 98%]
python\tests\v5\test_u6_guards.py::test_u6_profiles PASSED               [100%]

======================= 68 passed in 135.01s (0:02:15) ========================
```

---

## 6. 冷徹な自己内省・死角の開示（Ruthless Self-Critical Audit）

1. **誤爆（False Positive）**: 現行 11D 評価では、微修正帯（$\Delta L / L_{\max} < 15\%, \text{Dice} \ge 0.85$）の混在ペア（$n=9$）で 1窓側の `ac_*` 3次元が厳密に `0.0` であるにもかかわらず `|値 - 0|/σ` として評価されるため、`abs_max` 中央値 `2.3513` という高い偽スパイクが生じる。
2. **取りこぼし（False Negative）と標本数の限界**: 層別分析②の微修正帯における混在ペアは $n=9$（うち `< 5%` が 8 件）であり、統計的検出力には標本数の制約がある。また `sparsity-audit --skip-main` は場面注釈 4,576 場面の監査を正本固定したが、青空 46 作品の生テキスト段落側（`spaces`）は F-14/F-15 で別途計測済みのため本正本ではスキップしている。
3. **社会的先入観と物理的事実の乖離**: 混在ペア（84件）全体で 8D 化しても `abs_max` 中央値が `1.0751` に留まったため、当初想定した「11D $\to$ 8D のみで全混在ペアが一律 0.30 前後に落ちる」という単純仮説（ケース1単独）とは異なり、実態は「**混在ペアの 75.0% (63/84) が $\Delta L / L_{\max} \ge 30\%$ の大幅加筆・削除に集中している改稿量交絡（ケース3）**」と「**微修正帯での純粋な次元追加アーティファクト（ケース1）＋ `ac_mean_crest` 窓境界ジャンプ（ケース2）**」の複合であった。

---

## 7. Stage B-13 成果物・SHA-256 台帳

- スクリプト・テスト:
  - `scripts/run_stage_b13_analysis.py`
  - `python/tests/v5/test_b13_f17_ablation_and_sparsity.py`
- 成果物 JSON:
  - `docs/handoff/v10/real_revision_pairs_v1_f17_harvest_summary.json` (`6b650b64f143355d862eb3ae47f12a58e425405ba610339404219335a5bd8611`)
  - `docs/handoff/v10/sparsity_audit_106works_canonical.json` (LF: `d16d95b54c89ad7010cadea6da4c1a0d553c2188c24b30cbea6b0e52c8c6e64a` / CRLF: `efee120d503c929fd363f81a305cf8845a3c661c271faeea5867dcbb5f9c93a7`)
- レポート:
  - `docs/handoff/2026-11-13_usir_v5_stage_b13_report.md`
  - `Claude-Sync/inbox/2026-11-13_usir_v5_stage_b13_report.md`
  - Google Drive `Claude-Sync/inbox/2026-11-13_usir_v5_stage_b13_report.md`
