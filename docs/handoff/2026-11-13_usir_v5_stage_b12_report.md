# USIR v5 Stage B-12 完了報告書

**作成日**: 2026-11-13  
**対象リポジトリ**: `C:\dev\usir`（ブランチ `feat/fable-v5-shootout`）  
**起点 HEAD (Stage B-11)**: `ae992e5`  
**Stage B-12 実装コミットハッシュ**: `6e526995a12407cd0404034aa1238c70335f18f6`（`6e52699`）  
**参照ノイズ床 (F-15) SHA-256**: `5a6095659d4ef7defeab238f7ffb7aed3a980d6c8a9a83d0ca229f4c11a05910`  

---

## 0. ガバナンスおよび確定事項遵守チェック

| 項目 | 本ステージの状態 | 検証根拠 |
| :--- | :--- | :--- |
| Gate 3a 判定 | `SHIFT_QUANTIFIED_NOT_PASSED`（不変） | `data/usir_v5_eval/usir_v5_b12_canonical_eval.json` / `docs/handoff/v10/gate3a_register_centering_summary.json` |
| Gate 3b 判定 | `GATE3B_FROZEN`（不変） | 同上 |
| 妻の未発表原稿（層B-1） | **一切不使用（0 件）** | タスク 4 データ源はすべて青空文庫「作品著作権フラグ=なし」公開版のみ |
| 有料 API | **一切不使用** | 全処理をローカル Python 3.12 + Rust `nishiki.exe` で完結 |
| Stage B-13 設計・自動開始 | **未実施（禁止事項遵守）** | 設計文書・起動コード・予約実行なし |
| 凍結成果物（B-8〜B-11）上書き禁止 | **不変確認済み** | 下記 4 ファイルの SHA-256 一致を単体テストで検証 |

凍結済み成果物の SHA-256（改変なし実測値）：
- `data/usir_v5_eval/usir_v5_stage_b8_scale_separation_and_gate3a.json`: `975765a13db7fdf89f1b32a21a88bf02f470817a4308a750b0be1bd521671de1`
- `data/usir_v5_eval/usir_v5_b9_register_centering_and_level3.json`: `4a555708dbcba73f4b08ad6e13c1aca4828231b698421f7903b3e64dd09032a0`
- `data/usir_v5_eval/usir_v5_b10_gate3a_scene_scale_and_level3.json`: `2e0833d5c300273f73222c9ea3b039ec7358006849e9ff4957e4d9105c991c60`
- `data/usir_v5_eval/usir_v5_b11_gate3a_calibrated_and_f14_parity.json`: `929bbfe9e7340bcf4f648145403e9c9f82ef9d2c78e7101e09ffe7c417ca2d84`

---

## 1. タスク1：Dice ゲートの Python 実装（正本統一）

### 1.1 実装内容
Rust `crates/nishiki-cli/src/diff_harvester.rs` を正本とし、`python/usir_py/v5/f14_hybrid_noise_floor.py` および `experiments/scale_separated_level3.py` に非空白文字 2-gram Dice ゲートを実装した。
- 定数・関数追加：`ALIGN_MIN_SIMILARITY = 0.25`、`char_bigrams()`、`char_bigram_dice()`
- `evaluate_f14_hybrid_pair(..., apply_dice_gate=True)`:
  - `dice < 0.25` の場合：`kind = "unaligned"`、`n_evaluated = None`、`n_eval_rule = None`、`ac_sigma_source = "not_applicable"` として評価対象外記録を返す。
  - `dice >= 0.25`（または Set 1 事前アライン済み隣接ペアなど `apply_dice_gate=False` 明示指定時）：`kind = "pair"` として F-14 ハイブリッド規準（両側単窓 $\to 8$、片側以上複窓 $\to 11$）を適用する。

### 1.2 3 セット再検証実測結果（Python vs Rust `nishiki.exe harvest-diff`）

| 検証セット | 件数 | Dice ゲート設定 | Rust との不一致数 | 備考 |
| :--- | ---: | :--- | ---: | :--- |
| **Set 1**: 青空 46 作品隣接段落ペア | 10,859 | 適用外 (`False`) | **0** | Band S=8,258 / M=1,952 (`n_eval=8`: 621, `n_eval=11`: 1,331) / L=649 (`n_eval=11`: 649) すべて一致 |
| **Set 2**: U-6 86 ペア（Class A/B/C） | 86 | 適用 (`True`, `< 0.25`) | **0** | **77 ペアが aligned 一致**、**9 ペアが両側 unaligned 一致** |
| **Set 3**: 推敲 28 ペア × 3 粒度 | 84 (28×3) | 適用 (`True`, `< 0.25`) | **0** | `span_only`: 28 aligned (`n_eval=8`: 22, `n_eval=11`: 6, min Dice=`0.764706`), `local_context`: 28 aligned (`n_eval=8`: 7, `n_eval=11`: 21, min Dice=`0.916201`), `whole_scene`: 28 aligned (`n_eval=11`: 28, min Dice=`0.928898`) |

#### Set 2（U-6 86 ペア）における `unaligned` 一致 9 ペアの内訳
Python・Rust 両側で `kind="unaligned"`・`n_evaluated=None` として完全一致した 9 ペアと Dice 値：

| pair_id | 文字 2-gram Dice | Python 側判定 | Rust CLI 側判定 |
| :--- | ---: | :--- | :--- |
| `A_29` | `0.214286` | `unaligned` (`n_evaluated=None`) | `unaligned` |
| `A_35` | `0.228571` | `unaligned` (`n_evaluated=None`) | `unaligned` |
| `B_19` | `0.203642` | `unaligned` (`n_evaluated=None`) | `unaligned` |
| `B_20` | `0.176724` | `unaligned` (`n_evaluated=None`) | `unaligned` |
| `B_22` | `0.167832` | `unaligned` (`n_evaluated=None`) | `unaligned` |
| `C_19` | `0.064516` | `unaligned` (`n_evaluated=None`) | `unaligned` |
| `C_20` | `0.044444` | `unaligned` (`n_evaluated=None`) | `unaligned` |
| `C_21` | `0.103448` | `unaligned` (`n_evaluated=None`) | `unaligned` |
| `C_22` | `0.120000` | `unaligned` (`n_evaluated=None`) | `unaligned` |

Class 別 aligned 内訳（合計 77 aligned / 9 unaligned）：
- Class A (`n=42`): aligned `40` (`n_eval=8`: 40)、unaligned `2`
- Class B (`n=22`): aligned `19` (`n_eval=11`: 19)、unaligned `3`
- Class C (`n=22`): aligned `18` (`n_eval=8`: 18)、unaligned `4`

---

## 2. タスク2：一次警報・補助警報閾値の明示

`experiments/gate3a_register_centering_eval.py` に `PRIMARY_ALARM_THRESHOLD_KEY = "mc_indep_p95"`、`AUXILIARY_ALARM_THRESHOLD_KEY = "mc_r4corr_p95"`、`ALARM_THRESHOLD_SPEC`、`get_alarm_threshold_specification()`、`build_b12_gate3a_summary()` を追加し、`docs/handoff/v10/gate3a_register_centering_summary.json`（SHA-256: `6874a66eb78a8469c8c5878f90aa63a9aff5c34c6f01254d1d9664601cf2922b`）および `data/usir_v5_eval/usir_v5_b12_canonical_eval.json` に明示的に記録した。閾値数値および Gate 判定値は一切変更していない。

| 役割 | ラベル | 閾値キー | 閾値 ($d$) | $d^2$ | 帰無モデル |
| :--- | :--- | :--- | ---: | ---: | :--- |
| **一次警報 (`primary`)** | 一次警報 | `mc_indep_p95` | **6.9267** | 47.9796 | `independent_gaussian_24d_loo` |
| **補助警報 (`auxiliary`)** | 補助警報 | `mc_r4corr_p95` | **7.6574** | 58.6355 | `r4_correlated_gaussian_24d_loo` |

---

## 3. タスク3：KASANE8 インデックス統一検証と H1／H2 再検算

### 3.1 次元集合の差異検証
`experiments/scale_separated_level3.py::extract_kasane8_active` の旧実装（配列先頭 8 要素スライス `[:8]`）と `KASANE8_ACTIVE_DIM_NAMES` が選択する次元集合を比較し、**同一ではない（`same_index_set = False`）**ことを確認した。
- **旧スライス `[:8]` (`[0..7]`)**: インデックス 7（正本 `ac_mean_velocity` / Python 代理 `c_no`）を含み、インデックス 10（正本 `ac_mean_crest` / Python 代理 `mean_cl_div_100`）を落としていた。
- **統一 `KASANE8_ACTIVE_INDICES` (`(0, 1, 2, 3, 4, 5, 6, 10)`)**: `ac_mean_velocity` / `ac_max_sis` / `ac_coupling`（indices 7, 8, 9）を除外し、`ac_mean_crest` / `mean_cl_div_100`（index 10）を保持する。
- 対応変更：`scale_separated_level3.py` の `extract_kasane8_active` を `KASANE8_ACTIVE_INDICES` に統一（旧実装比較用に `extract_kasane8_active_legacy_slice` を保持）し、凍結済み B-8〜B-11 成果物は上書きせず、付属報告 `docs/handoff/v10/b12_kasane8_unification_h1_h2_recalc.json`（SHA-256: `0fdae7c4b66d06e26c1b4adc153fbdf1dbc33061f8a1c78c4116ec777770477f`）へ再検算結果を出力した。

### 3.2 H1／H2 再検算結果と凍結値との差分

| 検定空間・項目 | 凍結 B-8 値（旧 `[:8]`） | 統一再検算（`[0..6, 10]`） | 差分 ($\Delta$) |
| :--- | ---: | ---: | ---: |
| **H1 (32D コア, `n=42`, `n_quasi=8`)** | | | |
| `median_rho_perp` | 0.1462 | 0.1464 | +0.0002 |
| `median_theta_deg` | 8.41° | 8.42° | +0.01° |
| `mean_cos_spec` | 0.1083 | **0.1083** | **0.0** |
| `mean_cos_rand` | -0.0001 | 0.0020 | +0.0021 |
| `mean_cos_other` | -0.0128 | **-0.0128** | **0.0** |
| `vs_other` Wilcoxon $W$ | 654.0 | **654.0** | **0.0** |
| `vs_other` $p_{\text{one-sided}}$ / $p_{\text{two-sided}}$ | 0.005252 / 0.010504 | **0.005252 / 0.010504** | **0.0** |
| `vs_other` Bonferroni $p_{\text{two-sided}}$ | 0.042017 (`sig_005=True`) | **0.042017 (`sig_005=True`)** | **0.0** |
| `vs_random` Wilcoxon $W$ | 552.0 | 553.0 | +1.0 |
| `vs_random` $p_{\text{one-sided}}$ / $p_{\text{two-sided}}$ | 0.011889 / 0.023777 | 0.011463 / 0.022926 | -0.000426 / -0.000851 |
| `vs_random` Bonferroni $p_{\text{two-sided}}$ | 0.095109 (`sig_005=False`) | 0.091704 (`sig_005=False`) | -0.003405 |
| **H2 (5D 連続コア, `n=22`, `n_quasi=11`)** | *(KASANE8 非使用)* | | |
| `mean_cos_spec` / `rand` / `other` | 0.0849 / 0.0001 / -0.1089 | **0.0849 / 0.0001 / -0.1089** | **0.0 (ビット一致)** |
| `vs_other` $W$ / $p_{\text{two}}$ / Bonf $p_{\text{two}}$ | 214.0 / 0.003239 / 0.012957 | **214.0 / 0.003239 / 0.012957** | **0.0 (ビット一致)** |
| `vs_random` $W$ / $p_{\text{two}}$ / Bonf $p_{\text{two}}$ | 153.0 / 0.406033 / 1.000000 | **153.0 / 0.406033 / 1.000000** | **0.0 (ビット一致)** |

### 3.3 格付け・結論への影響判定
- **格付けの結論変更有無**: **変更なし（`rating_conclusion_changed = False`）**。
  - H1 の `vs_other` は完全不変（Bonferroni 補正後有意のまま）、`vs_random` は片側・両側名目有意（$p < 0.05$）・両側 Bonferroni 非有意（0.0917）とも凍結値と同一の判定区分に留まる。H2 は KASANE8 を使用しないため 1 ビットの差も生じない。

---

## 4. タスク4：実推敲ペア拡充データセット v1 の構築と F-15 実測

### 4.1 データセット構築仕様と出典（Provenance）
- スクリプト：`scripts/build_real_revision_pairs_v1.py`
- 出力データセット：`data/usir_v5_eval/real_revision_pairs_v1.jsonl`（SHA-256: `62da75271f453bd3a8f161d916fdf7b7b87138b773b564b737f90aee85b6b8e4`）
- 入力テキスト単位：青空文庫クリーンテキストを単一改行単位で分割し、**1 行＝厳密な単一段落**（全 415 件で `"\n" not in text_before and "\n" not in text_after`）に対して Rust `nishiki.exe harvest-diff` による段落アライメントを実行した。複数改行ブロック結合による章連結・疑似長段落化は排除している。
- 候補 39 作品組（青空文庫 `作品著作権フラグ == "なし"`）のうち **25 作品組** から計 1,212 件（S=837, M=245, L=130, mixed=84）の非同一（ただし語彙類似範囲内の改稿）段落ペアを抽出し、**Band M・Band L・混在ペアは水増しなしで全件採用**、純粋な Band S は先頭 40 件に上限設定して合計 **415 ペア** とした。

| 区分 | 目安目標 | 実績件数 | 達成状況 |
| :--- | ---: | ---: | :--- |
| 総件数 | — | **415** | — |
| Band S ($\le 250$ 字) | — | **40** | 上限 40 件（候補プール 837 件） |
| **Band M (251〜500 字)** | 30 件以上 | **245** | **達成** |
| **Band L (501 字以上)** | 20 件以上 | **130** | **達成** |
| **片側単窓／片側複窓 混在ペア** | 20 件以上 | **84** | **達成**（Band M 内 65 件、Band L 内 19 件） |

variant relationship 別件数：
- `first_publication_vs_book`（初出・雑誌掲載 vs 単行本／全集）：**221 件**
- `single_volume_vs_complete_works`（単行本／文庫 vs 全集・定本）：**81 件**
- `cross_edition_variant`（底本・版間異同）：**73 件**
- `initial_draft_vs_final`（草稿・初期稿 vs 定稿）：**40 件**

正書法別件数：
- 同一仮名遣い・文字遣い（`same_orthography = True`）：**381 件**
- 旧仮名 $\to$ 新仮名（`same_orthography = False`）：**34 件**（宮沢賢治『〔「銀河鉄道の夜」初期形一〕』 vs 『銀河鉄道の夜』）

出典作品組別件数（25 有効作品組／候補 39 組中抽出 0 件の 14 組も含め全記録は `docs/handoff/v10/real_revision_pairs_v1_f15_harvest_summary.json` に保存）：
- `horoki_first_vs_new` (林芙美子『放浪記（初出）』 vs 『新版 放浪記』): 採用 49 件 (M=28, L=1, mixed=16; S=20採用/元120)
- `oda_horo_first_vs_rev` (織田作之助『放浪』(初出) vs 定本全集1): 39 (M=31, L=8, mixed=1)
- `oda_ame_first_vs_rev` (織田作之助『雨』(初出) vs 文学全集72): 36 (M=19, L=17, mixed=21)
- `ginga_initial_vs_final` (宮沢賢治『銀河鉄道の夜』初期形一 vs 新潮文庫): 34 (S=20, M=10, L=4, mixed=7)
- `ichiyo_takekurabe_ed` (樋口一葉『たけくらべ』文藝倶樂部初出 vs 全集): 30 (M=10, L=20)
- `hisao_mugetsu_ed` (久生十蘭『無月物語』短篇選 vs 全集Ⅱ): 29 (M=26, L=3, mixed=9)
- `kusaka_saigo_ed` (久坂葉子『幾度目かの最期』単行本 vs 作品集): 28 (M=11, L=17, mixed=2)
- `miyazawa_ginga_ed` (宮沢賢治『銀河鉄道の夜』岩波文庫 vs 新潮文庫): 24 (M=22, L=2, mixed=5)
- `origuchi_shisha_ed` (折口信夫『死者の書』全集 vs 単行本): 22 (M=19, L=3, mixed=3)
- `ichiyo_nigorie_ed` (樋口一葉『にごりえ』文藝倶樂部初出 vs 全集): 20 (M=4, L=16, mixed=1)
- `ichiyo_jusanya_ed` (樋口一葉『十三夜』文藝倶樂部初出 vs 全集): 14 (M=3, L=11, mixed=2)
- `ichiyo_yukukumo_ed` (樋口一葉『ゆく雲』太陽初出 vs 全集): 13 (M=4, L=9, mixed=2)
- `ichiyo_utsusemi_ed` (樋口一葉『うつせみ』二全集異同): 11 (M=7, L=4, mixed=2)
- `yamamoto_aobeka_ed` (山本周五郎『青べか物語』単行本 vs 全集): 11 (M=9, L=2)
- `oda_wagamachi_first_vs_rev` (織田作之助『わが町』(初出) vs 名作選集9): 10 (M=10, mixed=4)
- `ichiyo_otsugomori_ed` (樋口一葉『大つごもり』太陽 vs 全集): 10 (M=2, L=8, mixed=1)
- `okamoto_rogisho_ed` (岡本かの子『老妓抄』文庫 vs 全集): 10 (M=10)
- `tanizaki_neko_ed` (谷崎潤一郎『猫と庄造と二人のをんな』単行本 vs 全集): 9 (M=4, L=5)
- `kishida_doin_draft1_vs_final` (岸田国士『動員挿話［第一稿］』 vs 最終稿): 5 (M=5, mixed=4)
- `kikuchi_tojuro_ed` (菊池寛『藤十郎の恋』二版異同): 4 (M=4, mixed=2)
- `kikuchi_irefuda_ed` (菊池寛『入れ札』二版異同): 2 (M=2, mixed=1)
- `ishikawa_yama_ed` (石川欣一『可愛い山』二版異同): 2 (M=2)
- `kishida_kotoba_draft1_vs_final` (岸田国士『言葉の魅力［第一稿］』 vs 定稿): 1 (M=1, mixed=1)
- `kishida_hogen_ed` (岸田国士『方言について』二版異同): 1 (M=1)
- `nakaya_alaska_ed` (中谷宇吉郎『アラスカの氷河』二版異同): 1 (M=1)

*(注: Band M/L/混在が 0 件だった作品組のうち Band S は一部先頭 40 件に含まれないため、総計は採用済み 415 件に一致する)*

### 4.2 F-15 ノイズ床での `harvest-diff` 実測結果（記述統計・格付け主張なし）
- **使用ノイズ床**: `docs/handoff/v10/paragraph_noise_floor_f15_all_novel.json`
  - **SHA-256**: `5a6095659d4ef7defeab238f7ffb7aed3a980d6c8a9a83d0ca229f4c11a05910`
- **Python vs Rust (`nishiki.exe harvest-diff`) パリティ不一致件数**: **0 件 / 415 ペア**
- **成果物**: `docs/handoff/v10/real_revision_pairs_v1_f15_harvest_summary.json`（SHA-256: `82e594f30741551bff62494b269758ae73e7f6e1aaae3900eb97f8183cd03456`）

#### 4.2.1 帯別 `n_evaluated` 分布および `per_pair_abs_ratio_median`／`mean`／`max`／`frac >= 1SD`

| サブセット (`n`) | `n_eval=8` | `n_eval=11` | `mixed` | median of `abs_median` [Q25, Q75] | median of `abs_mean` [Q25, Q75] | median of `abs_max` [Q25, Q75] | median / mean of `frac_above_1sd` |
| :--- | ---: | ---: | ---: | :--- | :--- | :--- | :--- |
| **overall** (`415`) | 196 | 219 | 84 | **0.0235** [0.0017, 0.2791] | **0.1252** [0.0223, 0.4229] | **0.5015** [0.0987, 1.3208] | 0.0000 / **0.0633** (Q75=0.0909) |
| **band_S** (`40`) | 40 | 0 | 0 | **0.0692** [0.0001, 0.3247] | **0.2564** [0.0679, 0.4341] | **0.8256** [0.2472, 1.3339] | 0.0000 / **0.0656** (Q75=0.1250) |
| **band_M** (`245`) | 106 | 139 | 65 | **0.0988** [0.0033, 0.3496] | **0.2016** [0.0404, 0.4834] | **0.6616** [0.2304, 1.4206] | 0.0000 / **0.0762** (Q75=0.1250) |
| **band_L** (`130`) | 50 | 80 | 19 | **0.0036** [0.0006, 0.0996] | **0.0311** [0.0057, 0.2021] | **0.1444** [0.0373, 0.7809] | 0.0000 / **0.0385** (Q75=0.0000) |
| **mixed_window_pairs** (`84`) | 0 | 84 | 84 | **0.4732** [0.2671, 0.6520] | **0.5875** [0.4515, 0.7369] | **1.6715** [1.3126, 2.2919] | **0.1818** / **0.1937** (Q75=0.2727) |
| **non_mixed_both_single_window** (`196`) | 196 | 0 | 0 | **0.0042** [0.0007, 0.1710] | **0.0458** [0.0098, 0.2537] | **0.2739** [0.0482, 0.7881] | 0.0000 / **0.0325** (Q75=0.0000) |
| **non_mixed_both_multi_window** (`135`) | 0 | 135 | 0 | **0.0077** [0.0021, 0.1141] | **0.0625** [0.0149, 0.1995] | **0.3248** [0.0753, 0.7803] | 0.0000 / **0.0269** (Q75=0.0000) |
| **band_M_n_eval_8** (`106`) | 106 | 0 | 0 | **0.0190** [0.0026, 0.2032] | **0.0919** [0.0300, 0.2814] | **0.4198** [0.1621, 0.8929] | 0.0000 / **0.0354** (Q75=0.0000) |
| **band_M_n_eval_11** (`139`) | 0 | 139 | 65 | **0.1768** [0.0050, 0.4727] | **0.3473** [0.0596, 0.6168] | **1.2042** [0.3232, 1.7390] | **0.0909** / **0.1073** (Q75=0.1818) |
| **band_L_mixed_window** (`19`) | 0 | 19 | 19 | **0.3725** [0.2639, 0.5561] | **0.5363** [0.4165, 0.7414] | **1.6901** [1.2463, 2.2700] | **0.1818** / **0.1962** (Q75=0.2727) |
| **band_L 内の非混在 (`111`)** *(JSONキー `band_L_both_multi_window`)* | 50 | 61 | 0 | **0.0023** [0.0005, 0.0222] | **0.0249** [0.0033, 0.0916] | **0.0978** [0.0189, 0.3575] | 0.0000 / **0.0115** (Q75=0.0000) |

全サブセットの詳細統計（`min` / `Q25` / `median` / `Q75` / `IQR` / `max` / `mean`、および `per_pair_frac_above_2sd`）は `docs/handoff/v10/real_revision_pairs_v1_f15_harvest_summary.json` に格納済みである。注記すべき観察事実として、Band L 内の非混在 111 ペア（summary JSON 内キー名 `band_L_both_multi_window`）の内訳は、片側が 501 字以上であるにもかかわらず両側とも文窓数 1 であった 50 件（`n_evaluated=8`）と、両側とも複数窓であった 61 件（`n_evaluated=11`）の合計で構成される。

---

## 5. `pytest python/tests/v5/` フル実行結果（最終コード編集後の実出力）

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.2.0, pluggy-1.7.0 -- C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe
cachedir: .pytest_cache
rootdir: C:\dev\usir\python
configfile: pyproject.toml
plugins: anyio-4.15.1
collecting ... collected 62 items

python\tests\v5\test_b11_gate3a_and_f14_parity.py::test_b11_gate_status_and_non_conflation_invariants PASSED [  1%]
python\tests\v5\test_b11_gate3a_and_f14_parity.py::test_b10_mc_quantiles_exact_reproducibility PASSED [  3%]
python\tests\v5\test_b11_gate3a_and_f14_parity.py::test_b11_old_vs_new_thresholds_on_64_benchmark PASSED [  4%]
python\tests\v5\test_b11_gate3a_and_f14_parity.py::test_f14_window_count_and_hybrid_rule_mechanics PASSED [  6%]
python\tests\v5\test_b11_gate3a_and_f14_parity.py::test_b11_f14_parity_artifact_and_zero_discrepancies PASSED [  8%]
python\tests\v5\test_b12_dice_kasane8_and_v1_dataset.py::test_task1_dice_formula_and_u6_unaligned_gate PASSED [  9%]
python\tests\v5\test_b12_dice_kasane8_and_v1_dataset.py::test_task2_gate3a_explicit_primary_auxiliary_alarm_thresholds PASSED [ 11%]
python\tests\v5\test_b12_dice_kasane8_and_v1_dataset.py::test_task3_kasane8_index_unification_and_h1_h2_recalc PASSED [ 12%]
python\tests\v5\test_b12_dice_kasane8_and_v1_dataset.py::test_frozen_b8_to_b11_artifacts_unmodified PASSED [ 14%]
python\tests\v5\test_b12_dice_kasane8_and_v1_dataset.py::test_task4_real_revision_pairs_v1_dataset_and_f15_harvest_diff PASSED [ 16%]
python\tests\v5\test_gate3a_scene_scale.py::test_gate_status_constants PASSED [ 17%]
python\tests\v5\test_gate3a_scene_scale.py::test_scene_variance_calculations_and_rank_deficiency PASSED [ 19%]
python\tests\v5\test_gate3a_scene_scale.py::test_condition_a_and_b_behavior PASSED [ 20%]
python\tests\v5\test_gate3a_scene_scale.py::test_b10_json_artifact_schema_and_values PASSED [ 22%]
python\tests\v5\test_modern_text_normalizer.py::test_normalize_modern_punctuation PASSED [ 24%]
python\tests\v5\test_modern_text_normalizer.py::test_web_novel_rubi_removal PASSED [ 25%]
python\tests\v5\test_modern_text_normalizer.py::test_dialogue_dominant_layout_and_nan_guards PASSED [ 27%]
python\tests\v5\test_modern_text_normalizer.py::test_cleanroom_irreversible_profile_invariant PASSED [ 29%]
python\tests\v5\test_modern_text_normalizer.py::test_mask_entities_to_black_square PASSED [ 30%]
python\tests\v5\test_narrative_pacing.py::test_hw22_metaphor_leap PASSED [ 32%]
python\tests\v5\test_narrative_pacing.py::test_hw24_cliffhanger PASSED   [ 33%]
python\tests\v5\test_narrative_pacing.py::test_hw25_uid_ratio PASSED     [ 35%]
python\tests\v5\test_narrative_pacing.py::test_extract_all PASSED        [ 37%]
python\tests\v5\test_ortho_v2.py::test_content_char_detection PASSED     [ 38%]
python\tests\v5\test_ortho_v2.py::test_content_char_count PASSED         [ 40%]
python\tests\v5\test_ortho_v2.py::test_v2_m1_non_comma_invariance PASSED [ 41%]
python\tests\v5\test_scene_splitter.py::test_all_106_works_present PASSED [ 43%]
python\tests\v5\test_scene_splitter.py::test_scene_continuity_per_work PASSED [ 45%]
python\tests\v5\test_scene_splitter.py::test_scene_length_bounds PASSED  [ 46%]
python\tests\v5\test_scene_splitter.py::test_usir_schema_strict_compliance PASSED [ 48%]
python\tests\v5\test_scene_splitter.py::test_degenerate_and_boundary_inputs PASSED [ 50%]
python\tests\v5\test_scene_splitter.py::test_pilot_50_scenes PASSED      [ 51%]
python\tests\v5\test_scene_splitter.py::test_fix1_melos_and_lemon_awase PASSED [ 53%]
python\tests\v5\test_scene_splitter.py::test_fix2_sadame_characters_and_edges PASSED [ 54%]
python\tests\v5\test_scene_splitter.py::test_fix3_kamae_opened_and_closed_hooks PASSED [ 56%]
python\tests\v5\test_scene_splitter.py::test_fix4_nagare_continuous_logp_and_mood_diversity PASSED [ 58%]
python\tests\v5\test_scene_splitter.py::test_fid_5_triggers_and_adjacent_dialogue_edges PASSED [ 59%]
python\tests\v5\test_scene_splitter.py::test_within_work_isf_hook_filtering PASSED [ 61%]
python\tests\v5\test_scene_splitter.py::test_pilot_50_split_is_schema_conformant_and_lossless PASSED [ 62%]
python\tests\v5\test_six_pole_atlas.py::test_narrative_rg_normalizer_features PASSED [ 64%]
python\tests\v5\test_six_pole_atlas.py::test_normalized_hsic_properties PASSED [ 66%]
python\tests\v5\test_six_pole_atlas.py::test_svd_lowdin_orthogonalization PASSED [ 67%]
python\tests\v5\test_six_pole_atlas.py::test_goldilocks_void_finder_logic PASSED [ 69%]
python\tests\v5\test_six_pole_atlas.py::test_order0_and_order1_calibration_artifacts PASSED [ 70%]
python\tests\v5\test_six_pole_atlas.py::test_order2_and_order3_diachronic_artifacts PASSED [ 72%]
python\tests\v5\test_u6_guards.py::test_numeral_parser PASSED            [ 74%]
python\tests\v5\test_u6_guards.py::test_numeral_guard_identical_value_style_polish PASSED [ 75%]
python\tests\v5\test_u6_guards.py::test_numeral_guard_value_change_content_rewrite PASSED [ 77%]
python\tests\v5\test_u6_guards.py::test_negation_guard_polarity_flip_content_rewrite PASSED [ 79%]
python\tests\v5\test_u6_guards.py::test_negation_guard_double_negation_exemption PASSED [ 80%]
python\tests\v5\test_u6_guards.py::test_negation_guard_lexical_prefix_ignored PASSED [ 82%]
python\tests\v5\test_u6_guards.py::test_numeral_closed_class_idiom_exclusions PASSED [ 83%]
python\tests\v5\test_u6_guards.py::test_numeral_closed_class_man_ga_ichi_exclusion PASSED [ 85%]
python\tests\v5\test_u6_guards.py::test_numeral_adverbial_juuppun_exclusion PASSED [ 87%]
python\tests\v5\test_u6_guards.py::test_negation_adverb_mazu_exclusion PASSED [ 88%]
python\tests\v5\test_u6_guards.py::test_negation_conjecture_modality_exclusion PASSED [ 90%]
python\tests\v5\test_u6_guards.py::test_negation_adjective_stems_exclusion PASSED [ 91%]
python\tests\v5\test_u6_guards.py::test_numeral_placeholder_orthographic_protection PASSED [ 93%]
python\tests\v5\test_u6_guards.py::test_numeral_fraction_ratio_jubun_no_ichi PASSED [ 95%]
python\tests\v5\test_u6_guards.py::test_numeral_single_unit_kanji_hyaku PASSED [ 96%]
python\tests\v5\test_u6_guards.py::test_u6_test_vectors_fixture_pass PASSED [ 98%]
python\tests\v5\test_u6_guards.py::test_u6_profiles PASSED               [100%]

======================= 62 passed in 129.95s (0:02:09) ========================
```

---

## 6. 冷徹な自己内省・死角の開示（Ruthless Self-Critical Audit）

1. **誤爆（False Positive）および異質要因混入のリスク**:
   - **混在ペア (`mixed_window_pairs`, n=84) における高比率スパイク**: `mixed_window_pairs` の `per_pair_abs_ratio_max` 中央値は **1.6715**（`per_pair_frac_above_1sd` 平均 **0.1937**）であり、`non_mixed_both_single_window`（`abs_max` 中央値 0.2739、`frac_above_1sd` 平均 0.0325）や `non_mixed_both_multi_window`（同 0.3248 / 0.0269）と比べて突出している。これは純粋な文体変化だけでなく、「1 窓から 2 窓以上への離散遷移」そのものによる窓境界統計量のジャンプ（窓数差アーティファクト）が重畳している可能性が高い。
   - **旧仮名・新仮名混在ペア（34 件）と版ズレ結合**: `ginga_initial_vs_final`（34 件）は `same_orthography = False`（新字旧仮名 vs 新字新仮名）かつ `p6_p309` のように段落長が大きく異なる（932 字 vs 358 字、Dice=`0.517081`）統合・分割を伴う版異同を含む。また当初版 `build_real_revision_pairs_v1.py` の `ratios_obj.get("ratios", {})` キー誤読（Rust 側実キーは `per_dim`）により一時 `per_pair_frac_above_1sd` が全件 0.0 と算出されたバグを本セッション内で検出し修正した。
2. **取りこぼし（False Negative）**:
   - **U-6 Set 2 における Dice `< 0.25` 除外 9 件（うち Class A=2 件 `A_29`, `A_35`, Class B=3 件 `B_19`, `B_20`, `B_22`）**: 表層 2-gram 重なりが 0.25 未満となる大胆な言い換え・構造改稿（例: `B_19` Dice=`0.203642`）や短文で 2-gram が変動しやすい例（`A_29` Dice=`0.214286`）は、`unaligned` として F-14/F-15 比率評価の分母から落ちる。これはミスマッチ段落のノイズ遮断と引き換えの構造的取りこぼしである。
   - **KASANE8 の Python 代理指標と Rust 正本特徴量の違い**: Task 3 で H1 再検算を行った `scale_separated_level3.py` 上の `kasane11` 列は Python 代理次元（`c_comma`, ..., `c_no`, ..., `mean_cl_div_100`）であり、Rust 正本 `kasane11_v1`（`dc_showing`, ..., `ac_mean_velocity`, ..., `ac_mean_crest`）そのものの直交残差検定ではない点は依然残存する制約である。
3. **社会的先入観と物理的事実の乖離**:
   - 「長い段落の版異同ほどノイズ比率が安定して高い／低い」「青空文庫の版異同は大半が大きく書き換わっている」といった直観に反し、`band_L 内の非混在`（`n=111`）では `per_pair_abs_ratio_median` 中央値 **0.0023**、`per_pair_abs_ratio_max` 中央値 **0.0978** と極めて小さい差に留まる（版間の微小文字修正や漢字・仮名表記揺れが中心）。他方、比率を大きく押し上げる主要因は段落長そのもの（S/M/L）よりも、むしろ**文窓数の片側単窓・片側複窓の跨ぎ（`mixed_window_pairs`）**および**初期稿 vs 定稿の骨格改訂**であった。また `band_L` でも両側が単窓 (`[1, 1]`) に留まるケースが 130 件中 50 件（38.5%）存在し、JSON キー名 `band_L_both_multi_window` の字面と異なり実態は `n_evaluated=8` が 50 件、`n_evaluated=11` が 61 件に分かれている。

---

## 7. Stage B-12 成果物・SHA-256 台帳

- 実装・テスト変更:
  - `python/usir_py/v5/f14_hybrid_noise_floor.py`
  - `python/usir_py/v5/__init__.py`
  - `experiments/scale_separated_level3.py`
  - `experiments/gate3a_register_centering_eval.py`
  - `scripts/build_real_revision_pairs_v1.py`
  - `python/tests/v5/test_b12_dice_kasane8_and_v1_dataset.py`
- データセット・参照ノイズ床・評価 JSON:
  - `docs/handoff/v10/paragraph_noise_floor_f15_all_novel.json` (`5a6095659d4ef7defeab238f7ffb7aed3a980d6c8a9a83d0ca229f4c11a05910`)
  - `docs/handoff/v10/gate3a_register_centering_summary.json` (`6874a66eb78a8469c8c5878f90aa63a9aff5c34c6f01254d1d9664601cf2922b`)
  - `docs/handoff/v10/b12_kasane8_unification_h1_h2_recalc.json` (`0fdae7c4b66d06e26c1b4adc153fbdf1dbc33061f8a1c78c4116ec777770477f`)
  - `data/usir_v5_eval/usir_v5_b12_canonical_eval.json` (`50b1efb38b7d15195ea1da00cd049e51126f5100e90e03497fc7123d0b5c96fc`)
  - `data/usir_v5_eval/real_revision_pairs_v1.jsonl` (`62da75271f453bd3a8f161d916fdf7b7b87138b773b564b737f90aee85b6b8e4`)
  - `docs/handoff/v10/real_revision_pairs_v1_f15_harvest_summary.json` (`82e594f30741551bff62494b269758ae73e7f6e1aaae3900eb97f8183cd03456`)
- レポートおよび引継ぎ台帳:
  - `docs/handoff/2026-11-13_usir_v5_stage_b12_report.md`
  - `Claude-Sync/inbox/2026-11-13_usir_v5_stage_b12_report.md`
  - `F:\マイドライブ\Claude-Sync\inbox\2026-11-13_usir_v5_stage_b12_report.md`
  - `HANDOFF_usir.md`
