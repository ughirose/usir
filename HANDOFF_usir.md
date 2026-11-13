# USIR v5 実装引継ぎ台帳 (HANDOFF)
**最終更新日時**: 2026-11-13 15:40:00 JST  
**ブランチ**: `feat/fable-v5-shootout` (`C:\dev\usir`, 実装コミット `6e526995a12407cd0404034aa1238c70335f18f6`)  
**関連リポジトリ**: `C:\dev\nishiki`（Lane B 境界線厳守・変更ゼロ保証）

---

## -0.2 Stage B-12 実装・検証セッション（2026-11-13 実施）

本セッションでは、Stage B-11（`ae992e5`）を起点として、`Gate 3a = SHIFT_QUANTIFIED_NOT_PASSED`・`Gate 3b = GATE3B_FROZEN`・層B-1不使用・有料API不使用・B-13非設計の確定事項を厳守し、以下の 4 タスクを完遂した。

### 1. タスク1：Dice ゲートの Python 実装（正本統一）
- `python/usir_py/v5/f14_hybrid_noise_floor.py` に `ALIGN_MIN_SIMILARITY = 0.25`、`char_bigrams()`、`char_bigram_dice()`、および `evaluate_f14_hybrid_pair(..., apply_dice_gate=True)` を実装。非空白文字 2-gram Dice `< 0.25` のペアを `kind = "unaligned"`・`n_evaluated = None` とする Rust `harvest-diff` 仕様を正本統一した。
- `experiments/scale_separated_level3.py` の `evaluate_f14_hybrid_alignment_and_parity()` を再検証した結果：
  - **Set 2 (U-6 86ペア)**: `77` ペアがアライン一致、exact 9 ペア（`A_29`, `A_35`, `B_19`, `B_20`, `B_22`, `C_19`, `C_20`, `C_21`, `C_22`）が両側 `unaligned` として Rust `nishiki.exe harvest-diff` と **不一致 0 件** で一致。
  - **Set 1 (10,859ペア, `apply_dice_gate=False`)**: Band S=8,258 / M=1,952 / L=649、**不一致 0 件**。
  - **Set 3 (28ペア×3粒度 = 84評価, `apply_dice_gate=True`)**: `span_only`・`local_context`・`whole_scene` すべて `unaligned=0`（最小 Dice=`0.764706`）、Rust 実測との **不一致 0 件**。

### 2. タスク2：一次・補助警報閾値の明示記録
- `experiments/gate3a_register_centering_eval.py`・`docs/handoff/v10/gate3a_register_centering_summary.json`（SHA-256: `6874a66eb78a8469c8c5878f90aa63a9aff5c34c6f01254d1d9664601cf2922b`）・`data/usir_v5_eval/usir_v5_b12_canonical_eval.json`（SHA-256: `50b1efb38b7d15195ea1da00cd049e51126f5100e90e03497fc7123d0b5c96fc`）に以下を明示記録（閾値・Gate 判定値は変更なし）：
  - **一次警報 (`primary_alarm`)**: `mc_indep_p95` ($d = 6.9267$, $d^2 = 47.9796$)
  - **補助警報 (`auxiliary_alarm`)**: `mc_r4corr_p95` ($d = 7.6574$, $d^2 = 58.6355$)

### 3. タスク3：KASANE8 インデックス統一検証と H1／H2 再検算
- `scale_separated_level3.py::extract_kasane8_active`（旧 `[:8]` = `[0..7]`、`ac_mean_velocity` / Python代理 `c_no` を含み `ac_mean_crest` を欠落）と `KASANE8_ACTIVE_DIM_NAMES`（`KASANE8_ACTIVE_INDICES = (0, 1, 2, 3, 4, 5, 6, 10)`、`ac_mean_crest` / Python代理 `mean_cl_div_100` を保持）が**異なる集合**（`same_index_set = False`）であることを特定し、統一側へ修正（旧スライスは監査比較用 `extract_kasane8_active_legacy_slice` として保持）。
- 凍結済み成果物（B-8/B-9/B-10/B-11）は上書きせず、付属報告 `docs/handoff/v10/b12_kasane8_unification_h1_h2_recalc.json`（SHA-256: `0fdae7c4b66d06e26c1b4adc153fbdf1dbc33061f8a1c78c4116ec777770477f`）へ再検算結果を出力：
  - **H1（32D コア, n=42）**: `mean_cos_spec = 0.1083 -> 0.1083`, `mean_cos_other = -0.0128 -> -0.0128`, `vs_other W=654.0, p_two=0.010504, bonf=0.042017`（差分 0.0）; `mean_cos_rand = -0.0001 -> 0.002`, `vs_random W=552.0 -> 553.0, p_two=0.023777 -> 0.022926, bonf=0.095109 -> 0.091704`。
  - **H2（5D 連続コア, n=22）**: KASANE8 非依存であり凍結値とビット一致（`mean_cos_spec = 0.0849`, `vs_other W=214.0, p_two=0.003239, bonf=0.012957`, `vs_random W=153.0, p_two=0.406033, bonf=1.0`）。
  - **格付け結論**: `rating_conclusion_changed = False`（結論不変）。

### 4. タスク4：実推敲ペア拡充データセット v1 の構築と F-15 実測
- 青空文庫公開著作権なし版異同 39 組を走査し、単一改行を厳密な 1 段落境界（`"\n" not in text`）として Rust `harvest-diff` でアラインした結果、25 作品組から **415 件** の一次資料段落ペアを構築：`data/usir_v5_eval/real_revision_pairs_v1.jsonl`（SHA-256: `62da75271f453bd3a8f161d916fdf7b7b87138b773b564b737f90aee85b6b8e4`）。
  - **内訳**: Band S=`40`（両窓1限定で上限40件抽出; 元候補プール837件）、Band M=`245`（目安30件以上 達成）、Band L=`130`（目安20件以上 達成）、片側複窓混在=`84`（目安20件以上 達成）。正書法同一=`381` 件、歴史的/現代仮名遣い異同=`34` 件（宮沢賢治『銀河鉄道の夜』初期形 vs 新潮文庫）。
- 現行 F-15 ノイズ床 `docs/handoff/v10/paragraph_noise_floor_f15_all_novel.json`（SHA-256: `5a6095659d4ef7defeab238f7ffb7aed3a980d6c8a9a83d0ca229f4c11a05910`）で全 415 ペアを実行・Python/Rust パリティ検証（不一致 **0 件**）：`docs/handoff/v10/real_revision_pairs_v1_f15_harvest_summary.json`（SHA-256: `82e594f30741551bff62494b269758ae73e7f6e1aaae3900eb97f8183cd03456`）。格付け主張なし。

### 5. 単体・統合テストおよびレポート同期
- `pytest python/tests/v5/ -v`: **62 passed in 129.95s**（`test_b12_dice_kasane8_and_v1_dataset.py` 5 件含む全 62 件 PASS、凍結 B-8〜B-11 SHA-256 不変検証含む）。
- 総合レポート：`docs/handoff/2026-11-13_usir_v5_stage_b12_report.md`、`Claude-Sync/inbox/2026-11-13_usir_v5_stage_b12_report.md`、`F:\マイドライブ\Claude-Sync\inbox\2026-11-13_usir_v5_stage_b12_report.md`。

---

## 0. Stage B-8 改訂・総合検証セッション（2026-11-12 実施）

本セッションでは、Stage B-7 における残存課題を一気通貫で解決し、U6 閉鎖系除外同期、Gate 3a 現代散文分布シフト実測、改稿ペア拡充（86件）・U6 ROC 較正・二階層 Level 3b 残差検定、および Kamae $\eta^2=1.00$ の構造解明を完遂した。

### 1. タスク1：U6「万が一」閉鎖系除外同期と 38 件テストベクトル確定
- **数詞除外同期**:
  - `python/usir_py/v5/u6_guards.py` に「万が一」「万一」を完全除外設定（`extract_numerals("万が一の事態") == []`）。
  - `extract_numbers = extract_numerals` のエイリアス互換性を保証。
- **38件テストベクトル確定**:
  - `docs/handoff/v10/u6_guards_test_vectors.json` を 38 件に更新（SHA-256: `4AC04322811EB30776260F8EC2DEA378FF962A5C44180823218973DBAA3F3EF7`）。
  - `F:\マイドライブ\Claude-Sync\inbox\u6_guards_test_vectors.json` に同期完了。
  - テスト `pytest python/tests/v5/`: **46 件全 PASS**。

### 2. タスク2：Gate 3a 現代散文マルチレジスター分布シフトの実測
- **ベンチマークデータ生成**:
  - `scripts/build_tier_c_modern_benchmark.py` を修正（全サンプル 408〜590 字に調整）。
  - `data/usir_v5_eval/tier_c_modern_benchmark.jsonl`（4 レジスター × 8 サンプル = 32 件、クリーンルーム規格・生テキスト完全非保持）を生成（SHA-256: `45F5F5D45895F591D075F432DE5AA16F67CA511E0FCE1486741301084E0642B1`）。
- **分布シフト解析結果**:
  - スクリプト: `experiments/gate3a_modern_shift_analysis.py`
  - 成果物: `data/usir_v5_eval/usir_v5_gate3a_modern_shift_report.json`（SHA-256: `4FD446ADD2C30B964F36A82D735F01BF7B8D4CC3B20660284A80BD77A6583E57`）。
  - PPM bits/token: R1=4.0672 ($d=-3.19$), R2=4.1087 ($d=-2.87$), R3=4.0636 ($d=-3.07$), R4=4.7203。
  - 閉クラス被覆率: R1=58.88%, R2=56.74%, R3=55.64%, R4=63.22%。
  - ortho_v2 31D 列中心化 $\eta^2$: Raw=0.4554 $\to$ Standardized=0.4429。
  - **死角・限界の冷徹な開示**: 初期ヒューリスティック「被覆率 85% 以上」は過大であり、自然日本語散文は 55%〜63% が実測値であるため、未較正初期値であることを明記（本番推奨閾値: 52.0%）。

### 3. タスク3：改稿ペア拡充・U6 ROC 較正・二階層 Level 3b 残差検定
- **一次テキスト改稿 86 ペア構築**:
  - `data/usir_v5_eval/u6_calibration_and_level3_dataset.jsonl`（SHA-256: `93C689E44D68DFBB711DDA62F5F4A99E3551FC82789549DE4DFCC0FCE1C28A4D`）。
  - Class A (ミクロ推敲 42 件), Class B (マクロ改稿 22 件: D10 16件 + 賢治異稿等 6件), Class C (内容変更 22 件)。LLM 合成ペア完全不使用。
- **U6 Guards ROC 較正**:
  - 層化 2 分割（43/43 件）グリッドサーチ。
  - Scope 1 (ミクロ推敲境界): 初期値 (0.55, 0.25) の Recall=0.4286 (低再現率) を特定。最適値 (min_jaccard=0.35, max_char_change=0.45) で Precision=0.9286, Recall=0.6190, **F1=0.7429**, FPR=0.0909（楽観バイアス -0.0586）。
  - Scope 2 (全改稿境界): マクロ改稿を含めると Recall=0.4062 に低下。表層 Jaccard 単独でのマクロ改稿検出限界を冷徹に開示。
- **二階層 Level 3b 残差検定（H1 & H2）**:
  - スクリプト: `experiments/scale_separated_level3.py`
  - 成果物: `data/usir_v5_eval/usir_v5_stage_b8_scale_separation_and_gate3a.json`（SHA-256: `B6441FE5B8EF1787BC89847BD11F64661D6A07D3A9DCAF770F008DE9CE915DC9`）。
  - H1 (Class A, n=42): ortho28 (vs random $p=0.6380$, vs other $p=0.2280$), ortho28+kasane11 (vs random $p=0.3330$, vs other $p=0.2280$)。準平行 0 件。
  - H2 (Class B, n=22): 6pole (vs random $p=0.4826$, vs other $p=0.6250$), 6pole+HW-25 (vs random $p=0.3369$, vs other $p=0.6250$)。準平行 0 件。
  - **科学的客観結論**: コーパス規範回帰方向を除去した純残差空間ではランダム摂動や他者方向と有意差を持たず（$p > 0.20$）、局所残差空間での個別作家引き込み仮説は非支持。
- **Kamae $\eta^2=1.00$ の構造解明と撤回**:
  - D10 の 4 原作において、フック判定が 0 個（Melos, Sangetsu）または 1 個（Ginga, Prince）の離散値に固定化され、トランスレーター定型句がフック開閉に影響を与えなかったことによる $SS_{\text{trans}}=0$ の「構成的アーティファクト」であることをコードから解明。「完全意味保存」としての成果引用を正式撤回。

### 4. 総合レポート同期
- 総合報告書: `docs/handoff/2026-11-12_usir_v5_stage_b8_revised_report.md` 作成。
- Google Drive 同期: `F:\マイドライブ\Claude-Sync\inbox\2026-11-12_usir_v5_stage_b8_revised_report.md` に同期完了。

---

## 0.1 Stage B-7 実装・実測セッション（2026-11-12 実施）

本セッションでは、Orion 指摘1〜3への対応（同義反復の撤回、空間再編、Level 3a/3b 分離、本人特異残差検定の実装実測）および U6 ガードの表記変換保護・端ケース拡充を一気通貫で完遂した。

### 1. タスク1：Orion 指摘1〜3への対応 — 同義反復撤回・本人特異残差検定
- **同義反復の完全撤回**:
  - `experiments/usir_v5_edited_span_level3.py` における「vs 劣化（$-cos_{\text{author}}$）」の Wilcoxon 検定は差分 $2 \cdot cos_{\text{author}}$ の検定（$cos_{\text{author}} > 0$ の言い換え）であるため独立証拠から撤回。有効な方向性対照を「vs 等方ランダム摂動」のみに限定。
- **空間再編**:
  - 前回 vs ランダムで非有意だった `ppm3d`（$p = 0.169$）を主空間から探索空間へ格下げ。
  - 主空間: `ortho28`（28D, 前回 $p = 1.22 \times 10^{-4}$）、`6pole`（6D, 前回 $p = 4.73 \times 10^{-3}$）。
  - 探索空間: `ppm3d`（3D）、`A55`（55D, Canonical Atlas Z24 + ortho_v1 31D）。
- **Level 3a / 3b の概念的分離**:
  - 微細推敲28件における規範回帰（vs ランダム有意）は「Level 3a：レジスター整合性チェック（計器健全性確認）」と位置づけ。
  - 本人特異性「Level 3b」の合否判定は、人手由来マクロ改稿ペアが揃うまで保留（DEFERRED）。LLM 疑似ペアは循環検証となるため一切使用禁止。
- **本人特異残差検定（Residual Orthogonalization Test）の実装と実測**:
  - 実験スクリプト: `experiments/usir_v5_residual_specificity_level3.py` 新設。
  - 推敲前座標 $z_{\text{before}}$ からコーパス全体平均 $z^*_{\text{corpus}}$ への単位ベクトル $u_{\text{corp}}$ を直交化で除去した本人残差ベクトル $u_{\text{spec}}(\text{author})$ を算出。
  - 事前固定基準 $\rho_{\perp} < 0.10$（$\theta < 5.74^\circ$）による準平行フラグ管理。
  - **実測結果（`span_only` 28ペア）**:
    - **ortho28 [主空間]**: 直交残差空間でも本人方向へ有意！
      - 全28件: $cos_{\text{spec}} = +0.1485$（vs ランダム $W=304.0, \mathbf{p=0.0102}$、vs 他作家9名平均 $W=340.0, \mathbf{p=5.86 \times 10^{-4}}$）。
      - 非平行27件（準平行1件除外）: $cos_{\text{spec}} = +0.1454$（vs ランダム $W=278.0, \mathbf{p=0.0159}$、vs 他作家 $W=314.0, \mathbf{p=9.29 \times 10^{-4}}$）。
    - **6pole [主空間・死角開示]**:
      - 準平行ペアが 7/28（25.0%）発生。
      - 直交残差空間では本人方向へ非有意（全28件: $cos_{\text{spec}} = -0.0865$, vs ランダム $p=0.832$, vs 他作家 $p=0.865$；非平行21件: $cos_{\text{spec}} = +0.0353$, vs ランダム $p=0.500$, vs 他作家 $p=0.603$）。
      - 【特定事実】6pole が微細推敲28件で捉えていたのは「作家個人の文体偏向」ではなく「文語散文としての規範的整流化（コーパス平均回帰）」であった。
    - **A55 [探索空間]**: $cos_{\text{spec}} = +0.0946$（vs ランダム $p=0.0657$、vs 他作家 $W=308.0, \mathbf{p=7.82 \times 10^{-3}}$ 有意）。
    - **ppm3d [探索空間]**: $cos_{\text{spec}} = -0.0899$（vs ランダム $p=0.887$、vs 他作家 $p=0.788$ 非有意）。
- **D10 多訳ペア（4 Sources x 4 Translators）での予備観測**:
  - TR_STANDARD $\to$ 他3文体の改稿12ペアにおいて、残差空間でも全空間が文体特異方向へ有意（ortho28 $p=2.44 \times 10^{-4}$, 6pole $cos_{\text{spec}}=+0.9257, p=2.44 \times 10^{-4}$, A55 $p=2.44 \times 10^{-4}$, ppm3d $p=0.0261$）。マクロ文体改稿での特異性検出力を実証。
  - 成果物: `docs/handoff/v10/usir_v5_residual_specificity_level3.json`

### 2. タスク2：U6 ガード表記変換保護と 32 件テストベクトル
- **表記変換保護の実装**:
  - `python/usir_py/v5/u6_guards.py` に漢数字・位取り記号・アラビア数字表記ゆれ保護、否定助動詞・二重否定接辞境界保護を実装。
- **32件テストベクトル確定と未較正初期値の明記**:
  - `docs/handoff/v10/u6_guards_test_vectors.json`: 32件全件で論理確定（`pass_count: 32`）。
  - **重要注記**: 表層乖離度閾値（漢字 Jaccard 0.55 / 文字数変化率 0.25）はいまだデータ駆動で最適化されていない初期値（`calibrated: false`）であることを明記。
- **検証テスト**: `python/tests/v5/test_u6_guards.py`（15件全パス）。

### 3. テスト・レポート・同期
- 単体テスト `pytest python/tests/v5/`: **45 件全パス（100% PASS）**。
- 総合レポート: `docs/handoff/2026-11-12_usir_v5_stage_b7_report.md` 作成。
- Google Drive 同期: `F:\マイドライブ\Claude-Sync\inbox\2026-11-12_usir_v5_stage_b7_report.md` にコピー完了。

---

## 0.1 Stage B-6 実装・実測セッション（2026-11-12 実施）

本セッションでは、v4 決定事項および Stage B-5 の課題に基づき、以下のタスク1〜3を一気通貫で完遂した。

### 1. タスク1：Level 3「編集スパン限定差分（Edited-Span Micro-Diff）」と4ベースライン比較（v4 D4 準拠）
- **実験スクリプト**: `experiments/usir_v5_edited_span_level3.py` 新設
- **段落ノイズ床 $\sigma_{\text{para}}$ 実測（3,829隣接段落対）**:
  - 6pole (6D): $\sigma_{\text{para}} = 1.4389$ ($2\sigma = 2.8778$)
  - ortho28 (28D): $\sigma_{\text{para}} = 4.0143$ ($2\sigma = 8.0286$)
  - ppm3d (3D): $\sigma_{\text{para}} = 0.5332$ ($2\sigma = 1.0664$)
  - 成果物: `docs/handoff/v10/paragraph_noise_floor.json`
- **編集スパン限定による推敲シグナル増幅**:
  - `span_only`: $\|\Delta z\|$ が場面全体（`whole_scene`）比で **約 4.0 〜 4.9 倍に大幅増幅**（ortho28: 0.5825 $\to$ 2.8864, 6pole: 0.0464 $\to$ 0.2121）。
  - ただし段落ノイズ床が大きいため、$2\sigma_{\text{para}}$ ゲート超過率は 0/28（PPM で 1/28 3.6%）にとどまり、絶対ノルム単独では自然な段落揺らぎに埋没することを開示。
- **v4 D4 準拠 4比較対象検定結果（`span_only`）**:
  - **ortho28**: 本人重心余弦 $+0.2651$
    - vs 逆方向劣化: $W=400.0, \mathbf{p=5.22 \times 10^{-8} < 0.01}$（極めて有意）
    - vs 等方ランダム: $W=355.0, \mathbf{p=1.22 \times 10^{-4} < 0.01}$（極めて有意）
    - vs コーパス平均: $W=239.0, \mathbf{p=0.2124}$（**有意差なし**）
  - **6pole**: 本人重心余弦 $+0.3416$
    - vs 逆方向劣化: $W=305.0, \mathbf{p=2.66 \times 10^{-3} < 0.01}$（有意）
    - vs 等方ランダム: $W=297.0, \mathbf{p=4.73 \times 10^{-3} < 0.01}$（有意）
    - vs コーパス平均: $W=246.0, \mathbf{p=0.0854}$（**有意差なし**）
  - **ppm3d**: 本人重心余弦 $+0.3774$
    - vs 逆方向劣化: $W=328.0, \mathbf{p=1.72 \times 10^{-3} < 0.01}$（有意）
    - vs 等方ランダム: $W=246.0, p=0.1693$
    - vs コーパス平均: $W=204.0, \mathbf{p=0.4955}$（**有意差なし**）
  - **核心的所見**: 推敲差分はランダムノイズ・劣化方向から極めて有意に識別できるが、**「目標作家重心方向」と「コーパス全体平均方向」の間には有意差がない**。推敲は特定作家への個別接近ではなく文語規範コーパス平均への回帰と統計的に区別不能であることを客観開示。
  - 成果物: `docs/handoff/v10/usir_v5_edited_span_level3_result.json`

### 2. タスク2：Gate 3a 現代文・Web小説前処理器と Tier C クリーンルーム抽出器
- **実装モジュール**: `python/usir_py/v5/modern_text_normalizer.py`
  - 約物正規化（三点リーダー `……`、ダッシュ `――`、感嘆符・疑問符 `！？`）
  - Web小説ルビ非破壊除去（`|親文字《ルビ》`、`親文字《ルビ》`、`親文字(ルビ)`）
  - 著作権法第30条の4・D3層C規約準拠: 固有名詞・内容語を `■` にマスクし、非可逆特徴量（`ortho_v2` 31D、閉クラス PPM bits/token、閉クラス被覆率）のみを出力して原文をメモリ外に残さない `extract_tier_c_irreversible_profile`。
  - 会話文比率 50〜80% の現代レイアウトおよび空文入力でゼロ除算・NaN を完全防止。
- **検証テスト**: `python/tests/v5/test_modern_text_normalizer.py`（5件全パス）。

### 3. タスク3：`u6_guards.py` 閉鎖系4除外同期と単体 import 対応
- **トップレベル遅延インポート化（PEP 562）**:
  - `python/usir_py/__init__.py` および `python/usir_py/v5/__init__.py` を遅延インポート化。`scipy` / `janome` のない minimal 環境で `import usir_py.v5.u6_guards` が標準ライブラリのみで即座に完結。
- **閉鎖系 4パターンの誤爆除外同期**:
  - 数詞除外：`一通り`、`一旦`、`一応`、`一緒`、`一層`、`一向`、`一斉`
  - 副詞的 `十分` 除外：直後に助数詞（間・後・前 等）を伴わない場合を非数詞として除外（`十分に休んだ` $\to$ `[]`、`十分前に出発` $\to$ `[10]`）
  - 否定副詞除外：`先ず` / `まず`
  - 否定モダリティ・形容詞除外：`違いない / ちがいない / かもしれない / かも知れない`、および形容詞語幹（`少 / 切 / 危 / 汚 / 儚`）＋`ない`
- **固定フィクスチャ**: `docs/handoff/v10/u6_guards_test_vectors.json`（30文対、100% PASS）。
- **検証テスト**: `python/tests/v5/test_u6_guards.py`（11件全パス）。

### 4. テスト・レポート・同期
- 単体テスト `pytest python/tests/v5/`: **41 件全パス（100% PASS）**。
- 総合レポート: `docs/handoff/2026-11-12_usir_v5_stage_b6_report.md` 作成。
- Google Drive 同期: `F:\マイドライブ\Claude-Sync\inbox\2026-11-12_usir_v5_stage_b6_report.md` にコピー完了。

---

## 0.1 Stage B-5 改訂実装・実測セッション（2026-11-12 実施）

本セッションでは、Orion策定・Fable査読統合の正本仕様に基づき、以下のタスク1〜4を一気通貫で完遂した。

### 1. タスク1：Python 側 ortho_v2 移植・突合・M1 感受性監査・fisher_v2 出力
- **全11,723窓完全突合**:
  - `python/usir_py/orthographic_prior.py` に `ortho_matrix_ext_v2` / `ortho_matrix_narr_v2` / `content_char_count` 等を追加（既存 v1 関数・`fisher_v1.json` は完全凍結）。
  - 青空文庫106作品・全11,723窓において、Rust 側 `nishiki-hardcore ortho` との最大絶対差が **$8.8818 \times 10^{-16} < 10^{-12}$** で完全一致（倍精度マシンイプシロン精度）。
  - 成果物: `docs/handoff/v10/ortho_v2_parity_result.json`
- **z0〜z23（24次元）の M1 感受性監査**:
  - 全11,723窓で読点削除（M1）時の変動量を測定。**24次元すべてが読点削除で変動（不変次元 0 次元）**。
  - 特に z16（$\max \Delta = 6.61$）、z04（$\max \Delta = 0.81$）が節分割と分散乗算により顕著に変動。
- **完全不変サブセット `v2_28` の評価**:
  - LOWO (46): 28/46 (60.87%, Wilson 95%: [0.465, 0.736])
  - Held-out (60): 45/60 (75.00%, Wilson 95%: [0.628, 0.842])
  - **M1 問い合わせ予測ラベル反転数: 0/46 (0.0%), Held-out 0/60 (0.0%)**。読点全削除攻撃に対する完全耐性を実証。
  - 正本 A55（LOWO 34/46, Held 50/60, M1反転 27/46）、Z24+v2_28（52D：LOWO 34/46, Held 48/60, M1反転 30/46）、Z24単体（24D：LOWO 26/46, Held 40/60, M1反転 36/46）との並行比較表を確定。
- **実験用アーティファクト出力**:
  - `artifacts/style_atlas/fisher_v2.json` 出力完了（正本 `fisher_v1.json` は無変更）。
  - 成果物: `docs/handoff/v10/z24_m1_sensitivity_and_fisher_v2.json`

### 2. タスク2：Level 3「固定境界ペア差分」と U6 ガード
- **固定境界ペア差分 $\Delta z_{\text{paired}}$ 実測**:
  - `experiments/usir_v5_paired_micro_diff_level3.py` を新設。場面境界ジッターノイズ $\sigma_{\text{jitter}}$ を数学的にゼロに相殺。
  - Tier A 既知 STYLE_POLISH 28件の実測において、同一場面内ブートストラップ揺らぎ $\sigma_{\text{paired}} = 1.0963$ に対し、$2\sigma_{\text{paired}}$ ゲート超過率は **0/28 件（0.0%）**。微小推敲差分がサンプリングノイズ床に埋没することを客観開示。
  - 目標作家プロトタイプ $z^*$ への方向余弦:
    - 実推敲差分: 平均 $+0.2913$（中央値 $+0.3450$）
    - ランダム等方摂動: 平均 $+0.0999$（中央値 $+0.0473$）
    - 逆方向劣化摂動: 平均 $-0.2913$（中央値 $-0.3450$）
  - Wilcoxon 符号順位検定:
    - 実推敲 vs 逆方向劣化: 統計量 297.0, **$p = 0.00473 < 0.01$**（極めて有意に劣化方向を上回る）。
    - 実推敲 vs ランダム摂動: 統計量 253.0, **$p = 0.06207$**（マージナル、有意水準 0.05 にはわずかに届かず）。
- **U6 分離器ガード実装と誤爆実測**:
  - `python/usir_py/v5/u6_guards.py` に数詞ガード（値正規化・表記ゆれ対応）・否定ガード（助動詞限定・二重否定除外）を実装。
  - 単体テスト `python/tests/v5/test_u6_guards.py`（6件全パス）。
  - 既知 28 件の STYLE_POLISH ペアでの誤爆（誤却下）は **0 件（0.0%）**。
  - 成果物: `docs/handoff/v10/paired_micro_diff_level3_result.json`

### 3. タスク3：HW-22／24／25 Python 試作と青空文庫 106 作品実測
- **プロトタイプ実装**: `python/usir_py/v5/narrative_pacing_and_metaphor.py`（単体テスト全パス）。全定数は未較正初期値（$\kappa=1.35, \gamma=1.5, \epsilon_0=0.1$）。
- **青空文庫全 11,723 窓での ANOVA 分析と信号強度判定**:
  - **HW-22（比喩跳躍距離 $\mathcal{D}_{\text{leap}}$）**: ANOVA $F=7.52, p=4.75 \times 10^{-11}$, 効果量 $\eta^2 = 0.0057$（**0.57%**） $\to$ **WEAK_SIGNAL**。比喩の局所スパイク性により窓内ノイズに埋没。**Rust 移植見送り**。
  - **HW-24（章末引き強度 $\mathcal{C}_{\text{cliff}}$）**: ANOVA $F=13.94, p=1.39 \times 10^{-22}$, 効果量 $\eta^2 = 0.0106$（**1.06%**） $\to$ **WEAK_SIGNAL**。機械的スライス窓末尾では窓内ノイズに埋没。**Rust 移植見送り**。
  - **HW-25（設定開示結合比 $\Gamma_{\text{UID}}$）**: ANOVA $F=150.44, p=2.88 \times 10^{-270}$, 効果量 $\eta^2 = 0.1036$（**10.36%**） $\to$ **MODERATE_SIGNAL**。漢字複合語負荷 vs 会話・動詞運動量の比率は文体リズムとして定常的な作家間差を保持。
  - 成果物: `docs/handoff/v10/narrative_pacing_prototype_results.json`

### 4. タスク4：Gate 3a（層C現代小説）への布石棚卸し
- 著作権法第30条の4（情報解析目的利用）、妻の原稿（層B-1）の完全物理隔離、クリーンルーム規約（Entity-Blanking / 非可逆特徴量化）を整理。
- 既存パイプラインの流用可能性（`ortho_v2`, `extractor_24d`, `PPM` 分布シフト検証は即流用可、ルビ・レイアウト前処理器は要改修）と不足分リストを確定。

### 5. テスト・レポート・同期
- 単体テスト `pytest python/tests/v5/`: **31 件全パス（100% PASS）**。
- 総合レポート: `docs/handoff/2026-11-12_usir_v5_stage_b5_revised_report.md` 作成。
- Google Drive 同期: `F:\マイドライブ\Claude-Sync\inbox\2026-11-12_usir_v5_stage_b5_revised_report.md` にコピー完了。

---

---

## 1. タスク概要・実施内容
本セッションでは、正本 v4（`Argon_Discussion_Pack_v4_Decisions_20261111.md`）および親計画書（`implementation_plan.md`）に基づき、以下の 4 大タスクを一気通貫で完遂した。

1. **ステップ1：`aozora_scene_splitter.py` および `python/usir_py/v5/six_pole_atlas.py` の改修**
   - **M1 変形不変性保証**:
     - 深層4極の文字数分母および計算から読点 `、`・`，` を完全に除外。`past_ratio` を句末基準化。
     - `Nagare` 位相項から `comma_ratio` を完全排除し、読点非依存の `neg_ratio` / `kanji_ratio` に変更。
     - 単語境界なしでの形態素連結誤爆（「わあ、われ」$\to$「あわれ」）を完全遮断するため、すべての出現頻度計測を事前 `strip_commas` 統制テキスト上で実施。
   - **M2 固有名詞不変性保証**:
     - 特定キャラクター名（メロス、ジョバンニ等）のハードコードおよび `abs(hash(name))` による名前ハッシュ依存を撤廃。
     - 一般化された主語検出正規表現とスロット直交基底＋閉クラス役割語・一人称/二人称・敬語・場面極性による決定論的 `axes_q` 算出ロジックへ刷新。
   - **D5 残差仕様の決定論的組み込み**:
     - `Awase` に発話極性と地の文極性の乖離候補箇所 `extract_polarity_divergence_spans` 出力機能を追加。
     - `Kamae` に作者宣言伏線台帳 `declared_hooks` 照合口を追加。
   - **アトラスモジュール化**:
     - `NarrativeRGNormalizer`, `normalized_hsic`, `svd_lowdin_orthogonalize`, `GoldilocksVoidFinder` を `python/usir_py/v5/six_pole_atlas.py` としてモジュール化。
     - 場面アノテーション列から直接 Rust `nishiki profile` と厳密整合する作品プロファイルを集計する `aggregate_work_profile_from_scenes` を実装。

2. **ステップ2：タスク1 & タスク2（`experiments/usir_v5_instrument_calibration.py`）の実装と全106作品実測**
   - **M1/M2/M3 摂動テスト全件合格**:
     - M1（読点全削除）: 深層4極 $|\Delta z| = 0.00 \times 10^{-6} < 10^{-6}$、Kasane 表面シフト $\Delta = 0.0960 > 0$。
     - M2（人名置換）: 全6極 $|\Delta z| = 0.00 \times 10^{-6} < 10^{-6}$。
     - M3（場面順序反転）: Kamae シフト True、Sadame 相互作用誤差 $1.11 \times 10^{-16}$、Awase 誤差 $0.00 \times 10^{-6}$。
   - **ノイズ床 $\sigma_{\text{noise}}(\Delta z)$ 確定と 2軸監査表**:
     - 場面分割閾値 $\pm 20\%$ 変動時の座標変化分布より各極の測定ノイズ床を確定（Kamae 0.1573, Sadame 0.0449, Nagare 0.0493, Awase 0.0644, Kasane 0.0820, Sasae 0.0650）。
     - Sadame（$R_{xx}=0.664$, nHSIC$=0.0885$）を**真の直交シグナル**として確定。
   - **Level 1 Wilson 95% CI & McNemar 検定**:
     - LOWO（46作品）: B_cal_center 39/46 [0.718, 0.924] vs 統合 E2_equal 41/46 [0.770, 0.953]、McNemar $p = 1.0000$（有意差なし、「深層極が識別を押し上げた」という主張は統計的に支持されないことを確認）。
   - **U6 内容 vs 文体分離 & Level 3 ノイズ床ゲート**:
     - U6 分離器: Precision 1.0000, Recall 1.0000, F1 1.0000。
     - Level 3 ノイズ床ゲート $\|\Delta z\| > 2\sigma_{\text{noise}}$: 文体推敲ペア 28件中 通過 0/28件（0%）。微細な推敲がノイズ床に物理的に埋没することを解明。
   - 成果物: `docs/handoff/v10/usir_v5_instrument_calibration.json`

3. **ステップ3：タスク3（`experiments/usir_v5_gate3a_and_diachronic.py`）の実装と実測**
   - **順序 2（D2 Gate 3a: PPM bits/token 分布シフト実測）**:
     - 明治期訓練（860,731 トークン）の閉クラス Order-3 PPM に対し、明治（2.8357）、大正（3.1096, $d=1.4956$）、昭和（3.1476, $d=1.6945$）、現代口語文（2.9209, $d=0.4884$, KS $p=0.5629$）を実測。
     - 現代口語文のシフト量が平易語彙により小さく、被覆率 59.2% を維持したため Gate 3a を通過。
   - **順序 3（D10: 多翻訳者・多文体交差実験 ANOVA）**:
     - 4原作 $\times$ 4訳者文体（計16セル）の二元配置分散分析。
     - Kamae（$\eta^2_{\text{src}} = 1.0000$）、Nagare（$\eta^2_{\text{src}} = 0.6379$）が原作支配、Kasane（$\eta^2_{\text{trans}} = 0.9971$）が訳者支配。
     - 事前登録仮説順位との Spearman 相関 **$\rho = 0.8000$**（$p=0.1041$）で支持。
   - **順序 3（§6: 作家経年軌道と教師なし変化点検出）**:
     - 漱石（1906 vs 1910）、芥川（1924 vs 1923、誤差1年）、太宰（1942 vs 1945）、鴎外（1909 vs 1912）の4文豪において、平均年数誤差 **2.75年** で文学史上の転換期を教師なしで再現。
   - 成果物: `docs/handoff/v10/usir_v5_gate3a_and_diachronic.json`

4. **ステップ4：単体テスト・総合レポート・Google Drive 同期・Git コミット**
   - 単体テスト `pytest python/tests/v5/ -v`: 全18件 100% PASSED。
   - 総合実測レポート: `experiments/2026-11-11_usir_v5_d0_calibration_and_diachronic_report.md` 作成。
   - Google Drive 同期: `F:\マイドライブ\Claude-Sync\inbox\2026-11-11_usir_v5_d0_calibration_and_diachronic_report.md` にコピー完了。

---

## 2. 定量的確定データ

### 2.1 ノイズ床確定値と Level 3 ゲート（v4 §3 更新値）
| 極名 | 測定ノイズ床 $\sigma_{\text{noise}}$ | Level 3 ゲート閾値 ($2\sigma_{\text{noise}}$) |
|---|---|---|
| **Kamae** | 0.1573 | 0.3147 |
| **Sadame** | 0.0449 | 0.0898 |
| **Nagare** | 0.0493 | 0.0986 |
| **Awase** | 0.0644 | 0.1288 |
| **Kasane** | 0.0820 | 0.1640 |
| **Sasae** | 0.0650 | 0.1300 |

### 2.2 Gate 3a PPM bits/token 分布シフト実測
| 時代 / サンプル群 | bits/token (平均 $\pm$ SD) | 閉クラス被覆率 | Cohen's $d$ vs 明治 | KS 検定 $p$ 値 |
|---|---|---|---|---|
| **明治期 (In-domain)** | 2.8357 $\pm$ 0.183 | 61.9% | 0.0000 | 1.0000 |
| **大正期** | 3.1096 $\pm$ 0.183 | 63.0% | +1.4956 | < 0.0001 |
| **昭和期** | 3.1476 $\pm$ 0.185 | 62.9% | +1.6945 | < 0.0001 |
| **現代口語文** | 2.9209 $\pm$ 0.165 | 59.2% | +0.4884 | 0.5629 |

### 2.3 D10 多訳交差実験 ANOVA
- Kamae: $\eta^2_{\text{src}} = 1.0000$, $\eta^2_{\text{trans}} = 0.0000$
- Nagare: $\eta^2_{\text{src}} = 0.6379$, $\eta^2_{\text{trans}} = 0.1349$
- Awase: $\eta^2_{\text{src}} = 0.1549$, $\eta^2_{\text{trans}} = 0.3803$
- Sadame: $\eta^2_{\text{src}} = 0.0620$, $\eta^2_{\text{trans}} = 0.6333$
- Kasane: $\eta^2_{\text{src}} = 0.0025$, $\eta^2_{\text{trans}} = 0.9971$
- Spearman $\rho = 0.8000$ ($p = 0.1041$)

---

## 4. Stage B-9 完了記録（2026-11-12）

### 4.1 実施内容と定量確定値
1. **タスク1：二階層レンズ Level 3b 残差特異性検証コードバグ修正・実測完了**
   - `experiments/scale_separated_level3.py` の Gram-Schmidt 直交化引数バグ（L363-364）および Class B 目標／他スタイル重心計算バグ（L463-508）を修正。
   - **H1 (Micro-scale, Class A ミクロ推敲, $n=42$)**:
     * 全結合空間（39D: ortho28 + kasane11）: $\text{mean } \cos_{\text{spec}} = +0.1085$, vs ランダム $p=0.0175$, vs 他作家 $p=0.0051$（有意）。
     * 有効連続サブセット（32D: ortho24_cont + kasane8_active）: $\text{mean } \cos_{\text{spec}} = +0.1083$, vs ランダム Bonferroni $p=0.0476$, vs 他作家 Bonferroni $p=0.0210$（両対照で有意成立）。
   - **H2 (Macro-scale, Class B マクロ改稿, $n=22$)**:
     * 全結合空間（7D: 6pole + HW-25）: $\text{mean } \cos_{\text{spec}} = +0.1365$, vs ランダム $p=0.0164$, vs 他作家 $p=0.0018$（有意）。
     * 有効連続サブセット（5D: 4pole_cont + HW-25）: $\text{mean } \cos_{\text{spec}} = +0.0849$, vs 他作家 Bonferroni $p=0.0065$（対他者で有意成立）。
   - 成果物: `data/usir_v5_eval/usir_v5_stage_b8_scale_separation_and_gate3a.json`

2. **タスク2：U6 ガード暫定較正プロファイル確定とテストベクタ固定**
   - `python/usir_py/v5/u6_guards.py` に `U6_PROFILES`（`uncalibrated_v1 = (0.55, 0.25)`, `provisional_micro_v2 = (0.25, 0.85)`）を追加。
   - `docs/handoff/v10/u6_guards_calibration_metadata.json` を更新。
   - `python/tests/v5/test_u6_guards.py` に 17 件のテストを追加（全件 PASS）。
   - 既存38件テストベクタ SHA-256（`4AC04322811EB30776260F8EC2DEA378FF962A5C44180823218973DBAA3F3EF7`）との整合性を完全維持。

3. **タスク3：Gate 3a レジスター別中心化・時代オフセット補正の比較実測**
   - ベンチマークを32件から64件（4レジスター $\times$ 16件、D3クリーンルーム準拠不可逆プロファイル）へと拡充（`scripts/build_tier_c_modern_benchmark_expanded.py`, `data/usir_v5_eval/tier_c_modern_benchmark_expanded.jsonl`）。
   - `experiments/gate3a_register_centering_eval.py` により、**Baseline A（R4対角マハラノビス基準）** と **Baseline B（青空106作品アトラス基準）** の双方で4スキームを LOO 比較実測し、タスク1〜3統合 JSON `data/usir_v5_eval/usir_v5_b9_register_centering_and_level3.json` を出力。
   - **Scheme 0 (Raw 31D)**: Baseline A で現代散文全体 3-sigma 62.50%(n=32) / 58.33%(n=64)（R2=75〜87.5%, R3=100%）、Baseline B で 95.83%（R1=87.5%, R2=100%, R3=100%）。
   - **Scheme 1 (Continuous Core 24D)**: `ortho24_cont` および `ortho24_no_sparse_finals` でも現代散文全体 3-sigma 64.58%〜100.0% が残存し、スパース次元除外のみでは不十分。
   - **Scheme 2 (LOO 時代オフセット補正)**: R1 現代小説は Baseline A（31D/24D）で $3\sigma$ **0.0%**・$\chi^2_{0.95}$ **0.0%**、Baseline B（24D, $n=64$）で $3\sigma$ **0.0%**（$0/16$件）へ収束。一方、現代散文プール（R1〜R3）の時代補正では Baseline A $\chi^2_{0.95}$ で **35.42%**・Baseline B $3\sigma$ で **72.92%** の外れ値率が残存し、ジャンル乖離と時代ドリフトが直交することを確認。
   - **Scheme 3 (Within-Register LOO 中心化)**: レジスター間分散比 $\eta^2_{\text{register}}$ を **100.0% 完全消去**（$0.4965 \to 0.0000$）、内部距離相関 Spearman $\rho = 1.0000$ を完全保持。レジスター内 LOO 外れ値率は、R4分散併用で **0.0% / 0.0%**、縮約 LOO 分散併用で **0.0% ($3\sigma$) / 4.17% ($\chi^2_{0.95}$)** に収束（非正則化 LOO 分散では Single-Nonzero LOO Variance Collapse により $d_{\max}=166.7$ となる死角を定量記録）。
   - **総合レポート**: `docs/handoff/2026-11-12_usir_v5_stage_b9_report.md` 更新、Google Drive inbox へ同期完了。

4. **総合テスト検証**:
   - `pytest python/tests/v5/`: **全47件 PASS** (82.25s)。

---

## 5. Stage B-10 完了記録（2026-11-13）

### 5.1 実施内容と定量確定値
1. **タスク1：Gate 3a 場面スケール分母再設計・有限標本 Student-t インフレ解明の実装と実測**
   - `experiments/gate3a_register_centering_eval.py` に `evaluate_b10_scene_scale_redesign` および状態定数 `GATE3A_STATUS = "SHIFT_QUANTIFIED_NOT_PASSED"`, `GATE3B_STATUS = "GATE3B_FROZEN"` を配備。
   - 拡充 64 サンプル（4レジスター × 16件）において、書籍全体分散 $\sigma^2_{\text{work}}$ から短文場面内分散 $\sigma^2_{\text{scene}, r}$ へ分母を再設計。
   - **Condition (a)（無中心化）**: 場面分散採用により、近代対照群 R4 の Atlas 3-sigma 外れ値率は B-9 の **62.50%（24D）／81.25%（31D） $\to$ 0.00%**（標本内）/ **6.25%**（24D 縮約LOO）へ改善。一方、現代散文（R1〜R3）は 3-sigma **89.58%〜100.0%**・$\chi^2_{0.95}$ **100.0%** と極めて高率の外れ値率を維持し、戦前文学規範からの客観的分布シフトを確認。
   - **Condition (b)（Scheme 3 レジスター内 LOO 中心化）**: レジスター間分散比 $\eta^2_{\text{register}}$ を **100.0% 完全消去**（$0.4965 \to 0.000000$）。内部距離構造は **Spearman $\rho = 1.0000$** で完全保存。縮約 LOO 分散併用時の現代散文全体外れ値率は 3-sigma **0.00%**、$\chi^2_{0.95}$ **4.17%**（$< 5.0\%$）に収束。
   - **共分散行列ランク欠陥診断**: レジスター標本数 $n_r = 16$ に対し LOO 自由度 $\nu = 14$。次元数 $d \in \{24, 31\} > 14$ のため共分散行列がランク欠陥を起こし、対角縮約（0.15）下でも零空間逆投影ノイズにより R4 で 81.25%（24D）〜 93.75%（31D）の偽外れ値が発生することを実証。一変量場面分散 $\sigma_{\text{scene}}$ 採用の数理的必然性を確立。
   - **R4 名目水準（$\chi^2_{0.95}$）超過 4 大要因の定量解明**:
     * 有効 23 次元スケール比: $\sigma_{\text{scene}} / \sigma_{\text{work}}$ は平均 1.4233倍（各次元分散比平均 2.2645倍）であり、B-9 の主歪みを解消。
     * 有限標本 LOO Student-$t$ インフレ: $m=15, \nu=14$ で $\mathbb{E}[D^2]$ が理論上 **$1.2444\times$** にインフレ。5,000回 Monte Carlo 検証により、真の正規分布でも漸近 $\chi^2_{0.95}$ 超過率は **22.29%**（独立正規）〜 **24.51%**（R4相関正規）となることを立証。R4 実測超過率（25〜31%）は有限標本効果として正常範囲。
     * R4 相関行列幾何: 非対角相関自乗和 $\sum \rho^2 = 45.9978$、有効ランク（Participation Ratio） $7.67 / 23$。
     * 31D 単一非ゼロ LOO 分散崩壊: `R4_16` が `v2_final_ぬ` を単独保持するため非正則化 LOO 分散が 0 に崩壊して距離が **52.79** に爆発（縮約 LOO で 18.38、標本内で 5.32 に防護）。
   - **Task 2 固定 H1/H2 格付け**: §0 規約（両側 Wilcoxon + Bonferroni $k=4$）に従い、全6空間で対他作家特異性が有意存続（$p_{\text{bonf}} \le 0.0420$）、対ランダムは名目有意止まりで Bonferroni 非有意、H2 5D 連続コア（`4pole_cont_plus_hw25`）では対ランダム効果が完全消失（$p_2 = 0.4060, p_{\text{bonf}} = 1.0$）であることを確定記録（禁止事項厳守）。
   - **Scheme 2 参考値注記**: 自己レジスター平均を減算した差分を他分布の分散で除算した同義反復的補正であり、独立した収束証拠とはみなさない参考値であることを明記。
   - **成果物 JSON**: `data/usir_v5_eval/usir_v5_b10_gate3a_scene_scale_and_level3.json`

2. **タスク2：単体テスト整備と全スイート回帰検証**
   - `python/tests/v5/test_gate3a_scene_scale.py` を新設（4 テスト全件 PASS）。
   - `pytest python/tests/v5/ -v`: **全 51 件 100% PASS** (90.12s)。

3. **タスク3：総合検証レポートの作成と同期**
   - 総合報告書作成: `docs/handoff/2026-11-13_usir_v5_stage_b10_report.md`
   - 同期先 1: `Claude-Sync/inbox/2026-11-13_usir_v5_stage_b10_report.md`
   - 同期先 2: `F:\マイドライブ\Claude-Sync\inbox\2026-11-13_usir_v5_stage_b10_report.md`

---

## 6. Stage B-11 完了記録（2026-11-13）

**実装コミットハッシュ**: `7592579` (`7592579812f4212faeb351f7753b2650cf5f39ce`, ブランチ: `feat/fable-v5-shootout`)

### 6.1 実施内容と定量確定値
1. **タスク1：Gate 3a 閾値の有限標本較正の実装反映**
   - `experiments/gate3a_register_centering_eval.py` において、24D 連続コア空間（`ortho24_cont`）および対角場面分散 $\sigma^2_{\text{scene}, r}$ 分母を正準とし、外れ値判定閾値を漸近 $\chi^2_{0.95}$ ($d=6.0345$) から B-10 §4 の有限標本 Monte Carlo 帰無分布分位点（`seed=42`, `5000` 回試行, $n_r=16, m=15, \nu=14$, LOO Student-$t$ インフレ $1.2444\times$）へ切り替えた（`get_or_compute_b10_mc_quantiles` / `compute_b10_mc_quantiles`、`usir_v5_b10_gate3a_scene_scale_and_level3.json` 優先読込・未格納時再計算）。
   - 24D 較正閾値: `mc_indep_p95 = 6.9267`, `mc_indep_p99 = 7.7032`, `mc_r4corr_p95 = 7.6574`, `mc_r4corr_p99 = 9.0381`。
   - 64件ベンチマーク（`tier_c_modern_benchmark_expanded.jsonl`）再評価結果（Condition a, 無中心化・$\sigma^2_{\text{scene}, r}$ 分母）:
     * **R4（$n=16$）**: `in_sample` $\chi^2_{0.95}=6.25\% \to$ MC 各閾値 **0.00%**; `shrinkage_loo` / `unregularized_loo` $\chi^2_{0.95}=25.00\% \to$ MC indep 95% **6.25%**, MC indep 99%・R4corr 95%/99% **0.00%**。
     * **R1〜R3 プール（$n=48$）**: `in_sample` / `unregularized_loo` は MC indep 95%/99%・R4corr 95% で **100.00%**（R4corr 99% で 93.75%/91.67%）; `shrinkage_loo` は MC indep 95% **100.00%**, MC indep 99%・R4corr 95% **95.83%**, R4corr 99% **87.50%**。
   - Condition (b)（Scheme 3 レジスター内 LOO 中心化、`shrinkage_loo`）は MC 閾値下で R4（6.25% / 0.00%）・R1〜R3（0.00%）となるが診断値に留め、Gate 3a 通過証拠に算入しない。Scheme 2 は参考値（同義反復的補正注記付き）のまま維持。
   - **ガバナンス**: 閾値変更は「判定基準の較正」であり、**Gate 3a は `SHIFT_QUANTIFIED_NOT_PASSED`、Gate 3b は `GATE3B_FROZEN` を維持**。

2. **タスク2：F-14 ハイブリッド判定・新 $\sigma$ 参照整合**
   - `python/usir_py/v5/f14_hybrid_noise_floor.py` を新設（`python/usir_py/v5/__init__.py` に公開）。Rust（`kasane_core::IncrementalAnalyzer::split_sentences` / `TrajectoryAnalyzer::analyze`）と完全同一の文分割・窓数判定およびハイブリッド判定ルール（両側 $\le 1$ 窓 $\to$ 8D `both_single_window:8`、片側でも $\ge 2$ 窓 $\to$ 11D `any_multi_window:11`）を実装。
   - F-14 再生成版テーブル `docs/handoff/v10/paragraph_noise_floor_all_novel.json`（SHA-256: `00C398144ABE13288451A6C297438E884574EEF42E0B221EE2369F302DB03435`, `--scope all-novel`, 46作品・10,859ペア, `ac_multi_window_per_dim_std`）および `docs/handoff/v10/ac_sigma_bootstrap_ci.json`（SHA-256: `FDFEF9522DE47BDB44C05A6AD2491875E048CC37E29D48D312E176BEBE80A964`, `nishiki` コミット `48b6eb7`/`fa14019`）を参照し、由来と非混同ガード（Gate 3a $\sigma_{\text{scene}, r}$ と harvest ノイズ床 $\sigma$ の峻別）を記録。
   - `experiments/scale_separated_level3.py::evaluate_f14_hybrid_alignment_and_parity()` による一致検証結果（成果物: `data/usir_v5_eval/usir_v5_b11_gate3a_calibrated_and_f14_parity.json`）:
     * **Set 1（青空文庫 10,859 隣接段落ペア）**: 不一致 **0 / 10,859 (0.00%)**（Band M 621件が 8D、1,331件が 11D）。
     * **Set 2（U-6 86改稿ペア）**: Rust `harvest-diff` CLI で段落アライメントが成立した **77 / 77ペア（Dice $\ge 0.25$）で不一致 0件**（Class A 40/40, B 19/19, C 18/18）。残り 9ペア（`A_29, A_35, B_19, B_20, B_22, C_19, C_20, C_21, C_22`）は Dice $< 0.25$ により CLI 側が `kind="unaligned"` として除外する一方、Python 事前対応済み評価では A/C 6件を 8D、B 3件を 11D として評価。
     * **Set 3（`STYLE_POLISH` 28ペア $\times$ 3粒度 = 84評価）**: `span_only` (22/6), `local_context` (7/21), `whole_scene` (0/28) 全84評価で Rust CLI 不一致 **0件**。

3. **検証テスト・レポート同期**
   - `python/tests/v5/test_b11_gate3a_and_f14_parity.py`（5テスト）を含む `pytest python/tests/v5/ -v`: **全 57 件 100% PASS** (133.62s)。
   - 総合報告書: `Claude-Sync/inbox/2026-11-13_usir_v5_stage_b11_report.md`、`docs/handoff/2026-11-13_usir_v5_stage_b11_report.md`、`F:\マイドライブ\Claude-Sync\inbox\2026-11-13_usir_v5_stage_b11_report.md` に同期。

---

## 7. 現在の門番状態および引継ぎ事項

- **現在のブランチ**: `feat/fable-v5-shootout`
- **Gate 3a 状態**: **`SHIFT_QUANTIFIED_NOT_PASSED`**（有限標本 MC 較正後も現代散文の分布シフトは定量維持・無条件通過却下）
- **Gate 3b 状態**: **`GATE3B_FROZEN`（凍結継続・着手禁止）**
  * 理由: 現代散文各ジャンル（R1〜R3）は青空文庫規範多様体から有意に乖離しており、手元原稿（層B-1）の安全吸入には Scheme 3 型のレジスター条件付き正規化パイプラインの配備と Orion 合意が不可欠。
- **Stage B-12 状態**: **未設計・未着手・自動開始厳禁**（Orion 監査およびコーディネーター指示待ち）。

