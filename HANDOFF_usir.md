# USIR v5 実装引継ぎ台帳 (HANDOFF)
**最終更新日時**: 2026-11-12 14:40:00 JST  
**ブランチ**: `feat/fable-v5-shootout` (`C:\dev\usir`)  
**関連リポジトリ**: `C:\dev\nishiki`（Lane B 境界線厳守・変更ゼロ保証）

---

## 0. Stage B-7 実装・実測セッション（2026-11-12 実施）

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

## 3. 次のセッションへの引継ぎ事項
- **現在のブランチ**: `feat/fable-v5-shootout`
- **完了コミット**: `feat(v5): complete D0 calibration, D11/U6 polish separation, Gate 3a shift, and diachronic shootout`
- **次のマイルストーン**:
  1. Gate 3b の盲検評価設計（妻による初稿マークおよび Cohen's $\kappa$ 一致度検証）。
  2. Level 3 専用の局所高解像度差分プローブの設計（微細推敲のノイズ床埋没を克服する差分アンプ）。
