# USIR v5 実装引継ぎ台帳 (HANDOFF)
**最終更新日時**: 2026-11-11 20:50:00 JST  
**ブランチ**: `feat/fable-v5-shootout` (`C:\dev\usir`)  
**関連リポジトリ**: `C:\dev\nishiki`（Lane B 境界線厳守・変更ゼロ保証）

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
