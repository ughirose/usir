# handoff 13 検証と是正の進捗 (2026-11-04, 記入者: Claude Code)

## 決定事項
- 13 番の「7欠陥」を実コードで検証。①②③⑤⑥は事実、⑦は一部古い(run_v4 は既に `run_authentic_grpo_lora_forge` を使用)、④は未再検証。計画: `~/.claude/plans/c-dev-usir-docs-handoff-13-*.md`。
- ③ 修正済み(未コミット): `drivers/syntax_driver.py` に連体節ルール(Rule 0)と既定 `HTHPolyLexicon` のモジュール共有キャッシュを追加。指摘文の max_depth 3→10。平文は 1〜3 のまま。テスト `tests/test_syntax_driver_nesting.py` 追加。pytest 253 passed / 4 skipped(既存)+新規3件 passed。

## 実測(① A/B, 40チャンク LOOCV)
| パス | 5相 | core3 | q12のみ |
|---|---|---|---|
| 旧(use_multiresolution=False) | 39/40 (97.5%) | 27/28 | 0.925 |
| 新(use_multiresolution=True, 未較正) | 27/40 (67.5%) | 21/28 | 0.65 |
- 新パスを既定にすると現状は大幅悪化。重心・標準偏差の再計算だけで 95% に届くかは未検証。旧パスの 97.5% は手調整特徴のため過学習の可能性があり、未検証。

## ② 実測 ‖p‖₂
Critical 中央2.73(2.19–3.80) / Void 1.44–2.67 / Turbulent 1.40–3.55 / Deadzone 0.10–0.19。‖p‖単独では Critical と Turbulent を分離できない。

## ① 追加測定(新特徴空間)
- `run_loocv` は各 fold の学習データから平均・標準偏差・クラス重心を再計算する。したがって新パス 67.5% は旧スケーラー流用の結果ではなく、新特徴空間そのものの値(依頼文の前提とは異なる)。
- 新パス: 5相 27/40, core3 21/28。誤分類の大半は Void→Critical 7件, Critical→Void 3件。
- 旧24D+新24Dの連結48次元: 5相 35/40 (87.5%), core3 26/28 (92.9%)。基準(5相≥90%かつcore3≥95%)に未達、かつ旧パス単独(97.5%/96.4%)より低い。
- 結論: 新パス既定化は見送り、旧パスを維持。連結アンサンブルも現状は採用しない(40チャンクでの探索的調整は過学習になるため追加の特徴選択はしていない)。

## ② QDA 境界(実装済み)
- novel: `sigma_crit_low=1.40`, `sigma_crit_high=3.90`(Critical 最大 3.80), 新規 Shock ゲート `shock_mean_max=0.60`(q[3,7,11] 平均: Critical 最大0.51, Turbulent 最小0.76), `gamma_shock=30`。article / cot_reasoning は実データが無いため未変更。
- 実測 d_3phase(novel): Critical 0.1–0.2, Void 0.1–0.6, Deadzone 18.7–21.7, Turbulent 0.9–1.1(Mahalanobis のみ), Crystalline 0.8–21.3。Crystalline は ‖p‖<1.40 のものが freeze 罰を受ける(意図的かは要判断)。閾値は40チャンクからの較正で、独立データでの検証はしていない。
- `nishiki_unified_field.py` の `sigma_crit_high=3.5` 上書きを撤廃。Python/TS 両方に実装。実データ不変量テスト追加。

## ④ Toulmin トートロジー(実装済み)
- 因果接続詞前後の文を文字bigram Jaccard>0.70 かつ新規概念≤10% で同語反復と判定し、warrant 加点を除去、`tautology_penalty`(重み1.5)を加算。形態素解析は使わないため精度は粗い。
- 主述ねじれの SyntaxTreeFrame ベース一般化は試作したが、正当な文(「彼は走ったが、私は疲れた」)の誤検出と、真陽性を作れない設計であったため採用せず撤回。既存ルールのまま(未達)。

## 未完了
- ⑥⑤⑦、主述ねじれの一般化(④の残り)。TS 側に構文ドライバは存在しないため③の移植は不要。
- `scripts/harness.py` は自動モード分類器に拒否されたため、pytest(262 passed/4 skipped)・pnpm test(338 passed)・scan_cleanroom_copyright(FAIL=0, WARN=5)を個別実行して確認。

## 次の一手
- ⑥ 関手の p 全12次元算出と音楽逆関手の24D駆動化(Python→TS)。
