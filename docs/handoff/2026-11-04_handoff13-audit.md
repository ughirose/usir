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

## ⑥ クロスモーダル関手(実装済み)
- 共役運動量: 視覚は `patch_features`/`patchFeatures`(画像の g×g ブロック列, g=min(3, 短辺/16))、音楽は `bar_features`/`barFeatures`(推定テンポ4拍ごとの小節窓, 無音小節も保持)の q 系列から、全12次元を窓間 RMS 変動速度として算出。TS 抽出器 (`extractVisualFeaturesFromImage`, `extractMusicFeaturesFromNotes`) が窓系列を自動生成。
- 限界: 窓系列を持たないサマリ入力では時間情報が無いため p は計測不能。従来の4次元プロキシ(p[0,2,6,11])を残し、他8次元は 0 のまま(仕様の「サマリ入力時も12次元」は未達)。
- 逆音楽関手: 固定ペンタトニック(`step % 5`)を撤廃。16スロットの八分格子で、Flux=音数密度, Void=休符(スロット除去), Lattice=跳躍抑制, Shock=跳躍確率/幅・五度圏の主音移動・三全音→半音解決, syncopation=1/4拍ずらし。黄金比系列による決定論的選択で乱数不使用。出力音数は可変(旧テストの「必ず8音」は撤回)。
- Python/TS パリティ: 同一状態で音列12音・time・gain の一致をテストで固定。テスト: pytest 269 passed/4 skipped, pnpm 343 passed。

## ⑤ WorldCraft 零空間調停(実装済み)
- `WorldCraftArbitrator.compile_constraint_jacobian(context_text, active_constraints)`: モード→凍結要素の単位行から J_C∈R^{k×12} を構築。evidence=Sg-08,12(idx 7,11), logic_voice=Sg-02,06(idx 1,5), stillness=Sg-03,07,11(idx 2,6,10)。未指定時は文面キーワードで選択、該当なしは k=0(P_null=I)、未知モードは ValueError。
- `HSMMLongformDirector(arbitrator=, scene_constraints={段落idx: [モード]})`: 段落の目標運動量を零空間へ射影し、切り捨て成分を `p_banked` に蓄電(Climax で放出)。統合テスト: 強制 Narrative→Climax で、調停ありの Climax 放出量が調停なしより大きいことを確認。
- 限界: ヤコビアンは単位行(要素凍結)のみ。要素間の線形結合制約は未対応。モード→要素の対応は設計上の仮定で、データによる検証なし。TS 側に Arbitrator は無く移植なし。

## ⑦ 監査結果(修正なし、事実のみ)
- `HSMMLongformDirector` は tests と `__init__` の export 以外から呼ばれていない(本番スクリプトに結線なし)。`run_v4_bidirectional_experiment.py` は `run_authentic_grpo_lora_forge` と実デコーダを使用しているが、長尺生成は Director 経由ではない。
- `evaluate_llm_judge`: run_v4:945 は `judge_model` を渡さないため、常にオフライン決定論ルーブリック(文長・接続詞・bigram 多様性・定型句)で採点。USIR 特徴は使わないが、ベースライン定数(4.2 等)の手調整ヒューリスティックで、LLM 審査ではない。
- `usir_forge` の Cholesky 精度行列: 実共分散(Welford M2)から算出されるが、固定シュリンケージ 0.2·S+0.8·I により、40チャンクで実測すると Σ の 0.8·I からの相対ずれは 7.7%、精度行列対角は 1.19–1.25(≒1.25·I)、非対角最大 0.056。特徴の分散(平均0.097)に対し単位行列項が支配的で、事実上単位行列に近い。Python/TS のバイナリ互換に影響するため、シュリンケージ方式の変更は未実施(設計判断が必要)。

## usir_forge シュリンケージ再設計(実装済み, Python/TS 同値)
- 変更: `Σ = 0.2·S + 0.8·I` → `Σ = a·S + (1−a)·(tr(S)/24)·I`, a=0.75(`SHRINKAGE_ALPHA`)。目標行列をデータ分散に合わせたため、標本相関が支配的になる。バイナリレイアウト(4096B, 3モードGMM, 下三角コレスキー300)は不変。分散ゼロ/N≤1 のモードは従来の 0.8·I。
- 条件数: S が半正定値なら cond(Σ) ≤ (24a+1−a)/(1−a) = 73 < 500(解析的上限)。
- 実測(コーパス全文を文単位で投入, モード件数 131/4/131): 精度行列の対角 7.9〜933(旧: 1.19〜1.25), 非対角最大 12〜378(旧: 0.056), 実条件数 28.8〜53.6, 精度の非対角Frobenius比 0.25〜0.39。
- 依頼の「実共分散寄与 70% 以上」について: 標本共分散 S の重みは Σ 内で 75%。精度行列の非対角成分が Frobenius ノルムに占める比は 25〜39% で、精度行列の性質上 70% には達しない(別指標)。旧実装は Σ の 0.8·I からのずれが 7.7% だった。
- 注意: TS の `dist/` は gitignore で古いままだったため `pnpm --filter ./packages/usir-core build` を実行。Python 側の interop テストは dist を使うので、TS 変更後は再ビルドが要る。
- テスト追加: 非スカラー精度・条件数<500、Python/TS コレスキー一致(rtol 1e-3)。pytest 274 passed/4 skipped, pnpm 343 passed。
- 未検証: 実コーパスでの下流効果(蒸留バイナリを使った生成/判別への影響)は測っていない。

## 未完了
- 主述ねじれの一般化(④の残り)、サマリ入力時の p、HSMM Director の本番結線、LLM ジャッジの実モデル化。TS 側に構文ドライバは存在しないため③の移植は不要。
- `scripts/harness.py` は自動モード分類器に拒否されたため、pytest(262 passed/4 skipped)・pnpm test(338 passed)・scan_cleanroom_copyright(FAIL=0, WARN=5)を個別実行して確認。

## 次の一手
- HSMM Director の本番スクリプト結線(run_v4)。
