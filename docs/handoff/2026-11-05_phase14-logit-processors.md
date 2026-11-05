# Phase 14: ロジット処理のモジュール化・lex_decay 再設計・格フレームゲート (2026-11-05, 記入者: Claude Code)

## 決定事項
- 未コミット2件の監査:
  - `python/artifacts/checkpoints/table2_progress.json`: ローカルの Table 2 ベンチ(v3.0 手法、2 sample)を再実行した副産物。差分はタイムスタンプ、mauve_mean/std(全手法で 0.004〜0.03 低下)、elapsed のみで、d2/beta_err/closure は不変。コード作業の仕掛かりではない。未コミットのまま。
  - `docs/handoff/14_claude_code_handoff_spec.md`: `d546da5` 時点のキックオフ仕様(アシスタント口調 / 方向ステアリング / 3B・7B 化)。3件とも `6746661` で実装・実証済みで、内容は古い。未追跡のまま。
- `realtime_decoder.py` のクロージャ4クラスをモジュールレベルへ移動し、numpy/torch 両対応の行ループ実装にした: `NarrationStyleLogitsMask`, `ContentRepLogits`, `ContextNgramBan`, `BoundaryBiasLogits`。純関数は `narration_row_flags`, `case_frame_ok`, `kanji_bigrams`, `lex_bigram_targets`。
- `lex_decay` を漢字バイグラム条件付き減衰に変更した。段落テキスト(prefix + 行 tail)の末尾文字が漢字 C1 で、(C1,C2) が `lex_min_count`(既定 1)回以上既出なら、先頭文字が C2 のトークン全てに `-lex_decay`。`_token_classes()` は `(content_mask, first_kanji_map, hira_only_mask)` を返す。
- `stem_guard`(既定 False、ランナーは True かつ anti_repeat 時): 行テキスト末尾がひらがなのとき、ひらがなのみのトークンを soft/hard n-gram 抑制から除外。
- `case_frame_gate`(既定 False、ランナーは True): noun_req スコープ(rhythm 短文 mode0 / 中文読点後)の句点正バイアス条件を `case_frame_ok` に置き換えた。条件は「体言・指示詞 + が/は/も」または「末尾に漢字/カタカナ 2字以上」。が/は/も の直後が る/り/れ/ら/っ/ろ の場合は動詞の送り仮名とみなす(広がった・上がる)。
- テスト: `python/tests/test_logit_processors.py`(MockTokenizer + numpy、44件)。pytest 366 passed / 4 skipped(着手前 322/4)。

## 未完了
- torch 経路は未検証。ローカルに torch がないため、テストはすべて numpy。行ループ化による速度変化も未計測(大きな index リストを行ごと・ステップごとに渡しているため、Colab で遅くなる可能性がある)。
- 「さえぎ」崩れと主語抜け短文が実際に減るかは未測定(Colab 実測が必要)。格フレームゲートはロジットの正バイアスを止めるだけで、LM 自身が出す「。」は封じない。
- 格フレーム判定は正規表現ヒューリスティックで、「雨がっ」等の稀な表記を誤判定し得る。

## 次の一手
- Phase 15(ローカル): M15.1 節スコープ主述ねじれ → M15.2 サマリ入力時12次元 p(Py/TS パリティ)→ M15.3 judge_model 結線(モックテスト)。
- Phase 16 の最初の Colab 実行で、torch 経路の動作・所要時間・新ゲートの効果を 1.5B で確認する。

# Phase 15: 主述ねじれ・サマリ運動量・LLM Judge 結線 (2026-11-05, 記入者: Claude Code)

## 決定事項
- M15.1: `python/usir_py/clause_twist.py` を新規作成(`split_clauses`, `detect_clause_twists`, `count_clause_twists`)。節境界(けれど/けど/ながら/のに/ので、述語末の直後の が/から/し、述語形の後の読点)で主語スコープをリセットし、節内の主語主辞と述語を汎用語彙クラス(刺激名詞/抽象名詞 × 知覚・意志動詞/動作主動詞)で照合する。自然現象の主語(擬人法)、コピュラ、補文、受身・使役、引用は対象外。`evaluate_toulmin_components(clause_twist=True)` で既存の `twist_count` に加算し、`clause_twist_count` を別途返す。「彼は走ったが、私は疲れた」=陰性、「風の音が耳を澄ました」=陽性。
- M15.2: `momentum_from_summary`(Py: nishiki_unified_field.py、TS: crossModalFunctors.ts `momentumFromSummary`)。p_i = 2.2·sqrt(g_i² + 0.02²)、g = Λ(q−0.5)。Λ は固定の4族結合行列(対角 1、同族 0.3、異族 0.1、正定値)。族の割り当てはこのモジュールの既存定義(Flux{0,3,7}/Lattice{1,4,8}/Void{2,5,9}/Shock{6,10,11})に従う。旧プロキシ4次元は hypot で合成。スケール 2.2 は音楽の小節系列51本(LCG 乱数)で窓経路と較正。窓経路は不変。Py/TS の固定参照値は 1e-6 以内で一致。
- M15.3: `evaluate_llm_judge(judge_fn=...)`, `parse_judge_scores`, `make_hf_judge_fn`、CLI `--judge-reuse-model`(既定 off)。出力に `judge_rubric` / `judge_llm` / `judge_llm_raw` / `judge_source` を追加。旧経路は torch 未 import の NameError を握りつぶしており、一度も動いていなかった。
- テスト: pytest 391 passed / 4 skipped、pnpm 346 passed。

## 未完了・注意
- clause_twist は既定 on のため、Toulmin の twist_count が増える入力では既存ベンチのスコアが変わり得る(既存テストは全通過)。語彙クラスは小さく再現率は低い。実コーパスでの誤検出率は未測定。
- サマリ運動量の Λ は学習済み Forge 精度行列ではない(このモジュールからは参照できない)。視覚側は未較正。音楽の固定12音列テストの参照値(time / gain)は新しい p で再生成した(音高列は同一)。
- 実モデルでの judge は未実行(Colab が必要)。`--judge-reuse-model` を使うと、Spearman 相関の対象は LLM のスコアに変わる。

## 次の一手
- Phase 16: Colab で 1.5B の1本を流し、Phase 14 の torch 経路(動作・速度)を確認する。その後、5 seeds × {Steered, Ctrl} を回す。

# Phase 16: Colab 複数シード統計 (2026-11-05, 記入者: Claude Code)

## 決定事項
- fe7433c: case_frame_gate を Shock 域(Climax 短文の row bias / ramp)にも適用。きっかけは、2878d5d の 3B で Steered の主語・体言止め欠落が Ctrl より多かったこと(32.8 対 20.2/本)。
- 結果(Steered−Ctrl、seed 対応、多重比較補正なし。詳細は `docs/handoff/v6/phase16_all_summary.json`):
  - delta_sf: 1.5B +0.236(d=2.56, p=.014, Ctrl n=4)/ 3B +0.287(d=4.88, p=.0004)/ 7B +0.169(d=2.17, p=.008)。3モデルとも Phase 16 の完了条件 d>1.0 を満たす。
  - 全 run で敬体・メタ・括弧・断片・活用崩壊は 0、8gram 最大は 2 以下。
  - bigram 未達(<0.55): 1.5B s3 の Steered(0.540)と Ctrl(0.479)。3B は fe7433c 後に 0 本(前回は 2 本)。
- Colab wall 累計 9156s(smoke 242 + 3B 2370 + p16b 6544)。全セッション成功。CU は取得していない。

## 未完了・注意
- Shock 域へのゲート拡張の効果は小さい。3B の case_frame 不合格は 32.8→30.8/本で、Ctrl 20.2 との差は残る。7B では Steered 24.8 / Ctrl 23.2 とほぼ同等。
- 3B s137 Steered は delta_sf 0.131 で基準未達(Climax 文長 18.0)。ゲートの副作用か seed 内のばらつきかは未分離。
- 1.5B s3 Ctrl: Climax の direction 指標が記録されていない。原因は未調査。
- `count_non_jis_kanji` は「梦」(U+68A6)を検出できない。Python の shift_jis でエンコードできてしまうため、JIS 判定は簡体字の除外として不完全。
- 3B Ctrl は前回データの再利用(fe7433c の影響を受けない前提で、未確認)。新規 run の本文は目視していない。

## 次の一手
- 簡体字の判定を、shift_jis ではなく JIS X 0208 の明示表に置き換えるかを検討する。
- 格フレーム不合格の差を分離するため、Steered の case_frame_gate=False の対照を組む。
- Phase 17: TS への WorldCraftArbitrator / HSMM 移植。

# Phase 17: WorldCraftArbitrator 一般化と TS 移植 (2026-11-05, 記入者: Claude Code)

## 決定事項
- Python: `compute_null_space_projector_svd`(P = I − J⁺J、tol = max(k,12)·eps·s_max)、`build_general_jacobian`、`WorldCraftArbitrator(projector_method="tikhonov"|"svd")`。既定は従来の tikhonov のままで、既定動作は不変。director に `arbitrate_with_jacobian` を追加。
- TS(usir-core): `dynamics/worldcraftArbitrator.ts`(一側 Jacobi SVD、外部依存なし)、`hsmmLongformDirector.ts` に `arbitrateSceneMomentum` / `arbitrateWithJacobian`。
- パリティ: `packages/usir-core/tests/fixtures/worldcraft_arbitrator_parity.json`(生成は `scripts/gen_worldcraft_parity_fixture.py`)。制約5種(空・単位行・7行・2行線形結合・rank 欠損3行)と director の10ステップ op 列。Py/TS は 1e-6 以内で一致。性質(対称・冪等・J·P≈0・rank)は 1e-12。
- テスト: pytest 405 passed / 4 skipped、pnpm 359 passed。

## 未完了・注意
- tikhonov(eps=1e-6)は単位行でも 1e-6 の残差があり、SVD とは約 1e-6 ずれる。既定を SVD に切り替えるかは未決定。
- 確率的な遷移のパリティは未比較(numpy と TS の乱数系列が異なるため)。Jacobi SVD は fixture の範囲でのみ収束を確認。
- LLM デコード部、Tier-1 hard mask は TS に移植していない。

## 次の一手
- 残る判断事項: Arbitrator の既定を SVD に切り替えるか、簡体字の検出を JIS X 0208 の明示表に置き換えるか、Steered の case_frame_gate=False 対照を Colab で回すか。
