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
