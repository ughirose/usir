# Phase 23 リズム・句読点・改行 (2026-11-05, 記入者: Claude Code)

Phase 21 出力と Aozora 46 作品の比較で見えた、文長の短さ・読点過多・連体形直後の「。」・語の崩壊反復・無改行に対する実装。Colab 実行は未実施。既定値はすべて従来動作と同一。

## 決定事項
- `ClauseTwistedSMCDecoder.rhythm_profile` (`python/usir_py/realtime_decoder.py`, `_rhythm_params`)
  - `cycle`: 従来動作（既定）。
  - `steady`: 「。」の正の押しも「、」の押しも出さない。文が `steady_min_chars`(16) 未満の間は「。」を -12 で保持。`steady_max_chars`(90) 超で +3。文中の読点が `steady_max_commas`(3) 以上で「、」-4。節トークン上限は 20。
  - `pulse`: cycle と同じ構造。正の「。」押しは +3 が上限。long モードの「、」+4 を 0 にした（「。」封印 -12 は残す）。medium モードの「、」+4 は残した。
  - 段落ごとに変更可能なプロパティ。不正値は ValueError。`rhythm=True` のときだけ効く。
- 連体形ガード `adnominal_guard`（`BoundaryBiasLogits`）
  - 直前が た/る/い/だ/な で、現在の文が `steady_max_chars` 未満のとき、「。」への正のバイアス合計を +2.0 に制限する。負のバイアスは変えない。
  - 全寄与の合計に対する上限。`force_close`（段落末の閉じ）は対象外にした。対象にすると段落を閉じられなくなるため。
- CV 帯域 `cv_ceil`: 設定時、SMC 重みに `-cv_gain*max(0, CV-cv_ceil)` を加える。None は従来どおり下限のみ。
- `lex_compound_exempt`: 末尾の漢字・カタカナ連続 R と候補先頭字 C2 の連結 R+C2 が、段落内で既出の 2 字以上の連続の接頭辞なら、lex_decay を当てない（`_lex_compound_filter`）。content_rep_penalty はそのまま。
- `collapse_guard` / `collapse_penalty`(8.0): 候補が完結・延長する文に、同一漢字 3 回以上または同一 2 字の漢字・カタカナ列 3 回以上があれば SMC 重みから減点。`stats["collapse_cands"]` に計数。
- 指標 `python/usir_py/text_metrics.py`: 平均文長、読点/100字、文内同一漢字最大反復、文内 2 字語反復最大、連体形分割疑い。
  - 連体形分割疑いは `(?<=[たるい])。(?=[一-鿿]{1,4}(?:く?の|[がをはに]))` の件数。ヒューリスティック。仕様の正規表現に「多くの」を拾う `く?の` を追加した。
- 整形 `python/usir_py/text_format.py: format_novel`
  - 「…」台詞は単独行（直後の と/って〜文末 は同じ行に付ける）。地の文は [0.6,1.4]×目標字数 の乱数（seed 固定）に達したら文境界で改行。
  - 台詞以外の行頭に全角空白。director 段落間は空行。
- ランナー `run_longform_director` (`python/colab/run_v4_bidirectional_experiment.py`)
  - 追加引数: `rhythm_profile`, `climax_rhythm_profile`, `adnominal_guard`, `cv_ceil`, `lex_compound_exempt`, `collapse_guard`, `target_line_chars`（指定時に `formatted_text`）。
  - Climax 判定: director が段落ごとに `decoder.current_well_idx`（well 番号。Climax=2）を設定するようにし（`hsmm_longform_director.py` に 2 行追加）、フックが `current_well_idx == 2` で `climax_rhythm_profile` に切り替える。
  - チャットテンプレートは `enable_thinking=False` 付きで呼び、TypeError なら付けずに再呼び出し。出力から `<think>…</think>` を除去（段落テキストと full_text）。
- `_token_classes` / `_demo_token_ids` は sentencepiece 形式の先頭 `▁` を除いて分類する。偽トークナイザのテストを追加した。

## 実測
テスト: `cd python && uv run pytest tests -q` で 508 passed / 4 skipped（基準 486 passed + 本 Phase の新規 22）。`test_phase_d_usir_forge` は dist 未ビルドで失敗したため `pnpm install && pnpm -r build` 後に再実行して通過。

指標（`corpus_40_benchmark.CORPUS` のチャンクと、`docs/handoff/v7/p21_*.json` の `runs[*].paragraph_texts` の連結。ランごとに計算し、条件内で平均）。

| 対象 | n | 平均文長 | 読点/100字 | 同一漢字最大(文内) | 2字語反復最大(文内) | 連体形疑い/1000字 | 平均字数 |
|---|---|---|---|---|---|---|---|
| corpus III_Void | 10 | 30.0 | 2.86 | 1.80 | 1.10 | 10.4 | 231 |
| corpus II_Critical | 10 | 33.8 | 4.58 | 2.30 | 1.60 | 7.5 | 371 |
| corpus II+III (実作家) | 20 | 31.9 | 3.72 | 2.05 | 1.35 | 8.6 | 301 |
| corpus 全体 | 40 | 43.2 | 3.43 | 2.27 | 1.57 | 6.4 | 281 |
| p21_3b_a ctrl | 2 | 22.2 | 7.58 | 6.50 | 2.00 | 16.8 | 2080 |
| p21_3b_a A | 2 | 20.7 | 5.55 | 3.00 | 1.00 | 19.1 | 2099 |
| p21_3b_a B | 2 | 19.2 | 5.65 | 4.00 | 1.00 | 18.8 | 2080 |
| p21_3b_b ctrl | 1 | 21.7 | 7.87 | 5.00 | 2.00 | 13.7 | 2046 |
| p21_3b_b A | 1 | 18.4 | 4.53 | 4.00 | 2.00 | 16.6 | 2106 |
| p21_3b_b B | 1 | 18.2 | 4.62 | 4.00 | 2.00 | 15.9 | 2082 |
| p21_7b ctrl | 2 | 26.2 | 4.86 | 2.00 | 2.00 | 16.3 | 2141 |
| p21_7b A | 2 | 24.0 | 4.31 | 2.00 | 1.50 | 16.5 | 2124 |
| p21_7b B | 2 | 23.2 | 3.82 | 3.00 | 1.00 | 14.1 | 2132 |

（corpus の I_Crystalline（文長 81.7）、IV_Turbulent、V_Deadzone は法典文・合成文のため表から省いた。`A_legacy` 条件は A と数値が同一なので省いた。）

読み取れること
- 平均文長: 生成 18–26 字に対し実作家チャンク 30–34 字。corpus チャンクは 200〜370 字、生成は約 2100 字で、文長の比較に支障はない。
- 読点/100字: 3B ctrl は 7.6–7.9 で実作家 2.9–4.6 を超える。A/B と 7B は 3.8–5.7 で範囲に近い。
- 同一漢字の文内最大反復と 2 字語反復は、生成の方が大きい。ただしこれらは「最大値の平均」で、テキスト長（生成は約 7 倍）に依存し、長さで正規化していない。
- 連体形疑いは実作家でも 7.5–10.4/1000字あり、生成は 13.7–19.1/1000字。正規表現が「た。彼が…」のような正当な文分割も拾うので、約 2 倍の差は上限側の目安にとどまり、精度は未検証。

限界・未検証
- Phase 23 の新機構（steady/pulse、ガード類）を実モデルで走らせた測定は無い。上表は Phase 21 の既存出力に対する指標のみ。
- steady の閾値（16/90/3）、pulse の +3、collapse の 3 回、連体形ガードの +2.0 は、設計時の仮置き。実データで較正していない。
- `_lex_compound_filter` は段落テキスト内の既出語のみを見る。`carry_context` 時の `para_prefix` は空文字なので、前段落の語は対象外。
- `adnominal_guard` は文字単位の終端判定（た/る/い/だ/な）。名詞止めや、連用形の「い」などで「。」が不必要に抑えられる場合がある。
- `<think>` 除去は最終テキストのみ。生成中のトークンでは除去していない。

## 未完了
- Colab で steady / pulse × ガード類の実生成（3B・7B）と、上表の指標・Judge・ΔLL への影響の測定。
- 連体形疑いの正解ラベル付き評価（現状は件数のみ）。
- 閾値の較正（Aozora 文長分布との照合）。
- `format_novel` の出力を読んだ上での目視評価（台詞検出は「」のみ。『』は対象外）。

## 次の一手
- `run_longform_director(rhythm_profile="steady", climax_rhythm_profile="pulse", adnominal_guard=True, cv_ceil=…, lex_compound_exempt=True, collapse_guard=True, target_line_chars=120)` を Colab で 3B・1 seed 実行し、本表と同じ指標で cycle との差を取る。
