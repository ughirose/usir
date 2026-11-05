# Phase 19 ミクロ日本語ゲート (2026-11-05, 記入者: Claude Code)

## 決定事項
- 変更は `python/usir_py/realtime_decoder.py` のみ（TS 移植なし）。`run_longform_director` に kwarg `demo_decay` / `short_prev_gate`（既定 False）を追加。
- M19.1 漢字 i 形容詞の連用止め: `_I_ADJ_STEMS`（薄白黒赤青暗重軽深高低強弱鋭鈍広狭浅濃淡長短早速遅古若熱厚固硬堅太細多良悪痛甘辛苦寒暑丸粗荒緩温酸渋憎怖眩無）。`_is_connective_end` は語幹+く で True、`_FRAGMENT_RE` も同じ語幹集合で追従。ペナルティは既存 gate_bias(-12.0) 経路。
  - 指定リストから除外: 近・遠（「遠く。」「近く。」は名詞の体言止めとして文末に来る）、惜（惜しく であり 惜く ではない）。godan 動詞語幹（行書聞吹鳴…）は含めない。追加: 無。
- M19.2 指示語減衰: `demo_decay`（既定 False）、`demo_pen1=4.0` / `demo_pen2=10.0`。段落内（para_prefix + 行の decode 済み tail）で、それ/その/そこ/あれ/あの/これ/この の出現数が 1 なら -4.0、2 以上なら -10.0 を、その指示語で始まるトークン（strip 後の表層）に加算。トークン ID は `_demo_token_ids()` でデコーダ単位にキャッシュ。cr_pen / lex_decay と独立に ContentRepLogits を構築。
  - 制約: 「そ」「の」が別トークンに分かれる語彙では検出されない（トークン表層が指示語で始まる場合のみ）。
- M19.3 短文直後の体言止め: `case_frame_ok(s, prev_len=None, short_prev=15)`。prev_len <= short_prev のとき名詞連続終止（_NOUN_END_RE）は不可、述語終止なら主語なしで可。`short_prev_gate`（既定 False）は `case_frame_gate` 経路上でのみ有効（BoundaryBiasLogits の noun 判定）。prev_len は prefix の直前の完結文長（空文除外）。
- 新規: `python/tests/test_phase19_micro_japanese.py`、`python/scripts/phase19_false_positive.py`。

## 実測
対象: `corpus_40_benchmark.CORPUS` の II_Critical / III_Void 20 チャンク、197 文（。！？で分割）。
- 規則1: 漢字+く で終わる文 2 件（動く、行く）。規則1で遮断される文 0 件（誤爆 0/197）。
- 規則3: 直前文が 15 字以下の文 36 件のうち、名詞連続終止は 1 件（2.8%）。case_frame_ok が True→False に変わるのは 1 件（2.8%）。全文での名詞連続終止は 1/197（0.5%）。
- 限界: 標本は 197 文で小さい。規則1の対象となる「漢字形容詞+く。」の実例は標本に 0 件のため、規則1の再現率（実際の断片を何割止めるか）は未測定。規則2はコーパス実測なし（ユニットテストのみ）。

## 未完了
- 実モデル（Colab）での効果測定なし。`demo_decay` / `short_prev_gate` は Colab 実験で未検証、既定は False。
- Qwen トークナイザでの指示語トークン分割の実態（「その」が 1 トークンか）未確認。
- TS 側への移植なし（依頼範囲外）。

## 次の一手
- Colab で 1.5B/3B を `demo_decay=True, short_prev_gate=True` と既定で n=5 seed 比較し、指示語の段落内重複率・「漢字+く。」断片数・短文後の体言止め率を計測する。
