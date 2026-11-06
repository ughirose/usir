# Phase 37: Gemini の Phase 36 監査を受けたローカル再解析と修正（Colab 不使用）

- 入力: Gemini の監査 `Usir_V4_Phase36_Audit_And_Phase37_Plan.md`（ユーザーがアップロード）。
- 方針: 主張を一つずつ、コードと JSON で照合してから実装した。
- 再解析スクリプト: `scripts/phase37_reanalysis.py`。出力は `docs/handoff/v10/p37_reanalysis.json`。

## 0. 照合
| Gemini の主張 | 照合結果 |
|---|---|
| `docs/paper/usir_v4_paper.md`、`scripts/generate_paper_tables.py`、`verify_paper_claims.py`、Table 5 / §4.5 / §4.6 | **存在しない**。実体は `usir_v4_paper.md`（ルート）、`build_paper_v4.py`、`verify_paper_numbers.py`、§3.3 / §5.3。実体の側を更新した |
| JSON は `artifacts/colab_run/` にある | 誤り。`docs/handoff/v10/` |
| well 2 の段落だけが pulse、他は steady | 正しい（`climax_rhythm_profile="pulse"`、フックは `current_well_idx == 2`）。ただし「第 3 段落」とは限らない。wells の並びは seed ごとに異なる |
| Steady の文長 ctrl 29.63 / gA 54.89 / gB 71.29、Climax の文数比 45.7%、Steady の gB − gA は平均 16.35（12.62〜21.11） | **再現した**（同じ分割関数 `sentence_lengths`）。ただし事前に固定した判定指標ではなく、結果を見たあとの分割 |
| 帯内の `pb = 0.0` により、節 1 つ分はみ出す | 機構は確認した。`_rhythm_params` は節の開始時に 1 回だけ評価される（`_phase_bias`、`_row_caps`）。gA の steady の文 180 文のうち 76 文が上限 55 字を超えている |
| `_THREE_DIGIT_RUN_RE`、`year_trigger_ids`、`self.rhythm.steady_min_chars` | **存在しない**。実体は `_YEAR_RE`、digit 表、`self.steady_min_chars` |
| 「〇」は (1277, 229) に分かれ、途中まで生成した時点の復号は「二�」 | 正しい（Phase 36 で確認済み） |
| `lora_grpo_forge.compute_shaped_grpo_reward`（L237） | **存在しない**。シェイピング報酬は `run_phase33_phase_c.make_shaped_reward` |
| cadence_alive は 72 件で二極化（低 25、高 47、中間 0） | 正しい。ただし評価時のテキストでの話で、訓練ロールアウトは保存していない。原因は `extractor_24d.py` の flat-slop ゲート（mean_len > 40 かつ CV_L < 0.28 かつ（abstract > 0.015 または sluggish > 0.15）で −0.68 の段差）。Gemini の書いた条件「です・ます比 ≥ 0.90」とは異なる |
| merged LoRA は token 上限到達が 6 → 12、改行が 2 → 8、文字数 590.7 → 618.1 | 上限到達と文字数は正しい。空行（`\n\n`）を含むものは 2 → 7（改行 1 つ以上なら 2 → 8） |
| GRPO は「途中切れによる報酬ハック」を学習した | **未検証**。訓練ロールアウトを保存していないので、因果は確かめられない。仮説として記載した |
| ステア α 0.12 は決定的な陽性（cadence 0.564 → 0.851、σ 0.972 → 1.526） | 数値は正しい。**ただし末尾の断片を除いて判定し直すと、Deadzone は 1 → 5/12**（ランダム 7 → 9、base 7 → 7）。較正方向 α 0.12 は 12 件中 8 件が文の途中で終わっていた。陽性結果の一部は途中切れの断片から来ている |
| ランダム「直交」方向の対照 | 誤り。ノルムを合わせたランダム方向（直交ではない） |

## 1. 実装（ローカル、単体テストのみ。モデルでの実行はしていない）
- `realtime_decoder.py`:
  - `byte_split_digit_pairs(tokenizer)` を追加した。2 つのトークンに分かれる数字文字を列挙する。
  - `ForbiddenWordsLogits(digit_pairs=...)` は、直前 3 文字が数字のとき、その文字の先頭バイトトークンを減点する。先頭バイトが出た直後で、文字を完成させると 4 桁になる場合は、第 2 バイトトークンを減点する。減点は `digit_penalty` を使う（Gemini 案の −50 ではない。既存の 1 つのつまみに揃えた）。
  - 「二〇」のように、前の数字が 1 文字だけなら止めない。
- `realtime_decoder._rhythm_params`: 帯内で `band_ramp × frac` の句点バイアスを加える。既定は 0 で、Phase 36 と同じ挙動。`proposal_from_prior(band_ramp=0.0)`。
- `lora_grpo_forge.strip_incomplete_tail` を追加した。`make_shaped_reward(..., strip_tail=False)`。
- テスト 3 件を `test_phase36_narrative_prior.py` に追加した（モックのバイトトークナイザ、帯内の傾斜、末尾の除去）。

## 2. 未解決
- `band_ramp`、「〇」マスク、`strip_tail` は、いずれも Colab で実行していない（ユーザー指示「Colab 不要」に従った）。効果は未測定。
- Phase C の評価は 400 トークンで切れる。陽性結果を確定させるには、完結した文での評価（たとえば max_new_tokens を上げて途中切れを 0 にする）が必要。
- GRPO の報酬ハック仮説を確かめるには、訓練ロールアウトの保存が必要。
