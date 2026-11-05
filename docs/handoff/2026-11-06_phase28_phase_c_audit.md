# Phase 28.2 Phase C 監査と実モデル Colab ドライバ (2026-11-06, 記入者: Claude Code)

対象: `python/colab/run_experiment_c_lora_forge.py`（以下「Table 3 ドライバ」）、`python/usir_py/activation_steerer.py`、`python/usir_py/lora_grpo_forge.py`、`python/tests/test_phase_c_grpo_and_steering.py`。
行番号は HEAD adfd9e2 時点（`lora_grpo_forge.py` は本コミットで編集したため、編集前の行番号を示す）。

## 結論

- Table 3 ドライバが出力する全ての数値は、実モデルの出力から計算されていない。テキストは手書きの固定文、レイテンシは定数または乱数、1/f 誤差は人工的に作った「サプライズ列」から得ている。
- `--mock_mode` は `store_true, default=True`（L590）で CLI から無効化できない。さらに `mock_mode=False` を渡しても生成は `generate_mock_sample`（L461）で、実モデル生成の分岐は存在しない。
- 結果 JSON `artifacts/table3_lora_grpo_forge_results.json` は `lora_telemetry.mode = "mock"` であり、この値が論文節 `scripts/paper_sections/sec4_6_table3.md`（L11-13, L20）に「Qwen-2.5-7B」「N=200」「0.00 ms vs 4.17 ms」「100.0% escape」として実測表の体裁で載っている。実際は mock、iterations_per_method=50、dataset_num_pairs=200（200 はデータセット件数であり試行数ではない）。
- docstring の「escape >= 94%」等（L3-9）は事前に書かれた期待値で、mock 生成器はその期待値を満たすように固定文と乱数パラメータを設計している（下表）。

## A. Table 3 ドライバが JSON に書く各値の出所

| 出力値 | 計算元 | 区分 | 根拠 (file:line) |
|---|---|---|---|
| `overhead_ms` Base / LoRA | 定数 `0.0` | ハードコード | L176, L223 |
| `overhead_ms` Adaptive-BAOAB | `rng.normal(4.21, 0.15)` | 乱数（期待値 4.2 ms を直接指定） | L192 |
| `latency_per_token_ms` | `rng.normal(12.5, ...)` | 乱数。さらに結果 JSON には書かれない（`record_summary` に無い） | L186, L212, L241, L472-484 |
| 生成テキスト | 手書き文リストの連結 | ハードコード（モデル出力ではない） | L177-185, L195-210, L226-239 |
| `len_cv`, `cadence_alive`, `z24` | 実装の `extract_usir_24d` を上記固定文に適用 | 計算は本物、入力が固定文 | L266-269 |
| `phi_void` | `q[2], q[6], q[10]` の正部分の平均（独自定義） | 計算は本物、入力が固定文。抽出器の `order_params.phi_void` とは別定義 | L272 |
| `grpo_reward` | `compute_v3_multiplicative_grpo_reward` | 計算は本物、入力が固定文。target はデータセットの `target_phase` | L275-280 |
| `escaped_deadzone` | `len_cv>=0.45 and cadence_alive>=0.35 and grpo_reward>=0.30` | ドライバ独自の手設定しきい値。リポジトリの Deadzone 分類器（`manifold_geodesic.py:157-158`, `||p||` が `sigma_crit_low` 未満）とは別物 | L284 |
| `beta_error` | 人工サプライズ列のスペクトル傾き | 乱数で作った列に傾き指数（Base: 白色雑音、BAOAB/LoRA: `freqs**0.52`/`0.50`）を指定して生成し、その傾きを回収 | L188, L214-219, L243-248, L289 |
| `beta_error` の代替値 0.85 / 0.12 | 定数 | ハードコード（`surprise_seq` が無い場合のみ。mock では発火しない） | L290 |
| `mauve` | `exp(-0.5 * ||mean(z_gen) - mean(z_ref)||^2)` | MAUVE ではない独自の近似。参照は `masterwork_texts[:16]`（CORPUS の先頭 16 件、位相が偏る）。入力が固定文 | L151-161, L378, L514 |
| `cleanroom_zero_copyright_verified` | 定数 `True` | ハードコード。検査は 3 名のブラックリスト一致とファイル内フラグ `cleanroom_7gram_passed` の読み取りのみで、7-gram は再計算しない | L117-126, L540 |
| `lora_telemetry` (実 GPU 時) | 12 ステップの DPO 風損失 | 実計算だが、このモデルは以降の生成に一切使われない。最終表の値と無関係 | L381-432 |
| `lora_telemetry` (既定) | `{"mode": "mock"}` | mock | L380 |
| `deadzone_escape_rate` ほか全統計 | 上記の集計 | mock 由来 | L499-529 |

結果 JSON の実値からの傍証: Base の `len_cv_std = 0.0`、`overhead_ms = 0.0 ± 0.0`、`phi_void_std = 0.0`（固定文が毎回同一のため）。

## B. レイテンシ主張 (「0.00 ms」) の検査

- Table 3 ドライバに計時コードは無い（`time` の使用は `datetime` による時刻刻印のみ。`perf_counter` / `synchronize` なし）。「0.00 ms」は定数 `0.0`（L176, L223）。
- LoRA が 0 オーバーヘッドとなる前提はマージ済み重みである。`lora_grpo_forge.py` にマージ実装は無く、`AttachedLoRALinear.forward`（L90-99）は有効時に毎回 fp32 へのキャスト + 2 回の行列積を追加する。マージしない限り 0.00 ms にはならず、これは未測定。
- Adaptive-BAOAB の「4.2 ms」は実装に対応する推論時ステアリングが存在せず、`rng.normal(4.21, 0.15)` の出力である。

## C. 「Deadzone escape」の算出元

- L284 の独自ルール（上表）。リポジトリの分類器は `compute_three_phase_qda_distance`（`manifold_geodesic.py:83-178`）の `phase_regime == "Phase_I_Deadzone"` で、`||p||_2 < sigma_crit_low - 0.05`（novel: 1.35 未満）。両者は別の基準で、論文の「Deadzone escape」がどちらを指すか文書上定義されていない。
- 実測の補足: CORPUS の V_Deadzone 8 件は全て分類器で `Phase_I_Deadzone`、II_Critical 10 件は全て `Phase_II_Critical`、III_Void 10 件も `Phase_II_Critical`（本セッションで実行して確認）。

## D. `activation_steerer.py` の監査

実モデルに対して動作する部分は、フック登録（L242-283）、`steer_tensor`（L185-220）、直交性検証（L285-298）。以下は実モデルで使う際の問題点。

1. 較正のモデル未指定経路（L154-162）は乱数の擬似隠れ状態を使い、`V_steer` はモデルと無関係になる。`model` と `tokenizer` を渡す経路（L141-153）のみが実データ。
2. 隠れ状態の添字がフック位置と 1 ずれる。フックは `layers[layer_idx]` の出力、つまり `hidden_states[layer_idx + 1]` に作用するが、較正は `hidden_states[layer_idx]`（L149-151）を使う。また較正は最後のトークンのみ（L151）、注入は全位置。
3. 既定コーパス 12 文（L112-125）では、中心化後のランクが最大 11 のため 12 列目が不定（L165-181）。`_ensure_orthonormal`（L71-86）の QR により 2 列目以降は前の列と混ざり、「軸 k = 第 k 要素」の対応が保たれない。
4. `d_model=1536`, `num_layers=28` が既定（L43-46）でモデルから自動取得されない。Qwen2.5-3B は 2048/36、Qwen3-8B は 4096/36。不一致のまま `reshape(-1)[:self.d_model]`（L140）で黙って切り詰められる。
5. 目標 `Δq*` を作る関数が無い。呼び出し側が与える必要がある。
6. `steer_tensor` は毎回 `torch.tensor(self.v_steer, ...)` でホストからデバイスへ転送する（L202-203）。層あたり毎トークンのオーバーヘッドになり、実測対象である。
7. テスト `test_steerer_forward_hook_live_injection_and_removal`（L454）等はモック層のみで、実 HF モデルのフック（デコーダ層が tuple か Tensor かはバージョン依存）は未検証だった。フック関数自体は tuple / Tensor の両方に対応している（L262-272）。

## E. `lora_grpo_forge.py` の監査

実装されている本物の経路: `run_authentic_grpo_lora_forge` の PyTorch 分岐（L383-539）。実生成、実 PPL（LoRA 無効）、実報酬、GRPO 損失、KL、AdamW 更新を行う。以下は問題点。

1. `trainable_params = [p for p in model.parameters() if p.requires_grad]`（L385）。HF モデルは既定で全パラメータが `requires_grad=True` のため、基底モデルを事前に凍結しないとモデル全体が最適化対象になる（3B/8B で OOM）。凍結は呼び出し側の責務。
2. ロールアウトの長さが 48 トークン固定（L417）で、評価用の長さと乖離する。
3. `log_softmax` / `cross_entropy` を bf16 のロジットに適用（L433, L471, L476）。bf16 では絶対値 16-32 付近の分解能が 0.125 程度で、KL 推定と比率が粗くなる。
4. 全ロールアウトの損失を `torch.stack` して 1 回 backward（L508-510）。プロンプト数 × G 個の計算グラフを同時保持し、8B では 24GB に収まらない。
5. シミュレーション分岐（L541-645）: 報酬は本物の関数だが、入力は固定文 4 件（L556-561）または渡された候補、PPL は `25.0 + 4.0*i`（L582）、対数確率は `-1.2 + 0.1*adv - 0.02*step`（L603）、パラメータ更新は乱数 `grad_sim`（L616-620）、損失は `-0.5*sum|adv| + beta*KL`（L625）。戻り値の `mode = "simulation_cpu"`（L638）は明示されるが、`mean_kl >= 0` は式 `exp(x)-x-1` の性質で必ず成り立つ。
6. `tokenizer` を渡さない、またはモデルが `nn.Module` で学習可能パラメータを持たない場合は、黙ってシミュレーション分岐に落ちる（L377-383）。
7. マージ機能が無い（B 参照）。
8. torch が入っている環境では `AttachedLoRALinear` がモック（NumPy 重み）でエラーになる。本セッションで torch を入れた環境で既存テストを実行し、`test_phase_c_grpo_and_steering.py` の 5 件が失敗することを確認した（原因: L60-62 の凍結ループ、L72-74 の dtype 取得、`attach_lora_to_model` の `p_count`、NumPy 入力の forward）。既存の CI（torch 無し）では発現しない。

## F. テストの検査

- `test_phase_c_grpo_and_steering.py::test_run_authentic_grpo_lora_forge_loop`（L211-245）は `tokenizer` 無しのモックで呼ぶためシミュレーション分岐を通る。`final_mean_reward > 0` と `mean_kl >= 0` は、固定文と式の性質で必ず成り立つ。PyTorch 分岐は実モデルでは未実行だった。
- `test_colab_experiment_c_pipeline.py`（L97, L102, L107, L121, L132, L143, L182）は mock 生成器の設計結果（Base は Deadzone に残る、他は escape する、overhead が 0.0 / 3.5 超）を assert しており、期待値の固定になっている。

## G. 変更内容（本コミット）

`lora_grpo_forge.py`（既定動作は維持）:
- `run_authentic_grpo_lora_forge` に追加引数: `max_new_tokens=48`, `batched_rollouts=False`, `per_rollout_backward=False`, `prompts_per_step=None`, `progress_fn=None`。ステップごとの `sec` をログに追加。
- 認証パス内の `log_softmax` / `cross_entropy` を fp32 に変換（E-3）。fp32 モデルでは数値が変わらない。
- `AttachedLoRALinear`: 凍結ループと dtype 取得を torch.Tensor のみに限定、NumPy 入力の forward を追加、`attach_lora_to_model` の `p_count` を torch でも動作させる（E-8）。

追加:
- `python/colab/run_phase28_phase_c_real.py`（実モデル測定ドライバ）。
- `artifacts/colab_run/build_colab_p28c.py`（自己完結ペイロード生成）。生成物 `p28c_*.py` は `.gitignore` に追加。
- `python/tests/test_phase28_phase_c_real_driver.py`。

未変更: Table 3 ドライバ、`activation_steerer.py`、論文節、`artifacts/table3_lora_grpo_forge_results.json`。論文の Table 3 を mock 由来と明記して差し替える作業は別途必要（本コミットでは触れていない）。

## H. 実モデルドライバの設計

`python/colab/run_phase28_phase_c_real.py --model-id Qwen/Qwen2.5-3B-Instruct`（`Qwen/Qwen3-8B` も bf16 で可）。

- 入力: 手書きの汎用書き出し 12 件（評価）と、重ならない 12 件（GRPO 訓練）。作家名、作品名、登場人物名を含まない（テストで CORPUS の author/title との部分一致を検査）。生成はチャットテンプレート無しの続き書き、サンプリング（temperature 0.8, top_p 0.95）、プロンプトごとのシードを全条件で共通化。
- 指標（全て生成テキストから算出）: 抽出器の `len_cv`（CV_L）、`cadence_alive`、`order_params`、`||p||_2`、QDA の `phase_regime`、`in_deadzone = (phase_regime == "Phase_I_Deadzone")`、GRPO 報酬（PPL なし）、distinct char bigram、q/p 12 次元ベクトル。ドライバ独自のしきい値は無い。
- (a) base: プロンプトごとに `cuda.synchronize` で括った `generate` の壁時間、生成トークン数、ms/token。計測前に 16 トークンのウォームアップ（統計に含めない）。
- (b) ステアリング: 層 `floor(0.65 * L)`。`V_steer` はモデル自身の隠れ状態（`hidden_states[layer_idx+1]` のトークン平均、位置 0 を除外）を CORPUS 40 件に対してリッジ回帰（モジュールの `calibrate_12_element_axes`）。`Δq*` = CORPUS の II_Critical と III_Void の平均 q から (a) の生成の平均 q を引いた実測差。`alpha_act ∈ {0.05, 0.15, 0.30}` のスイープ、同じ `Δq*` をランダム直交方向に適用する対照条件、(a) との対応ありの差分（`paired_vs_base`）を記録。
- (c) GRPO-LoRA: 基底モデルを凍結、`attach_lora_to_model`（q_proj, v_proj, r=8, alpha=16）、学習可能パラメータ数が LoRA パラメータ数と一致することを検査、`run_authentic_grpo_lora_forge`（既定 40 ステップ、ステップあたり 2 プロンプト × G=4、ロールアウト 96 トークン、バッチ化、ロールアウトごとの backward）。`mode != "authentic_pytorch"` なら例外にする（シミュレーションを受け入れない）。学習後は LoRA をマージした重みで評価プロンプトを生成。
- レイテンシ: 固定トークン数（`min_new_tokens = max_new_tokens = 64`）の貪欲生成を、状態を巡回させて 5 回インターリーブ測定。比較は {base, steer}、{original, lora_unmerged, lora_merged}。状態切替は `LoraStateSwitcher`（original は重み複製から復元、merged は W + scaling·B·A）。
- 整合性チェック（`checks`）: フックによるロジット変化量、フック除去後のロジット差（0 であるべき）、LoRA マージ前後のロジット差、復元後と学習前のロジット差（0 であるべき）、LoRA 効果の KL。
- 失敗の扱い: 例外は該当プロンプトまたは該当ステージの `error` / `errors` に記録し、既定値で置換しない。モデルロード失敗もアーカイブ行に `fatal_error` として出力する。
- 出力: 単一 JSON（全プロンプトの生値 + 要約統計 + Wilson 区間）、最終行 `ARCHIVE_B64 <crc32hex> <base64(gzip(json))>`（`build_colab_p27.py` BODY 末尾と同形式）。

## I. 検証結果と限界

- `cd python && uv run pytest tests -q`: 553 passed, 7 skipped, 0 failed。この環境（torch 無し）では、新規テストのうち torch を使う 2 件は skip。ワークツリーに node_modules が無く、`test_phase_d_usir_forge.py` の 2 件は最初失敗したため、`pnpm install --frozen-lockfile` と `pnpm -C packages/usir-core build`（いずれも gitignore 対象）を実行して解消した。
- 別の一時 venv（CPU 版 torch + transformers）で、ランダム初期化の小型 Qwen2（hidden 64, 4 層）と文字トークナイザにより全ステージを CPU で実行するスモークテストを通した（新規テスト 7 件、既存 Phase C 関連 22 件を含め 29 passed）。これはコードパスと整合性チェックの確認であり、実モデルでの効果や速度については何も示さない。
- GPU 実行、実 Qwen モデルのロード、実モデルでの HF フック挙動（デコーダ層の戻り型）は、本セッションでは未検証。
- 実行時間の見積もり（未実測の推定）: Qwen2.5-3B / L4 でモデル取得とロードに 2-4 分、(a) 約 2 分、(b) の 4 条件で約 8-10 分、(c) の学習 10-20 分 + 評価 2-3 分 + レイテンシ測定数分、合計 30-45 分程度。Qwen3-8B は A100 40GB で同程度から 1.5 倍程度、L4 24GB では GRPO ステージの余裕が小さい（重み 16.4GB + 1 グラフ分）。
- n=12 は小標本であり、レート差の統計的有意性は主張できない。`paired_vs_base` は符号の個数と平均差のみを記録する。
- ステアリング軸の同一性（D-3）は修正していないため、`V_steer` の第 2 列以降は q の各要素に対応しない。効果は生成物の q と指標で測る。

## 次の一手

- `python artifacts/colab_run/build_colab_p28c.py` で生成し、`python scripts/run_colab_file.py artifacts/colab_run/p28c_q25_3b.py docs/handoff/v9 --gpu L4 --timeout 5400` で実行。結果 JSON を読んでから、論文 Table 3 の mock 由来記述の扱いを決める。
