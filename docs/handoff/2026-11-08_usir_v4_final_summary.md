# USIR v4.0 開発総括（Phase 18〜49、最終）

作成日: 2026-11-08。USIR v4.0 の開発はこの文書をもって完了とする（ユーザー指示 2026-11-08）。出典は `docs/handoff/` 配下のファイルと `git log`。表の各行・各項目に出典ファイル名を付ける。
`docs/handoff/` は以下 `h/` と略す。数値は各ファイルの記載をそのまま写した。「未確認」は docs から確認できなかった項目。

## 1. 構成

- パッケージ（`packages/`、実在を ls で確認）: `usir-core`、`usir-mcp`、`usir-bridges`、`usir-wasm`。Python は `python/usir_py`（uv 管理）。
- Python の主要モジュール（`python/usir_py` の ls より）:
  - 抽出・解析: `extractor_24d.py`、`fisher_atlas.py`、`style_atlas.py`、`orthographic_prior.py`、`genome_fit.py`、`linguistic_gates.py`
  - 生成: `hsmm_longform_director.py`、`bidirectional_bridge.py`、`activation_steerer.py`、`realtime_decoder.py`、`text_format.py`、`text_metrics.py`
  - 学習・評価: `lora_grpo_forge.py`、`grpo_reward.py`、`pairwise_judge.py`
  - 実験ドライバ: `python/colab/run_phase33_phase_c.py` ほか。Colab 実行は `scripts/run_colab_file.py`、割り当て確認は `scripts/colab_assignments.py`
- テスト件数
  - Python: 599 passed / 18 skipped（2026-11-08 実測、`uv run pytest`。torch 依存テストは `uv run --with torch --with transformers` で別途 38 passed）。
  - Python の docs 上の最新値: 595 passed / 11 skipped（2026-11-06, Phase 37）。出典 `h/2026-11-06_usir_v4_true_completion_report.md`（追補 Phase 37）。
  - TypeScript: 378 passed（36 files、`pnpm test`、2026-11-08 実測）。
  - 推移: 開始時（`0764c82`）436 passed / 4 skipped、pnpm 364（`h/2026-11-05_usir_v4_complete_final_report.md`）→ Phase 22 時点 486/4、pnpm 371（同）→ Phase 29 時点 562/7、pnpm 378（`h/2026-11-06_usir_v4_grand_completion_report.md`）→ Phase 35 時点 585/11、pnpm 378（true_completion_report §6）。
- 論文: `usir_v4_paper.md` は `scripts/build_paper_v4.py` が実 JSON から生成。登録数値は Phase 35 で 228 件、Phase 37 で 402 件が一致（`h/2026-11-06_usir_v4_true_completion_report.md`）。v2/v3 論文は撤回バナー付き。
- cleanroom スキャン: PASS 195 / WARN 5 / FAIL 0（200 件中）。WARN 5 は Set A の禁止語警告で、中身は未精査（同ファイル §6）。

## 2. フェーズ別の成果表

| Phase | 内容 | 主な実測結果 | 限界・未達 | 出典 |
|---|---|---|---|---|
| 18 | 原典仕様の監査、ゲノムによる条件付き運動量（Py/TS パリティ）、article/cot の QDA 較正、青空文庫取得 | 監査 53 項目: 実装済み 12 / 簡略化 36 / 未実装 3 / 意図的不採用 2。QDA 帯域 article 0.30〜1.40→1.03〜2.07、cot 1.35〜2.65→0.86〜2.12 | cot は 48 チャンク・単一モデルで推定精度が低い。article は和辻 2 作の比重が大きい。Tier-M の消費者は未接続だった | `h/2026-11-05_phase18_spec_audit.md`、`h/2026-11-05_usir_v4_complete_final_report.md`、`h/2026-11-05_phase18-22.md` |
| 19 | ミクロ日本語ゲート（漢字形容詞の連用止め、指示語減衰、短文後の名詞止め抑制） | 40 チャンクの原典で規則 1 の誤爆 0 件。規則 3 は 36 文中 1 文（2.8%）に影響 | 規則 1 は原典に該当例がなく、捕捉率は測れていない | `h/2026-11-05_phase19_micro_japanese.md`、complete_final_report |
| 20 | 10 作家・46 作品・5,528 窓の 24 次元アトラス、MW2、LOWO | LOWO 正答率 scaled 0.543 / legacy 0.413（偶然 0.100）、top-3 0.739 / 0.717 | 谷崎・安吾は 0 件正答。窓の 99.8% 以上が 1 モード。82〜99% が II_Critical で相の判別力は低い | `h/2026-11-05_aozora_style_atlas.md`、complete_final_report |
| 21 | scene_beats、style_genome、ΔLL、短編生成（Colab L4、3B と 7B、計 18 本） | 統計ゲート 18/18 通過。B ステアで ΔLL が ctrl より低下（5 組すべて） | A ステアで ΔLL は上がらない。A_legacy は A と出力一致。Judge は天井（7B 自己採点 93.8）。目視で造語・反復崩壊・連体形直後の「。」・筋の不整合を確認 | complete_final_report |
| 22 | 整理、最終報告 | pytest 486/4、pnpm 371。cleanroom FAIL 0 | 未完了 8 項目（§4 参照） | complete_final_report |
| 23 | 句読点・リズム（steady/pulse）・改行・造語・モデル選定 | 11 モデルの制御なし比較。第一候補 Qwen3-Swallow-8B（平均文長 31〜37 字）。造語の原因を 3 段階で特定し 2 段階を修正 | 造語の残存③（段落間 n-gram 照合）は未修正・原因は未検証。3B 以下は制御に不向き。new_B の CV_L が高い（0.43〜0.62） | `h/2026-11-06_phase23_report.md`、`h/2026-11-05_phase23_rhythm.md` |
| 24 | 候補の一括採点とプロンプト KV 共有（kv_share） | Qwen3-8B / L4: 309.1 s → 173.7 s（1.78x、kv_fallbacks 0）。一致誤差 fp32 1e-5、bf16 0.035 | 全機能が既定オフ。最初の試行は 176 回フォールバック（run1 として保存） | `h/2026-11-06_usir_v4_grand_completion_report.md`、`git log`（3c6644c） |
| 25 | 造語防止、grace_close、直喩減衰、引用符整形 | 両モデル 40 本で造語 0・途中切れ 0。Swallow の「まるで」10.0→1.0 本/話 | 造語検出は固定パターンの正規表現で、網羅的ではない | grand_completion_report §3 |
| 26 | quantile 3 モード Forge、モード別 G_style、TS 側 OPEN-01/02/04 移植 | LOWO scaled 0.522（rule 0.543）、legacy 0.413。G の非対角最大 scaled 0.287 / legacy 0.006 | モード比率は構成上ほぼ固定（0.56/0.25/0.19）。n=46・標準誤差約 0.07 で差は有意でない | `h/2026-11-06_phase26_forge_modes.md`、`h/2026-11-06_phase24-29.md` |
| 27 | Swallow-8B / Qwen3-8B × 2 台本 × 2 seed × 5 条件（L4、各 20 本） | 造語 0、途中切れ（new_* 条件）0、「まるで」≤2、scaled≠legacy 8/8 | **ΔLL の方向分離は未達**（Swallow gA 2.28 対 gB 2.25、Qwen3 は gB 7.35 > gA 6.61）。Qwen3.5-9B は未実行 | grand_completion_report §3 |
| 28 | ペアワイズ Judge、Phase C 監査と実機計測（Qwen3-8B、A100） | Judge の先頭選択率 0.78（両 Judge）。Phase C: base Deadzone 0/12、steer α0.3 で 6/12、LoRA merged 3/12。遅延 steer +0.07 ms/tok、merged +0.19、unmerged +12.99 | 論文 Table 3（mock）の「Deadzone 脱出 100%」は再現せず（base が 0/12）。8B Judge は条件差の測定に使えない | grand_completion_report §4〜5、`h/2026-11-06_phase28_phase_c_audit.md` |
| 29 | 検証と報告 | pytest 562/7、pnpm 378、cleanroom FAIL 0 | 論文 Table 3 の本文は未改稿のまま（当時） | grand_completion_report §6、§8 |
| 30 | 書記統計 27 次元＋クラス均衡 Fisher の LOWO | top-1 0.674（31/46）、top-3 0.870。GMM ゲノム 0.522、Fisher z24 のみ 0.565 | 目標 0.70 未達。谷崎 1/5、安吾 1/4。正則化 1e-3〜1e-1 で 30〜32/46 | `h/2026-11-06_phase30_atlas_fisher.md`、true_completion_report §1 |
| 31 / 31b | ゲノム→proposal 結合、芥川 A と谷崎 B の双方向ステア（A100、Qwen3-8B と Swallow-8B） | ΔF_AB は gA>ctrl>gB が 6/6（3 構成すべて）。造語 0・途中切れ 0 | **文長は A/B で分離しない**（gA>gB 2/6〜3/6）。ΔF はステアと同じ特徴空間の指標。Swallow の gB で漢数字の年号が 3 seed に漏れた | `h/2026-11-06_phase31_genome_proposal.md`、true_completion_report §2 |
| 32 | logit 脱バイアス Judge（交差採点、A100 218.9 s） | 位置バイアス 2.0〜6.7 nats。脱バイアス後の選好 \|·\|≤0.6 | 符号が条件間で一貫しない。Qwen3 Judge は回答質量 0.017 で信頼できない。品質差は主張不能 | `h/2026-11-06_phase32_logit_judge.md`、true_completion_report §3 |
| 33 | Phase C 再実測（instruct 説明文 12 本、400 トークン） | base Deadzone 7/12。較正 steer(last, α0.15) 1/12 対ランダム 10/12。merged latency −0.17% | 較正側は ref PPL 2.70（base 1.55）で 3/12 が反復に退化。GRPO 報酬は上昇せず（先頭 10 step 0.081→末尾 0.077）、merged LoRA の Deadzone 8/12（base 7/12）で不成立 | `h/2026-11-06_phase33_phase_c.md`、true_completion_report §4 |
| 34 | 論文を実 JSON から再生成、数値検証、モック除去 | 228 数値すべて一致。v2/v3 の Table 1・4 は合成コーパス、Table 2・3 はモックと判明し撤回 | Table 1/4 の撤回は「合成で AUC を構成上較正していた」ため | `h/2026-11-06_phase34_paper_provenance_audit.md`、true_completion_report §5 |
| 35 | 検証・報告 | pytest 585/11、pnpm 378、cleanroom FAIL 0 | LOWO 0.70、文長分離、merged LoRA による Deadzone 低減の 3 目標が未達（報告時点） | true_completion_report §0、§6 |
| 36 | 地の文統計 4 次元を追加、文長 A/B、年号マスク、単位方向ステア、shaped GRPO | LOWO 34/46 = 0.739（top-3 0.913）。単位方向ステア Deadzone 5 対 7、1 対 7、退化 0、PPL 最大 1.873（base 1.551） | LOWO は同じ 46 作品の診断後の設計で保持データ検証ではない。文長 gB−gA 5.75 字で基準 10 字未達。年号 1 件漏れ（基準 0）。shaped GRPO は 7/12 対 7/12 で未達 | `h/2026-11-06_phase36_pinpoint.md`、true_completion_report 追補 |
| 37 | ローカル再解析（Colab 不使用） | steady 段落のみで gB−gA 平均 16.35 字。末尾断片を除くと較正 α0.12 の Deadzone は 1/12→5/12（ランダム 9/12、base 7/12） | 結果を見た後の分割で、事前基準（5.75 字）は未達のまま。新実装（〇マスク、band_ramp、strip_tail）は単体テストのみで効果未測定 | `h/2026-11-06_phase37_local_reanalysis.md` |
| 38 | 事前登録: 38-1 切り詰めなしのステア再評価（n=36）、38-2 保持 30 作品の作家識別 | 38-2: 27/30 = 0.900（基準構成 26/30）で達成。38-1: Deadzone 較正12 対ランダム27 対 base21、3/3 系列で成立 | 38-1 は退化 1 件（bigram 0.496）で総合未達（4 基準中 3 達成）。38-2 は学習が 46 作品全部で LOWO と直接比較不可 | `h/2026-11-06_phase38_preregistration.md`、true_completion_report 追補 |
| 39 | 文学の場面プロンプトでのステア再評価（事前登録） | 全条件で Deadzone 0/36。退化 0、PPL は基準内 | 床効果で base 比較は判定不能、較正<ランダムは 0 対 0 で未達 | `h/2026-11-06_phase39_preregistration.md` |
| 40 | CV_L を主指標とした文学プロンプトでの再評価（4 系列、n=48） | 較正 α0.12 − base +0.020（p 0.071）、− ランダム +0.0004（p 0.57） | 未達。Phase 39 の探索での差は再現せず | `h/2026-11-06_phase40_preregistration.md`、session_phase38-42_handoff |
| 41 | 第 2 保持組 30 作品での作家識別（方式 A 作家平均、方式 B 最近傍作品） | A 23/30、B 24/30。2 組合計は両方式とも 50/60（= 0.833） | B の基準は 1 作品で満たすが採用せず。後期芥川作品は読点が少なく誤識別（診断） | `h/2026-11-06_phase41_preregistration.md`、session_phase38-42_handoff |
| 42 | 末尾除去の shaped 報酬による GRPO（40 step、事前登録） | 報酬 0.226→0.259（+0.033、並べ替え p 0.18）、Deadzone 6/12 対 7/12、退化 1 | 未達。学習の効果も末尾仮説も確認・否定できない（1 系列）。報酬は主に CV_L と cadence_alive が決める | `h/2026-11-06_phase42_preregistration.md`、session_phase38-42_handoff |
| 43 | 入力者の表記慣習との交絡検査（60 作品、事前登録） | 入力者共有あり 24/31 = 0.774、なし 26/29 = 0.897（差 −0.122）。校正者 15/18 対 35/42 | 判定「交絡の兆候なし」だが n が小さく、交絡がないことの証明ではない | `h/2026-11-06_phase43_preregistration.md` |
| 44 | 評価生成のバッチ化（`--gen-batch`）、A100 を許可制に | 同一作業量の生成時間 A100 バッチ1 1413 s → L4 バッチ12 233 s（約 6.1 倍）。エラー 0、上限到達 0 | バッチ化で乱数列が変わり過去値と直接比較不可。最長行まで計算が続く無駄が約 10% | session_phase38-42_handoff（2026-11-07 追記）、`git log`（5b7c946） |
| 45 | 速度検討（モデル・バッチ別スループット、mixed-steer、フック負荷） | L4 Qwen3-8B 15.0 tok/s（B=1）→158.7（B=12）→270.7（B=24）→460.3（B=48）。mixed-steer でステア段 187 s→130 s（1.44x）、ジョブ 422→361 s。フック負荷 ±1% 以内 | static cache は遅い（85.0 対 75.6 ms/step）、`torch.compile` は CUDAGraphs エラー。小型モデルは較正やり直しのため不採用 | session_phase38-42_handoff（Phase 45 節）、colab_speed_cu_report |
| 46 | 複数 seed を 1 ジョブ化、CU 実測 | Phase 40 全量（240 本）を L4 1 ジョブで 926 s、0.48 CU（実測、1.94 CU/h）。A100 の 6593 s から 7.1 倍短縮 | 変更前の CU は記録なし（推定のみ）。GRPO は `--seeds` 非対応 | `h/2026-11-07_colab_speed_cu_report.md` |
| 47 | GRPO を L4 で実測（Phase 42 と同設定） | 2,267 s、1.13 CU（実測）、44.0 s/step（A100 37.9）、ピーク 18.7 GB、エラー 0。最終平均報酬 0.2655、mean KL 0.0104 | 評価サンプルがバッチ化で Phase 42 と別物のため品質は直接比較しない | colab_speed_cu_report §4 |
| 48 | 参照 forward の共有（`share_ref_forward`） | L4 6 ステップ A/B: 44.35→42.66 s/step（−3.8%）、ピーク 18.69→18.01 GB。ベンチ 698 s、0.29 CU。CPU パリティテストあり | GPU ではビット一致でなく統計的同等（2 ステップ目以降分岐、bf16 丸め差と推定）。40 step で約 68 s・約 0.04 CU の削減で小さい | colab_speed_cu_report §5 |
| 49 | GRPO ロールアウトの合算（`merge_step_rollouts`、`--grpo-merge-rollouts`） | L4 6 ステップ A/B: 43.58→25.96 s/step（−40.4%）、ピーク 18.03 GB で同じ。ベンチ 570 s、0.19 CU | サンプルが変わるので統計的同等。報酬の差は n=6 ステップで比較しない。40 ステップの本番 GRPO では未実行 | colab_speed_cu_report §6 |

## 3. 実験で確認できたこと / 確認できなかったこと

### 3.1 確認できたこと
- 作家識別（作品単位の保持）: LOWO 0.739（34/46、同一 46 作品の診断後設計）。保持 60 作品で 50/60 = 0.833（Phase 38: 27/30、Phase 41: 23/30）。偶然水準 0.100。出典 true_completion_report、session_phase38-42_handoff。
- 入力者の表記慣習との交絡は、60 作品の範囲で兆候なし（Phase 43、ただし n が小さい）。
- 生成の制御が効いた項目: 造語 0、途中切れ段落 0、「まるで」≤2 本（Phase 25/27、固定パターン検出）。ΔF_AB は 3 構成で gA>ctrl>gB が 6/6（Phase 31）。出典 grand_completion_report、true_completion_report。
- 速度: 生成のバッチ化と L4 移行で、同一作業量の実行時間が 7.1 倍短縮し、CU は 0.48 と実測（Phase 44〜46）。フック自体の速度低下は ±1% 内で検出されず。GRPO は L4 で動作し（Phase 47）、1 ステップは参照 forward 共有で −3.8%、ロールアウト合算で −40.4%（Phase 48〜49）。出典 colab_speed_cu_report、session_phase38-42_handoff。
- merged LoRA の推論遅延はほぼ 0（Phase 28: +0.34%、Phase 33: −0.17%）。unmerged は +23% 前後。出典 grand_completion_report §5、true_completion_report §4。

### 3.2 確認できなかったこと・否定的結果
- **ΔLL のゲノム方向分離（Phase 27）**: 未達。符号が一貫しない。Phase 31 でも ΔLL は部分的、文長は不成立（grand_completion_report、true_completion_report）。
- **文長の A/B 分離（Phase 31/31b/36）**: 未達。gB−gA 5.75 字（基準 10 字）。steady 段落のみの事後分割で 16.35 字だが、事前基準は満たさない（Phase 37）。
- **LOWO 0.70（Phase 30）**: 未達（0.674）。Phase 36 で 0.739 に達したが保持データ検証ではない。
- **論文の旧 Table 3 の主張**: 「Deadzone 脱出率 100%」は再現せず。ステア・LoRA は Deadzone を増やした（Phase 28.2）。旧 Table 1/4 は合成コーパス、Table 2/3 はモックで、撤回（Phase 34）。
- **LLM Judge**: 8B Judge は位置バイアス（先頭選択率 0.78）で条件差を測れない。脱バイアス後も符号が一貫しない（Phase 28、32）。7B 自己採点は天井 93.8（Phase 21）。
- **較正ステアの効果**: Phase 33/36 では Deadzone が較正<ランダムだったが、Phase 37 で末尾断片除去後に効果が縮小（1/12→5/12）。Phase 38-1 は退化 1 件で未達。Phase 39 は床効果で判定不能。Phase 40 は較正−ランダム +0.0004（p 0.57）で未達。
- **Phase 38-1 の Deadzone 減少の内訳（探索）**: 主に固定の抽象語リストの語が減ったことによる（「重要」51→15、ランダム 50）。CV_L はほとんど変化せず。結果を見た後の分析。出典 session_phase38-42_handoff。
- **GRPO（Phase 33/36/42）**: 報酬が有意に上がらない。Phase 33: 0.081→0.077、Phase 36: 0.286→0.281、Phase 42: +0.033（p 0.18）。merged LoRA の Deadzone は base 以下にならず（Phase 33: 8/12 対 7/12、Phase 36: 7/12 対 7/12、Phase 42: 6/12 対 7/12 で差の有意性は未検証）。
- **Phase 41 方式 B（最近傍作品）**: 基準は満たすが方式 A と 2 組合計で同数（50/60）のため不採用。
- 事前登録で不支持だった仮説: Phase 36（文長、年号、shaped GRPO）、38-1（総合）、39（判定不能）、40、42。
- 生成物の質: 筋をたどる短編にはなったが「本格短編の水準」ではない（Phase 21）。「〜かのように」の定型句、人物の出入り、年代整合の破綻が残る（Phase 27 の目視、grand_completion_report §8）。

## 4. 未完了・既知の問題（docs が未完了と書く項目）

### 4.1 Phase 22 時点の未完了（`h/2026-11-05_usir_v4_complete_final_report.md`）
1. 連体形直後の「。」を抑える統語ゲート。
2. 反復ペナルティから複合語を除外し造語を防ぐ。
3. 統計ゲートへの反復崩壊（n-gram 反復）指標の追加。
4. 天井効果のない Judge（別系統モデル、対比較、人手評価）。
5. ゲノム精度行列を生成へ反映する経路（`momentum_from_summary(genome=)` が生成経路から未呼び出し）。
6. モード分類器が 1 モードに偏る問題。
7. 監査表の簡略化項目（`G_style`、HSMM 遷移と well 中心の実データ較正、TS 側ゲート移植）。
8. Table 1/2 の再完遂、GRPO/LoRA、多重解像度特徴抽出器の較正。
- 状態の注記: 6 は Phase 26 の quantile 割当で一部対応（LOWO 0.522）、5 は Phase 26 の `genome_modes` で経路を追加。TS 側ゲートは Phase 26 で移植（`git log` 728fb01）。ただし各項目を解消済みと確認した記述は未確認。

### 4.2 仕様監査の残り（`h/2026-11-05_phase18_spec_audit.md`「未完了（優先順）」）
- Tier-M ゲノムの消費者（一部 Phase 21/26 で対応）、Fisher 計量の本番結線、TS 側の欠落（Ghost History とベイズ収縮、BAOAB 摩擦、Toulmin/統一場）、Phase A 仕様の正本決定（S10 か B10）、H_syntax の実エントロピー化、不応期の ‖ṗ‖² 条件付き化、Activation Steering の較正コーパス（自作 12 文）、OPEN-04 の接続、Tier-M 署名 SHA-256・Wasm ビルド・Judge の LLM 化・`EQ_GROUPS` の確認。

### 4.3 アトラス・モード（`h/2026-11-05_aozora_style_atlas.md`、`h/2026-11-06_phase26_forge_modes.md`）
- 3 モード退化（窓の 0.998〜1.000 が Mode 2）、5 相診断の分解能、LOWO が単一分割（作家 10 名・46 作品）、作家ごとの窓数不均衡（梶井 96〜漱石 1726）、article 実測で和辻 2 作が 82%。
- 会話シグナルは括弧文字の比率のみ。作家内クラスタリング未実施。TS の `OSWEStreamingCompiler` に `mode_assign` 未追加。

### 4.4 生成・リズム（`h/2026-11-05_phase23_rhythm.md`、`h/2026-11-06_phase23_report.md`）
- 造語の残存③（段落間 n-gram 照合、原因は未検証）、new_B の CV_L が高い、3B の会話書式（半角 `"…"`）で `format_novel` が誤動作、連体形疑いの正解ラベル付き評価、閾値較正（青空文庫文長分布との照合）、`format_novel` の目視評価、Qwen3.5-9B の制御あり実行、段落境界での文切れの調査。

### 4.5 Phase 24〜29（`h/2026-11-06_usir_v4_grand_completion_report.md` §8）
- ΔLL が作家方向に動かない原因（候補 3 つ、いずれも未検証: G_style が二次形式の重みだけを変える／目標重心 q* の離れ不足／600 字窓 ΔLL の感度不足）。
- 8B Judge の位置バイアス（より大きい Judge か人手評価が必要）。Qwen3.5-9B 未実行。論文 Table 3 の本文改稿（Phase 34 で実データから再生成済みのため解消の可能性。確認は未）。

### 4.6 Phase 30〜37（`h/2026-11-06_usir_v4_true_completion_report.md` §8〜9・追補、`h/2026-11-06_phase33_phase_c.md` §5、`h/2026-11-06_phase37_local_reanalysis.md` §2）
- 年号漏れ（「〇」のバイト単位トークン）の再実行。Phase 37 の `band_ramp`・〇マスク・`strip_tail` は Colab 未実行（効果未測定）。
- gB の文長不足の段落別診断。GRPO の報酬が上がらない原因（step 数・lr・群内 advantage の分散は未検証）。
- ステアの退化検出（bigram ガード）なし。Phase C は 400 トークンで評価が切れる。n=12 前後で有意性は主張しない。
- ΔF がステアと同じ特徴空間の指標である点、ランダム方向が 1 系列に 1 本である点（`h/2026-11-06_session_phase38-42_handoff.md`）。

### 4.7 Phase 38〜43（`h/2026-11-06_session_phase38-42_handoff.md`）
- A100 ランタイム 2 件（`gpu-a100-s-usc1b0-3kyhjzqzbijl3`、`gpu-a100-s-usw4b0-167gj4qwekq29`）の手動解放。2026-11-06 23:08 の記録では残存。2026-11-07 追記では「2026-11-07 時点で 0 件」（同ファイル、A100 許可制の節）。解放時点の経緯は未確認。
- `1b6444b`（解放前のトークン再取得）の長時間ジョブでの動作は未確認。keepalive スレッドが開始時トークンを使い続ける点（1 時間超で問題の可能性）は未確認。
- 今後の候補（未実施）: 対照の拡張（−Δq、Δq の置換、ランダム方向 8 本の null 分布）、抽象語リスト外の指標（55 次元 Fisher）での採点。

### 4.8 速度・CU（`h/2026-11-07_colab_speed_cu_report.md`）
- GRPO の 1 ステップの内訳（生成、forward/backward）は直接計測していない。Phase 49 の差から生成が約 35 s/step 以上と推定。`--grpo-share-ref --grpo-merge-rollouts` を付けた 40 ステップの本番 GRPO ジョブは未実行。
- A100 の実レートは未計測（許可なしに割り当てない方針）。CU レートは Colab 側で変わりうる。
- Phase 45 のベンチと Phase 44 ジョブの CU は記録を確認できなかった（未確認）。

### 4.9 Gemini 相談記録（`h/2026-11-07_gemini_discussion_record.md`）
- 46 作品のうち 32 作品の作家内訳、マクネマー検定、機能語 JSD の実装・採用可否（事前登録が先）、Phase 36-2/36-3 の結果ファイル。
- 注記: 同ファイルは 2026-11-07 付で「Phase 36-2/3 の結果ファイルなし」と書く。一方 `git log` と true_completion_report には 36-2/36-3 の結果が記録済み（8efc643、4458e41）。同ファイルの当該項目は古い可能性が高い。

## 5. 計算資源

### 5.1 実測（CU 記録あり、L4）
| ジョブ | 実行時間 | CU | 出典 |
|---|---|---|---|
| Phase 46 4 系列 1 ジョブ | 926 s | 0.48（228.73→228.24） | colab_speed_cu_report §2 |
| Phase 47 GRPO | 2,267 s | 1.13（228.24→227.11） | 同 §4 |
| Phase 48 A/B ベンチ | 698 s | 0.29（227.11→226.83） | 同 §5 |
| Phase 49 A/B ベンチ | 570 s | 0.19（226.83→226.63） | 同 §6 |
- 実測合計 2.09 CU（上の 4 行の和）。L4 のレート 1.94 CU/h（実行中の ccu-info）。残高は 226.63 CU（Phase 49 後、割り当て 0 件を確認）。
- Phase 44（422 s）、Phase 45 のベンチの CU は未記録（未確認）。

### 5.2 CU 未記録の過去実行（推定。実測ではない）
- L4（Phase 21）: 合計 5,850 s（失敗含む。cot 63+509、p21_3b_b 1,044、p21_7b 2,267、p21_3b_a 1,967。割当失敗 412 は 0 s）。`h/2026-11-05_usir_v4_complete_final_report.md`。
- L4（Phase 27）: p27_q3_8b 4,724 s、p27_swallow_8b 5,322 s。`h/2026-11-06_usir_v4_grand_completion_report.md` §7。
- A100 の実行時間（記録分）: Phase 28 p28j 347 s、p28c 1,884 s、Phase 31〜33 計 15,727 s（3,803 + 4,404 + 218.9 + 1,945 + 5,356）、Phase 36 計 7,715 s、Phase 38 計 4,534 s、Phase 39 計 4,803 s、Phase 40 計 6,593 s、Phase 42 の p42g 2,460 s。出典は各報告の Colab 台帳。
- 上記 A100 合計は本ドラフトでの算出で約 44,063 s（約 12.2 時間）。公開レート 5.40 CU/h（McCormick, 2026-03 測定）なら約 66 CU、10.59 CU/h（2024 年の比較記事）なら約 130 CU。レートは `colab_speed_cu_report` §2 の仮定を流用した。算出は本ドラフトによるもので docs には合計値の記載がない。
- 上の合計に含まれないもの: 強制停止した p31 の 2 回（時間計上不能）、割当失敗の試行（ReadTimeout、ConnectionError、412 は GPU 時間を消費しないとされるが、ReadTimeout 後のランタイム残存の有無は未確認）、解放されず残った A100 ランタイム（Phase 40 の s2・s4、Phase 42 の p42g。残存時間の記録なし）。`colab_speed_cu_report` は「1 件が 1 時間残ると 5〜11 CU」と記す。
- Phase 40 の比較: A100 6,593 s は推定 9.9〜19.4 CU（5.40 / 10.59 CU/h 仮定）。L4 実測 0.48 CU で約 21〜40 倍の差。Phase 42 の A100 CU は推定 3.4〜6.7（colab_speed_cu_report §4）。

### 5.3 方針の変更
- 2026-11-07: ユーザー指摘（CU の消費が想定外に速い）を受け、A100 を禁止し、その後ジョブごとの許可制（`--allow-a100`）に変更。割り当て確認は `scripts/colab_assignments.py`（読み取り専用）。出典 session_phase38-42_handoff、`git log`（a43147a、0689e7e）。
- 残高で回せる量の見積り（L4 実測ベース）: 約 1,900 系列、約 114,000 本、約 118 L4 時間（残高 228.24 CU 時点、colab_speed_cu_report §3）。残高 226.63 CU では約 117 L4 時間。GRPO 40 ステップは Phase 47 の方式で 1.13 CU/回（合算・共有後は未実測。Phase 47 の 2,267 s から学習短縮分 (44.0−25.96)×40 ≈ 722 s を引いた約 1,545 s × 1.94 CU/h で推定約 0.83 CU/回）。

## 6. 資料間で見つけた不一致
- Phase 28 の Judge 実行時間: 316 s（grand_completion_report §4）と 347 s（同 §7 の台帳）。
- Phase 42 の A100 ジョブ時間: ジョブ全体（kernel 内）2,290 s、ランナー記録 2,460 s（colab_speed_cu_report §4 と true_completion_report 追補）。測定範囲の違いで、矛盾ではない可能性が高い。
- Phase 18 終了時の pytest: 449 passed / 4 skipped（`h/2026-11-05_phase18-22.md`）。complete_final_report は Phase 22 終了時 486 passed / 4 skipped。時点が異なる。
- A100 ランタイムの残存: 2026-11-06 の記録は 2 件残存、2026-11-07 の記録は 0 件（session_phase38-42_handoff）。解放の経緯は記載なし。
- Gemini 相談記録の「Phase 36-2/3 結果ファイルなし」は、true_completion_report の記載と食い違う（§4.9）。

## 7. 今後使う標準設定（Colab）
- GPU は L4。A100 はジョブごとのユーザー許可が必要（`--allow-a100`）。
- 生成評価: `--gen-batch 24 --mixed-steer --latency-reps 0 --seeds ...`。
- GRPO: `--grpo-share-ref --grpo-merge-rollouts`（`--seeds` とは併用不可）。
- 毎ジョブ後に `python scripts/colab_assignments.py` で割り当て 0 件を確認し、`<名前>.ccu.json` で CU を記録する。
