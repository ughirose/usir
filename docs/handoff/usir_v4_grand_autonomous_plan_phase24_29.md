# USIR (錦 Nishiki) v4.0 原典仕様完全結線 ＆ 本格文学生成 完成マスタープラン（Phase 24〜Phase 28）

**対象リポジトリ**: `C:/dev/usir` (`https://github.com/ughirose/usir.git`)
**策定日**: 2026-11-06
**策定者**: Gemini Argon（チーフアーキテクト兼リサーチディレクター）
**参照したGoogleドライブ原典仕様書**:
1. `【完全日本語版論文】普遍スタイル周期表（USIR v2.0）` / `Usir V2 Camera Ready Paper`
2. `Usir Adversarial Stress Test And Feasibility Audit`（全60節・9大時限爆弾監査）
3. `10_SPEC_PHASE_A_HEXAGONAL_AND_HTH_LEXICON.md`（128-bit多義辞書・OPEN-01〜04・Contrastive Fisher計量）
4. `11_SPEC_PHASE_B_DUAL_RATE_BAOAB_SMC_HSMM.md`（デュアルレートBAOAB・O(1)バッチTwisted SMC・OPEN-10不応期）
5. `12_SPEC_PHASE_C_D_E_GRPO_LORA_STEERING_AND_FUNCTORS.md`（USIR-GRPO QLoRA・12元素Activation Steering・OSWE Forge）
6. `Usir Antigravity Handover And Implementation Plan.md`（`@nishiki/usir-mcp`・`usir-wasm`・`usir-bridges`）

## 1. Googleドライブ原典と現時点（Phase 23）の徹底突合せ監査：何が達成され、どこが「簡略化（未結線）」されているか

Phase 18〜23 における Claude Code（Opus 5.5 + Sonnet 5.5）の実装・実験により、テスト件数は **pytest 508 passed / pnpm 371 passed** に達し、青空文庫10作家・46作品（5,528窓）の24次元アトラス構築、11モデルのベイクオフ（**`Qwen3-Swallow-8B-RL-v0.2` および `Qwen3-8B` が最良**と特定）、および `cProfile` による推論ボトルネック特定まで進んだ。

しかし、Googleドライブの原典仕様書（`10_SPEC`〜`12_SPEC`）と、Claude Code 自身が作成した監査表（実装済み12 / **簡略化36** / 未実装3）および Phase 23 レポートの「構想全体の内省」を照合すると、**「なぜ `.usir` ゲノムでステアしても A（芥川）と A_legacy の出力が1文字も変わらなかったのか」「なぜ1本315秒もかかるのか」「なぜ造語（懐中時針等）や段落境界の文切断が起きるのか」** の真因がすべて原典仕様からの逸脱（簡略化）にあることが判明した。

### 原典仕様 vs 現実装の7大ギャップ（本計画で完全解消するターゲット）

| ギャップID | Googleドライブ原典仕様の定義 | 現実装（Phase 23時点）の簡略化・バグ | 引き起こされている実害 |
| :--- | :--- | :--- | :--- |
| **Gap 1: 推論速度とO(1)バッチ採点の逸脱** | **`11_SPEC_PHASE_B` 第2.1節**: 「GPU推論時は `model(input_ids)` の **1回のforwardパス（バッチサイズ $P=4$）** で対数尤度を $O(1)$ バッチ計算する」 | `_compute_hf_logprob`（`realtime_decoder.py:1844`）が `for` ループで候補1つずつ `self.model(input_tensor)` を呼び、**計633回も600トークンのプロンプトを再計算**している（時間の34%＝113秒を浪費）。さらに生成時もプロンプトKVキャッシュを再利用していない。 | L4で1本315秒もかかり、反復実験や複数シード検証のスループットを著しく阻害している。 |
| **Gap 2: `.usir` 精度行列 $L_k$ と計量 $G_{\text{style}}$ の未結線** | **`10_SPEC` 第4節 / `11_SPEC` 第1.2節 / `12_SPEC` 第3.2節**: BAOABの勾配 $\nabla_{\mathbf{q}} V = G_{\text{style}}(\mathbf{q}-\boldsymbol{\mu}^*)$ およびSMCの追従誤差 $\|\hat{\mathbf{q}}-\mathbf{q}^*\|_{G_{\text{style}}}^2$ に、**`.usir` バイナリ内の下三角Cholesky精度行列 $\Sigma_k^{-1} = L_k L_k^\top$（およびContrastive Fisher計量）** を使う。 | `G_style` が **単位行列 $I_{12}$ のままハードコード**され、`style_genome` ステアリングが「モード重心の平均 $\boldsymbol{\mu}$」しか見ていない。また3モードGMMのクラスタリングが1モードに99.8%偏っている。 | 新シュリンケージ（$\alpha=0.75$）と旧（`0.2S+0.8I`）で生成文が**1文字も変わらない**という致命的空回りが発生。作家の特徴相関（精度行列）が生成に1ミリも効いていない。 |
| **Gap 3: 保護語（テーマ語・役割語）へのN-gram誤爆による造語** | **`Usir Adversarial Audit` 爆弾③**: BPE生成途中の自立語内部では構文・語彙保護ゲートを働かせ、単語の途中を切って別の漢字にすげ替える造語を起こさせない。 | `ContextNgramBan`（`hist_map`）が前段落に出たテーマ語（「懐中時計」「老職人」）の3〜4文字目（「計」など）を2段落目以降で禁止するため、モデルが苦し紛れに **「懐中時針」「懐中時刻」「懐中時鐘」** という造語を吐く。 | 小説のキーワードである「懐中時計」が2回目以降すべて奇妙な造語に化ける。 |
| **Gap 4: 段落境界での文の途中切断** | **`11_SPEC_PHASE_B` 第3節**: 段落は必ず完結した文末（句点「。」）で閉じ、次段落へ文の途中を持ち越さない。 | Qwen3 p23s で「時計は、また、」で第 $k$ 段落が打ち切られ、第 $k+1$ 段落が「止まったままのままだった」で始まる**段落またぎ文切断**が発生している（`max_para_chars` や `max_clauses` 到達時の打ち切りバグ）。 | 小説としての段落構造が壊れる。 |
| **Gap 5: Group IV (Shock) の直喩ハック（「まるで〜」連発）** | **`10_SPEC_PHASE_A` 第4節**: Group IV (Shock) は本来「HTH Octree 双曲跳躍距離 $D_{\text{jump}}$ ＋ 認知残像 $\tilde{\mathbf{s}}^-$」で測る。 | 旧特徴抽出器の表層プロキシが「まるで」「かのよう」等の比況マーカーに高い Shock スコアを与えるため、Shock ステアリングをかけると Swallow-8B が **1本で9〜13回も「まるで〜」を連発する（Goodhartハック）**。 | 小説が「まるで〜のようだった」だらけのクドい比喩文になる。 |
| **Gap 6: OPEN-10 逆U字審美効用・不応期とHSMM実データ遷移の未接続** | **`11_SPEC_PHASE_B` 第3.3節 (OPEN-10)**: Climax で大放出が起きた直後の段落は、審美疲労（Purple Prose）を防ぐため **強制的に Void（静寂・余韻）井戸へ遷移させる不応期ゲート** を稼働させる。また HSMM の3モード重心を `.usir` の3モード（Narrative / Dialogue / Climax）から直接ロードする。 | HSMM の遷移行列と well 中心が手作業の固定値のままで、`.usir` の3モード重心と連動しておらず、Climax 後の不応期（Void 強制遷移）や TS 側の OPEN-01/02/04 ゲートが未接続。 | Climax 後も文長 CV が荒れたままになり、余韻（間）が生まれない。 |
| **Gap 7: Level 2 中間層 Activation Steering ＆ USIR-GRPO 鍛造の Qwen3/Swallow 8B 実証と LLM Judge の天井解消** | **`12_SPEC_PHASE_C` 第1・2節**: ① `SymplecticActivationSteerer`（中間層 Residual Stream への12元素直結介入）と ② `AttachedLoRALinear` による実モデル USIR-GRPO 鍛造（推論遅延 0.00ms）、③ 自己採点天井（93.8張り付き）を排した **ペアワイズ（A/B対比較）ブラインド評価**。 | 単体採点 LLM Judge が全条件 93.8 に飽和して判別不能。また Phase 23 で特定した本命モデル（`Qwen3-Swallow-8B-RL-v0.2` / `Qwen3-8B`）での Activation Steering / GRPO 比較が未実施。 | 論文 Table 2 / Table 3 に相当する「制御なし vs SMC制御 vs Activation Steering / LoRA」の決定的な優位性証明が完結していない。 |

## 2. 本日〜土曜02:00リセットに向けた全6フェーズ（Phase 24〜Phase 29）完全自律計画

現在残っている **週間トークン枠 70%（Opus 5.5 司令塔 ＋ Sonnet 5.5 並列実装ワーカー）** と **Colab A100 / L4** をフル回転させ、以下の Phase 24〜Phase 29 をノンストップで完遂する。

```
[Phase 24: デコーダ推論 2倍速化（案A+B両方：O(1)バッチ採点 ＋ プロンプトKVキャッシュ共有）]
   ├─ M24.1: _compute_hf_logprob の P候補一括バッチ forward 化（633回個別呼び出しの全廃）
   ├─ M24.2: プロンプト prefix の past_key_values (KVキャッシュ) 1回計算と generate / 採点での共有
   ├─ M24.3: 個別計算との対数尤度一致単体テスト（bf16/fp32 許容誤差内）の追加
   └─ M24.4: Colab 実測で 8B 1本 315秒 → 160秒以下（A100なら 80秒台）への高速化実証 & push
        ↓
[Phase 25: 造語ゼロ・段落切断ゼロ・直喩インフレ抑制・フォーマット完全化（ミクロ表現完遂）]
   ├─ M25.1: 保護語（プロンプト内の2字以上漢字列・テーマ語・ロール語）の単語途中 N-gram 禁止免除
   ├─ M25.2: 段落末の文途中切断（「時計は、また、」で終わるバグ）の根絶と完全句点終止保証
   ├─ M25.3: 直喩マーカー（「まるで」「あたかも」「かのよう」）の段落内2回目以降のロジット減衰 & 特徴量上限キャップ
   ├─ M25.4: format_novel の引用符拡張（「」『』""“”対応・閉じ忘れ自動補完）
   └─ M25.5: 単体テスト追加 & push
        ↓
[Phase 26: 原典仕様の核心結線（.usir 3モード GMM 分離・Cholesky 精度行列 G_style・OPEN-10 不応期）]
   ├─ M26.1: usir_forge の3モード割り当て改善（極座標フーリエ位相 + 分位クラスタリングで1モード99.8%偏りを解消し Narrative/Dialogue/Climax の3モードを均等分離）
   ├─ M26.2: BAOAB & Twisted SMC の距離計算に .usir のモード別 Cholesky 精度行列 G_style = L_k L_k^T (または Contrastive Fisher 計量) を直結（α=0.75 と legacy が明確に異なる軌道を生成することの証明）
   ├─ M26.3: HSMM Director に .usir の3モード重心・遷移行列ロードと OPEN-10（Climax直後のVoid不応期ゲート）を実装
   ├─ M26.4: TypeScript 側（packages/usir-core）への OPEN-01/02/04 言語ゲートおよび新計量の完全移植（Py/TS パリティ固定）
   └─ M26.5: 単体テスト・パリティテスト通過 & push
        ↓
[Phase 27: 本命 8B/9B モデル（Qwen3-Swallow-8B / Qwen3-8B / Qwen3.5-9B）× 文豪ゲノム本格短編小説の実証]
   ├─ M27.1: Qwen3-Swallow-8B-RL-v0.2 を既定モデルに昇格 + Qwen3.5-9B（Textバックボーン抽出）のデコーダ対応
   ├─ M27.2: 複数プロット脚本（①時計職人と嵐の夜、②廃駅の最終列車と手紙、③古書店の琥珀と記憶）の整備
   ├─ M27.3: Colab A100/L4 で {制御なし(min), USIR新制御(new_ctrl), 芥川ゲノム(A: α=0.75), 太宰ゲノム(B: α=0.75), 旧シュリンケージ(A_legacy)} の本格短編生成
   ├─ M27.4: 造語=0件・段落途中切れ=0件・「まるで」≤2回/本・A vs A_legacy の非同一性・ΔLL の双方向シフト（Aで正シフト、Bで負シフト）を実証 & push
        ↓
[Phase 28: 原典 Phase C 完遂（中間層 Activation Steering & USIR-GRPO QLoRA 鍛造）＋ ペアワイズ LLM Judge]
   ├─ M28.1: 天井効果（93.8張り付き）を解消する「ブラインド A/B ペアワイズ比較 Judge（位置バイアス相殺のため A vs B と B vs A の双方向判定）」の実装
   ├─ M28.2: SymplecticActivationSteerer（中間層 0.65L への12元素ステアリング行列 V_steer 直結）の実モデル（8B/3B）稼働と相転移実証
   ├─ M28.3: AttachedLoRALinear による USIR-GRPO QLoRA 鍛造（参照PPLバリア付き4重積報酬）の実モデル稼働と推論遅延 0.00ms での Deadzone 脱出実証
   └─ M28.4: ペアワイズ勝率・統計量の記録 & push
        ↓
[Phase 29: エコシステム統合（@nishiki/usir-mcp & Wasm）と最終グランドレポート作成]
   ├─ M29.1: packages/usir-mcp（inspect_style_24d, diagnose_phase, steer_rewrite CLI/MCP）と packages/usir-bridges の最新 v4.0 コア接続確認
   ├─ M29.2: 全テストスイート（pytest 550+ / pnpm 400+ / クリーンルーム監査 FAIL=0）の完全グリーン化と git status クリーン化
   └─ M29.3: 最終総括レポート `docs/handoff/2026-11-06_usir_v4_grand_completion_report.md`（生成された傑作短編小説全文・24Dアトラス・全ベンチマーク表入り）の作成 & push
```

## 3. 各フェーズの具体的実装仕様（Claude Code への厳密指示）

### Phase 24：デコーダ推論の高速化（質問への回答：案Aと案Bの「両方」を実装せよ）

Claude Code がプロファイリングで特定した通り、`realtime_decoder.py` の実行時間の96%はモデル forward であり、そのうち **34%（113秒）が `_compute_hf_logprob` の候補1件ずつの再計算**、さらに **約25%（60〜80秒）が `generate` 時の同一プロンプト4行複製再計算** である。
**結論：案Aと案Bの「両方」を実装する。**

1. **案A（4候補採点の $O(1)$ バッチ化 — 原典 `11_SPEC_PHASE_B` 第2.1節通り）**:
   - `_compute_hf_logprob_batch(self, prompt_ids, cand_ids_list)` を実装する。
   - $P$ 個の候補節をパディング（右パディング＋`attention_mask`）して1つのバッチテンソル $(P, L_{\text{max}})$ にまとめ、**1回の `self.model(input_ids, attention_mask=...)` 呼び出し**で全候補のトークン対数確率を同時算出する。
   - 1件ずつ計算した従来関数との差が `bf16` 丸め誤差（`atol=1e-2`、`fp32` なら `1e-5`）以内に収まることを単体テストで保証する。
2. **案B（プロンプト KV キャッシュ `past_key_values` の1回計算と共有）**:
   - 各節ステップの冒頭で、確定済みプレフィックス `prompt_ids`（長さ $L_p$）に対して **1回だけ `self.model(prompt_ids, use_cache=True)` を呼んで `prefix_kv`（`DynamicCache` または tuple）を取得** する。
   - 候補生成（トークン1つずつのロールアウトループ）および候補採点（`_compute_hf_logprob_batch`）では、`prompt_ids` 全体を再投入せず、**`prefix_kv` を $P$ バッチに展開（`expand` / `repeat_interleave` または `copy.deepcopy`）して候補トークン部分（10〜20トークン）だけを forward に通す**。
   - ※ `transformers` のバージョン差で `DynamicCache` の複製が難しい場合に備え、簡単なヘルパー `_clone_kv_cache(past_key_values, batch_size)` を書き、万一非対応のモデル型が来た場合のみ案A（フルプロンプトのバッチ1回計算）へ安全にフォールバックする設計にする。これによりリスクゼロで最大の高速化（1本315秒→150秒前後、A100なら70〜90秒）が得られる。

### Phase 25：造語・段落途中切断・直喩過多・会話整形の完全修正

1. **保護語（Protected Compounds）による造語（「懐中時針」「懐中時刻」）の根絶**:
   - プロンプト（`theme`, `scene_beats`, `roles`）および段落テキスト内で初出した **2文字以上の漢字熟語（例：「懐中時計」「老職人」「歯車」「作業台」）** を保護語セット $\mathcal{W}_{\text{prot}}$ に自動登録する。
   - `ContextNgramBan` および `ContentRepLogits` / `lex_decay` において、現在生成中の行末が保護語 $w \in \mathcal{W}_{\text{prot}}$ の真の接頭辞（例：「懐中時」）と一致している場合、**その保護語を完成させる後続トークン（例：「計」）に対するすべての負のバイアス（n-gram 禁止 `-1e4`、soft 4-gram `-4.0`、反復ペナルティ）を強制的に `0.0`（免除）** にする。これにより、熟語の最後の1文字だけが禁止されて「懐中時＋針/鐘/刻」にすげ替わる造語現象は100%消滅する。
2. **段落境界での文途中切断（「時計は、また、」→次段落「止まったまま…」）の根絶**:
   - `realtime_decoder.py` の段落終了判定において、`max_chars` や `max_clauses` に達した際に読点「、」で終わった節のままループを抜けて最後の「、」を機械的に削る／残すのではなく、**段落ループの脱出条件を「必ず句点『。』または閉じ括弧『。」』『」』で終わった節が採択された時点」に限定**する。
   - もし `max_chars` を超えても文が閉じていない場合は、追加のグレース期間（最大3節・30トークン）を与え、読点「、」のロジットを `-30.0` で完全封印し、統語終止ゲートを満たす最初の終止形＋句点「。」で確実に文を完結させてから段落を閉じる。
3. **Swallow-8B 等の直喩インフレ（「まるで〜」9〜13回/本）の抑制**:
   - **ロジット側**: `ContentRepLogits`（または `DemonstrativeDecay`）に比況副詞リスト `["まるで", "あたかも", "さながら", "かのよう", "みたいに"]` を追加し、**1つの段落内で1回出現したら `-8.0`、作品全体（全段落）で2回出現したら以降すべての段落で `-12.0`** の強い減衰をかける。これで「まるで〜」は作品全体で最大1〜2回（ここぞという場面のみ）に制限される。
   - **特徴量側**: 24次元特徴抽出器の Shock 成分において、「まるで／かのよう」の表層カウントによる寄与に上限キャップ（1チャンク1回分まで）をかけ、モデルが「まるで」を連発して Shock スコアを稼ぐ Goodhart ハックを封じる。
4. **`format_novel` の会話引用符の正規化と閉じ補完**:
   - 半角ダブルクォート `"..."`、スマートクォート `“...”`、二重鉤括弧 `『...』`（会話として使われた場合）を検出し、未閉鎖のまま改行・文末に達した場合は自動で閉じ括弧を補完した上で標準の `「...」` として独立行整形する。

### Phase 26：原典仕様の核心結線（`.usir` 精度行列 $G_{\text{style}}$・3モード分離・OPEN-10 不応期）

Claude Code が Phase 20 / 21 / 23 で鋭く指摘した「3モードGMMの99.8%が1モードに偏る」「ステアリングが重心しか使っておらず精度行列を使っていないため、`scaled (α=0.75)` と `legacy` の出力が完全一致する」という構造的欠陥をここで完全に解決する。

1. **`usir_forge` の3モード（Narrative / Dialogue / Climax）分離の正常化**:
   - なぜ窓の99.8%が1モードに入ったかといえば、600字スライディング窓全体で計算すると会話率やShockが平均化され、固定閾値クラスタリングではほぼすべてが Mode 0（Narrative）に落ちるからである。
   - **修正**: `usir_forge`（Py/TS両方）において、原典 `12_SPEC` 第3.1節の**極座標フーリエ位相アライメント**を適用した上で、コーパス内の全窓（または文塊）に対して **分位点ベースの3モード割り当て**（① 会話率または対話特徴が上位25%（または閾値以上）を Mode 1 [Dialogue]、② 残りのうち Shock・運動量ノルム $\|\mathbf{p}\|$ が上位25%を Mode 2 [Climax]、③ 残りの50%を Mode 0 [Narrative]）を行うように修正する。これにより、どの作家コーパスでも必ず3つのモードすべてに十分なサンプル（各20〜50%）が割り当てられ、3つの異なる24次元重心 $\boldsymbol{\mu}_k$ と3つの非単位 Cholesky 精度行列 $L_k$ が蒸留される。
   - 修正後、Phase 20 の青空文庫10作家の `.usir` バイナリ（20個）を再蒸留して更新する。
2. **デコーダ＆ディレクターへのモード別精度行列 $G_{\text{style}}^{(k)} = L_k L_k^\top$ の直結**:
   - `HSMMLongformDirector` に `style_genome`（`.usir` バイナリ）が渡された際：
     - HSMM の3つの Well（0: Narrative, 1: Dialogue, 2: Climax）の目標重心 $\boldsymbol{\mu}_k^*$ と遷移行列 $T$ を、手作業の固定値ではなく **`.usir` バイナリ内の3モード重心 $\boldsymbol{\mu}_k$ と遷移行列 $T$** からブレンド設定する。
     - さらに、各段落の Well $k \in \{0, 1, 2\}$ における `ClauseTwistedSMCDecoder` の計量行列 `G_style` に、単位行列 $I_{12}$ ではなく **`.usir` のモード $k$ の Cholesky 因子 $L_k$ から再構成した $12\times 12$ 位置精度ブロック $G_{\text{style}}^{(k)} = (L_k L_k^\top)_{qq}$（トレース正規化済み）** を注入する。
     - これにより、BAOAB の復元力 $\nabla_{\mathbf{q}} V = G_{\text{style}}^{(k)}(\mathbf{q} - \boldsymbol{\mu}_k^*)$ と SMC のマハラノビス追従項 $-\frac{\beta}{2}(\hat{\mathbf{q}} - \mathbf{q}^*)^\top G_{\text{style}}^{(k)}(\hat{\mathbf{q}} - \mathbf{q}^*)$ が、作家の実共分散構造（どの特徴とどの特徴が連動し、どの特徴の逸脱を厳しく罰するか）を直接反映するようになり、**`scaled (α=0.75)` と `legacy (0.2S+0.8I)` で明確に異なる生成軌道・異なる出力テキストが生まれる**（単体テストおよび実生成で `text_scaled != text_legacy` を検証する）。
3. **OPEN-10（Climax 直後の Void 不応期ゲート）の実装**:
   - 原典 `11_SPEC` 第3.3節に基づき、`HSMMLongformDirector` で Climax（Well 2）段落が実行された直後の段落 $k+1$ では、マルコフ抽選にかかわらず **強制的に Well 0（Narrative / Void 余韻モード：`rhythm_profile="steady"`, 静寂・余白重視）へ遷移させる不応期（Refractory Period）** を実装する。これにより、`3B s137` のように Climax が連続したり、Climax 後の段落までリズムが荒れ続ける現象が解消される。
4. **TypeScript 側への OPEN-01/02/04 言語ゲート移植**:
   - Phase 18 の監査表で「TS側に未移植」とされた OPEN-01（具体度減衰）、OPEN-02（否定・様相双対テンソル）、OPEN-04（バフチン2チャネル分離）を `packages/usir-core` に移植し、Python/TS パリティテストを追加する。

### Phase 27：本命 8B/9B モデル × 複数プロット × 文豪ゲノムによる「本当に面白い短編小説」実生成

1. **既定モデルの `Qwen3-Swallow-8B-RL-v0.2` への切り替えと `Qwen3.5-9B` 対応**:
   - Phase 23 の11モデル比較で「青空文庫の実作家帯（平均文長31〜37字）に唯一自然に入り、最も日本語が自然」と実証された **`tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2`（および `Qwen/Qwen3-8B`）** を長尺生成の主力（Tier-1 既定）とする。
   - また、制御なしで最も文学的（CV 0.27）だった `Qwen3.5-9B`（`AutoModelForImageTextToText` 等のマルチモーダルラッパー構造）についても、言語モデルバックボーン（または `AutoModelForCausalLM` / `language_model` サブモジュール）を取り出してデコーダで動かせるようにアダプタを追加する。
2. **プロット脚本と世界観整合性（退場人物の再登場防止）の強化**:
   - Phase 21 で「工房を去ったはずの老職人がその場にまだいる」という筋の不整合が起きたのは、各段落の `scene_beats` に「この段落で画面上に存在する人物／不在の人物」が明示されていなかったためである。
   - 各段落の `scene_beats` に **「現在場の状態（誰がどこにいて、何が起きた後か）」** を1文で明記した2種類の本格短編プロット（①『止まった懐中時計と嵐の工房』改訂版、②『霧の廃駅と琥珀の切符』）を用意する。
3. **Colab A100（または L4）での本格生成と実証**:
   - Phase 24 の高速化（約2倍速）を活かし、`Qwen3-Swallow-8B-RL-v0.2`（および `Qwen3-8B`）を用いて、以下の条件で複数シード生成を実行する：
     - `min`（素のモデル生成・制御なし）
     - `new_ctrl`（USIR 構文・リズム制御のみ、特定作家ゲノムなし）
     - `genome_A_scaled`（芥川龍之介 `.usir` ゲノム、新シュリンケージ $\alpha=0.75$、精度行列 $G_{\text{style}}$ 有効）
     - `genome_A_legacy`（芥川龍之介 `.usir` ゲノム、旧シュリンケージ `0.2S+0.8I`）
     - `genome_B_scaled`（太宰治または谷崎潤一郎 `.usir` ゲノム、新シュリンケージ $\alpha=0.75$）
   - **検証項目**:
     - `genome_A_scaled` と `genome_A_legacy` の出力が明確に分岐し、かつ `genome_A_scaled` の方が芥川ゲノムに対する対数尤度（またはマハラノビス適合度）が優れていること。
     - `genome_A` ステアで $\Delta\text{LL}_{AB}$ が A 側へ、`**genome_B**` ステアで B 側へ双方向にシフトすること。
     - 造語＝0件、段落途中切断＝0件、「まるで」$\le 2$回/本、整形済みテキスト（`formatted_text`）が人間が読んでそのまま楽しめる文学品質に達していること。

### Phase 28：原典 Phase C（Activation Steering ＆ USIR-GRPO）実モデル検証とペアワイズ LLM Judge

1. **天井効果（93.8張り付き）を打破する「ブラインド A/B ペアワイズ比較 Judge」の実装**:
   - 単一テキストに点数をつけさせると LLM は 90点台に飽和する（Phase 21/23 で確認済み）。
   - `evaluate_pairwise_judge(text_a, text_b, judge_model)` を実装する。2つの小説（例：`min` vs `new_ctrl`、または `ctrl` vs `genome_steered`）を匿名（作品1・作品2）で並べ、**位置バイアス（先に提示された方を好む癖）を完全に相殺するため `(A, B)` の順と `(B, A)` の順の2回評価**させ、5つの文芸評価軸（①文長リズムの緩急と呼吸、②表現の自然さと造語・破綻のなさ、③情景の鮮明さと余韻［間］、④クライマックスの迫力、⑤総合的な文学的魅力）でどちらが優れているか（勝敗とマージン $-2 \sim +2$）を判定させる。
2. **原典 Phase C（`SymplecticActivationSteerer` ＆ `AttachedLoRALinear` USIR-GRPO）の実機検証**:
   - Googleドライブ原典 `12_SPEC_PHASE_C_D_E` に規定された **Level 2: 中間層 Residual Stream 12元素直結（`activation_steerer.py`）** と **Level 3: 実モデル USIR-GRPO LoRA 鍛造（`lora_grpo_forge.py`）** を Colab 上で実行し、推論追加遅延 `0.00 ms`（またはフックのみの数ミリ秒）で Deadzone から脱出できることを実測データとして確定させる。

### Phase 29：エコシステム統合（`@nishiki/usir-mcp`・`usir-bridges`）と最終総括レポート出力

1. **`@nishiki/usir-mcp` / `usir-bridges` / `usir-wasm` の総点検**:
   - Googleドライブ原典 `Usir Antigravity Handover And Implementation Plan.md` に記載されているパッケージ群（`usir-core`, `usir-mcp`, `usir-bridges`, `usir-wasm`）がすべて最新の v4.0 スキーマ・SVD 零空間調停器・新 QDA 境界と整合してビルド・テスト通過することを確認する。
2. **全テスト・クリーンルーム監査と最終グランドレポート作成**:
   - `pytest` 全件、`pnpm test` 全件、`scan_cleanroom_copyright`（FAIL=0）を通す。
   - 生成された最良の短編小説（`Qwen3-Swallow-8B-RL-v0.2` による芥川風・太宰風・USIR制御版の整形済み全文）、ペアワイズ Judge 勝率、推論高速化ベンチマーク、原典53項目の最終解消状況をすべて収録した決定版レポート `docs/handoff/2026-11-06_usir_v4_grand_completion_report.md` を作成し、`origin/main` へ commit & push して完了する。
