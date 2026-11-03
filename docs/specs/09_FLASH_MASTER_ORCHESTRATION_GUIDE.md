# USIR 完全構想実装マスターガイド（Gemini 3.8 Flash 専属オーケストレーション＆フェーズ別実行指示書）

> **対象ワークスペース**: `C:/dev/usir`（GitHub: `https://github.com/ughirose/usir.git`）  
> **Google ドライブ連携パス**: `F:\\マイドライブ\\USIR_Session_Handoff_20261102\\`（Drive Folder ID: `13f8sUR57RFRQhiVB_P5iytWsbtZVpUkQ`）  
> **稼働モデル前提**: Antigravity の Gemini 4 Argon クォータ枯渇に伴い、**メインセッション（司令塔）および全サブエージェント（`invoke_subagent`）の双方を `Gemini 3.8 Flash (High)` で完遂する専用設計**。

## 1. 基本方針とアーキテクチャ決定事項（ユーザー確定仕様）

1. **外部3万語辞書（`hth_poly_30k.bin`）の後回しと「自前軽量オートマトン＋GPU実隠れ層」による即時解決**
   - 大規模外部辞書（WordNet 30,000語ビルド）は最終フェーズ（Phase E 末尾）へ後回しとする。
   - **CPU / TypeScript / Wasm 単体動作時**：外部の形態素解析ライブラリ（Janome / Sudachi 等）を USIR 本体に組み込むことはせず、USIR 内部に **「外部依存ゼロの決定論的・文節（Bunketsu）＆係り受け木オートマトン（約120行）」** と **「256部首ラプラシアン拡散HTH＋オノマトペ音韻象徴合成器（`OPEN-03`）」** を実装する。これにより Python と TypeScript の 100% ビット完全一致とゼロ依存性を死守しつつ、`OPEN-01〜05` を完全稼働させる。
   - **重計算・意味多様体・強化学習時（Google Colab 上位GPU活用）**：Colab 実行時はケチらず **L4 (24GB VRAM) または A100 (40GB/80GB VRAM)**（利用不可時のみ T4 フォールバック）を指定し、`Qwen/Qwen2.5-3B-Instruct` または `Qwen/Qwen2.5-7B-Instruct` の**実モデルの隠れ層ベクトル（Hidden States）と実トークン対数確率**を直接抽出して計算する。

2. **Gemini 3.8 Flash で 100% の品質を出すための「5つの絶対規律（Flash Guardrails）」**
   - **規律1（設計書の事前読み込み必須）**：実装を始める前に、必ず `F:\\マイドライブ\\USIR_Session_Handoff_20261102\\` 内の対応する詳細設計書（`10_SPEC_PHASE_A...`, `11_SPEC_PHASE_B...`, `12_SPEC_PHASE_C_D_E...`）を `read_file` で読み込むこと。数式やアルゴリズムを自己流で簡略化することを厳禁とする。
   - **規律2（ダミー・ハリボテ実装の完全禁止）**：
     - LLM本体と繋がっていない単体行列 `LinearLoRAAdapter(64, 64)` に乱数 `torch.randn` を入れるようなモック学習を厳禁とする。
     - `evaluate_llm_judge` の中で `_embed_sentence_unit` や `cv_l` を使って自己採点する偽装 Judge を厳禁とする。
     - `USIRPyTorchLogitsProcessor` で固定ベクトルのコサイン類似度を足すだけの処理を「BAOAB」と呼ぶことを厳禁とする。
   - **規律3（60秒 SSE EOF タイムアウト防止のための小規模差分編集）**：
     - 1回のサブエージェントが読み込むコンテキストは対象ファイル 1〜2 個（500行単位の分割 `read_file`）に限定する。
     - `replace_file_content` / `multi_replace_file_content` は **1回あたり最大 40〜60 行以内の関数単位差分** に分割し、ファイル全体の一括置換を絶対に行わないこと。
   - **規律4（サブエージェント起動時の `Model="flash"` 明示）**：
     - `invoke_subagent` を呼ぶ際は、必ず `Model="flash"` を明示指定し、1つのサブエージェントには直交する 1〜2 ファイルのタスクのみを割り当てること。
   - **規律5（2ストライク・即時転換）**：
     - 同一コマンドや同一アプローチで 2 回エラーが出た場合は、同じリトライを繰り返さず即座に原因を特定して別アプローチへ切り替えること。

## 2. 全体実装ロードマップ（Phase A 〜 Phase E）

| フェーズ | 参照すべき補助設計書（Google Drive / `F:\\マイドライブ`） | 実装する核心モジュール（未実装領域の本実装） | 完了検証ゲート（Exit Criteria） |
| :--- | :--- | :--- | :--- |
| **Phase A**<br>（最優先） | `10_SPEC_PHASE_A_PARSER_FREE_AND_MULTIRESOLUTION.md` | 1. **外部依存ゼロ・文節＆係り受け木オートマトン** (`bunketsu_automaton.py` / `bunketsuAutomaton.ts`)<br>2. **`OPEN-01〜05` 完全実装**（具体度減衰 $\\mathcal{C}(u)$、否定・様相双対感覚テンソル $(\\tilde{\\mathbf{s}}^+, \\tilde{\\mathbf{s}}^-)$、オノマトペ音韻象徴合成 $\\Psi_{\\text{mimetic}}$、地の文/キャラ別会話バフチン2チャネル分離 $\\mathbf{z}_{\\text{nar}} \\perp \\mathbf{z}_{\\text{dia}}^{(c)}$）<br>3. **Stage 1 ($\\mathbf{s}_m \\in \\mathbb{R}^4$) $\\to$ Stage 2（対数3等分ウェーブレット × Presence/Void直交分解 × ISTA形態素分離 × RQA）＆ Contrastive Fisher Distillation 重み学習器**<br>4. **実モデル隠れ層 / 文節構文木ベースの真の `_embed_sentence_unit`** | ・「冷たい氷（物理）」vs「冷たい視線（抽象）」、「炎が燃え盛る（肯定）」vs「炎などなかった（否定）」が24D空間で明確に分離されること。<br>・未知オノマトペ（「ぎらぎら」「しんと」「ぷつり」）の感覚ベクトルが非ゼロで正確に合成されること。<br>・Python / TS 全単体テスト 100% PASS（誤差 $< 10^{-5}$）。 |
| **Phase B**<br>（生成制御の本丸） | `11_SPEC_PHASE_B_DUAL_RATE_BAOAB_SMC_HSMM.md` | 1. **階層的デュアルレート 5ステージ Adaptive-BAOAB 積分器**（節境界での24D実測フィードバック $\\hat{\\mathbf{q}}_m$ ＋ 構文エントロピー連動可変摩擦 $\\Gamma_t(H_{\\text{LLM}})$ ＋ `OPEN-08` Ghost History カルマン収縮）<br>2. **節単位 Twisted SMC（`OPEN-07` & 第21.2節、$P=4$ 粒子フィルタ）**（4粒子並列節生成＋Terminal Reachability $\\ln P_{\\text{LLM}}(W_{\\text{end}}^* \\mid w_{<k}\\circ c^{(p)})$ ＋ 毎節リサンプリング）<br>3. **4KB 3モード HSMM 多井戸スイッチング ＋ Momentum Banking** による **3,000字規模（複数段落〜1章）の長尺連続生成** | ・デコード中に毎節 BAOAB の $(\\mathbf{q}_m, \\mathbf{p}_m)$ が動的発展し、Twisted SMC が4粒子から最適節を選択するテレメトリが出力されること。<br>・Colab GPU（L4/A100/T4）実機で **3,000字の長文生成** を完走し、後半でも $CV_L \\in [0.40, 0.70]$ と $1/f$ ゆらぎが維持され、文途中切断が 0件であること。 |
| **Phase C**<br>（学習・内部回路直結） | `12_SPEC_PHASE_C_D_E_GRPO_LORA_STEERING_AND_FUNCTORS.md`（前半） | 1. **ダミー `LinearLoRAAdapter` の完全撤去と、実LLM（Qwen2.5）直結 QLoRA / USIR-GRPO 鍛造**（参照モデル PPL バリア $\\mathcal{M}_{\\text{semantic}}(\\text{PPL}_{\\text{ref}})$ ＋ KL正則化 $\\beta_{\\text{KL}} D_{\\text{KL}}$ ＋ 実トークン対数尤度への勾配逆伝播）<br>2. **Level 2: 中間層 Residual Stream 12元素直結（`SymplecticActivationSteerer`）**（`register_forward_hook` による12元素ステアリング軸の抽出・直接注入）<br>3. **自己採点を廃した独立 LLM-as-a-Judge ブラインド評価器** | ・Colab GPU 上で実 LLM の LoRA 重みが更新され、訓練後アダプタ単体（**推論追加遅延 `0.00 ms`**）で Deadzone 脱出と品質向上が実証されること。<br>・中間層 Activation Steering のゲイン操作だけで「結晶相 $\\leftrightarrow$ 超流動ヴォイド相（間）$\\leftrightarrow$ 臨界相」へ相転移すること。 |
| **Phase D**<br>（蒸留・高速化） | `12_SPEC_PHASE_C_D_E_GRPO_LORA_STEERING_AND_FUNCTORS.md`（中盤） | 1. **OSWE ストリーミング蒸留 ＆ 極座標フーリエ位相アライメント CLI (`usir-forge`)**（4,096B 3モードGMM＋コレスキー精度行列の実コーパス完全フィッティング）<br>2. **C++17 Wasm SIMD128 (`usir_simd.cpp`) の実 `.wasm` ビルド統合** | ・実コーパス（$K \\ge 5$ パブリックドメイン作家群）から 4,096 Bytes の `.usir` バイナリがゼロ埋めなしで完全蒸留されること。<br>・1チャンク計装が $<1\\text{ms}$ で動作すること。 |
| **Phase E**<br>（統一場・6モダリティ・辞書） | `12_SPEC_PHASE_C_D_E_GRPO_LORA_STEERING_AND_FUNCTORS.md`（後半） | 1. **Nishiki Toulmin 論理監査ポテンシャル $U_{\\text{Toulmin}}$ とジャンル温度 $\\tau_{\\text{genre}} \\in [0, 1]$ 統一場（`OPEN-06`）**<br>2. **WorldCraft オントロジー制約との辞書式ヌル空間射影調停（`OPEN-09`）**<br>3. **視覚関手 $\\mathcal{F}_{\\text{visual}}$（実画像 Gabor/Quadtree/Saliency 解析）＆ 音楽関手 $\\mathcal{F}_{\\text{music}}$（GTTM/五度圏解析 $\\leftrightarrow$ WebAudio/MIDI 合成）**<br>4. （オプション最終工程）外部3万語辞書 `hth_poly_30k.bin` コンパイラ | ・$\\tau_{\\text{genre}}$ スイープで仕様書・記事・小説の監査/生成が連続遷移すること。<br>・実画像・MIDIデータと小説24D軌道の間で双方向トランスパイルが実動すること。 |

## 3. Antigravity 投入用：フェーズ別コピペ実行プロンプト集

Antigravity で新規チャットを開き、モデルを **`Gemini 3.8 Flash (High)`** に設定した上で、以下のプロンプトを Phase A から順番に投入してください。

### 【Prompt 1：Phase A 投入用（文節・係り受けオートマトン ＆ OPEN-01〜05 ＆ 多重解像度計装の本実装）】

```markdown
/boost USIR Phase A Implementation: Zero-Dependency Bunketsu/Kakari-Uke Automaton, OPEN-01..05 Linguistic & Literary Modules, and Stage 1->2 Multiresolution Wavelet/ISTA/Fisher Core

あなたは USIR プロジェクトのリード実装統括（Gemini 3.8 Flash）です。
まず最初に、ローカルにマウントされている Google ドライブフォルダ `F:\\マイドライブ\\USIR_Session_Handoff_20261102\\` にある以下の2つの設計仕様書を `read_file` で必ず読み込んでください：
1. `F:\\マイドライブ\\USIR_Session_Handoff_20261102\\09_FLASH_MASTER_ORCHESTRATION_GUIDE.md`
2. `F:\\マイドライブ\\USIR_Session_Handoff_20261102\\10_SPEC_PHASE_A_PARSER_FREE_AND_MULTIRESOLUTION.md`

### 絶対遵守ルール（Flash Guardrails）:
- サブエージェントを起動する際は、必ず `invoke_subagent(TypeName="DeepCoder", Model="flash", Workspace="inherit", ...)` を使用し、1サブエージェントあたり1〜2ファイルの直交タスクに分割すること。
- 60秒 SSE EOF タイムアウトを防ぐため、1回のファイル編集（`replace_file_content`）は最大40〜60行の関数単位差分とし、巨大ファイルの一括置換を絶対に行わないこと。
- 外部の形態素解析ライブラリ（Janome/Sudachi等）や3万語外部辞書は使わず、`10_SPEC_PHASE_A_PARSER_FREE_AND_MULTIRESOLUTION.md` に定義された「外部依存ゼロの決定論的・文節（Bunketsu）＆係り受け木オートマトン」と「256部首HTH＋オノマトペ音韻象徴合成器」によって Python と TypeScript の 100% 一致実装を行うこと。

### 委譲ワークストリーム（直交分割）:
1. **WS-A1（Python 文節・係り受け木オートマトン ＆ OPEN-01〜04 実装）**:
   - `python/usir_py/bunketsu_automaton.py` を新規作成し、`10_SPEC_PHASE_A_PARSER_FREE_AND_MULTIRESOLUTION.md` の仕様通りに文節分割、Head-Final 係り受け木 $T_m=(V_m, E_m)$ 構築、`OPEN-01`（具体度減衰ゲート $\mathcal{C}(u)$）、`OPEN-02`（否定・推量スコープ双対感覚テンソル $\tilde{\mathbf{s}}^+, \tilde{\mathbf{s}}^-$）、`OPEN-03`（オノマトペ音韻象徴合成 $\Psi_{\text{mimetic}}(w)$＋Mimetic Immunity Protocol）、`OPEN-04`（地の文 $\mathbf{z}_{\text{nar}}$ とキャラ別会話文 $\mathbf{z}_{\text{dia}}^{(c)}$ のバフチン2チャネル分離＆Dialogic Collapse距離）を実装せよ。
   - `python/usir_py/extractor_24d.py` および `python/usir_py/manifold_geodesic.py`（`_embed_sentence_unit`）を更新し、文字ハッシュ近似を撤去して文節係り受け木・双対感覚テンソル・対数3等分ウェーブレット・ISTAソリトン分離・RQA・Contrastive Fisher Distillation（`fit_contrastive_fisher_metric`）を接続せよ（GPU利用時は実モデルの hidden_states 平均プーリングにも対応させること）。
   - `python/tests/test_phase_a_linguistic_core.py` を作成し、「冷たい氷」vs「冷たい視線」（OPEN-01）、「紅蓮の炎が燃える」vs「紅蓮の炎などなかった」（OPEN-02）、未知オノマトペ「ぎらぎら」「しんと」「ぷつり」（OPEN-03）、地の文と複数キャラ会話の分離（OPEN-04）を検証せよ。

2. **WS-A2（TypeScript 完全同型移植 ＆ パリティ検証）**:
   - `packages/usir-core/src/sensor/bunketsuAutomaton.ts` を新規作成し、Python 側と 100% 同一のアルゴリズムで文節係り受け木・`OPEN-01〜04`・Stage 1 $\to$ Stage 2 多重解像度抽出を実装し、`symplectic24d.ts` および `manifoldGeodesic.ts` に統合せよ。
   - `packages/usir-core/tests/phase-a-linguistic-core.test.ts` を作成し、Python 側との出力誤差 $< 10^{-5}$ および全既存テストを含む `python scripts/harness.py test` 全件 PASS を確認してコミット＆Pushせよ。
```

### 【Prompt 2：Phase B 投入用（本物のデュアルレート BAOAB × 節単位 Twisted SMC × 3,000字長尺HSMM生成）】

```markdown
/boost USIR Phase B Implementation: Genuine Dual-Rate 5-Stage Adaptive-BAOAB, Clause-Level Twisted SMC (P=4 Particles), and 3,000-Char 3-Mode HSMM Multi-Well Long-Form Rollout on Colab GPU

まず最初に、`F:\\マイドライブ\\USIR_Session_Handoff_20261102\\11_SPEC_PHASE_B_DUAL_RATE_BAOAB_SMC_HSMM.md` を `read_file` で読み込み、設計仕様を正確に把握してください。

### 絶対遵守ルール:
- サブエージェントは必ず `Model="flash"` を明示指定し、関数単位の小規模編集（40〜60行以内）で進めること。
- Colab 実行時は上位GPU（`accelerator="L4"` または `"A100"`、フォールバック `"T4"`）および十分なタイムアウト（`timeout=600`）を設定し、モデルには `Qwen/Qwen2.5-3B-Instruct`（または `7B`）を使用すること。

### 委譲ワークストリーム:
1. **WS-B1（デュアルレート 5ステージ Adaptive-BAOAB ＆ 節単位 Twisted SMC の本実装）**:
   - `python/usir_py/realtime_decoder.py`（および対応する TS 側 `packages/usir-core/src/dynamics/`）を `11_SPEC_PHASE_B_DUAL_RATE_BAOAB_SMC_HSMM.md` に従って全面アップグレードせよ：
     1) **Outer Loop（節境界 $m \to m+1$）**：確定した節 $s_m$ から `extract_usir_24d` で実測 $\hat{\mathbf{q}}_m$ を抽出し、`OPEN-08` Ghost History カルマン収縮 $\hat{\mathbf{q}}_m^{\text{shrunk}} = \alpha_m \hat{\mathbf{q}}_m + (1-\alpha_m)\boldsymbol{\mu}^*$ を適用した上で、**5ステージ Adaptive-BAOAB 積分器（B-A-O-A-B）** を1ステップ回して次節の目標 $(\mathbf{q}_{m+1}^*, \mathbf{p}_{m+1}^*)$ を算出する。Stage O ではベースLLMの構文エントロピー $H_{\text{LLM}}$ に連動する異方性摩擦 $\Gamma_t(H_{\text{LLM}})$ の下でガウス平衡分布 $\mathcal{N}(0, \tau M)$ を厳密に保つこと。
     2) **Clause-Level Twisted SMC（$P=4$ 粒子フィルタ）**：次の読点・句点まで $P=4$ 本の候補節 $\{c_{m+1}^{(1..4)}\}$ を並列ロールアウトし、①節のモデル対数尤度 $\ln P_{\text{LLM}}(c^{(p)})$、②文末・段落着地アンカー $W_{\text{end}}^*$ への到達可能性 $\ln P_{\text{LLM}}(W_{\text{end}}^* \mid w_{\le m} \circ c^{(p)})$、③USIR計量距離 $-\frac{\beta}{2}\|\hat{\mathbf{q}}(c^{(p)}) - \mathbf{q}_{m+1}^*\|_{G_{\text{style}}}^2$ の捻り重み $w^{(p)}$ で毎節リサンプリングする `ClauseTwistedSMCDecoder` を実装せよ。

2. **WS-B2（3モード HSMM 多井戸スイッチング ＋ Momentum Banking による 3,000字長尺生成実験）**:
   - `python/usir_py/hsmm_longform_director.py` を新規作成し、4KB プロファイルの 3モード（`Narrative` [地の文] $\leftrightarrow$ `Dialogue` [会話] $\leftrightarrow$ `Climax` [臨界破調]）遷移行列 $T_{ij}$ と `OPEN-09` Momentum Banking（未解放運動量 $\Delta \mathbf{p}_{\text{banked}}$ の蓄電と次文での解放）および `OPEN-10` Wundt-Berlyne 不応期疲労ペナルティに従って、複数段落・全 3,000 字規模の長編を連続生成するエンジンを実装せよ。
   - Colab GPU 実機で 3,000字長尺生成ベンチマーク（Vanilla vs USIR Dual-Rate BAOAB+SMC+HSMM）を実行し、第1段落から最終段落まで Deadzone（$CV_L < 0.28$）への減衰が起きず、文途中切断 0件で完走することを検証して成果物を `F:\\マイドライブ\\USIR_Session_Handoff_20261102\\` および GitHub へ保存せよ。
```

### 【Prompt 3：Phase C 投入用（実モデル直結 USIR-GRPO QLoRA 鍛造 ＆ 中間層 Activation Steering ＆ 独立 LLM Judge）】

```markdown
/boost USIR Phase C Implementation: Authentic Transformer-Attached USIR-GRPO LoRA Weight Forge, Level-2 Residual Stream 12-Element Activation Steering, and Independent Blind LLM Judge

まず最初に、`F:\\マイドライブ\\USIR_Session_Handoff_20261102\\12_SPEC_PHASE_C_D_E_GRPO_LORA_STEERING_AND_FUNCTORS.md` の Phase C セクションを `read_file` で読み込んでください。

### 委譲ワークストリーム:
1. **WS-C1（ダミー `LinearLoRAAdapter` の完全撤去と、実モデル直結 QLoRA / USIR-GRPO 鍛造＆参照PPLバリア実装）**:
   - `python/usir_py/manifold_geodesic.py` の `compute_multiplicative_grpo_reward` に、原典第22.2節の **参照モデル・パープレキシティ意味一貫性バリア $\mathcal{M}_{\text{semantic}}(y \mid x) = \sigma(\text{PPL}_{\text{thresh}} - \text{PPL}_{\text{ref}}(y \mid x))$** を追加せよ。
   - `python/colab/run_v4_bidirectional_experiment.py` 内のダミー `LinearLoRAAdapter(64, 64)`（`torch.randn` を入れていたモック）を完全削除し、実モデル（`Qwen2.5`）の Attention 層（`q_proj`, `v_proj`）に直接装着される本物の ` LoRALinearHook`（または `peft` LoRA）を実装せよ。
   - 同一プロンプトから $G=4\text{〜}8$ 本のロールアウトを生成し、実トークン列の対数確率 $\sum_t \ln \pi_\theta(y_{i,t} \mid x, y_{i,<t})$ と参照モデルとの KL ダイバージェンス $\beta_{\text{KL}} D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})$ を計算して本物の GRPO 政策勾配更新を行い、学習後のアダプタ重みで **推論時介入なし（追加遅延 `0.00 ms`）** でも Base モデルより $CV_L$・$\mathcal{D}_{\text{3-phase}}$・品質スコアが向上することを Colab GPU 実機で実証せよ。

2. **WS-C2（Level 2: 中間層 Residual Stream への 12元素神経回路直結 `SymplecticActivationSteerer` ＆ 独立 LLM Judge）**:
   - `python/usir_py/activation_steerer.py` を新規作成し、PyTorch の `register_forward_hook` を用いて LLM の中間層（例：$L/2 \sim 3L/4$ 層）の Residual Stream から 12元素（`Sg-01..12`）の対照ステアリングベクトル $\mathbf{v}_k \in \mathbb{R}^{d_{\text{model}}}$ を抽出・直交化し、推論時に $\mathbf{h}'_t = \mathbf{h}_t + \sum_{k=1}^{12} \Delta q_k^* \mathbf{v}_k$ として直接電位注入する Neuro-Symbolic Activation Steerer を実装せよ。
   - `evaluate_llm_judge` から自前の `_embed_sentence_unit` / `cv_l` による自己採点ロジックを撤去し、独立した Judge システムプロンプトによる実 LLM 推論採点（オフライン単体テスト時のみ決定論的ルーブリック検証）へ刷新せよ。
```

### 【Prompt 4：Phase D & E 投入用（OSWE `usir-forge`・Wasm SIMD・Nishiki統一場 $\tau_{\text{genre}}$・WorldCraftヌル空間・6大モダリティ関手）】

```markdown
/boost USIR Phase D & E Implementation: OSWE Polar-Fourier Phase-Aligned usir-forge, Wasm SIMD, Nishiki Genre Temperature Unified Field, WorldCraft Null-Space Arbitration, and Cross-Modal Visual/Music Functors

まず最初に、`F:\\マイドライブ\\USIR_Session_Handoff_20261102\\12_SPEC_PHASE_C_D_E_GRPO_LORA_STEERING_AND_FUNCTORS.md` の Phase D および Phase E セクションを `read_file` で読み込んでください。

### 委譲ワークストリーム:
1. **WS-D1（OSWE ストリーミング蒸留・極座標フーリエ位相アライメント・4KB 3モードGMM `usir-forge` ＆ Wasm SIMD）**:
   - `python/usir_py/usir_forge.py` および `packages/usir-core/src/schema/usirForge.ts` を実装し、Welford オンライン更新＋極座標フーリエ位相アライメント（クライマックス位相角 $\theta_{\text{peak}}$ 整列による逆位相相殺防止）＋ 3モード GMM（地の文・会話・クライマックスの各 $\boldsymbol{\mu} \in \mathbb{R}^{24}$ と下三角コレスキー精度行列 $L \in \mathbb{R}^{300}$、マルコフ遷移行列 $T_{ij} \in \mathbb{R}^{3\times 3}$）を厳密に 4,096 Bytes のバイナリへ出力し、Dirty Room 揮発＆ $K \ge 5$ クラスタ匿名性ゲートを通すパイプラインを完成させよ。
   - `packages/usir-wasm` の SIMD 距離・内積カーネルをビルド・検証せよ。

2. **WS-E1（Nishiki 統一場 $\tau_{\text{genre}}$・WorldCraft 辞書式ヌル空間調停・視覚/音楽クロスモーダル関手）**:
   - `OPEN-06`: Toulmin 論理監査ポテンシャル $U_{\text{Toulmin}}(x)$（Claim-Data-Warrant 構造、推論深さ制限、主述ねじれ、用語一義性）を実装し、$\mathcal{F}_{\text{total}}(x) = (1-\tau_{\text{genre}})U_{\text{Toulmin}}(x) + \tau_{\text{genre}}\mathcal{F}_{\text{USIR}}(x)$ によって仕様書（$\tau=0.1$）・論説（$\tau=0.5$）・文芸（$\tau=0.9$）を連続制御せよ。
   - `OPEN-09`: WorldCraft オントロジー制約（呼称・生死・知識境界）のハードマスクと制約ヤコビアンの零空間射影 $(I - J_C^\dagger J_C)\nabla \mathcal{F}_{\text{USIR}}$ を実装せよ。
   - 第27・28節の視覚関手 $\mathcal{F}_{\text{visual}}$（2D画像配列からの Gabor 空間周波数・色空間エントロピー・Spatial Quadtree・Saliency Scanpath 抽出 $\leftrightarrow$ Diffusion 制御パラメータ）および音楽関手 $\mathcal{F}_{\text{music}}$（ノート列からの五度圏トーラス・GTTM木・不協和度・休符比抽出 $\leftrightarrow$ 小説24D軌道に完全同期する WebAudio/MIDI シンセサイザ）を本実装し、`python scripts/harness.py test` 全件 PASS を確認せよ。
```
