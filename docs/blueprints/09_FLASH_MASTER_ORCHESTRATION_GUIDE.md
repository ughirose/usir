# USIR 完全構想実装マスターガイド【迎合排除・本質定礎版】（Gemini 3.8 Flash 専属オーケストレーション）

> **対象ワークスペース**: `C:/dev/usir`（GitHub: `https://github.com/ughirose/usir.git`）  
> **Google ドライブ同期パス**: `F:\マイドライブ\USIR_Session_Handoff_20261102\`（Folder ID: `13f8sUR57RFRQhiVB_P5iytWsbtZVpUkQ`）  
> **稼働モデル前提**: Antigravity の Gemini 4 Argon クォータ枯渇に伴い、**メインセッション（司令塔）および全サブエージェント（`invoke_subagent`）の双方を `Gemini 3.8 Flash (High)` で完遂する専用設計**。

---

## 1. 迎合の完全排除と確立された「真のアーキテクチャ3大原則」

1. **「物差し（辞書）」の事前確定（Step 0 先行実行）**
   - 辞書を後回しにして文字種カウントや部首テーブルだけで進める妥協を全廃する。
   - **Step 0 として、まず Colab 上位GPU（L4/A100）またはローカル環境で `compile_hth_clean_poly.py` を一括実行し、480,000 Bytes（30,000語 × 16B）の多義対応バイナリ `hth_poly_30k.bin` を確定させる**。以降の全フェーズはこの固定バイナリを高速参照する。
2. **ヘキサゴナル・アーキテクチャ（依存性の逆転）による構文・言語層の純粋性死守**
   - 「自前の正規表現オモチャパーサ」を作る退行を完全禁止する。
   - USIR コア（`packages/usir-core` / `python/usir_py`）は、外部の形態素解析器ライブラリ（Janome / Sudachi 等）に一切依存せず、**正規化された構文フレーム `SyntaxTreeFrame`（トークン列、品詞、基本形、係り受け辺、深さ）を受け取るだけの純粋な数理・状態遷移エンジン（Port）** として設計する。
   - 形態素や係り受けの解決は、外部のアダプタ層（Driver / CLI / Runner）が担当してコアへ注入する。これにより、USIR コアのゼロ依存性と Python / TypeScript 間の出力誤差 $< 10^{-5}$ を両立する。
3. **Gemini 3.8 Flash 専属完遂のための「5大絶対規律（Flash Guardrails）」**
   - **規律1（設計書の事前読み込み）**：実装前に必ず `C:\dev\usir\docs\blueprints\` 内の詳細仕様書を `read_file` で読み込ませること。
   - **規律2（ダミー・ハリボテ実装の厳禁）**：単体 `LinearLoRAAdapter(64,64)` への乱数学習、`cv_l` を使った自己採点型 Judge、固定コサイン加算だけの名ばかり BAOAB を完全に排除する。
   - **規律3（小規模差分編集）**：60秒 SSE EOF タイムアウトを防ぐため、1回のファイル変更は最大 40〜60 行以内の関数単位差分とし、ファイル一括置換を行わないこと。
   - **規律4（`Model="flash"` の明示）**：`invoke_subagent` 呼び出し時は必ず `Model="flash"` を明示指定し、1サブエージェントあたり 1〜2 ファイルの直交タスクに限定すること。
   - **規律5（2ストライク・即時転換）**：同一アプローチで 2 回失敗した場合は即座に別アプローチへ切り替えること。

---

## 2. 全体実装ロードマップ（Step 0 ＋ Phase A〜E）

```text
【Step 0: 物差しの確定（Colab L4/A100 / ローカルによる 480KB 辞書ビルド）】
  ・WordNet + Lancaster感覚規範 + 分類語彙表から 128-bit 多義対応辞書 `hth_poly_30k.bin` をビルド・確定。
       ▼
【Phase A: ヘキサゴナル構文インターフェース ＆ OPEN-01〜05 ＆ 多重解像度計装】
  ・USIRコアは純粋な SyntaxTreeFrame 受容ポートとして定義。
  ・480KB 辞書と構文木上で OPEN-01〜04、対数3等分ウェーブレット、ISTAソリトン分離、Contrastive Fisher を本物化。
       ▼
【Phase B: 階層的デュアルレート BAOAB × 節単位 Twisted SMC × 3,000字 HSMM 長尺生成】
  ・節境界での 24D 実測フィードバック＋5ステージ Adaptive-BAOAB（可変摩擦熱浴）＋Twisted SMC（P=4粒子）。
  ・Colab L4/A100 で 3,000字長文生成を完走。
       ▼
【Phase C: 実Transformer直結 QLoRA / USIR-GRPO 鍛造 ＆ 中間層 Activation Steering ＆ 独立Judge】
  ・実モデル（Qwen2.5）の Attention 重みに直結した本物の GRPO 鍛造（推論遅延 0.00ms）と潜在空間電位注入。
       ▼
【Phase D & E: OSWE usir-forge ＆ Wasm SIMD ＆ Nishiki 統一場 ＆ 普遍クロスモーダル関手】
```

---

## 3. Antigravity 投入用コピペ実行プロンプト集

Antigravity で新規チャットを開き、モデルを **`Gemini 3.8 Flash (High)`** に設定した上で、以下のプロンプトを順に投入してください。

### 【Prompt 0：Step 0 投入用（480KB 多義対応辞書 `hth_poly_30k.bin` の一括ビルド）】

```markdown
/boost USIR Step 0: Build Deterministic 128-bit Dual-Sense Lexicon `hth_poly_30k.bin` (Strictly 480,000 Bytes)

あなたは USIR プロジェクトのリード実装統括（Gemini 3.8 Flash）です。
まず最初に、以下のワンライナーを実行して Google ドライブ同期フォルダ（`F:\マイドライブ\USIR_Session_Handoff_20261102\09_USIR_FLASH_BLUEPRINTS_BUNDLE.json`）から詳細設計書群を `C:\dev\usir\docs\blueprints\` に自動展開してください：
```powershell
python -c "import json, pathlib; b = json.load(open(r'F:\マイドライブ\USIR_Session_Handoff_20261102\09_USIR_FLASH_BLUEPRINTS_BUNDLE.json', encoding='utf-8')); out = pathlib.Path('docs/blueprints'); out.mkdir(parents=True, exist_ok=True); [(out / k).write_text(v, encoding='utf-8') for k, v in b.items()]; print('Unpacked:', list(b.keys()))"
```
展開後、`C:\dev\usir\docs\blueprints\10_SPEC_PHASE_A_HEXAGONAL_AND_HTH_LEXICON.md` の第1節を `read_file` で読み込んでください。

### 絶対遵守事項:
- 辞書を後回しにせず、最初に物差しを確定させます。
- `python/scripts/compile_hth_clean_poly.py` を実装・実行し、30,000語 × 16 Bytes ＝ 厳密に 480,000 Bytes のバイナリ `python/usir_py/data/hth_poly_30k.bin` をビルドしてください。
- 外部ファイルが存在しない環境でも即座に検証できるよう、決定論的シンセティック補完モード（`--synthetic`）と実データ結合モードの双方をサポートし、単体テスト `python/tests/test_hth_poly_lexicon.py` を 100% 通過させて成果物をコミット＆Pushしてください。
```

### 【Prompt 1：Phase A 投入用（ヘキサゴナル構文インターフェース ＆ OPEN-01〜05 ＆ 多重解像度計装の本実装）】

```markdown
/boost USIR Phase A: Hexagonal SyntaxTreeFrame Interface, 480KB HTH Binding, OPEN-01..05 Modules, and Stage 1->2 Multiresolution Core

まず最初に、`C:\dev\usir\docs\blueprints\10_SPEC_PHASE_A_HEXAGONAL_AND_HTH_LEXICON.md` の第2節〜第4節を `read_file` で読み込んでください。

### 絶対遵守ルール（迎合の完全排除）:
- サブエージェントは必ず `invoke_subagent(TypeName="DeepCoder", Model="flash", Workspace="inherit", ...)` で起動し、1サブエージェントあたり1〜2ファイルに限定すること。
- 自前の正規表現オモチャパーサを作る退行を厳禁とする。USIR コアは外部依存ゼロの「`SyntaxTreeFrame` 入力インターフェース」として設計し、形態素・係り受けは外部 Driver（Adapter）から注入させること。
- 1回のファイル編集は最大 40〜60 行以内の関数単位差分とし、ファイル一括置換を行わないこと。

### 委譲ワークストリーム:
1. **WS-A1（ヘキサゴナル SyntaxTreeFrame ＆ 480KB HTH 参照 ＆ OPEN-01〜04 実装）**:
   - `python/usir_py/syntax_frame.py`（および TS 側 `packages/usir-core/src/schema/syntaxFrame.ts`）に `SyntaxNode`、`SyntaxEdge`、`SyntaxTreeFrame` の純粋データ構造を定義せよ。
   - `python/usir_py/hth_lexicon.py`（および TS 側 `packages/usir-core/src/sensor/hthLexicon.ts`）を実装し、Step 0 で作成した `hth_poly_30k.bin`（480KB）を直接メモリマップまたは ArrayBuffer で $O(1)$ 参照して、物理語義（High 64bit）と抽象語義（Low 64bit）を瞬時に取得する関数を実装せよ。
   - `python/usir_py/linguistic_gates.py` に `OPEN-01`（具体度減衰 $\mathcal{C}(u)$）、`OPEN-02`（否定双対感覚テンソル $\tilde{\mathbf{s}}^+, \tilde{\mathbf{s}}^-$）、`OPEN-03`（オノマトペ音韻象徴合成 $\Psi_{\text{mimetic}}$＋Mimetic Immunity）、`OPEN-04`（地の文 $\mathbf{z}_{\text{nar}}$ / 会話文 $\mathbf{z}_{\text{dia}}^{(c)}$ バフチン分離）を実装せよ。
   - 外部 Driver（`python/usir_py/drivers/syntax_driver.py`）を作成し、外部形態素・係り受け解析器（利用可能な環境では Janome/Sudachi、CI環境ではクリーンルーム構文フィクスチャ）から `SyntaxTreeFrame` を構築して USIR コアへ注入するパイプラインを確立せよ。

2. **WS-A2（Stage 1 $\to$ Stage 2 多重解像度ウェーブレット × ISTA × RQA ＆ Contrastive Fisher 計量学習）**:
   - 文系列の 4基礎信号 $\mathbf{s}_m \in \mathbb{R}^4$ から、対数3等分ウェーブレット分解、Presence/Void 直交指示関数分解、ISTA スパース形態素分離、RQA リカレンス率 $I_{\text{arch}}$ を経て 12元素 $(\mathbf{q}, \mathbf{p}) \in \mathbb{R}^{24}$ を抽出するパイプラインを `extractor_24d.py` および `symplectic24d.ts` に実装せよ。
   - `manifold_geodesic.py` に `fit_contrastive_fisher_metric` を実装し、正例/負例クラスタから引き戻しリーマン計量 $G_{\text{style}}$ を対照学習で推定せよ。
   - `python scripts/harness.py test` を実行し、全テスト 100% PASS を確認してコミット＆Pushせよ。
```

### 【Prompt 2：Phase B 投入用（デュアルレート BAOAB × 節単位 Twisted SMC × 3,000字 HSMM 長尺生成）】

```markdown
/boost USIR Phase B: Genuine Dual-Rate 5-Stage Adaptive-BAOAB, Clause-Level Twisted SMC (P=4 Particles), and 3,000-Char 3-Mode HSMM Long-Form Generation on Colab GPU

まず最初に、`C:\dev\usir\docs\blueprints\11_SPEC_PHASE_B_DUAL_RATE_BAOAB_SMC_HSMM.md` を `read_file` で読み込んでください。

### 絶対遵守ルール:
- サブエージェントは必ず `Model="flash"` を指定し、関数単位の小規模編集（40〜60行以内）で進めること。
- Colab 実行時は上位GPU（`accelerator="L4"` または `"A100"`、フォールバック `"T4"`）および十分なタイムアウト（`timeout=600`）を設定し、モデルには `Qwen/Qwen2.5-3B-Instruct`（または `7B`）を使用すること。

### 委譲ワークストリーム:
1. **WS-B1（デュアルレート 5ステージ Adaptive-BAOAB ＆ 節単位 Twisted SMC の本実装）**:
   - `python/usir_py/realtime_decoder.py` に、節境界ごとの 24D 実測フィードバック $\hat{\mathbf{q}}_m$、`OPEN-08` Ghost History カルマン収縮、および可変摩擦 $\Gamma_t(H_{\text{LLM}})$ 下でギブス平衡分布を不変に保つ 5ステージ Adaptive-BAOAB 積分器（定理5.1）を本実装せよ。
   - `ClauseTwistedSMCDecoder` を実装し、1節ごとに $P=4$ 粒子を展開して文末到達尤度 $\ln P_{\text{LLM}}(W_{\text{end}}^* \mid w_{\le m} \circ c^{(p)})$ と $G_{\text{style}}$ 追従誤差の和で毎節リサンプリングする粒子フィルタを構築せよ。
2. **WS-B2（3モード HSMM ＆ Momentum Banking による 3,000字長尺生成ベンチマーク）**:
   - `python/usir_py/hsmm_longform_director.py` を実装し、地の文・会話・クライマックスの3井戸遷移行列 $T_{ij}$ と `OPEN-09` Momentum Banking、`OPEN-10` Wundt-Berlyne 不応期ゲートを連動させよ。
   - Colab GPU 実機で全 3,000 字規模の長文生成実験を完走させ、後半段落でも $CV_L \in [0.40, 0.70]$ が維持されることを検証して成果物をコミット＆Pushせよ。
```

### 【Prompt 3：Phase C 投入用（実モデル直結 USIR-GRPO QLoRA 鍛造 ＆ 中間層 Activation Steering ＆ 独立LLM Judge）】

```markdown
/boost USIR Phase C: Transformer-Attached USIR-GRPO LoRA Forge, Level-2 Residual Stream Activation Steering, and Independent Blind LLM Judge

まず最初に、`C:\dev\usir\docs\blueprints\12_SPEC_PHASE_C_D_E_GRPO_LORA_STEERING_AND_FUNCTORS.md` の Phase C セクションを `read_file` で読み込んでください。

### 委譲ワークストリーム:
1. **WS-C1（実モデル Attention 直結 QLoRA / USIR-GRPO 鍛造の実装）**:
   - `python/colab/run_v4_bidirectional_experiment.py` 内のダミー `LinearLoRAAdapter(64,64)` を完全削除し、実モデル `model`（`Qwen2.5`）の Attention 層（`q_proj`, `v_proj`）に直結する `AttachedLoRALinear` を実装せよ。
   - 参照モデル PPL バリア $\mathcal{M}_{\text{semantic}}(\text{PPL}_{\text{ref}})$ と KL 項 $\beta_{\text{KL}} D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})$ を含む4重積報酬で実トークン対数確率を政策勾配更新し、訓練後アダプタ単体（推論追加遅延 `0.00 ms`）で Deadzone 脱出と品質向上を実証せよ。
2. **WS-C2（Level 2: 中間層 12元素神経回路直結 `SymplecticActivationSteerer` ＆ 独立 LLM Judge）**:
   - `python/usir_py/activation_steerer.py` を実装し、中間層 Residual Stream から 12元素の直交ステアリング行列 $V_{\text{steer}}$ を抽出して `register_forward_hook` で直接電位注入せよ。
   - `evaluate_llm_judge` から数式自己採点を撤去し、独立した Judge システムプロンプトによる実 LLM 盲検採点へ刷新せよ。
```

### 【Prompt 4：Phase D & E 投入用（OSWE `usir-forge`・Wasm SIMD・Nishiki統一場・WorldCraftヌル空間・6大モダリティ関手）】

```markdown
/boost USIR Phase D & E: OSWE Polar-Fourier usir-forge, Wasm SIMD, Nishiki Genre Temperature Unified Field, WorldCraft Null-Space Arbitration, and Cross-Modal Functors

まず最初に、`C:\dev\usir\docs\blueprints\12_SPEC_PHASE_C_D_E_GRPO_LORA_STEERING_AND_FUNCTORS.md` の Phase D 及び E セクションを `read_file` で読み込んでください。

### 委譲ワークストリーム:
1. **WS-D1（OSWE ストリーミング蒸留・極座標フーリエ位相アライメント・4KB 3モードGMM `usir-forge` ＆ Wasm SIMD）**:
   - 極座標フーリエ位相アライメント（クライマックス位相角 $\theta_{\text{peak}}$ 整列）により、長編全集から 4,096 Bytes の 3モード GMM 精度行列バイナリを完全出力する `usir_forge.py` / `usirForge.ts` を実装せよ。
   - `packages/usir-wasm` の SIMD カーネルをビルド・検証せよ。
2. **WS-E1（Nishiki 統一場 $\tau_{\text{genre}}$・WorldCraft ヌル空間調停・視覚/音楽クロスモーダル関手）**:
   - Toulmin 論理監査ポテンシャル $U_{\text{Toulmin}}$ とジャンル温度 $\tau_{\text{genre}}$ 統一場（`OPEN-06`）、WorldCraft オントロジー制約の零空間射影 $(I - J_C^\dagger J_C)\nabla \mathcal{F}_{\text{USIR}}$（`OPEN-09`）、視覚関手 $\mathcal{F}_{\text{visual}}$（実画像 Gabor/Quadtree/Saliency 解析 $\leftrightarrow$ Diffusion 制御）、および音楽関手 $\mathcal{F}_{\text{music}}$（GTTM/五度圏解析 $\leftrightarrow$ WebAudio/MIDI 合成）を本実装し、`python scripts/harness.py test` 全件 PASS を確認せよ。
```
