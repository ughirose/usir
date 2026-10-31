# 【Stage 3〜4 深層開拓ロードマップ】USIR の「未踏の6大フロンティア」と Antigravity 実装指令書
> 対象ワークスペース: `C:/dev/usir`（Stage 1 & Stage 2 完了・コミット `a67331e` 時点）
> 監査結論: Stage 2 完了によって「1段落の静的24D計装器（Sensor）」と「16チャンクのサンプル実証」は完成したが、**USIR（Universal Style Genome）本来の構想の約60%（動的軌跡HSMM・マイ文体蒸留コンパイラ・様式代数・ランタイムLLM閉ループ・ミクロ禁則自動ゲート・可視化オシロスコープ）はまだこれから実装すべき本丸として残っている。**

---

## 1. 冷徹な現在地監査：なぜ「もっとやることがある」という直観が100%正しいのか？

Stage 2 の `walkthrough.md` を精査すると、現在の `C:/dev/usir` が達成したことと、**まだ「静的デモ・1段落平均」に留まっている限界** が明確に浮かび上がる。

### 1.1 Stage 2 までに完成したもの（全体の約40%：基礎計装と静的証明）
* **1段落の平均24次元ベクトル $(\mathbf{q}, \mathbf{p})$ を測る物差し**（LOOCV 97.5%）。
* **ハミルトン変調プロンプト文字列を出力する関数**（`generateRewriteDirectives`）。
* **サブエージェントがセッション中に作成・手直しした16個の固定サンプル文字列**（`closedLoopCorpus.ts`）。

### 1.2 まだ実装されていない「6つの本丸（残りの60%）」
1. **【死角①：実はまだ「任意の新しい文章」を自動閉ループ変換するランタイムが存在しない】**
   * Stage 2 の16チャンク（`closedLoopCorpus.ts`）は、Antigravity のサブエージェントが開発セッション中に生成・修正して TypeScript ファイルにハードコード保存した「静的スナップショット」である。
   * 今日あなたが新しい原稿（例：note記事の下書きや小説の1シーン）を入力したとき、裏で **ローカルLLM（RTX 4070 上の Ollama / LM Studio）またはクラウドAPI（Gemini / Claude）を実際に呼び出し、自動で「生成 $\to$ 24D計測 $\to$ ミクロ禁則検査 $\to$ 誤差 $\Delta\mathbf{z}$ 再フィードバック」を回して収束させるランタイム実行器（`LiveClosedLoopRunner`）** がまだ実装されていない。
2. **【死角②：Stage 2 の盲検監査が見つけた「6つの意味破綻・重言・時代錯誤」がまだ自動コード化されていない】**
   * Stage 2 で Blind QA サブエージェントが摘出した6件の欠陥（`青い青葉` という重言、`匂いが土を濡らす` という主述破綻、`汽車が跳躍した` というキーワード無理押し、`猛スピードで` という時代レジスター違反）は、人間の代わりにサブエージェントが目視で直したものである。
   * これを **決定論的な「ミクロ禁則・レジスター・主述共起ゲート（`MicroConstraintGate`）」としてコード化** しなければ、実運用で LLM にリライトさせた際に同じ破綻が再発する。
3. **【死角③：1段落を「1個の平均点」に潰しており、Tier M 本来の『3モードHSMM（序・破・急の時系列波形）』がまだ動いていない】**
   * Tier M（4096B）のバイナリ枠は作ってあるが、現在の分類器はチャンク全体の平均ベクトルしか見ていない。
   * 志賀直哉の『城の崎にて（いもりの死）』がなぜ凄いかといえば、「静かな観察（Mode 0: 序）」$\to$「石が当たる突発的衝撃（Mode 1: 破・Shock）」$\to$「尾が静かに下りて動かなくなる絶対静寂（Mode 2: 急・Deep Void）」という **文単位の相転移軌跡（Trajectory）** があるからである。文ごとのスライディング窓で **Viterbi 状態遷移系列（序破急の波形）を抽出・設計するエンジン** が未実装である。
4. **【死角④：任意のテキスト群から『4KBのマイ・スタイル粒子（`.usir` ファイル）』を蒸留するコンパイラがない】**
   * 今あるのは固定された4つのプリセット（琥珀・黒曜・白磁・紫電）だけである。
   * あなた自身の過去の note 記事や小説、あるいは特定のターゲット文体群（Markdown フォルダ）を読み込ませ、Dirty Room $\to$ 7-gram 監査ゲート $\to$ Clean Room を経て **あなた固有の `zero_style.usir`（4096B）を自動生成する `usir extract-genome` コンパイラ** がまだ存在しない。
5. **【死角⑤：『様式代数（Style Algebra）』と『ジャンル間移調（Transposition）』が未実装】**
   * 2つの `.usir` 粒子の引き算（例：`あなたの文体 - AIスロップ成分` の直交射影）、リーマン測地線上の連続モーフィング、および「論理骨格（第II族 Lattice）を100%固定したまま、肌理と呼吸（第III族 Void / 第IV族 Shock）だけを別ジャンルへ移調する」演算子が未実装である。
6. **【死角⑥：長編・章単位の『呼吸オシロスコープ UI（USIR Studio）』とローカルSLM用の DPO/GRPO データ自動生成工場】**
   * 数千〜数万字の原稿を流し込んだときに、「どこで AI Deadzone（中だるみ）に落ちているか」を一目で可視化するオシロスコープと、RTX 4070 (12GB) でローカルSLM自体を鍛えるための合成データ生成パイプラインが未実装である。

---

## 2. これから実装すべき「6大フロンティア」の全体マップ（Stage 3 & Stage 4）

| Stage | フロンティア領域 | 新規実装モジュール | これが完成すると何が起きるか（提供価値） |
| :--- | :--- | :--- | :--- |
| **Stage 3-A** | **ミクロ禁則・レジスター・重言自動ゲート**<br>（Stage 2 摘出欠陥の完全アルゴリズム化） | `packages/usir-core/src/cleanroom/microConstraintGate.ts` | Stage 2 で盲検監査が見つけた「重言（`青い青葉`）」「時代レジスター違反（和文・時代小説内の現代カタカナ俗語）」「禁止直喩リーク」「キーワード無理押し」を **ミリ秒単位で自動検知・リジェクト** できるようになる。 |
| **Stage 3-B** | **Tier M 本丸：3モード HSMM 軌跡デコーダ＆序破急シーケンサ** | `packages/usir-core/src/dynamics/trajectoryHsmm.ts` | 文章を1個の平均点ではなく、文単位のスライディング窓で **`[Mode 0: 静/序] → [Mode 1: 動/破] → [Mode 2: 余韻/間]` の状態遷移波形** として解読・設計できるようになり、長文の「中だるみ（Deadzone 陥落区間）」をピンポイントで特定できる。 |
| **Stage 3-C** | **マイ・スタイル粒子コンパイラ（`extract-genome`）＆様式代数（Style Algebra）** | `packages/usir-core/src/algebra/genomeCompiler.ts`<br>`styleAlgebra.ts` | 自分の過去記事や任意のコーパスから **7-gram 漏洩ゼロの固有 `.usir`（4096B）を1コマンドで蒸留** でき、「AI臭の直交引き算（$\mathbf{z} \ominus \mathbf{z}_{\text{slop}}$）」や「論理仕様書 $\to$ 臨界エッセイへの移調」が可能になる。 |
| **Stage 3-D** | **実機ランタイム閉ループ・エグゼキュータ**<br>（Ollama RTX 4070 / Gemini API 対応） | `packages/usir-mcp/src/runtimeLoop.ts` | 固定サンプルではなく、**任意のテキストを投入すると裏でローカル Ollama (RTX 4070) または LLM API を最大3イテレーション自動呼び出しし、24Dゲート＋ミクロ禁則ゲートを両方パスするまで自動推敲して返す** 本物の自律調律エンジンになる。 |
| **Stage 4-A** | **RTX 4070 用 DPO / GRPO 選好ペア自動工場** | `python/usir_py/preference_factory.py` | 任意の生テキストから `(Rejected: Deadzone, Chosen: Void/Critical)` の学習ペア（JSONL）を自動大量生産し、ローカルSLM / Narrative-Nano の LoRA 学習データセットを無尽蔵に作れる。 |
| **Stage 4-B** | **USIR Studio（5相オシロスコープ＆イコライザー GUI）＋ Plotailor 結合** | `packages/usir-bridges/src/studioVisualizer.ts` | 原稿全体の文単位の5相軌跡と12元素スペクトルをブラウザ上で視覚化し、Deadzone 陥落段落を1クリックで相転移させる GUI が動く。 |

---

## 3. コピペ即稼働：Antigravity 投入用 `/goal Stage 3` マスタープロンプト

`C:/dev/usir` の Antigravity に以下のプロンプトをそのまま貼り付けることで、上記の **Stage 3（A〜D の4大コア拡張）** を一気通貫で実装・検証できます。

```markdown
/goal Stage 3: USIR を「1段落の静的分析器」から「時系列軌跡（3-Mode HSMM）・マイ文体蒸留コンパイラ・様式代数・ミクロ禁則ゲートを備えた動的ランタイムエンジン」へ進化させる。

### 目的と背景:
Stage 2 までの実装で1段落の平均24D計装とプロンプト出力は完成したが、まだ以下の4つの本質的機能が未実装である：
1. Stage 2 の盲検監査が人間的に摘出した欠陥（重言「青い青葉」、時代レジスター逸脱「猛スピードで」、禁止直喩のすり抜け、無理押し動詞）を自動検知する **決定論的ミクロ禁則ゲート（`microConstraintGate.ts`）**。
2. 1段落を1点に平均化するのではなく、複数文・複数段落にわたる **「序（Setup）→ 破（Rupture/Shock）→ 間（Deep Void）」の時系列状態遷移（3-Mode HSMM Trajectory）を解読・中だるみ検知する `trajectoryHsmm.ts`**。
3. 任意のテキスト群から Clean Room 監査を経て **固有の 4096B `.usir` 粒子ファイルを蒸留する `genomeCompiler.ts`** と、**AIスロップ直交引き算・リーマン測地線モーフィング・ジャンル移調を行う `styleAlgebra.ts`**。
4. 任意の入力テキストに対して「プロンプト生成 → LLM/SLMアダプタ呼び出し → 24D計測＋ミクロ禁則検査 → 残差 $\Delta\mathbf{z}$ 再フィードバック」を自動実行する **ランタイム閉ループ実行器（`runtimeLoop.ts`：ローカル Ollama HTTP エンドポイント / Gemini API / Pluggable Provider 対応）**。

### 実行ステップ（Flash サブエージェント並列体制＋最終 Blind QA 監査で遂行せよ）:

#### Step 1: `microConstraintGate.ts`（Stage 2 摘出欠陥の自動ゲート化）
`packages/usir-core/src/cleanroom/microConstraintGate.ts` を新規作成し、以下の5つのミクロ品質検査を実装せよ：
1. **重言・同根反復検知 (`detectTautology`)**: 同一文節・隣接語での同字反復（例：`青い青葉`, `白い白雪`, `暗い闇夜` 等、同一漢字または訓読み語根が直近3文字以内に重複するパターン）を検知。
2. **時代・文体レジスター逸脱検知 (`detectRegisterViolation`)**: `registerMode: 'classical_literary' | 'modern_neutral' | 'technical_crystalline'` に応じ、`classical_literary` 指定時に現代カタカナ俗語（`スピード`, `システム`, `レベル`, `ストレス`, `タイミング`, `イメージ`, `メリット` 等）が混入した場合にペナルティを返す。
3. **ターゲット相別ハード禁則検査 (`verifyPhaseHardConstraints`)**:
   - `III_Void` 指定時：直喩マーカー（`まるで`, `ように`, `ごとき`）、感嘆符・疑問符（`！`, `？`）、および `ABSTRACT_FILLERS` が **1件でも存在すれば即座に `passed: false`** とし、違反語と行位置を返す。
   - `II_Critical` 指定時：`ABSTRACT_FILLERS` の残存が `0` であり、かつ第IV族 Shock 要素が物理描写を伴っていることを検証する。
4. **不自然キーワード無理押し検知 (`detectForcedKeywords`)**: 文脈と無関係に指標を稼ぐためだけの唐突なメタ語彙（`跳躍した`, `臨界`, `破調`, `シンプレクティック` 等）の混入を検知する。

#### Step 2: `trajectoryHsmm.ts`（3-Mode HSMM 時系列波形解読＆中だるみ Deadzone ピンポイント検知）
`packages/usir-core/src/dynamics/trajectoryHsmm.ts` を新規作成し、以下を実装せよ：
1. **スライディング窓24D軌跡抽出 (`extractTrajectoryWave`)**:
   - 長文（複数段落または10文以上）を入力とし、3文スライディング窓（ストライド1〜2文）で局所24次元状態 $\mathbf{z}_t = (\mathbf{q}_t, \mathbf{p}_t)$ と局所相 $P_t \in \{\text{Crystalline, Critical, Void, Turbulent, Deadzone}\}$ の時系列配列を算出する。
2. **Viterbi 3-Mode デコーダ (`decodeHsmmModes`)**:
   - 軌跡系列 $\{\mathbf{z}_t\}$ に対し、`Mode 0 (静・序: Low Shock / High Lattice or Void)`、`Mode 1 (動・破: High Shock & Momentum)`、`Mode 2 (余韻・間: High Ma-Void & Clean Cut)` の3状態隠れマルコフ遷移確率行列 $A \in \mathbb{R}^{3\times 3}$ と持続長分布を推定し、実際に `usirBinarySchema.ts` の **Tier M (`4096B`) 構造体の3モード領域へ実データをパッキング** できるようにせよ。
3. **中だるみ（Deadzone Sinkhole）ピンポイント検出 (`detectDeadzoneSinkholes`)**:
   - 長文の中で局所的に $CV_L < 0.28$ かつ $\kappa_{\text{kin}} < 0.25$ に陥落している「中だるみ段落・文インデックス範囲」を特定し、そこだけを局所外科手術（Targeted Micro-Intervention）するための診断レポートを出力せよ。

#### Step 3: `genomeCompiler.ts` & `styleAlgebra.ts`（マイ文体4KB蒸留＆様式代数）
`packages/usir-core/src/algebra/` 配下に以下を実装せよ：
1. **`genomeCompiler.ts` (`compileStyleGenomeToTierM`)**:
   - 複数のサンプルテキスト配列 `texts: string[]` を受け取り（Dirty Room）、全チャンクの24D状態と3モードHSMM遷移行列・Ledoit-Wolf 収縮コレスキー下三角精度因子 $L_k$ を計算し、`threeRoomGate`（7-gram 漏洩ゼロ監査）を通過させた上で、厳密な **4096バイトの `.usir` バイナリバッファ** を出力するコンパイラ。
   - 実際に青空文庫の志賀直哉チャンク群から蒸留した `shiga_void.usir` (4096B) と、芥川・梶井チャンク群から蒸留した `kajii_critical.usir` (4096B) を生成・検証するテストを含めよ。
2. **`styleAlgebra.ts`（様式代数の3大演算子）**:
   - **① 反スロップ直交射影 (`projectOutSlopSubspace(z, alpha)`)**: 状態ベクトル $\mathbf{z}$ から `V_Deadzone` 重心方向の成分を直交減算し、純粋な作家固有の特徴ベクトルを抽出・増幅する演算。
   - **② リーマン・コレスキー測地線補間 (`interpolateGeodesic(particleA, particleB, t)`)**: 2つの Tier M 粒子間を $t \in [0, 1]$ で滑らかにモーフィングする演算。
   - **③ ジャンル間移調 (`transposeStyleRegister(sourceText, skeletonGroup, targetTextureParticle)`)**: 原文の **第II族（Lattice：論理構造・主張骨格）** を維持したまま、**第III族（Void）または第IV族（Shock）と運動量 $\mathbf{p}$** だけをターゲット粒子へ移調するプロンプト・目標ベクトル合成器。

#### Step 4: `runtimeLoop.ts`（Pluggable LLM / Ollama ランタイム閉ループ実行器）＆ CLI 拡張
`packages/usir-mcp/src/runtimeLoop.ts` を新規作成し、`packages/usir-mcp/src/cli.ts` を拡張せよ：
1. **`runClosedLoopPhaseTransition(text, targetPhase, llmProvider, maxIterations = 3)`**:
   - `llmProvider: { generate(prompt: string): Promise<string> }` インターフェース（ローカル Ollama `http://localhost:11434/api/generate`、OpenAI互換エンドポイント、Gemini API、またはテスト用モックプロバイダを差し替え可能）を受け取る。
   - 各イテレーションで `generateRewriteDirectives` / `compileIterativeFeedbackPrompt` $\to$ `llmProvider.generate()` $\to$ `extractUsir24d()` ＋ `microConstraintGate()` を実行し、**「24Dターゲット相到達」かつ「ミクロ禁則違反ゼロ」** の両方を満たした時点で即座に収束・返却する本物の閉ループランタイムを実装せよ。
2. **CLI サブコマンド追加 (`usir-audit`)**:
   - `extract-genome <input-json-or-txt> -o <output.usir>`: マイ文体 `.usir` (4096B) の蒸留・クリーンルーム出力
   - `trajectory <file>`: 長文の文単位5相軌跡と中だるみ（Deadzone 陥落箇所）の診断出力

#### Step 5: 独立 Blind Adversarial QA サブエージェントによる全機能検証
- すべての実装と単体テスト完了後、履歴を遮断した Blind QA サブエージェントを起動し、以下を検証せよ：
  1. Stage 2 で問題になった6つの欠陥テキスト（`青い青葉`、`匂いが黒い土を濡らす`、`汽車が跳躍した`、`猛スピードで`、`冷たい砂のように` 等）を `microConstraintGate` に通した際、**100% すべて違反として自動検出・ブロックされること**。
  2. 長文の中に1段落だけ `DEAD-01`（AIスロップ）を混ぜたハイブリッド長文を `trajectoryHsmm` に通した際、**その段落のインデックスが正確に「Deadzone Sinkhole（中だるみ箇所）」としてピンポイント特定されること**。
  3. `compileStyleGenomeToTierM` で生成された `4096B` バイナリが 7-gram 漏洩ゼロであり、デシリアライズ後にそのまま `mixStyleParticles` や `interpolateGeodesic` で演算できること。
```
