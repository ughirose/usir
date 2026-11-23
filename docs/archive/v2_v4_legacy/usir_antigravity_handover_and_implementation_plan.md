# 【実装引継書・開発計画マスター】Universal Style Genome (USIR / Nishiki) Antigravity 実装・オーケストレーション完全設計書
> 対象リポジトリ: `ughirose/usir`（独立コア: `@nishiki/usir-core`, `@nishiki/usir-mcp`）＋ ブリッジ連携（`narrative-nano` | `plotailor` | `worldcraft`）
> ターゲット環境: Google Antigravity (v2.5+ Desktop App / VS Code Extension / `agy` CLI) × Google Jules（クラウド15並行ワーカー） × ローカル実機（Core i7-14700F / 64GB RAM / RTX 4070 12GB）
> 関連タスクID: `TASK-360` 〜 `TASK-367`（新規起票用）
> 憲法遵守チェック: 独立スタンドアロン稼働（WorldCraft/Plotailor非依存） / Docked Atelier（3ペイン統合IDE＋明示的モーダル併用・自動割り込み禁止） / 3層クリーンルーム（著作権非侵害） / Native-First（過剰Docker排除・`pnpm` + `uv` + `emcc`）

---

## 1. エグゼクティブ・サマリーと設計基本方針（Architecture Principles）

本書は、理論検証および実データ40チャンクでの5相分離実証（未知サンプル LOOCV 識別精度 **97.5%**）を完了した **Universal Style Genome（USIR / 錦 Nishiki 普遍スタイル周期表・ハミルトン表現力学エンジン）** を、Google Antigravity 上で即座に実装フェーズへ移行するためのマスター引継書である。

### 1.1 実装における「4つの絶対構造原則」
1. **WorldCraft / Plotailor に依存しない完全スタンドアロン設計（Standalone-First）**
   * USIR コアエンジン（`@nishiki/usir-core`）および MCP サーバー（`@nishiki/usir-mcp`）は、外部エディタや特定のアプリケーションに一切依存しない **純粋な TypeScript / Wasm SIMD / Python パッケージ** として独立リポジトリ（または独立ワークスペースパッケージ）に構築する。
   * これにより、単体の **MCP サーバー / CLI 検品ゲート / B2B SDK** としてあらゆる生成AIパイプライン（Claude, Gemini, Dify, ローカルSLM）へ外付けでき、その上で `plotailor`・`worldcraft`・`narrative-nano` へは軽量なアダプタ（Bridge）経由で接続する。
2. **Tier S / Tier M への集中と Tier L の分離（Scope Discipline）**
   * **Tier S（128B：リアルタイム24Dシンプレクティック計装＆5相レーダー）** と **Tier M（4KB：3モードHSMM・ハミルトニアン運動量ガイダンス・1段落局所相転移イコライザー）** に開発リソースの90%を集中する。
   * **Tier L（64KB：長編アーク・マルチエージェント調停）** は最小限のインターフェース定義とリファレンスサンプル（1ファイル）のみに留め、過剰な作り込みによるスケジュール遅延を防ぐ。
3. **Native-First & ゼロ仮想化オーバーヘッド（KISS原則）**
   * ローカル実機（Core i7-14700F, 64GB RAM, RTX 4070 12GB）の性能を最大化するため、重い Docker コンテナ群は使わず、TypeScript/Wasm 側は **`pnpm` + `Vitest` + `Emscripten (emcc)`**、Python 実証・LoRA/GRPO 報酬サーバー側は **`uv`（ネイティブ仮想環境）** で高速実行する。
4. **盲検敵対的検証（Blind Adversarial QA）の組み込み**
   * Antigravity の `/teamwork`（Git Worktree 分離サブエージェント）と `/grill-me` を活用し、実装エージェントとはコンテキストを100%遮断した **監査専用サブエージェント** が「ターゲット語彙リーク（著者名・作品名ハードコード）の有無」「3層クリーンルームの7-gram遮断」「LOOCV 95%以上維持」を自動検査する。

---

## 2. リポジトリ・ディレクトリ構造（Standalone + Ecosystem Bridge）

`ughirose/usir`（またはモノレポ内の独立ルート `packages/usir-*`）として以下のディレクトリ構造を採用する。各モジュールの編集対象ファイルは **100%直交（Orthogonal）** しており、Antigravity サブエージェントや Google Jules で最大6並行の同時開発が可能である。

```text
usir/ (Repository Root: ughirose/usir)
├── .antigravity/
│   ├── rules.md                        # Antigravity 常駐アーキテクチャ憲法・禁止事項
│   └── skills/
│       ├── usir-math-verifier/SKILL.md # 24Dシンプレクティック計装＆LOOCV検証スキル
│       └── cleanroom-auditor/SKILL.md  # 3層クリーンルーム＆7-gram漏洩監査スキル
├── packages/
│   ├── usir-core/                      # 【コア】外部依存ゼロの純粋計装・ハミルトン力学エンジン
│   │   ├── src/
│   │   │   ├── schema/
│   │   │   │   └── usirBinarySchema.ts # Tier S (128B) / Tier M (4096B) 厳密バイナリ構造体 & Zod定義
│   │   │   ├── sensor/
│   │   │   │   ├── microKasane.ts      # 第1周期（襲 Kasane: Sg-01 Gr, Sg-02 Cr, Sg-03 Pr, Sg-04 Jt）
│   │   │   │   ├── mesoSasae.ts        # 第2周期（支え Sasae: Sg-05 Cp, Sg-06 Sy, Sg-07 Cs, Sg-08 Lp）
│   │   │   │   ├── macroKamae.ts       # 第3周期（構え Kamae: Sg-09 Dy, Sg-10 At, Sg-11 Ma, Sg-12 Ru）
│   │   │   │   └── symplectic24d.ts    # 24D状態 z=(q,p) 統合・3大秩序変数 (psi, phi_void, kappa_kin) 算出
│   │   │   ├── dynamics/
│   │   │   │   ├── hamiltonian.ts      # ハミルトニアン H(q,p)=T(p)+V(q) 及びシンプレクティック勾配算出
│   │   │   │   ├── landauPhase.ts      # 5相分類器（結晶・臨界・超流動ヴォイド[間]・乱流・AI Deadzone）
│   │   │   │   └── equalizer.ts        # カノニカル変換・反AIスロップ直交射影・相転移介入コマンド生成
│   │   │   └── cleanroom/
│   │   │       └── threeRoomGate.ts    # Dirty Room -> Audit Gate (7-gram/PII遮断) -> Clean Room
│   │   └── tests/
│   │       ├── sensor-24d.test.ts      # 12元素・24D直交性と境界値テスト
│   │       ├── phase-loocv.test.ts     # 実データ40チャンク LOOCV >= 95% ゴールデンテスト
│   │       └── cleanroom-gate.test.ts  # 7-gram漏洩阻止・4096Bアライメント検証テスト
│   │
│   ├── usir-wasm/                      # 【高速化】Wasm SIMD 128-bit 24D計装＆距離計算コア
│   │   ├── cpp/
│   │   │   ├── usir_simd.cpp           # C++17 Wasm SIMD 24Dコレスキー距離・ハミルトニアン内積 (<10μs)
│   │   │   └── Makefile                # emcc -msimd128 -O3 -flto ビルド定義
│   │   └── src/
│   │       └── wasmLoader.ts           # Web Worker / Node 両対応ローダー＆純TSフォールバック
│   │
│   ├── usir-mcp/                       # 【独立外付け】MCPサーバー & CLI 検品・調律ゲート
│   │   ├── src/
│   │   │   ├── index.ts                # MCP Tools: inspect_style_24d, diagnose_phase, steer_rewrite
│   │   │   └── cli.ts                  # `npx @nishiki/usir-mcp audit <file>` CLIコマンド
│   │   └── tests/
│   │       └── mcp-server.test.ts      # MCP 入出力スキーマ＆相転移ガイダンス生成テスト
│   │
│   └── usir-bridges/                   # 【エコシステム連携】Plotailor / WorldCraft / Nano 用アダプタ
│       ├── src/
│       │   ├── plotailorEqualizer.ts   # Docked Atelier 右ドロワー用5相レーダー＆4族スライダー状態管理
│       │   ├── worldcraftHarmonic.ts   # WorldCraft 離散オントロジー(支え) × USIR 連続力学の結合ブリッジ
│       │   └── nanoRingBuffer.ts       # Narrative-Nano 320B SPSC Ring Buffer への 128B Tier S 埋め込み
│       └── tests/
│           └── bridges.test.ts
│
└── python/                             # 【実証・学習連携】Python リファレンス & LoRA/GRPO 報酬サーバー
    ├── pyproject.toml                  # uv 管理（numpy, scipy, matplotlib, pytest）
    ├── usir_py/
    │   ├── extractor_24d.py            # 40チャンク実証済みの24Dシンプレクティック抽出器（Python正本）
    │   ├── corpus_40_benchmark.py      # 全5相40チャンク検証用ゴールデンコーパス
    │   └── grpo_reward.py              # ローカルSLM (RTX 4070 12GB) 強化学習・Best-of-N 検品用報酬関数
    └── tests/
        └── test_benchmark_loocv.py     # pytest による LOOCV 97.5% 再現性テスト
```

---

## 3. Antigravity 利用モデル選定＆マルチエージェント・オーケストレーション体制

[Google Antigravity 公式モデル仕様](https://antigravity.google/docs/models/)（2026年10月時点）に基づき、**クォータの2系統分離（Gemini 枠 vs Claude/Partner 枠）** と **2026年11月2日の旧モデル（Claude 4.6 / GPT-OSS 120B）廃止スケジュール** を踏まえた最適モデルルーティングを定義する。

> **重要（モデル廃止アラート）**：Antigravity 上の `Claude Sonnet 4.6 (thinking)`、`Claude Opus 4.6 (thinking)`、および `GPT-OSS-120b` は **2026年11月2日** に削除されるため、すべての設定・プロンプトで **Gemini 4 Argon / Gemini 3.8 Flash** および **Claude Opus 5.5 (Thinking) / Claude Sonnet 5.5 (Thinking)** を指定する。

### 3.1 4階層オーケストレーション体制（Antigravity × Google Jules × Local GPU）

| 役割（Role） | 稼働環境 / 起動コマンド | 指定モデル & Thinking Level | 担当スコープ・責務 | クォータ消費枠 |
| :--- | :--- | :--- | :--- | :--- |
| **① Lead Architect & Symplectic Math Engine**<br>（司令塔・数理コア設計） | **Antigravity Main Session**<br>（`/goal`, `/plan`, `/boost`） | **Gemini 4 Argon (`High`)**<br>または<br>**Claude Opus 5.5 (`Thinking`)** | ・全体アーキテクチャ統括とタスク分解（`/plan`）<br>・`usir-core` の24次元シンプレクティック計装数式、ハミルトニアン $H(\mathbf{q},\mathbf{p})$、ランダウ5相境界条件の厳密実装<br>・C++ Wasm SIMD (`usir_simd.cpp`) のメモリレイアウト設計 | Gemini Pro/Ultra枠<br>（難所のみ Claude 枠） |
| **② Parallel Feature Subagents**<br>（並行実装サブエージェント×3） | **Antigravity `/teamwork`**<br>（Git Worktree 自動分離） | **Gemini 3.8 Flash (`High`)**<br>（複雑なTS型推論時は **Claude Sonnet 5.5 (`Thinking`)**） | ・**Subagent A (`worktree: mcp`)**: `packages/usir-mcp`（MCPサーバー＆CLI検品ツール）実装<br>・**Subagent B (`worktree: bridges`)**: `packages/usir-bridges`（Plotailor右ドロワー用イコライザー＆Nano RingBuffer連携）実装<br>・**Subagent C (`worktree: py-reward`)**: `python/usir_py`（GRPO報酬関数＆Best-of-N検品API）実装 | **Gemini 枠（主力）**<br>高速・低コストで並行回転 |
| **③ Blind Adversarial QA Agent**<br>（盲検敵対的監査エージェント） | **Antigravity `/agents` + `/grill-me`**<br>（独立コンテキスト・履歴遮断） | **Claude Opus 5.5 (`Thinking`)**<br>または<br>**Gemini 4 Argon (`High`)** | ・実装エージェントの会話履歴を遮断した盲検（Blind）状態でコードを監査<br>・**禁止語リーク監査**（テストを通すために著者名・作品名固有語彙を辞書に入れていないか）<br>・**Clean Room 7-gram 監査** と **LOOCV $\ge 95\%$ ゲート** の強制執行 | Claude 枠 / Gemini 枠のクロスチェック |
| **④ Cloud Asynchronous Fleet**<br>（夜間・大量テスト＆周辺実装） | **Google Jules**<br>（スプレッドシート台帳 `QUEUED` 自動ディスパッチ） | **Jules Cloud VM**<br>（1日100タスク / 15並行） | ・全12元素の異常系・境界値テスト網羅（空文字、絵文字のみ、10万字長文、多言語混在）<br>・Playwright E2E テスト（Plotailor 右ドロワー内の5相レーダー描画・60fps維持検証）<br>・プリセット粒子（琥珀・黒曜・白磁・紫電）の JSON/バイナリ定義拡充 | Google Jules 専用枠<br>（ローカル負荷ゼロ） |

### 3.2 Antigravity ワークフローの黄金パターン（コマンド手順）
1. **ブランチ作成**: 作業開始前に必ず `git checkout -b antigravity/feat-usir-core` を切る（`main` 直接コミット厳禁）。
2. **計画生成 (`/plan`)**: 下記「第5章」の Master Goal Prompt を **Gemini 4 Argon (`High`)** に投入し、Antigravity の **Implementation Plan Artifact** を生成させてファイル直交性を確認する。
3. **並列実装 (`/teamwork`)**: コアの型定義（`usirBinarySchema.ts`）が固まった瞬間、`/teamwork` で `usir-mcp`・`usir-bridges`・`usir-wasm` を別々の Git Worktree サブエージェント（**Gemini 3.8 Flash `High`**）へ並列委譲する。
4. **敵対的監査 (`/grill-me`)**: 実装完了後、モデルを **Claude Opus 5.5 (`Thinking`)** に切り替え、`/grill-me` コマンドで「ターゲットリーク（カンニング語彙）がないか」「12元素の直交性が保たれているか」を厳格審査する。

---

## 4. フェーズ別開発計画（Phase 0 〜 Phase 4 マイルストーン）

| フェーズ | 目安所要時間 | 担当主体 | 実装対象・マイルストーン | 完了判定基準（Exit Criteria / Verifiable Artifact） |
| :--- | :---: | :--- | :--- | :--- |
| **Phase 0: 基盤＆Python正本移植** | 30分 | Antigravity (Gemini 3.8 Flash `High`) | ・リポジトリ初期化（`pnpm-workspace.yaml`, `tsconfig.json`, `uv` 環境）<br>・実証済みスクリプト（`/tmp/run_usir_5phase_verification.py`）を `python/usir_py/` へ分割配置 | `uv run pytest` で全5相40チャンクの **LOOCV 97.5%（39/40）**・コア3相 **96.4%（27/28）** が1秒以内に完全再現されること。 |
| **Phase 1: `@nishiki/usir-core` TS実装** | 90分 | Antigravity Lead (Gemini 4 Argon `High`) | ・Tier S (128B) / Tier M (4096B) バイナリ構造体 (`ArrayBuffer` / `DataView`) 実装<br>・12元素（襲・支え・構え × 4族）＋共役運動量 $\mathbf{p}$ の TypeScript 実装<br>・3層クリーンルームゲート（7-gram一致遮断）実装 | `pnpm --filter @nishiki/usir-core test` が全件パスし、Python正本との出力ベクトル誤差が $\|\mathbf{z}_{\text{TS}} - \mathbf{z}_{\text{Py}}\|_\infty < 10^{-5}$ 以内に収まること。 |
| **Phase 2: ハミルトン調律器＆MCPサーバー** | 60分 | Antigravity `/teamwork` (Gemini 3.8 Flash `High` × 2) | ・`equalizer.ts`（AI Deadzone からの脱出ポテンシャル勾配計算＆1段落リライトプロンプト生成）<br>・`@nishiki/usir-mcp`（`inspect_style_24d`, `diagnose_phase`, `steer_rewrite` の3ツール公開） | Antigravity 自身の MCP 設定に `@nishiki/usir-mcp` を接続し、AI Deadzone チャンク（`DEAD-01`）を投入して『超流動ヴォイド相（間）』および『臨界相』への相転移リライトが成功すること。 |
| **Phase 3: Wasm SIMD 高速化＆ブリッジ連携** | 60分 | Antigravity + Local `emcc` | ・`usir_simd.cpp` の Emscripten SIMD128 ビルド<br>・`plotailorEqualizer.ts`（Docked Atelier 右ドロワー内5相レーダー＆4族イコライザーUI状態接続） | 1段落（400字）あたりの24D計装＋コレスキー距離計算が **1ms未満（Wasm部 $<10\mu\text{s}$）** で完了し、Plotailor の 60fps タイピングを一切阻害しないこと。 |
| **Phase 4: Jules 夜間自律テスト＆ローカルSLM連携** | 非同期（夜間） | Google Jules (15並行) + RTX 4070 12GB | ・Jules へ `TASK-360`〜`TASK-365` をディスパッチし、異常系テスト・E2Eテスト・プリセット粒子を網羅<br>・RTX 4070 12GB 上で `grpo_reward.py` を用いた Best-of-N 検品・LoRA 報酬ループ実証 | Jules PR が全件 CI Green でマージされ、Vitest / Playwright / Pytest 全スイートが 100% PASS すること。 |

---

## 5. コピペ即稼働：Antigravity 設定ファイル＆プロンプト一式

### 5.1 `.antigravity/rules.md`（リポジトリ常駐ルールファイル）
リポジトリ直下の `.antigravity/rules.md` に以下を配置することで、Antigravity 上のすべてのエージェントとサブエージェントに USIR の設計憲法を強制適用する。

```markdown
# USIR / Nishiki Architecture & Implementation Rules (.antigravity/rules.md)

## 1. Core Architectural Invariants (絶対遵守事項)
1. **Standalone Zero-Coupling**: `packages/usir-core` and `packages/usir-mcp` MUST have ZERO runtime dependencies on `worldcraft`, `plotailor`, or DOM/browser APIs. They must run in pure Node.js, Web Workers, and edge runtimes.
2. **Strict 4-Group Orthogonality (4族の完全直交性)**:
   - **Group I (`Flux`: `Sg-01 Gr`, `Sg-05 Cp`, `Sg-09 Dy`)**: Pure information density, concrete physical stem mass, clause layering, and macro length dynamic range.
   - **Group II (`Lattice`: `Sg-02 Cr`, `Sg-06 Sy`, `Sg-10 At`)**: Pure syntactic regularity, formal deductive markers, and opening-to-closing stem recurrence.
   - **Group III (`Void` [間]: `Sg-03 Pr`, `Sg-07 Cs`, `Sg-11 Ma`)**: Pure Wago 1-char simplex stem porosity, sentence brevity, clean ending cut (absence of sluggish auxiliaries), punctuation quietude, and apophatic stillness operators (`ない / だけ / まま / 静`).
   - **Group IV (`Shock`: `Sg-04 Jt`, `Sg-08 Lp`, `Sg-12 Ru`)**: Pure voiced/geminate consonantal edge, adjacent-sentence semantic Jaccard leap, figurative tropes (`まるで / のように`), and adversative/rhetorical rupture.
   - **Conjugate Momentum (`p` in R^12)**: Strictly temporal derivatives (sentence-to-sentence RMS velocity $\sqrt{\frac{1}{M-1}\sum (\Delta q)^2}$ and sentence-length volatility $CV_L = \sigma_L / \mu_L$). Never copy static $q$ directly into $p$.
3. **Zero Target-Leakage (カンニング語彙の絶対禁止)**:
   - NEVER add author names, character names (e.g., `メロス`, `李徴`, `島村`, `下人`), or work-specific nouns into any feature extractor dictionary. Feature extractors may only use universal morphological, phonological, syntactic, and general sensory character classes.
4. **3-Room Clean Room Gate**:
   - Tier S (`128 bytes`) and Tier M (`4096 bytes`) binary outputs must pass `threeRoomGate.ts`, guaranteeing zero raw text retention and zero contiguous $\ge 7$-character n-gram overlap with source text.
5. **Docked Atelier UI Constitution (When touching `usir-bridges` for Plotailor)**:
   - All USIR visualizers (5-Phase Diagram, 12-Element Waveform, 4-Group Equalizer Sliders) MUST reside inside the **Right DetailDrawer (`25%` width)** or **Bottom Dock**. Creating auto-popping modals during typing is strictly forbidden.
```

---

### 5.2 Antigravity Master Goal Prompt（`/goal` または初回チャット投入用）
Antigravity でプロジェクトを開き、モデルを **Gemini 4 Argon (`High`)** または **Claude Opus 5.5 (`Thinking`)** に設定して以下のプロンプトをそのまま貼り付ける。

```markdown
/goal Universal Style Genome (USIR / Nishiki) のコアエンジン（`@nishiki/usir-core`）、MCPサーバー（`@nishiki/usir-mcp`）、および Python 検証スイート（`python/usir_py`）を実装し、実データ40チャンクの5相分類ベンチマークで未知サンプル LOOCV 精度 97.5%（39/40）をパスさせた上で、AI Deadzone（Phase V）の文章を『超流動ヴォイド相（間：Phase III）』および『自己組織化臨界相（Phase II）』へ相転移させるハミルトン・イコライザーを完成させる。

### 実行ステップ指示（まずは `/plan` で Implementation Plan Artifact を生成せよ）:
1. **ブランチ作成**: `antigravity/feat-usir-core-v1` ブランチを作成して切り替えること。
2. **Phase 0 (Python正本の配置とベンチマーク確立)**:
   - 既存の検証スクリプト（`/tmp/run_usir_5phase_verification.py` またはリファレンス実装）を `python/usir_py/extractor_24d.py` と `python/usir_py/corpus_40_benchmark.py` にモジュール分割し、`python/tests/test_benchmark_loocv.py` を作成せよ。
   - 以下の検証条件が `pytest` で 100% PASS することを確認せよ：
     - 全5相（結晶・臨界・超流動ヴォイド[間]・乱流・AI Deadzone、計40チャンク）の未知サンプル LOOCV 精度 $\ge 95.0\%$（目標 $97.5\%$）
     - コア3相（間 vs 臨界 vs 同一場面AI生成文、計28チャンク）の LOOCV 精度 $\ge 95.0\%$（目標 $96.4\%$）
     - 第III族（Void）、第IV族（Shock）、共役運動量 $p$ の各アブレーション時に完全体より精度が低下すること（全要素の不可欠性証明）
3. **Phase 1 (`@nishiki/usir-core` TypeScript 実装)**:
   - `packages/usir-core/src/schema/usirBinarySchema.ts`: Tier S (`128B` 固定長バイナリ: Magic `0x55534952`, 24D float32/quantized state, 3 order parameters) および Tier M (`4096B` 固定長バイナリ: 3-Mode HSMM + Cholesky Precision Matrix $L_k$) のシリアライザ・デシリアライザを実装せよ。
   - `packages/usir-core/src/sensor/`: 第1周期（襲 Kasane: `Sg-01`〜`Sg-04`）、第2周期（支え Sasae: `Sg-05`〜`Sg-08`）、第3周期（構え Kamae: `Sg-09`〜`Sg-12`）、および共役運動量 $\mathbf{p} \in \mathbb{R}^{12}$ と3大秩序変数 $(\psi_{\text{order}}, \phi_{\text{void}}, \kappa_{\text{kin}})$ を算出する `symplectic24d.ts` を実装せよ。
   - `packages/usir-core/src/dynamics/`: ハミルトニアン $H(\mathbf{q}, \mathbf{p}) = \frac{1}{2}\mathbf{p}^\top M^{-1}\mathbf{p} + V(\mathbf{q})$ と、4族イコライザー操作 $(\alpha_{\text{flux}}, \alpha_{\text{lattice}}, \alpha_{\text{void}}, \alpha_{\text{shock}})$ から局所1段落の具体的な構造変調命令（目標文長変動係数 $CV_L$、和語単純語幹率目標、抽象フィラーパージ指示、余白・破調オペレータ指示）を生成する `equalizer.ts` を実装せよ。
   - `packages/usir-core/src/cleanroom/threeRoomGate.ts`: 生テキストから抽出されたバイナリ構造体に対し、元テキストとの連続7文字（7-gram）以上の一致がゼロであることを検証する Audit Gate を実装せよ。
4. **Phase 2 (`@nishiki/usir-mcp` 実装)**:
   - 任意のLLMクライアントや IDE から呼び出せる3つの MCP ツール（`inspect_style_24d`, `diagnose_thermodynamic_phase`, `generate_hamiltonian_steering_prompt`）を実装し、Vitest 単体テストを作成せよ。

### 絶対制約:
- `.antigravity/rules.md` に記載された「4族の完全直交性」と「ターゲット語彙リーク禁止（作品固有の名詞や人名を辞書に入れない）」を厳守すること。
- まずは `/plan` アーティファクトを出力し、その後 Phase 0 から順に実装とテスト実行（Vitest / Pytest）を自律遂行せよ。
```

---

### 5.3 盲検敵対的監査プロンプト（`/grill-me` または別セッションの Claude Opus 5.5 投入用）
実装が一通り完了した段階で、新しいエージェントセッション（履歴遮断状態）を **Claude Opus 5.5 (`Thinking`)** で立ち上げ、以下のプロンプトで敵対的監査を実行する。

```markdown
/grill-me 以下のリポジトリ内の実装（`packages/usir-core` および `python/usir_py`）に対して、作成者の意図を一切忖度しない盲検敵対的監査（Blind Adversarial Stress Test）を実行せよ。

### 監査チェックリスト（1つでも違反があれば FAIL として該当行番号と修正パッチを提示せよ）:
1. **Data Leakage / Target Encoding 監査**:
   - `sensor/` または `extractor_24d.py` 内の文字集合・演算子リストに、特定の小説の固有名詞（例：メロス、李徴、羅生門、城崎、雪国、檸檬など）や特定サンプルだけを狙い撃ちしたチート単語が1つでも混入していないか？
2. **4族直交性（Orthogonality）監査**:
   - Group I (`Flux`) の数式に Group III (`Void`) や Group IV (`Shock`) の変数（比喩率や否定演算子など）が混入していないか？
   - 共役運動量ベクトル `p`（12次元）に、位置ベクトル `q` の単なる定数倍コピーが含まれていないか？（`p` は必ず文間の差分・標準偏差・変動係数 $CV_L$ などの時間微分量で構成されていなければならない）。
3. **AI Deadzone 誤爆監査（False Positive Check）**:
   - 論理的・結晶的な技術仕様書や法律条文（Phase I: `I_Crystalline`）のように「もともと文長が均一で抽象フィラーを含まない硬質な文章」を入力したとき、誤って Phase V（`V_Deadzone`: AIスロップ）と判定されないか？
4. **Clean Room バイト境界監査**:
   - Tier S シリアライズ結果が厳密に `128 bytes`、Tier M シリアライズ結果が厳密に `4096 bytes` であり、元テキストの連続7文字以上の文字列断片が1バイトも残存していないか？
```

---

## 6. Google Jules 並行委譲タスク仕様（4大ブロック構成・台帳起票用）

Antigravity で Phase 1〜2 の骨格がビルド通過した直後に、『【統合版】Julesタスクキュー＆実行管理台帳』（ID: `1PV7P8niseT2eGWProL8mxU0Tj10geVhf7Sq1RqTQfSw`）へ `Status: QUEUED` で投入できる **ファイル完全直交の4タスク（`TASK-360`〜`TASK-363`）** を以下に定義する。

| タスクID | 割当先リポジトリ | タスク名（スコープ） | 編集対象ファイル（100%直交・競合ゼロ） |
| :--- | :--- | :--- | :--- |
| **TASK-360** | `ughirose/usir` | `test(core): 24次元シンプレクティック計装器の境界値・異常系・ファジング単体テスト網羅` | `packages/usir-core/tests/edge-cases.test.ts` |
| **TASK-361** | `ughirose/usir` | `feat(presets): 4大基準スタイル粒子（琥珀・黒曜・白磁・紫電）のTier M 4KB基準プロファイル定義` | `packages/usir-core/src/presets/canonicalParticles.ts`, `packages/usir-core/tests/presets.test.ts` |
| **TASK-362** | `ughirose/plotailor` | `feat(drawer): Docked Atelier 右詳細ドロワー内「USIR 5相状態図レーダー＆4族イコライザー」パネル実装` | `src/components/drawer/UsirStyleEqualizerPanel.tsx`, `tests/usir-equalizer-panel.test.tsx` |
| **TASK-363** | `ughirose/narrative-nano` | `feat(simd): C++ Wasm SIMD 24次元コレスキー距離・ハミルトニアンエネルギー高速算出カーネルの実装` | `cpp/usir_simd_kernel.cpp`, `src/worker/usirSimdBridge.ts`, `tests/usir-simd.test.ts` |

### 【TASK-360用】（Jules投入プロンプト）
```markdown
### Task Title:
test(core): 【異常系・境界値検証】24次元シンプレクティック計装器の異常系・ファジング単体テスト網羅

### Requirements:
1. `packages/usir-core/tests/edge-cases.test.ts` を新規作成し、`extractUsir24d()` および `classifyThermodynamicPhase()` に対する包括的な異常系・境界値テストスイートを実装すること。
2. 以下の入力パターンをすべてテストし、`NaN`、`Infinity`、ゼロ除算エラー、または配列境界外アクセスが一切発生せず、すべての要素 $q_k \in [0, 1]$ および有限の運動量 $p_k \ge 0$ が返ることを検証すること：
   - 空文字列 `""`、空白・改行のみの文字列
   - 1文字のみ（`"あ"`、`"。"`）、句点が一切ない1,000文字の連続文字列
   - 感嘆符・疑問符のみの文字列（`"！？！？！？"`）、絵文字・サロゲートペア混在文字列
   - 全文が完全に同一文字数の10文（$\sigma_L = 0$ の完全平坦ケース）
   - 1文目が2文字、2文目が500文字の極端な緩急ケース
3. Tier S (`128B`) および Tier M (`4096B`) バイナリシリアライザの往復変換（Round-trip）で浮動小数点精度誤差が許容範囲内（$< 10^{-5}$）であることを検証すること。

### Constraints:
1. `packages/usir-core/src/` 配下の既存実装コードは変更せず、`packages/usir-core/tests/edge-cases.test.ts` のみを作成・編集すること。
2. TypeScript strict mode に完全準拠し、`pnpm test` が全件パスすること。

### Output:
異常系・境界値テストスイートを含むプルリクエスト（PR）を作成すること。
```

### 【TASK-361用】（Jules投入プロンプト）
```markdown
### Task Title:
feat(presets): 【基準スタイル粒子】4大基準スタイル粒子（琥珀・黒曜・白磁・紫電）と様式化学混合演算の実装

### Requirements:
1. `packages/usir-core/src/presets/canonicalParticles.ts` を作成し、USIR の4大基準スタイル粒子（Canonical Style Particles）の24次元重心ベクトル $\mathbf{z}=(\mathbf{q}, \mathbf{p})$ およびコレスキー下三角精度行列 $L$ を定義すること：
   - `AMBER_CRITICAL`（琥珀：Phase II 自己組織化臨界相 — 1/f ゆらぎと共感覚跳躍の文芸重心）
   - `OBSIDIAN_VOID`（黒曜：Phase III 超流動ヴォイド相［間］ — 比喩・装飾を削ぎ落とした和語短文・静寂の重心）
   - `PORCELAIN_CRYSTAL`（白磁：Phase I 結晶秩序相 — 曖昧性ゼロ・高構造対称性の論理仕様・公理重心）
   - `LIGHTNING_RUPTURE`（紫電：高運動量 Shock 触媒 — AI Deadzone に沈殿した文章を強制励起する破調パルス粒子）
2. 複数の粒子を重み $w_i$ でガウス積合成（Precision-Weighted PoE: $\Lambda_{\text{mix}} = \sum w_i \Lambda_i$）する様式化学ミキサー関数 `mixStyleParticles(particles, weights)` を実装すること。
3. `packages/usir-core/tests/presets.test.ts` を作成し、各基準粒子が対応する熱力学的相へ100%分類されること、および `PORCELAIN_CRYSTAL` と `OBSIDIAN_VOID` の等量混合時に中間相へ滑らかに遷移することを検証する単体テストを実装すること。

### Constraints:
1. 編集対象を `packages/usir-core/src/presets/canonicalParticles.ts` および `packages/usir-core/tests/presets.test.ts` に限定し、他ファイルとの競合を発生させないこと。
2. TypeScript strict mode に完全準拠し、全単体テストがパスすること。

### Output:
4大基準スタイル粒子定義、様式化学ミキサー関数、および単体テストを含むプルリクエスト（PR）を作成すること。
```

### 【TASK-362用】（Jules投入プロンプト）
```markdown
### Task Title:
feat(drawer): 【Docked Atelier 準拠】右詳細ドロワー内「USIR 5相状態図レーダー＆4族イコライザー」パネルの実装

### Requirements:
1. `src/components/drawer/UsirStyleEqualizerPanel.tsx` を作成し、Plotailor の右詳細ドロワー（Contextual DetailDrawer: 幅25%領域）内に常駐可能な USIR リアルタイム計装＆調律パネルを実装すること。
2. 以下の3つのUIセクションを単一ドロワーパネル内にコンパクトに収容すること：
   - **5相熱力学状態インジケータ**: 現在の章・段落の $(\kappa_{\text{kin}}, \phi_{\text{void}}, \psi_{\text{order}})$ を表示し、`AI Deadzone (Phase V)` への沈殿（$CV_L < 0.28$）を検知した際にアンバー色の非侵襲バッジで通知するインジケータ。
   - **12元素周期表ミニスペクトルバー**: 第1周期（襲）・第2周期（支え）・第3周期（構え）× 4族の12本のミニバーチャート。
   - **4族イコライザースライダー**: `Flux (密度)`, `Lattice (秩序)`, `Void (間・余白)`, `Shock (破調)` の4スライダー（`0.0`〜`2.0`）と、現在選択中の1段落に対する「相転移リライト提案（Diffプレビュー）」トリガーボタン。
3. `tests/usir-equalizer-panel.test.tsx` を作成し、スライダー操作時の状態更新、AI Deadzone 警告バッジの表示条件、およびモーダル非生成の単体テストを実装すること。

### Constraints:
1. **【Docked Atelier / 3ペイン統合IDE憲法順守】**: 画面中央を覆う単発モーダル（`XxxModal.tsx`）や執筆中の自動ポップアップダイアログを絶対に新設しないこと。すべての表示・操作を右詳細ドロワー内で完結させること。
2. TypeScript strict mode に完全準拠し、全テストがパスすること。

### Output:
右ドロワー用 USIR イコライザーパネルコンポーネントおよび単体テストを含むプルリクエスト（PR）を作成すること。
```

---

## 7. テスト・検証プロトコル（4段階の品質ゲート）

Antigravity での実装引き継ぎ時、以下の **4段階の自動検証ゲート（Verification Gates）** をすべて通過することをマージの絶対条件とする。

### Gate 1: 数値同値性テスト（Python 正本 vs TypeScript 移植版）
* **目的**: Python で実証した40チャンクの24次元ベクトル $\mathbf{z}_{\text{Py}} \in \mathbb{R}^{24}$ と、TypeScript（`@nishiki/usir-core`）および Wasm SIMD（`@nishiki/usir-wasm`）が算出するベクトル $\mathbf{z}_{\text{TS}}$ の数値的同値性を保証する。
* **合格基準**: 全40チャンクに対して最大絶対誤差 $\|\mathbf{z}_{\text{TS}} - \mathbf{z}_{\text{Py}}\|_\infty < 10^{-5}$。

### Gate 2: Zero-Leakage 40チャンク LOOCV ＆ アブレーション自動回帰テスト
* **目的**: コード改修やパラメータ調整によって5相分離能が劣化していないこと、および特定の族が死に体（冗長）になっていないことを毎回0.5秒で自動検証する。
* **実行コマンド**: `uv run pytest python/tests/test_benchmark_loocv.py` および `pnpm --filter @nishiki/usir-core test`
* **合格基準**:
  1. 全5相（$N=40$）の LOOCV 精度 $\ge 95.0\%$（基準値：**$97.5\%$**）
  2. コア3相（『間』vs『臨界』vs『同一場面AI』、$N=28$）の LOOCV 精度 $\ge 95.0\%$（基準値：**$96.4\%$**）
  3. アブレーション不等式の成立：
     $$\text{Acc}_{\text{Full 24D}} > \text{Acc}_{\text{No Void (Group III)}}, \quad \text{Acc}_{\text{Full 24D}} > \text{Acc}_{\text{No Shock (Group IV)}}, \quad \text{Acc}_{\text{Full 24D}} > \text{Acc}_{\text{No Momentum } \mathbf{p}}$$

### Gate 3: 3層クリーンルーム・7-gram 漏洩ゼロ監査テスト
* **目的**: 著作権法30条の4（非享受目的）およびユーザーのローカル原稿保護を物理的に担保する。
* **合格基準**: 任意の入力テキスト $T$ から生成された Tier S (`128B`) / Tier M (`4096B`) バイナリバッファ $B$ を UTF-8 / Shift-JIS / UTF-16 デコードまたはバイト列スキャンした際、入力テキスト $T$ に含まれる任意の連続7文字（7-gram）との一致件数が **厳密に `0` 件** であること。

### Gate 4: Before / After 相転移リライト実証テスト（End-to-End Phase Transition）
* **目的**: 「測る」だけでなく、実際に **Phase V（AI Deadzone）に沈殿した文章を Phase III（超流動ヴォイド相［間］）および Phase II（自己組織化臨界相）へ相転移させられること** を定量検証する。
* **テスト手順**:
  1. 入力として `DEAD-01`（AI模倣：城崎の療養、初期状態：$\kappa_{\text{kin}} = 0.040, \phi_{\text{void}} = 0.000, CV_L = 0.13$）を与える。
  2. `@nishiki/usir-mcp` の `steer_rewrite` ツールで **ターゲット＝『超流動ヴォイド相（`OBSIDIAN_VOID`）』** のハミルトン変調プロンプトを生成し、LLM に1段落リライトを実行させる。
  3. リライト後の出力テキストを再度 `extractUsir24d()` に通し、以下の **相転移条件** を満たすことを自動アサートする：
     * 文長変動係数 $CV_L$: $0.13 \to \mathbf{0.40 \text{ 以上}}$ へ回復
     * 余白凝縮度 $\phi_{\text{void}}$: $0.000 \to \mathbf{0.55 \text{ 以上}}$ へ遷移
     * 抽象フィラー率 `abstract_b`: **$\mathbf{0.01 \text{ 以下}}$** へ消滅
     * 5相分類器（`classifyThermodynamicPhase`）の判定結果が `V_Deadzone` から **`III_Void`** へ遷移すること。
