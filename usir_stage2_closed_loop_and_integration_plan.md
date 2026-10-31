# 【Stage 2 実行計画＆プロンプト】USIR 閉ループLLM相転移実証・Goodhart監査・Plotailor実機結合
> 対象ワークスペース: `C:/dev/usir`（ブランチ `antigravity/feat-usir-core-v1` コミット `1538100` 完了後）
> 前提ステータス: Phase 0〜4（全95テスト：Pytest 5件＋Vitest 90件）PASS、盲検QAによる `equalizer.ts` ハードコード撤去完了
> Stage 2 の核心ゴール: 「ルールベースの機械的文分割（センサーハック）」から脱却し、**実LLM / ローカルSLMとの閉ループ（Closed-Loop）で自然文としての文学的品質と5相遷移を両立できること** を証明した上で、MCP実稼働および `plotailor` 本体へ結合する。

---

## 1. 現状の冷徹な客観評価（何が達成され、何がまだ「スタブ」か）

今回の Antigravity 実行（特に `Blind Adversarial QA Auditor` が `equalizer.ts` 内の `蜂・温泉` ハードコード分岐を摘出・撤去したプロセス）は極めて健全に機能している。
一方で、**「盲検監査報告の項目3（ルールベース節分割の限界）」** に記載されている通り、現在の `C:/dev/usir` には次の **構造的ギャップ（Goodhartの法則リスク）** が残っている。

1. **既に100%完成している領域（本物の資産）**:
   * **24次元シンプレクティック計装器（`symplectic24d.ts` / `extractor_24d.py`）**: Python と TypeScript 間で誤差 $1.11 \times 10^{-15}$ で完全一致し、実データ40チャンクをチート語彙ゼロで **97.5%（39/40）** 分離できる「物差し（Sensor）」は完全に完成している。
   * **Tier S (128B) / Tier M (4096B) バイナリ＆3層クリーンルーム**: 7-gram 漏洩ゼロのバイナリ仕様も完成している。
2. **まだ「機械的スタブ」にとどまっている領域（今から潰すべき急所）**:
   * 現在の `equalizer.ts` の `steerPhaseRewrite()` は、テスト（Gate 4）を単体で通すために **「接続助詞で機械的に文をぶつ切りにし、辞書にある抽象語（`ABSTRACT_FILLERS`）を文字列置換で削除する」** という決定論的ルールで $CV_L \ge 0.40, \phi_{\text{void}} \ge 0.55$ を達成している。
   * これは **「センサーの数値をクリアするためだけに文章を機械的に切り刻む（Goodhart's Law / センサーハック）」** 状態であり、そのままでは「抽象的な説明（例：『強い後悔と悲しみを覚えた』）を、志賀直哉のように『具体的な身体動作と静寂（例：いもりの反らした尾が自然に静かに下りて来た）』へ昇華する」という本来の文芸的リライトにはなっていない。

したがって、次の一手（Stage 2）は以下の **3ステップ** を順番に踏み抜くのが最もレバレッジが高い。

---

## 2. Stage 2 の優先順位と3大アクションプラン

| 優先度 | ステップ名 | 所要時間 | 目的・ゴール |
| :---: | :--- | :---: | :--- |
| **Step 1**<br>（最優先） | **Closed-Loop 相転移ハーネス（実LLM / ローカルSLM 結合）と「自然度＋相転移」二重監査** | 約45分 | ルールベースの文切り刻みではなく、`@nishiki/usir-mcp` が出力するハミルトン変調指示書（`generateRewriteDirectives`）を **実際の生成モデル（Antigravity サブエージェント / Gemini API / ローカル RTX 4070 Ollama）** に渡して `DEAD-01`〜`08` を再執筆させ、**① USIRセンサーで `III_Void` / `II_Critical` に到達し、かつ ② 盲検監査で「機械的なぶつ切りではなく自然な日本語散文として成立している」こと** を実証する。 |
| **Step 2**<br>（即戦力化） | **`@nishiki/usir-mcp` の Antigravity 実機 MCP 登録 ＆ Git リモート確定** | 約15分 | `C:/dev/usir/packages/usir-mcp` をビルドし、Antigravity 自身の MCP 設定にローカル登録する。これにより、今日から Antigravity 上で note 記事や小説、仕様書を書く際にリアルタイムで `inspect_style_24d` / `steer_rewrite` を呼び出せる「生きた武器」にする。 |
| **Step 3**<br>（本体統合） | **`plotailor` 実機リポジトリへの `@nishiki/usir-bridges` 統合（Docked Atelier 右ドロワー）** | 約45分 | `C:/dev/plotailor` から `@nishiki/usir-core` / `@nishiki/usir-bridges` を `pnpm` リンクし、Plotailor の右詳細ドロワー（25%幅）に「5相状態図レーダー＆4族イコライザー」を実機マウントしてブラウザ（`/browser`）で60fps動作を確認する。 |

---

## 3. コピペ即稼働：Antigravity 投入用 Stage 2 マスタープロンプト

Antigravity（`C:/dev/usir` ワークスペース）を開き、以下のプロンプトをそのまま貼り付けて実行してください。

```markdown
/goal Stage 2: ルールベースの機械的文分割（Goodhartの法則リスク）を超え、`@nishiki/usir-mcp` のハミルトン変調ディレクティブ（`generateRewriteDirectives`）を用いた「実生成AI閉ループ相転移（Closed-Loop Generative Phase Transition）」を実証し、MCPサーバーの実機ビルド確定および Before/After 比較レポートを生成する。

### 背景と課題意識:
Phase 0〜4 の実装により、24次元シンプレクティック計装器（Gate 1〜3）は完璧に完成した。しかし、`equalizer.ts` の決定論的 `steerPhaseRewrite` は接続助詞での文分割と抽象語削除を行うルールベース処理であるため、センサー上の数値（$CV_L \ge 0.40, \phi_{\text{void}} \ge 0.55$）はクリアできても、「抽象的な説明を具体的な五感・身体描写へ書き換える」という本来の生成リライトは LLM/SLM との閉ループ結合なしには完結しない。

### 実行ステップ（Flash サブエージェント＋独立 Blind QA サブエージェントで遂行せよ）:

1. **Step 1: `@nishiki/usir-core` の `generateRewriteDirectives` 高度化（数値フィードバック付きハミルトン・プロンプトコンパイラ）**
   - `packages/usir-core/src/dynamics/equalizer.ts` の `generateRewriteDirectives(text, targetPhase)` を拡張し、単なる定型文ではなく、入力テキストの現在の24次元状態 $\mathbf{z}=(\mathbf{q}, \mathbf{p})$ とターゲット相（例：`OBSIDIAN_VOID` [超流動ヴォイド相・間] または `AMBER_CRITICAL` [1/f臨界相]）との **具体的な差分ベクトル $\Delta \mathbf{z}$** に基づく定量的な編集命令書（Prompt Payload）を出力するようにせよ：
     - 現在の文長変動係数 $CV_L$ と目標 $CV_L \in [0.45, 0.70]$ を達成するための「短文言い切り（8〜18文字）と中長文の具体的な配置リズム指示」
     - 原文中に検出されたすべての `ABSTRACT_FILLERS`（例：`雰囲気`、`思われた`、`感じられた`、`様々な`、`テーマ` 等）の具体的なリストと、それらを「削除または視覚・聴覚・触覚の物理名詞（1文字和語語幹）と身体動作へ置換せよ」という禁止・置換リスト
     - ターゲットが `III_Void`（間）の場合は比喩（`まるで〜のように`）や感嘆符を全面禁止し、`ない / だけ / まま / 静` 等の引き算の余白演算子と潔い過去形言い切り（`〜た。`）で結ぶ指示
     - ターゲットが `II_Critical`（臨界相）の場合は、静から動への急激な破調（逆接 `しかし / 不意に` や鮮烈な共感覚直喩）を1箇所だけ点火する指示

2. **Step 2: 全8チャンク（`DEAD-01`〜`DEAD-08`）に対する「実AI生成 閉ループ相転移ベンチマーク」の実行**
   - ルールベースの文字列切り刻み（`steerPhaseRewrite`）ではなく、実際に `generateRewriteDirectives` が出力したハミルトン変調プロンプトを使って、サブエージェント（Generator Subagent）に `DEAD-01`〜`DEAD-08` の8チャンクを **① `III_Void`（超流動ヴォイド相・間）向け** と **② `II_Critical`（自己組織化臨界相）向け** の2通り（計16チャンク）で自然文としてリライト生成させよ。
   - 生成された16チャンクを `extractUsir24d()` で実測し、もし1発目でターゲット相に入らなかったチャンクがあれば、その実測誤差 $\Delta \mathbf{z}$ を再度フィードバックしてリライトさせる **Closed-Loop Iterative Steering（最大3イテレーション）** を回し、何イテレーションで相転移が完了したかを記録せよ。
   - 結果を `tests/closed-loop-rewrite.test.ts` および `artifacts/before_after_phase_transition_samples.md` に保存し、**原文（AI Deadzone） vs ルールベース分割 vs ハミルトン閉ループ生成（Void相 / Critical相）の実際の文章と24Dスコア比較表** を出力せよ。

3. **Step 3: 独立した Blind Adversarial QA サブエージェントによる「自然度＆センサーハック有無」の監査**
   - 履歴を遮断した Blind QA サブエージェントを起動し、Step 2 で生成された16チャンクの日本語散文に対して：
     - (A) 機械的な文ぶつ切りや不自然な日本語破綻（センサーの数値を稼ぐだけのハック）が起きていないか？
     - (B) 原文のあらすじ・場面設定（蜂の死、いもりの死、羅生門、檸檬、桜、山月記、雪国、メロス）を保ったまま、本当に文豪作品に近い「間（Void）」と「臨界の張り詰め（Critical）」が生まれているか？
     を100点満点のルーブリック（自然度・具象性・余白の緊張感）で辛口採点させよ。

4. **Step 4: `@nishiki/usir-mcp` の本番ビルド＆起動確認**
   - `packages/usir-mcp` を `pnpm build` でビルドし、スタンドアロン実行（`node packages/usir-mcp/dist/cli.js` および MCP stdio 起動）がエラーゼロで動作することを確認し、Antigravity / Claude Desktop 用の `mcp_config.json` スニペットを出力せよ。
```

---

## 4. その後の展開ロードマップ（Step 1〜2 完了後の選択肢）

上記の Stage 2 プロンプトが完了すると、**「センサーの精度（97.5%）」だけでなく「実際にAIの駄文（AI Deadzone）を、自然で美しい『間（Void）』や『臨界（Critical）』の文章へ書き換えられる」という決定的な Before/After 実例（16本の対照サンプル）** が手に入ります。

そこからの分岐は以下の3つです：
1. **ルートA（Plotailor 実機統合）**：`C:/dev/plotailor` の右詳細ドロワーに `plotailorEqualizer.ts` を組み込み、エディタ上でリアルタイムに5相レーダーが動き、1クリックで選択段落を相転移リライトできる UI を完成させる。
2. **ルートB（ローカルGPU RTX 4070 12GB での Best-of-$N$ / GRPO 実証）**：`python/usir_py/grpo_reward.py` を使い、手元の RTX 4070（Ollama / LM Studio の Qwen-2.5-7B や Gemma-2-9B 等）の出力を USIR 報酬関数でフィルタリング・強化学習し、「4KBのUSIRファイルだけでローカルSLMの文体が激変する」ことを実証する。
3. **ルートC（note記事・対外発信用の実証ショーケース化）**：今回得られた「文豪作品 vs 同一場面AI生成文の文長変動係数 $CV_L$ 崩壊データ」と「Before/After 相転移サンプル」を、ペンネーム（ゼロさん）名義の技術・創作エッセイ記事の一次検証素材としてパッケージ化する。
