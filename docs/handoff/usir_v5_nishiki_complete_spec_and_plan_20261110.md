# 【完全統合正本】USIR v5 × Nishiki 6極：作家性抽出の数理設計・特化型SLM蒸留・3レーン段階的監査＆実装マスター仕様書（2026-11-10 完全自己完結版）

* **文書ステータス**: 全AIエージェント・新セッション引き継ぎ用 単一正本（Single Source of Truth / 外部Google Docs参照不要の100%自己完結Markdown）
* **対象ローカルリポジトリ**:
  * `C:\dev\nishiki`（GitHub: `ughirose/nishiki` — 純Rust 6極コンパイラ・数理カーネル・Wasm/PyO3/CLI）
  * `C:\dev\kasane`（GitHub: `ughirose/kasane` — 32B `CognitiveSpan`・二重サプライザル・呼吸波形・BYOR評価器 → `C:\dev\nishiki` へ統合対象）
  * `C:\dev\usir`（GitHub: `ughirose/usir` — 青空文庫アトラス・55D Fisher LDA・ローカルRTX 4070 SLM蒸留・SMC生成制御）
  * `C:\dev\worldcraft-workspace`（Plotailor / WorldCraft — **Phase 4までUI改修凍結**、6極エンジン完成後にWasm接続）
* **ローカルGoogleドライブ（Fドライブ）実在同期パス（検証済み）**:
  * **エージェント共通受け渡しルート（最優先・浅階層）**: `F:\マイドライブ\Claude-Sync\`
    * **Gemini → エージェント指示・仕様書配置先**: `F:\マイドライブ\Claude-Sync\USIR_v5_Nishiki_Complete_Spec_and_Plan_20261110.md`（本ファイル）
    * **エージェント → Gemini/人間 監査・結果レポート提出先**: `F:\マイドライブ\Claude-Sync\inbox\`
    * **Claude Code 自動会話ログ同期先（1時間周期 `ClaudeCodeToDriveSync`）**: `F:\マイドライブ\Claude-Sync\Claude-Code\`（目次: `_INDEX.md`）
  * **プロジェクト別アーカイブパス（深階層）**:
    * `F:\マイドライブ\01_プロジェクト・仕事\02_AI副業・収益化\プロトタイプ・ツール\05_Nishiki_錦\`
    * `F:\マイドライブ\01_プロジェクト・仕事\02_AI副業・収益化\プロトタイプ・ツール\WorldCraft_GitHub_Sync\`

## 0. 反省と設計原則：なぜ「いきなり実装」ではなく「Phase 0（構想・実コード冷徹監査）」から始めるのか

過去のセッション（[20261104-1705_0d6552a0.md](https://drive.google.com/file/d/1PZ5TIWemPWzmoFPe6NOZOARRiJnonldj/view?usp=drivesdk) および [05_Nishiki実コード監査確定事実](https://drive.google.com/file/d/1AhGn2bl861kAVQRoBujl4xlwqLLRwjBU/view?usp=drivesdk)）において、ローカルの実コードを見ずに書かれた仕様書をそのまま無批判に実装させた結果、「既存の型定義との衝突」「架空ファイルパスの参照」「実テキストで機能しない過剰な数式」による手戻りが発生した。

この教訓を踏まえ、本計画では **工程の省略を一切禁止** し、以下の **5段階の厳格なエンジニアリング工程** を踏む：

1. **【Phase 0：構想・数式・実コード整合性の冷徹な事前監査（Claude Fable 5.2 担当）】**:
   コードを1行も書き換える前に、**Claude Fable 5.2** が本仕様書とローカルの3リポジトリ（`C:\dev\nishiki`, `C:\dev\kasane`, `C:\dev\usir`）の実コードを突き合わせ、「この構想・数式のどこが本当に有効で、どこが過剰設計・破綻リスクか」「既存コードの何を残し何を統合すべきか」を遠慮なく批判・修正した **監査レポート兼・確定実装計画書** を出力する。
2. **【Phase 1：人間＋Gemini による監査結果レビューと仕様確定（Gate 1）】**:
   Fable 5.2 が出力した監査レポート（`F:\マイドライブ\Claude-Sync\inbox\2026-11-10_phase0_fable_critical_audit.md`）を人間とGeminiが確認し、不要と判定された要素を削ぎ落とした上で実装GOサインを出す。
3. **【Phase 2：Fable 5.2 特務レーン（Lane F）による最難関共通カーネルの先行構築】**:
   11月16日までの Fable 5.2 開放枠（＋11月17日 16:59 JSTまでの無料リセット権）を使い、Sonnet 5.5 ではバグを出しやすい「純Rust数値線形代数（SVD/固有値/球面幾何）」「kasane→nishiki 32B型統合」「日本語ゼロ代名詞の中心化理論ソルバー」「Ripser型TDA未踏領域発見器」をゼロ依存クレートとして先行実装・証明する。
4. **【Phase 3：Lane A（Claude Sonnet 5.5 + Haiku 5.5）× Lane B（Antigravity 2.0 + RTX 4070）の並列本実装と青空文庫3,000場面蒸留】**:
   Phase 2 の共通カーネルの上に6極クレート・PyO3・CLIを組み上げ（Lane A）、並行してローカルRTX 4070（API課金 $0.00）で青空文庫3,000場面の教師生成・特化型SLM蒸留・作家識別＆未踏領域アトラス検証（Lane B）を完遂する。
5. **【Phase 4：実証完了後の Plotailor「離見の見」6レンズWasm接続】**:
   青空文庫実データで6極の分離精度と閾値が確定した後に初めて、`C:\dev\worldcraft-workspace\plotailor` に `nishiki-wasm` を接続する。

## 1. 結果の共有方法・受け取りプロトコル（人間・Claude Code・Antigravity・Gemini Web 間の双方向同期）

「誰が・どこに・どの形式で結果を出し、どう受け取るか」を以下の3経路に完全固定する。

| 共有経路 | 書き出し側（エージェント）の動作 | 出力ファイルパス（Windows実パス） | 受け取り側（人間 / Gemini Web / 他エージェント）の確認方法 |
| :--- | :--- | :--- | :--- |
| **経路①：フェーズ完了・監査レポートの直接書き出し（メイン経路）** | Claude Code（Fable 5.2 / Sonnet 5.5）および Antigravity 2.0 は、各フェーズ（特に Phase 0 監査と各 STAGE）の完了時に、必ずリポジトリ内の `docs/handoff/` と **Fドライブの `Claude-Sync\inbox\` の両方** に同一の `.md` レポートおよび `.json` 実測値を書き出す。 | 1. リポジトリ内正本:<br>`C:\dev\nishiki\docs\handoff\HANDOFF.md`<br>`C:\dev\usir\docs\handoff\HANDOFF.md`<br>2. **Drive即時共有先**:<br>`F:\マイドライブ\Claude-Sync\inbox\2026-11-10_phase0_fable_critical_audit.md`<br>`F:\マイドライブ\Claude-Sync\inbox\HANDOFF_nishiki.md`<br>`F:\マイドライブ\Claude-Sync\inbox\HANDOFF_usir.md` | ・**人間**: エクスプローラまたはエディタで `F:\マイドライブ\Claude-Sync\inbox\` を開くだけで即座に閲覧可能。<br>・**Gemini Web**: チャットで「inboxの監査結果を読んで」と1行指示するだけで、Google Drive ツール経由で直接全文を読み取り、次フェーズの判断・仕様修正を行う（コピペや手動アップロード一切不要）。 |
| **経路②：Claude Code 全会話履歴の自動Markdown同期（サブ経路）** | 各フェーズの終了時（およびWindowsタスクスケジューラの1時間周期 `ClaudeCodeToDriveSync`）に、既存の `python C:\dev\claude-drive-sync\sync_claude_to_drive.py` を実行する。 | `F:\マイドライブ\Claude-Sync\Claude-Code\_INDEX.md`<br>および各プロジェクト別フォルダ（`C_dev`, `C_Dev_usir`, `C_Dev_kasane` 等）内のセッション `.md` | 途中の思考過程やツール実行ログ、エラーの詳細まで Gemini Web から直接検索・精査できる。 |
| **経路③：Gemini 日中・夜間スケジュールタスクからの宿題追記** | Gemini の自律スケジュールタスク（毎日 02:00 / 10:00 / 13:00 / 16:00 JST）が宿題（`HW-01`〜`HW-31`）を消化するたびに、必ず **純粋な `.md` ファイル（非変換バイナリ）** として `F:\マイドライブ\Claude-Sync\` 内の本仕様書・ランブックを更新する。 | `F:\マイドライブ\Claude-Sync\USIR_v5_Nishiki_Complete_Spec_and_Plan_20261110.md` | Claude Code と Antigravity 2.0 はローカルの `F:\マイドライブ\Claude-Sync\...md` を `Read` ツールで直接読み込む。 |

### 引き継ぎレポート（`HANDOFF.md` および監査レポート）の必須8節フォーマット
ユーザーのグローバル規約（`C:\Users\user\.claude\templates\handoff.md` / [arXiv:2611.05306](https://arxiv.org/abs/2611.05306) 準拠）に従い、エージェントが出力するすべての引き継ぎ・監査ファイルは以下の8節構成を厳守する：
1. **タスク原文（Original Prompt / Scope）**
2. **現在の意図（Current Objective）**
3. **計画と進捗（Plan & Status）**
4. **完了条件と検証結果（Verification & Test Logs）**
5. **成果物・変更ファイルポインタ（File Pointers & Commit Hash）**
6. **確定事項と除外・却下した方針（Confirmed Decisions & Rejected Ideas）**
7. **未解決・未検証・限界（Open Issues & Unverified Limits — 誇張ゼロで正直に記述）**
8. **次の1手（NEXT Action）**

## 2. USIR と Nishiki の関係性定義およびリポジトリ統合構想（Phase 0 での重点監査対象）

### 2.1 概念上の関係性
1. **USIR（Universal Style & Structure Intermediate Representation：普遍表現ゲノム規格）**:
   小説・評論（note）・音楽・映像（Cinema / MV）を同一の数学的位相空間にマッピングする **「普遍中間表現データフォーマット（6極テンソル座標系）」**。作品および場面の状態ベクトル $\Xi_{\text{USIR}} = (\text{Kasane}, \text{Sasae}, \text{Kamae}, \text{Sadame}, \text{Nagare}, \text{Awase})$ のスキーマ定義。
2. **Nishiki（錦：6極静的型システム＆構造計装コンパイラ）**:
   生テキスト（またはマルチモーダル特徴量）を入力として受け取り、決定論的構文解析と特化型SLM出力から **USIRの6極座標を算出・型検査・監査する「コンパイラ／計装エンジン本体」**。
   * 旧Nishiki（[15_Nishiki_全層制覇完了台帳](https://drive.google.com/file/d/14KLkgvQOXVAiXktaWVxJlG8mskTLPHMt/view?usp=drivesdk)）はアリストテレス[『弁論術』](https://plato.stanford.edu/entries/aristotle-rhetoric/)の3極（Kasane 襲 / Sasae 支 / Kamae 構）による技術文書・対話リンタであった。
   * 新Nishiki（v5）は、アリストテレス[『詩学』](https://en.wikipedia.org/wiki/Poetics_(Aristotle))および東洋美学・近代文学理論（バフチン多声性、ブレヒト異化効果、世阿弥「風情の序破急」・『花鏡』「離見の見」、夏目漱石『文学論』〈F+f〉）から導出した創作3極（Sadame 定 / Nagare 流 / Awase 合）を加えた **全6極（母音 `[a-a-e]` 統一）のUSIRコンパイラ** とする。

### 2.2 Nishiki 6極の全体定義と「離見の見」6診断レンズの1対1対応
世阿弥が『花鏡』で定義した「離見の見」（[東京大学博士論文データベース『成就の詩学－世阿弥能楽論の芸術論的特質－』](https://www.l.u-tokyo.ac.jp/postgraduate/database/2018/7308.html)）は、書き手の主観（我見）を離れて客席から全身を映す単一の鏡であり、内部に6極と1対1対応する6つの診断レンズを持つ（「なぜ6極なのに4つのミラーなのか？ Kamaeがない理由は？」という疑問への回答：4ミラーは旧案の不備であり、正しくは **6極すべてに対応する6診断レンズ** である）。

| 軸番号・名称（母音 `[a-a-e]`） | 担当クレート（`C:\dev\nishiki`） | 哲学・文芸理論の根拠 | 抽出する作家性（USIR v5 構造ゲノム） | Plotailor「離見の見」における診断レンズの役割 |
| :--- | :--- | :--- | :--- | :--- |
| **1. Kasane（襲）**<br>【表層・身体肌理】 | `crates/kasane-core` | アリストテレス[『弁論術』Pathos / Lexis](https://plato.stanford.edu/entries/aristotle-rhetoric/) ＋ [USIR v4 55次元書記統計](https://drive.google.com/file/d/1ZjaLnod_7_Le3yZ1_Cg5RlMCdmv15zHY/view?usp=drivesdk) | 地の文/会話文分離の文長・句読点呼吸周期、二重サプライザル（視覚画数×音韻モーラ）、五感モダリティ（視覚・聴覚・触覚・嗅覚・内臓感覚）配合比 | **【呼吸・五感偏りレンズ】**: リズムの単調化（Deadzone）や、視覚説明ばかりで身体感覚（触覚・温度・匂い）が欠落している箇所を検知。 |
| **2. Sasae（支）**<br>【論理・設定・視点】 | `crates/sasae-core` | アリストテレス『弁論術』Logos / Toulmin論証 / [日本語中心化理論](https://aclanthology.org/J94-2003.pdf) | 命題の根拠接地度、時系列・物理整合性、ゼロ代名詞（省略主語）と自由間接話法（内的視点）の帰属、登場人物の知識境界 | **【設定・時系列・視点汚染レンズ】**: 年表矛盾や、登場人物がまだ知り得ない情報を前提に行動・発言する「視点・知識汚染（POV Breach）」を検知。 |
| **3. Kamae（構）**<br>【姿勢・異化と余白】 | `crates/kamae-core` | アリストテレス『弁論術』Ethos / [ブレヒト「異化効果」](https://eprints.bournemouth.ac.uk/19482/1/Narrative-Approaches-to-Wellbeing.Jul2011.JF.pdf) / [世阿弥「秘する花」](https://www.l.u-tokyo.ac.jp/postgraduate/database/2018/7308.html) | 謎・伏線（Hole）の終幕未解決残余率 $K_{\text{Kamae}}$、重大場面での省略圧縮重力 $\Gamma_{\text{omission}}$、「秘する花（芸術的余白）」と「単なる説明不足」の分離 | **【説明過多・秘する花・迎合監査レンズ】**: 設定や感情を説明し尽くす「説明過多」と、逆に五感接地を欠いた「筆力不足の飛躍」を峻別し、無根拠な迎合（`E005`）を遮断。 |
| **4. Sadame（定）**<br>【世界観・人物公理】 | `crates/sadame-core` | アリストテレス[『詩学』Dianoia](https://en.wikipedia.org/wiki/Poetics_(Aristotle)) / [バフチン「ポリフォニー」](https://shs.cairn.info/revue-internationale-de-philosophie-2009-2-page-165?lang=en) | 6大価値軸に対する報い構造（世界観重力 $\Omega_k$）、人物間の思想的多声性（直交O(6)接続ラプラシアン $E_{\text{inter}}$）と1人称内面分裂度 $\text{Var}_t(x_u)$ | **【人物の操り人形化・公理ブレレンズ】**: 登場人物が作者の都合で動く単声の記号に堕ちていないか、作品固有の世界観の報い構造がブレていないかを診断。 |
| **5. Nagare（流）**<br>【時間・展開軌道】 | `crates/nagare-core` | アリストテレス『詩学』Mythos vs [世阿弥「風情の序破急」](https://shoryobu.kunaicho.go.jp/Publication/PDF/000/kiyo018a043.pdf)・起承転結 | 出来事駆動 $R_t^{(\text{event})}$ と空気感・心象駆動 $R_t^{(\text{mood})}$ の比率（世阿弥指数 $\Phi_{\text{Zeami}}$）、球面Fisher-Rao 1階運動量 $p_t$・2階共変加速度 $a_t$・局所屈曲度 $\tau_w(t)$ | **【筆のみの序破急・展開停滞レンズ】**: 事件の因果だけで話が進み「風情（空気感・内面）の変容」が死んでいる箇所や、加速度 $\|a_t\|_2 \to 0$ の停滞を検知。 |
| **6. Awase（合）**<br>【多層連動・対位法】 | `crates/awase-core` | [夏目漱石『文学論』〈F+f〉公式（大阪大学機関リポジトリ OUKA）](https://ir.library.osaka-u.ac.jp/repo/ouka/all/54390/mra_047_051A.pdf) | 焦点的出来事（$F_t$）と文体情緒（$f_t$）の順接（同調）、逆接アイロニー（太宰・道化型 $\Psi_{\text{irony}}$）、非情抑制（志賀直哉・ハードボイルド型 $\Psi_{\text{detachment}}$） | **【感情ベタ張り・対位法レンズ】**: 悲しい場面（$F_t < 0$）で「悲しい」と直接書くベタ張りを検知し、乾いた風景や日常所作を重ねる対位法を提示。 |

## 3. 全6極の完全数理仕様と「5大実装壁」の事前解消ロジック（完全収録版）

[Nan Z. Da (2019, Critical Inquiry)](https://jonathanstray.com/papers/Computational-Literary-Studies.pdf) の計算文学批判を踏まえ、小規模・離散的な特化型SLM出力でも発散・過学習・ゼロ除算を起こさない確定数式体系を以下に定義する。

### 3.1 第4極 `sadame-core`：6大価値軸・直交O(6)接続ラプラシアン・私小説内面分散・通時プロクラステス補正（HW-03統合済）
1. **6大価値軸（$d = 6$）の定義（[Kozlowski, Taddy & Evans 2019, ASR](https://arxiv.org/abs/1803.09288) 準拠）**:
   * $k=1$: **生死観**（`-1.0`: 死・虚無・滅び $\leftrightarrow$ `+1.0`: 生・生命力・再生）
   * $k=2$: **秩序・規範**（`-1.0`: 反逆・逸脱・破壊 $\leftrightarrow$ `+1.0`: 秩序・義理・遵法）
   * $k=3$: **世俗・神秘**（`-1.0`: 世俗・金銭・打算 $\leftrightarrow$ `+1.0`: 神秘・聖性・超越）
   * $k=4$: **利己・献身**（`-1.0`: 利己・支配・収奪 $\leftrightarrow$ `+1.0`: 献身・自己犠牲・庇護）
   * $k=5$: **理性・情念**（`-1.0`: 冷徹・打算・抑制 $\leftrightarrow$ `+1.0`: 情念・狂気・耽美）
   * $k=6$: **宿命・自由意志**（`-1.0`: 宿命・諦念・業 $\leftrightarrow$ `+1.0`: 自由意志・開拓・抗い）
   各場面 $t$ における人物 $u$ の価値茎ベクトル $x_u(t) \in [-1.0, +1.0]^6$ は、5段階整数順序尺度 $q \in \{-2, -1, 0, +1, +2\}$ から決定論的写像 $x = q / 2.0$ によって得る。
2. **【実装壁①の解消】人物間の対話ポリフォニー $E_{\text{inter}}$ と1人称私小説の内面分裂度 $E_{\text{intra}}$ の凸結合**:
   [Hansen \& Ghrist (2020) の言説層ラプラシアン](https://www.google.com/search?q=https://epubs.siam.org/doi/10.1137/19M1290458)および[Barbero et al. (2022) の直交O(d)接続ラプラシアン](https://arxiv.org/abs/2202.02536)に基づき、人物 $u, v$ 間の対話辺 $e=(u,v)$ における軸 $k$ の合意・対立符号を $\sigma_{uv}^{(k)} \in \{-1, 0, +1\}$、発話帰属信頼度を $\gamma_{uv} \in [0, 1]$、対話重みを $w_{uv}^{(k)} \ge 0$ とする：
   $$E_{\text{inter}} = \frac{1}{\sum_{e, k} \gamma_{uv} w_{uv}^{(k)} + \varepsilon} \sum_{e=(u,v) \in E} \sum_{k=1}^{6} \gamma_{uv} w_{uv}^{(k)} \frac{\left(x_{u,k} - \sigma_{uv}^{(k)} x_{v,k}\right)^2}{\|x_u\|_2^2 + \|x_v\|_2^2 + \varepsilon}$$
   しかし、梶井基次郎『檸檬』や志賀直哉『城の崎にて』、および現代のnoteエッセイのような **「1人称独白（対話辺 $|E| = 0$）」** では $E_{\text{inter}} = 0$ となり、「内面の激しい葛藤」が「単声の平板な文章」と誤判定される。これを防ぐため、同一主体の時間軸上の内面価値分散：
   $$E_{\text{intra}} = \frac{1}{|V|} \sum_{u \in V} \frac{1}{T_u} \sum_{t=1}^{T_u} \left\| x_u(t) - \bar{x}_u \right\|_2^2$$
   を導入し、有効対話辺数 $|E_{\text{eff}}| = \sum_e \gamma_{uv}$ に応じた重み $\lambda_{\text{inter}} = \frac{|E_{\text{eff}}|}{|E_{\text{eff}}| + 3.0}$ で凸結合した **総合Sadameエネルギー** を定義する：
   $$E_{\text{Sadame}}^{\text{total}} = \lambda_{\text{inter}} E_{\text{inter}} + (1 - \lambda_{\text{inter}}) E_{\text{intra}}$$
3. **世界観重力ベクトル（作家の公理署名） $\Omega_k$**:
   各人物 $u \in V$ の作品前半の平均価値茎 $\bar{x}_{u,k}^{\text{init}}$ と、終盤の運命利得変化 $\Delta u_u \in [-1.0, +1.0]$（救済・達成 vs 死・破滅・喪失）の重み付き共分散として算出する：
   $$\Omega_k = \operatorname{Cov}_{u \in V}\!\left(\bar{x}_{u,k}^{\text{init}}, \Delta u_u\right)$$
4. **【HW-03 完了仕様】72語の時代不変アンカー辞書 $A_{\text{anchor}}$ と直交プロクラステス通時補正 `EraProcrustesAligner`**:
   [Hamilton, Leskovec & Jurafsky (2016, ACL)](https://arxiv.org/pdf/1605.09096) の通時意味変化2法則（高頻度基本語の意味不変性・多義制度語の意味ドリフト）および[国立国語研究所『分類語彙表 増補改訂版』](https://core.ac.uk/download/234727868.pdf)に基づき、以下の72語を **時代不変アンカー語彙** として固定する：
   * **第1軸（生死）**: 正極 `{生きる, 命, 息, 芽吹く, 脈, 蘇る}` / 負極 `{死ぬ, 殺す, 亡骸, 息絶える, 墓, 朽ちる}`
   * **第2軸（秩序）**: 正極 `{守る, 律する, 誓う, 従う, 掟, 償う}` / 負極 `{背く, 破る, 逆らう, 盗む, 乱す, 欺く}`
   * **第3軸（神秘）**: 正極 `{祈る, 魂, 神, 霊, 祟る, 拝む}` / 負極 `{銭, 稼ぐ, 勘定, 売買, 損得, 腹}`
   * **第4軸（献身）**: 正極 `{尽くす, 救う, 庇う, 捧げる, 慈しむ, 譲る}` / 負極 `{奪う, 貪る, 見捨てる, 陥れる, 妬む, 蔑む}`
   * **第5軸（情念）**: 正極 `{燃える, 狂う, 泣く, 悶える, 恋う, 震える}` / 負極 `{計る, 抑える, 澄む, 検める, 静まる, 省みる}`
   * **第6軸（自由意志）**: 正極 `{挑む, 選ぶ, 拓く, 切り開く, 決める, 踏み出す}` / 負極 `{宿る, 定まる, 逃れぬ, 因果, 運命, 諦める}`
   明治・大正・昭和初期の固有制度語（`{家督, 奉公, 書生, 女給, 不義, 華族, 爵位, 離縁, 勘当, 屯田, 丁稚, 妾}` 等）はアンカー回転推定からマスク除外して2部グラフ辞書 `EraInstitutionMap` で基本軸へ写像し、[Schönemann (1966)](https://aclanthology.org/R19-1139.pdf) の直交プロクラステス変換：
   $$M = X_{\text{era}}^\top X_{\text{mod}} = U \Sigma V^\top, \quad R_{\text{era} \to \text{mod}} = U \operatorname{diag}\!\left(1, 1, 1, 1, 1, \det(UV^\top)\right) V^\top \in SO(6)$$
   によって現代価値空間へ等長整列させる。

### 3.2 第5極 `nagare-core`：場面境界分割・Jeffreys平滑化・α-歪JSD共鳴度・Fisher-Rao球面ハミルトン力学（HW-07統合済）
1. **【実装壁②の解消】決定論的場面境界分割器 `SceneSegmenter`**:
   段落 $p_m$ と $p_{m+1}$ の境界において、①連続空行または章・節見出し・約物区切り（`＊＊＊`, `◇`, `○`）、②時空間転換マーカー（`翌朝`, `その夜`, `数年後`, `一方` 等）、③前後3文ブロック間の登場人物・固有名詞Jaccard係数 $J(E_{\text{prev}}, E_{\text{next}}) \le 0.25$ のいずれかが成立し、かつ現在場面の文字数が下限 $L_{\min} = 200$ 字以上であるときに場面を分割する（上限 $L_{\max} = 1,600$ 字を超過した場合は最大段落境界で強制分割）。
2. **16次元出来事コード（`E01`〜`E16`）と16次元風情コード（`M01`〜`M16`）の完全定義（HW-01確定仕様）**:
   * **出来事カテゴリ $\mathcal{C}_{\text{event}}$（筆の序破急：16次元）**:
     `E01`: 遭遇・邂逅 / `E02`: 移動・旅立ち / `E03`: 対話・交渉 / `E04`: 対立・口論 / `E05`: 暴力・戦闘・傷害 / `E06`: 死・葬送・別離 / `E07`: 発見・露見・覚醒 / `E08`: 隠蔽・欺瞞・密謀 / `E09`: 労働・制作・日常所作 / `E10`: 祝祭・飲食・享楽 / `E11`: 恋愛・接触・情교 / `E12`: 病気・衰弱・狂気 / `E13`: 契約・誓約・審判 / `E14`: 逃走・追跡・潜伏 / `E15`: 救済・赦し・和解 / `E16`: 停滞・待機・反復
   * **風情カテゴリ $\mathcal{C}_{\text{mood}}$（風情の序破急：16次元）**:
     `M01`: 静寂・澄明・冷気 / `M02`: 陰翳・幽玄・薄暮 / `M03`: 焦燥・逼迫・窒息感 / `M04`: 哀愁・無常・喪失感 / `M05`: 滑稽・道化・自嘲 / `M06`: 官能・耽美・陶酔 / `M07`: 畏怖・崇高・怪異 / `M08`: 倦怠・憂鬱・アンニュイ / `M09`: 憤怒・憎悪・殺気 / `M10`: 慈愛・温もり・安らぎ / `M11`: 孤独・疎外・寂寥 / `M12`: 狂騒・祝祭・熱狂 / `M13`: 虚無・白昼夢・非現実 / `M14`: 決意・凛然・緊張 / `M15`: 郷愁・追憶・ノスタルジア / `M16`: 乾いた非情・無機質
3. **Jeffreys事前分布平滑化と α-歪ジェンセン・シャノン共鳴度**:
   Top-3選択重み $s_t \in \mathbb{R}_{\ge 0}^{16}$ に対し、Jeffreys事前分布 $\beta_0 = 0.5$ を加えてゼロ確率を完全排除した平滑化分布 $\tilde{P}_t(i) = \frac{s_{t,i} + \beta_0}{\sum_{j=1}^{16} (s_{t,j} + \beta_0)} \in \Delta_{\circ}^{15}$ を構築する。
   [Nielsen (2020, Entropy)](https://www.mdpi.com/1099-4300/22/2/221) の $\alpha$-歪JSD（$\alpha = 0.2$, $M_\alpha(P, Q) = (1-\alpha)P + \alpha Q$）：
   $$D_{\text{JS}}^{(\alpha)}(P \parallel Q) = (1-\alpha) D_{\text{KL}}(P \parallel M_\alpha(P, Q)) + \alpha D_{\text{KL}}(Q \parallel M_\alpha(P, Q))$$
   を用い、オフライン全文解析時は双方向共鳴度 $R_t^{(\text{offline})} = D_{\text{JS}}^{(\alpha)}(\tilde{P}_t \parallel \tilde{P}_{<t}) - D_{\text{JS}}^{(\alpha)}(\tilde{P}_t \parallel \tilde{P}_{>t})$、オンライン執筆・生成時は片側EWMA速度 $V_t^{(\text{online})} = D_{\text{JS}}^{(\alpha)}(\tilde{P}_t \parallel \hat{P}_{t-1}^{\text{EMA}})$ を算出する。
   **世阿弥風情駆動指数**:
   $$\Phi_{\text{Zeami}} = \frac{\sum_{t=1}^{T} \|p_t^{(\text{mood})}\|_2}{\sum_{t=1}^{T} \|p_t^{(\text{event})}\|_2 + \sum_{t=1}^{T} \|p_t^{(\text{mood})}\|_2 + \varepsilon}$$
4. **【HW-07 完了仕様】平方根超球面 $S_+^{15}$ 上の Fisher-Rao 1階共役運動量 $p_t$ と 2階共変加速度 $a_t$**:
   [ReSFlow (arXiv:2610.13225)](https://arxiv.org/html/2610.13225v1) に基づき、平滑化分布 $\tilde{P}_t$ を平方根写像 $\psi_{t,i} = \sqrt{\tilde{P}_t(i)}$ によって単位超球面の正象限 $S_+^{15}$（$\|\psi_t\|_2 = 1$）へ等長リフトする。
   * **球面角距離（Fisher-Rao 測地距離の半分）**: $\theta_t = \arccos\!\left(\operatorname{clamp}(\langle \psi_{t-1}, \psi_t \rangle, -1.0, 1.0)\right)$
   * **1階共役運動量ベクトル（球面対数写像 $\operatorname{Log}_{\psi_{t-1}}(\psi_t)$）**:
     $$p_t = \begin{cases} \left(1 + \frac{\theta_t^2}{6} + \frac{7\theta_t^4}{360}\right)\left(\psi_t - \cos\theta_t \cdot \psi_{t-1}\right) & (\theta_t < 10^{-4}\text{：テイラー展開特異点ガード}) \\ \frac{\theta_t}{\sin\theta_t}\left(\psi_t - \cos\theta_t \cdot \psi_{t-1}\right) & (\theta_t \ge 10^{-4}) \end{cases}$$
   * **大円測地線に沿った平行移動 $\Gamma_{\psi_{t-1} \to \psi_t}(p_t) \in T_{\psi_t} S_+^{15}$ と 2階リーマン共変加速度 $a_t$**:
     $$\Gamma_{\psi_{t-1} \to \psi_t}(p_t) = p_t - \frac{\langle p_t, \psi_t \rangle}{1.0 + \cos\theta_t + 10^{-12}}\left(\psi_{t-1} + \psi_t\right), \quad a_t = p_{t+1} - \Gamma_{\psi_{t-1} \to \psi_t}(p_t)$$
   $\|p_t\|_2 \to 0 \land \|a_t\|_2 \to 0$ が **AI Deadzone（熱的死）**、$\|p_t\|_2 > 0 \land \|a_t\|_2 \approx 0$ が **序・承（定常測地推進）**、$\|a_t\|_2$ のスパイクが **破・転（外力による進行方向の急屈折）** を表す。

### 3.3 第6極 `awase-core`：品詞ハードマスキングによる〈F+f〉物理分離と2成分対位法指数
1. **【実装壁③の解消】入力形態素ストリームの物理分離 `split_f_streams`**:
   * **焦点的出来事ストリーム $F_{\text{stream}}$**: 名詞・代名詞・動詞・格助詞・助動詞（打消・過去・受身・使役）のみを残し、形容詞・形容動詞・副詞・連体詞・比喩マーカー（`まるで`, `ような`, `ごとき`）・終助詞をすべて `[MASK]` に置換したトークン列。ここから出来事ベクトル $F_t = (\text{Valence}_F, \text{Arousal}_F) \in [-1, 1]^2$ を算出する。
   * **付着情緒ストリーム $f_{\text{stream}}$**: 形容詞・形容動詞・副詞・比喩スパン・色彩/五感語・文末助動詞/終助詞のみを残し、生死・暴力・契約等のイベント動詞を `[MASK]` に置換したトークン列。ここから文体情緒ベクトル $f_t = (\text{Valence}_f, \text{Arousal}_f) \in [-1, 1]^2$ を算出する。
2. **2成分対位法指数**:
   $$\Psi_{\text{irony}}(t) = \max\!\left(0, -\langle F_t, f_t \rangle\right), \quad \Psi_{\text{detachment}}(t) = \|F_t\|_2 \cdot \max\!\left(0, \|F_t\|_2 - \|f_t\|_2\right)$$

### 3.4 第3極 `kamae-core`：未回収フック生存率・省略圧縮重力・秘する花／説明不足分離器（HW-04統合済）
1. **終幕宙づり指数 $K_{\text{Kamae}}$ と省略圧縮重力 $\Gamma_{\text{omission}}(t)$**:
   $$K_{\text{Kamae}} = \frac{\sum_{j \in H_{\text{open}}(T) \setminus H_{\text{closed}}(T)} \omega_j}{\sum_{j \in H_{\text{open}}(T)} \omega_j + \varepsilon}, \quad \Gamma_{\text{omission}}(t) = \frac{\|F_t\|_2}{\log(e + L_t / L_{\text{median}})}$$
2. **【HW-04 完了仕様】「秘する花（芸術的余白）」と「単なる説明不足」の分離方程式 `HisuruHanaClassifier`**:
   [Kintsch (1988, 1998) の構築・統合モデル](https://verbs.colorado.edu/~mpalmer/Ling7800/Kintsch.pdf)、[Zwaan & Radvansky (1998) の事象インデックス状況モデル](https://sites.ualberta.ca/~dmiall/Cognitive/Readings/Zwaan_Radvansky_1998.pdf)、および[Connell & Lynott (2012) 知覚強度基準](https://www.lancaster.ac.uk/staff/connelll/papers/Connell-Lynott-2012-Cognition.pdf)に基づき、直前・当該場面の五感具体物密度 $c_{\text{sens}}(t)$（100字あたり）と抽象説明語比率 $a_{\text{abst}}(t)$ からなる **Kasane五感具象接地ゲート** $G_{\text{concrete}}(t) = \sigma\!\left(\frac{c_{\text{sens}}(t) - 1.5 a_{\text{abst}}(t) - 2.5}{1.0}\right)$ と、場面 $t-1 \to t$ 間の主要人物・小道具Entity参照連結率からなる **Sasae状況連続性ゲート** $G_{\text{bridge}}(t) \in [0, 1]$ を乗じる：
   $$\mathcal{H}_{\text{flower}}(t) = \Gamma_{\text{omission}}(t) \cdot G_{\text{concrete}}(t) \cdot G_{\text{bridge}}(t) \quad (\text{芸術的余白：秘する花})$$
   $$\mathcal{D}_{\text{deficit}}(t) = \Gamma_{\text{omission}}(t) \cdot \left(1.0 - G_{\text{concrete}}(t) \cdot G_{\text{bridge}}(t)\right) \quad (\text{単なる説明不足・設定抜け})$$

### 3.5 第2極 `sasae-core`：日本語ゼロ代名詞の中心化理論ソルバー ＆ 自由間接話法5トリガー（HW-02統合済）
1. **【HW-02 完了仕様】自由間接話法（内的視点構成）の5大形態素トリガー `FidResolver`**:
   * **T1（重み $0.30$）**: 過去「タ形」基調文脈（直前3文のタ形率 $\ge 0.6$）における非過去「ル形」または体言止めへの遷移（[愛知大学『川端康成「雪国」のタ形とル形』](https://taweb.aichi-u.ac.jp/tgoken/bulletin/pdfs/NO41/05p069-082%20YAMAMOTO.pdf)）。
   * **T2（重み $0.25$）**: カギ括弧外における近称・発話時ダイクシス（`今`, `いま`, `ここ`, `この〜`, `明日`, `きのう`, `こうやって`）の出現（[山口大学『英語と日本語の話法』](https://ypir.lib.yamaguchi-u.ac.jp/bg/349/files/135619)）。
   * **T3（重み $0.35$）**: 三人称文脈における主観感情・内臓感覚形容詞（`悲しい`, `苦しい`, `痛い`, `寂しい`, `憎い`, `恋しい` 等）の非ガル化裸出。
   * **T4（重み $0.35$）**: カギ括弧外における認識モダリティ・疑問・詠嘆・言いさし文末（`〜にちがいない`, `〜のではなかろうか`, `〜かしら`, `〜のだ`, `〜か。`, `〜……`）（[京都大学『自由間接話法と語りのフレーム』](https://repository.kulib.kyoto-u.ac.jp/bitstreams/d0438f96-325f-4eb8-9918-4b26316ca445/download)）。
   * **T5（重み $0.25$ ＋ 焦点人物 $v_{\text{focal}}$ 確定）**: 直前1〜2文における知覚・認知動詞（`見る`, `眺める`, `見上げる`, `聞く`, `気づく`, `思う`, `考える`, `感じる`）の活性化と、そのハ格／ガ格主語 $v_{\text{focal}}$ の特定。
   内的視点スコア $I_{\text{FID}}(s_k) = \min(1.0, \sum_{m=1}^5 w_m \mathbb{I}(\text{T}_m)) \ge 0.50$ の文 $s_k$ は、語り手の客観事実ではなく焦点人物 $v_{\text{focal}}$ の主観内面として帰属させる。

### 3.6 第1極 `kasane-core`：USIR v4 55次元書記・地の文分離統計 ＋ 二重サプライザル ＋ 呼吸波形
`C:\dev\usir` Phase 36 で LOWO 73.9%（34/46）・Held-out 83.3%（50/60）を実証した **55次元特徴量**（24次元リズム特徴 ＋ 27次元拡張書記統計 ＋ **会話文を除外した地の文専用4統計：`narration_log_mean_len`, `narration_p90_len`, `narration_comma_rate`, `dialogue_char_ratio`**）と、`C:\dev\kasane` の二重サプライザル（$S_{\text{vis}} \times S_{\text{phon}}$）・呼吸波形・五感モダリティ分布 $m_{\text{sense}} \in \Delta^4$ を統合する。

## 4. 青空文庫30作家×100場面サンプリング仕様＆ローカル8B GBNFスキーマ（HW-01 完全収録版）

### 4.1 対象30作家リスト（4大潮流・各100場面＝計3,000場面）
1. **グループA：既存アトラス核10作家**
   夏目漱石（`000148`）、芥川龍之介（`000879`）、谷崎潤一郎（`001383`）、太宰治（`000035`）、宮沢賢治（`000081`）、梶井基次郎（`000074`）、森鴎外（`000129`）、泉鏡花（`000050`）、坂口安吾（`001095`）、中島敦（`000119`）
2. **グループB：自然主義・白樺派・私小説・内省派（7作家）**
   志賀直哉、島崎藤村、田山花袋、有島武郎、武者小路実篤、堀辰雄、永井荷風
3. **グループC：新感覚派・モダニズム・幻想・耽美派（7作家）**
   川端康成（※パブリックドメイン対象作または研究検証用保持枠）、横光利一、夢野久作、江戸川乱歩、小栗虫太郎、岡本かの子、室生犀星
4. **グループD：プロレタリア・大衆文芸・女性作家・戦後派（6作家）**
   小林多喜二、葉山嘉樹、林芙美子、樋口一葉、与謝野晶子、織田作之助

**層化抽出ルール**: 各作家の作品群から、物語位置 $u \in [0, 1]$ の **序盤（$[0.0, 0.25)$：25場面）・中盤（$[0.25, 0.75)$：50場面）・終盤（$[0.75, 1.0]$：25場面）** を層化抽出し、1場面あたり $200 \le L_t \le 1,600$ 字に切り出す。

### 4.2 ローカルRTX 4070（Qwen3-8B / Swallow-8B）用 GBNF 強制 JSON スキーマ
連続小数の出力を一切禁止し、5段階整数順序尺度 `{-2, -1, 0, 1, 2}` と Top-3 カテゴリコード選択のみを強制する：

```json
{
  "scene_id": "000879_rashomon_s004",
  "focal_character": "下人",
  "awase": {
    "F_event_valence": -2,
    "F_event_arousal": 1,
    "f_style_valence": 0,
    "f_style_arousal": -1
  },
  "nagare": {
    "event_top3": [
      {"code": "E07", "rank_weight": 3},
      {"code": "E01", "rank_weight": 2},
      {"code": "E16", "rank_weight": 1}
    ],
    "mood_top3": [
      {"code": "M02", "rank_weight": 3},
      {"code": "M16", "rank_weight": 2},
      {"code": "M07", "rank_weight": 1}
    ]
  },
  "sadame": {
    "character_stalks": [
      {
        "name": "下人",
        "axes_q": [1, -1, -1, -2, 0, 1]
      },
      {
        "name": "老婆",
        "axes_q": [2, -2, -1, -2, -1, 0]
      }
    ],
    "dialogue_edges": [
      {
        "u": "下人",
        "v": "老婆",
        "sigma": [-1, -1, 0, -1, -1, 0],
        "confidence": 3
      }
    ]
  },
  "kamae": {
    "opened_hooks": ["老婆が死人の髪を抜く真意"],
    "closed_hooks": []
  }
}
```

## 5. 【工程を省略しない再計画】Phase 0〜Phase 4 の全体開発プロセスと3レーン分担

### Phase 0：【最優先・コード変更前の必須ゲート】Claude Fable 5.2 による構想・数式・実コード整合性の冷徹監査
* **実行者**: **Claude Fable 5.2**（`C:\dev\nishiki` または `C:\dev` で起動、1Mコンテキスト活用）
* **入力**:
  1. 本仕様書（`F:\マイドライブ\Claude-Sync\USIR_v5_Nishiki_Complete_Spec_and_Plan_20261110.md`）
  2. ローカルの3リポジトリの実コード（`C:\dev\nishiki`, `C:\dev\kasane`, `C:\dev\usir`）
* **実施する4つの監査項目**:
  1. **実コード棚卸しと差分特定**: `C:\dev\nishiki`（226 tests）、`C:\dev\kasane`（86+ tests）、`C:\dev\usir`（599 tests）の現在のクレート・モジュール構成を実測確認し、本仕様書の提案クレート構成と衝突・重複する箇所をすべて列挙する。
  2. **6極数式とSLMパイプラインの批判的検証（忖度・迎合の完全禁止）**:
     本仕様書の第3章・第4章・第6章（Fable 6大課題）について、「ここは数学的・言語学的に筋が良い」「ここは計算量やデータスパース性の観点で破綻する／もっと単純な式にすべき」という冷徹なダメ出しと対案を出す。
  3. **Fable 5.2 自身が解くべき難関タスク（STAGE-F1〜F6）の優先順位と仕様確定**:
     後述する6つのFable特務課題（純Rust線形代数、kasane統合、ゼロ代名詞ソルバー、Ripser TDA、SMCデコーダ、HSIC/繰り込み群）を精査し、実装順序とインターフェース（関数シグネチャ）を確定する。
  4. **監査レポートの書き出しと共有**:
     結果を `C:\dev\nishiki\docs\handoff\2026-11-10_phase0_fable_critical_audit.md` および **`F:\マイドライブ\Claude-Sync\inbox\2026-11-10_phase0_fable_critical_audit.md`** の双方に書き出し、さらに `python C:\dev\claude-drive-sync\sync_claude_to_drive.py` を1回実行して同期を完了させる。

### Phase 1：監査レポートの受け取り・仕様確定と Phase 2 への進行（人間 ＆ Gemini Web）
* Fable 5.2 が `F:\マイドライブ\Claude-Sync\inbox\2026-11-10_phase0_fable_critical_audit.md` を出力したら、Gemini Web（または人間）がそれを確認し、Fable 5.2 が策定した修正計画に基づいて Phase 2（Fable 5.2 による難関カーネル実装）へ進む。

### Phase 2：【Lane F：Claude Fable 5.2 特務執刀レーン】難関数理・アルゴリズム・3リポジトリ統合の先行突破（STAGE-F1〜STAGE-F6）
11月16日までの Fable 5.2 開放枠＋1回無料リセット権（11月17日 16:59 JSTまで有効）を投入し、Fable 5.2 の実力をテストしつつ、**「これさえ先に純Rustで作っておけば、後の Sonnet 5.5 や Antigravity 2.0 の開発が劇的に楽になる6大難所」** を解き切る。

| ステージID | 対象クレート / ファイル | Fable 5.2 が解くべき難解テーマと厳密実装仕様 | 完了条件（Verification） |
| :--- | :--- | :--- | :--- |
| **STAGE-F1** | `crates/nishiki-linalg`<br>（新規・外部依存ゼロ数理カーネル） | **【純Rust・Wasm完全互換の数値線形代数＆超球面幾何カーネル】**<br>OpenBLAS/LAPACK等のC依存を一切使わず純Rust（`f64`）で以下を実装：<br>① **[Drmač & Veselić (2008) の前処理付き片側ヤコビSVD](https://www.netlib.org/lapack/lawnspdf/lawn169.pdf)** と `det(UV^T) = +1` 符号補正による直交プロクラステス変換器 `orthogonal_procrustes(a, b) -> Matrix6x6`（`HW-03` 用）。<br>② **巡回ヤコビ対称固有値ソルバー** `symmetric_eigen(mat)`（`Sadame` 接続ラプラシアン用）と Tikhonov正則化付き一般化Fisher判別器 `generalized_fisher_lda(sw, sb, reg)`。<br>③ **単位超球面 `S_+^15` 幾何カーネル**（`HW-07` 用）：$\theta < 10^{-4}$ でのテイラー展開特異点ガード付き球面対数写像（1階運動量 `p_t`）、大円平行移動 `parallel_transport`、2階共変加速度 `covariant_accel`。 | ランク落ち・悪条件行列（条件数 $10^{10}$）・$\theta = 0$・$\theta \to \pi/2$ を含む退化テストで `NaN`/`Inf` がゼロ、かつ直交性誤差 $\|R^\top R - I\|_F < 10^{-12}$ を達成すること。 |
| **STAGE-F2** | `crates/usir-schema`<br>`crates/kasane-core` | **【`C:\dev\kasane` 資産の `C:\dev\nishiki` 無血統合と 32B ABI 統一】**<br>`C:\dev\kasane` の二重サプライザル（$S_{\text{vis}} \times S_{\text{phon}}$）、生理的呼気波形 `BreathWaveEngine`、宣言型美学ルールブック `BYOR`（格フレーム選択制限・シグモイド飽和・TF-IDF手癖減衰）、および 32B `CognitiveSpan`（UTF-16/UTF-8変換）を、`C:\dev\nishiki` の既存226テストを壊さずに完全統合する。 | 既存の `nishiki` 全テスト＋移植した `kasane` 全テストが単一ワークスペースで 0 warning / 0 error で PASS すること。 |
| **STAGE-F3** | `crates/sasae-core/src/zero_anaphora.rs` | **【日本語の「主語省略（ゼロ代名詞）・多人数対話・自由間接話法」統合決定論ソルバー】**<br>[Walker, Iida & Cote (1994) の日本語中心化理論（Centering Theory）](https://aclanthology.org/J94-2003.pdf) に基づき、前向き中心リスト $Cf$ の優先順位（主題ハ ＞ 視点・授受補助動詞「くれる/もらう/やる/来る/行く」 ＞ 敬語レジスター ＞ 主格ガ ＞ 与格ニ ＞ 対格ヲ）と状態遷移（`CONTINUE > RETAIN > SMOOTH-SHIFT > ROUGH-SHIFT`）から省略主語・目的語を復元し、カギ括弧ターンスタックおよび `HW-02` の自由間接話法5トリガー `FidResolver` と結合する。 | 漱石『こころ』・川端『雪国』型の主語省略・無記名対話・自由間接話法テスト20本で人物帰属正解率 90% 以上かつ $O(N)$ 線形時間で動作すること。 |
| **STAGE-F4** | `crates/nishiki-tda`<br>（`HW-09` & `HW-10` 先行解決） | **【Ripser型 Vietoris-Rips パーシステント・ホモロジー × 認知エルゴノミクス未踏領域発見器】**<br>[Ulrich Bauer (2021) の Ripser アルゴリズム](https://arxiv.org/abs/1908.02518)（陰的余境界作用素・Apparent Pairs・Clearing最適化による $\mathbb{Z}/2\mathbb{Z}$ 境界行列簡約）を純Rustで実装し、6極空間の点群から0次・1次（ループ $H_1$）・2次（空洞 $H_2$）のパーシステンス・バーコードと空白重心座標 $c_i$ を抽出。さらに認知エルゴノミクス制約（`HW-10`：読解破綻ペナルティによるフィルタ）を適用して「読者が快感を得られる未踏領域（Viable Aesthetic Voids）」を出力する。 | 円環 $S^1$・トーラス $T^2$・球面 $S^2$ の合成点群でベッチ数 $(\beta_0, \beta_1, \beta_2)$ と穴の重心が正確に復元されること。 |
| **STAGE-F5** | `crates/nishiki-smc`<br>（`HW-27` 先行解決） | **【ローカル8Bデコーダ用「6極ハミルトン誘導 Clause-Twisted SMC & BPE境界ロジットマスク」高速Rustコア】**<br>`C:\dev\usir` Phase 36 の `realtime_decoder.py` の課題（長文目標時の読点ロック、結合句点BPEトークン漏れ）を解消する文長比例・動的読点スケジューラと、節単位4粒子SMCで6極ポテンシャルを `< 0.1ms` で採点するRust/PyO3エンジンを実装する。 | 1ステップ `< 0.1ms` で動作し、芥川型（短文・高対位法）と谷崎型（長文・多読点）の粒子重みが分離すること。 |
| **STAGE-F6** | `crates/nishiki-cli/tests/fable_invariants.rs`<br>（`HW-08` & `HW-11` 先行解決） | **【6極のHSIC直交性検定・長短スケール繰り込み群変換（RG）・極限退化ファジング証明スイート】**<br>RBFカーネルによるヒルベルト・シュミット独立性規準（HSIC）＋Löwdin対称直交化（`HW-08`）と、短編（20場面）と大長編（500場面）のスケール差を補正する繰り込み群変換（`HW-11`）を実装し、100種の極限退化入力でパニック・`NaN` ゼロを証明する。 | 20場面と200場面の繰り込み後ゲノム距離 $< 0.05$、全退化ファジングテスト 100% PASS。 |

### Phase 3：【Lane A：Sonnet 5.5 + Haiku 5.5】×【Lane B：Antigravity 2.0 + RTX 4070】の並列本実装
Phase 2 の共通カーネル完成後（または独立モジュールから並行して）、Lane A と Lane B を走らせる。

* **Lane A（`C:\dev\nishiki`：Sonnet 5.5 + Haiku 5.5）**:
  * **STAGE-A1**: `crates/usir-schema` の6極データ構造とJSON/バイナリシリアライズ実装。
  * **STAGE-A2**: `crates/sadame-core`（直交O(6)接続ラプラシアン＋私小説内面分散凸結合＋世界観重力 $\Omega_k$＋`HW-03` 時代不変アンカープロクラステス補正）および `crates/kamae-core`（$K_{\text{Kamae}}$＋$\Gamma_{\text{omission}}$＋`HW-04` 秘する花分離器 `HisuruHanaClassifier`）の実装。
  * **STAGE-A3**: `crates/nagare-core`（場面分割器＋Jeffreys平滑化＋α-歪JSD共鳴度＋`HW-07` 超球面 $S_+^{15}$ 運動量 $p_t$・共変加速度 $a_t$）の実装。
  * **STAGE-A4**: `crates/awase-core`（品詞ハードマスキング `split_f_streams`＋$\Psi_{\text{irony}}$・$\Psi_{\text{detachment}}$）および `crates/kasane-core`（USIR v4 55次元書記統計統合）の実装。
  * **STAGE-A5**: `crates/sasae-core`（`STAGE-F3` ゼロ代名詞・自由間接話法ソルバーの統合）および `crates/nishiki-py`（PyO3）・`crates/nishiki-cli`（`nishiki profile <PATH>`）の統合エクスポートと全テスト通過。
* **Lane B（`C:\dev\usir`：Antigravity 2.0 + ローカルRTX 4070 12GB VRAM / API課金 $0.00）**:
  * **STAGE-B1**: `python/usir_py/v5/aozora_sampler.py` — 第4.1節の30作家×100場面（計3,000場面）層化サンプリングと $F_{\text{stream}} / f_{\text{stream}}$ 分離JSONL生成。
  * **STAGE-B2**: `python/usir_py/v5/local_zero_cost_teacher.py` — ローカルRTX 4070上の `Qwen3-8B` & `Qwen3-Swallow-8B` に対し、第4.2節のGBNF JSONスキーマを適用した3,000場面合議アノテーション実行（API課金 $0.00）。
  * **STAGE-B3**: `python/usir_py/v5/train_specialized_slms.py` — `Nano-Dual-Valence`、`DeBERTa-v3-Ja-Sheaf-NLI`、`Nano-FID-Head` の蒸留とINT8 ONNXエクスポート。
  * **STAGE-B4**: `python/usir_py/v5/era_procrustes_calib.py` & `evaluate_usir_v5_atlas.py` — 明治・大正・昭和の72語アンカー直交プロクラステス行列算出、USIR v4（LOWO 73.9% / Held-out 83.3%）対比の作家識別精度検証、および `nishiki-tda` を用いた青空文庫・未踏領域（Viable Aesthetic Voids）レポートの出力。

## 6. 並行調査・宿題バックログ全31項目一覧（HW-01〜HW-31）

| ID | 状態 | テーマ | 要点・反映先 |
| :--- | :--- | :--- | :--- |
| **HW-01** | **DONE** | 青空文庫3,000場面サンプリング＆離散順序GBNF JSONスキーマ・30作家リスト確定 | 本仕様書 第4章に全文収録（STAGE-B1/B2）。 |
| **HW-02** | **DONE** | 日本語小説の「自由間接話法（内的視点構成）」5大形態素トリガーと `FidResolver` | 本仕様書 第3.5節に全文収録（STAGE-F3 / STAGE-A5）。 |
| **HW-03** | **DONE** | 『Sadame』6大価値軸の72語時代不変アンカー辞書と直交プロクラステス通時補正 | 本仕様書 第3.1節に全文収録（STAGE-F1 / STAGE-A2 / STAGE-B4）。 |
| **HW-04** | **DONE** | 『Kamae』における「芸術的な余白（秘する花）」と「単なる説明不足」の架橋推論分離フィルタ | 本仕様書 第3.4節に全文収録（STAGE-A2 / STAGE-B4）。 |
| **HW-05** | OPEN | 女性登場人物への「象徴投影（巫女化・ファムファタール化・母性化）」対称性破れテンソル抽出 | グレマス行為項の非対称遷移行列×第3軸共分散。 |
| **HW-06** | OPEN | 日本語の「人称代名詞・役割語・呼称変化マトリクス」による関係性力学の抽出 | 一人称・二人称・呼称変化からの急接近/決裂検知。 |
| **HW-07** | **DONE** | USIR v2「ハミルトン2階運動量 $p_t$」と6極軌道の統合（平方根超球面 $S_+^{15}$ 共変加速度 $a_t$） | 本仕様書 第3.2節に全文収録（STAGE-F1 / STAGE-A3）。 |
| **HW-08** | OPEN | 6極間の「直交性検定（HSIC監査）」とLöwdin対称直交化 | STAGE-F6 で先行実装・実データ検証。 |
| **HW-09** | OPEN | 位相的データ解析（Ripser型パーシステント・ホモロジー）による「未踏空白地帯」自動発見 | STAGE-F4 で先行実装・実データ検証。 |
| **HW-10** | OPEN | 「読解不能な破綻領域」と「未踏の傑作領域（Goldilocks Zone）」を分ける認知エルゴノミクス境界 | STAGE-F4 で自由エネルギー制約として統合。 |
| **HW-11** | OPEN | 短編（20場面）vs 大長編（500場面）のスケール不変性と繰り込み群（Renormalization Group） | STAGE-F6 で先行実装・実データ検証。 |
| **HW-12** | OPEN | note記事・現代エッセイにおける「読者共感（スキ率・読了率）」への6極適応 | 小説と評論のジャンル境界パラメータ較正。 |
| **HW-13** | OPEN | 音楽領域への拡張①：機能和声進行（T-SD-D）と調性重力の `Sadame` / `Nagare` 写像 | 音楽理論における緊張・解決の球面運動量モデル。 |
| **HW-14** | OPEN | 音楽領域への拡張②：日本語歌詞のモーラ・アクセントとメロディ高低の対位法（`Awase`） | 明るい曲調×暗い歌詞（ボカロ・J-POP対位法）の定量化。 |
| **HW-15** | OPEN | 映像（Cinema / MV）への拡張①：モンタージュ編集リズムとカット割りの `Kasane` / `Nagare` 写像 | ショット長分布・カメラ運動量のスペクトル解析。 |
| **HW-16** | OPEN | 映像（Cinema / MV）への拡張②：画面外の音（Off-screen sound）と視覚的余白の `Kamae` 写像 | 映さないことによる恐怖・余韻の定量化。 |
| **HW-17** | OPEN | 小説 $\leftrightarrow$ 音楽 $\leftrightarrow$ MV の6極等長クロスモーダル変換関手（Functor）の設計 | 同一の6極波形を持つ小説と楽曲・映像の相互生成。 |
| **HW-18** | OPEN | Meta Muse エージェントによる現代Web小説・noteトレンドの非同期収集プロトコル | ローカルGitと競合しない外部トレンド斥候設計。 |
| **HW-19** | OPEN | Plotailor「離見の見」6診断レンズのリアルタイムWasm UI表示仕様（Phase 4用） | 3ペインDocked Atelier準拠・モーダル禁止UI設計。 |
| **HW-20** | OPEN | ローカルRTX 4070 + 64GB RAM における6極バッチ推論のメモリ・スループット極限最適化 | 3,000場面を30分以内で処理するバッチ設計。 |
| **HW-21** | OPEN | 同一作家の「初期・中期・晩年」通時変遷軌道（Chronicle Trajectory）のベクトル場定式化 | 漱石・芥川の加齢・時代に伴う6極シフト解析。 |
| **HW-22** | OPEN | 比喩・隠喩の「概念跳躍距離（Metaphorical Leap Distance）」の定量化 | 平凡な直喩と鮮烈な隠喩（『檸檬』等）の双曲距離。 |
| **HW-23** | OPEN | 会話文における「言外のサブテキスト（建前と本音の乖離）」の抽出手法 | セリフ字面極性と直前直後の身体・小道具動作のズレ。 |
| **HW-24** | OPEN | Web連載小説における「章末の引き（Cliffhanger）」と読者継続率の数理モデル | 章末の $\Delta H_{\text{open}}$ と共変加速度 $\|a_t\|_2$。 |
| **HW-25** | OPEN | 「世界観の説明（設定開示）」を退屈させずに読ませる『情報開示ペース配分』の黄金比 | `Sasae` 新規シンボル導入率と `Nagare` の最適比。 |
| **HW-26** | OPEN | 読者ペルソナ別の認知負荷・好悪シミュレーション（純文学選考委員 vs Web読者） | 6極重みベクトルによるマルチペルソナ評価関数。 |
| **HW-27** | OPEN | 6極ターゲット座標からの「ローカル8B 生成時ロジット・SMC制御」の再設計 | STAGE-F5 で Rust/PyO3 コアとして先行実装。 |
| **HW-28** | OPEN | 多言語普遍性（英語・仏語・中国語古典）への6極スキーマ適用と翻訳不変性の検証 | 原文と英訳における深層4極の波形保存性検定。 |
| **HW-29** | OPEN | クリーンルーム著作権安全ゲート（Three-Room Gate v5）の6極拡張証明 | 作風（6極統計）の継承と表現一致（7字以上）の完全遮断。 |
| **HW-30** | OPEN | USIR v5 × Nishiki 6極の統合実証論文（Paper v5）自動ビルド＆全数値機械検証パイプライン | `verify_paper_v5.py` によるモックゼロ証明。 |
| **HW-31** | OPEN | 『Kamae × Kasane』T.S.エリオット「客観的相関物（Objective Correlative）」としての小道具反復追跡 | 省略された感情を運搬する象徴小道具の共起グラフ。 |

## 7. 【コピペ投入用】精度を極限まで高めたフェーズ別プロンプト集

### 7.1 【まず最初に実行：Phase 0 構想＆実コード冷徹監査プロンプト】（Claude Code でモデルを `Fable 5.2` にして投入）

```markdown
/plan
あなたは USIR × Nishiki プロジェクトのチーフアーキテクト兼・冷徹な技術監査役（Claude Fable 5.2）です。
現在、Gemini Web 側で策定された新構想『USIR v5 × Nishiki 6極（完全自己完結版仕様書）』が以下のローカルGoogleドライブ同期パスに配置されています：

- 仕様書パス: `F:\マイドライブ\Claude-Sync\USIR_v5_Nishiki_Complete_Spec_and_Plan_20261110.md`

【重要ルール：まだプロダクションコードの実装・書き換えは行わないでください】
いきなり実装に入るのではなく、まず **Phase 0（構想・数式・実コード整合性の冷徹な事前監査と実装計画の確定）** を実行してください。お世辞や迎合は一切不要です。個人共通ルール（平坦な事実・論理・データ、未検証事項の明記）に従い、以下の手順を実行してください：

### 手順1：仕様書の精読とローカル3リポジトリの実コード棚卸し
1. `F:\マイドライブ\Claude-Sync\USIR_v5_Nishiki_Complete_Spec_and_Plan_20261110.md` を全文読み込んでください。
2. ローカルの以下3つのリポジトリの実際のディレクトリ構造・既存クレート・主要ファイル・テスト件数を調査してください：
   - `C:\dev\nishiki`（既存の kasane-core, sasae-core, kamae-core, nishiki-cli, nishiki-wasm 等の実態）
   - `C:\dev\kasane`（32B CognitiveSpan, सुप्राईзал, BreathWave, BYOR 等の実態と未コミット差分の有無）
   - `C:\dev\usir`（Phase 14〜49 で確立された 55次元 Fisher LDA, realtime_decoder.py, 青空文庫コーパス等の実態）

### 手順2：構想・数式・Fable特務課題（STAGE-F1〜F6）に対する冷徹な批判的監査
実コードと照らし合わせ、以下の4点を遠慮なく批判・検証してください：
1. **【構想の良し悪しと数式の妥当性】**:
   - 第2章・第3章の6極（Kasane, Sasae, Kamae, Sadame, Nagare, Awase）の数式および完了済み宿題（HW-01, HW-02, HW-03, HW-04, HW-07）の中で、実データ（青空文庫）で確実に機能する本物の設計はどれか？
   - 逆に、計算量・スパース性・日本語文法の観点から「このまま実装すると破綻する／形骸化する／過剰に複雑すぎる」箇所はどこか？ どう修正・簡素化すべきか？
2. **【リポジトリ統合（kasane → nishiki）と usir 連携の現実解】**:
   - `C:\dev\kasane` を `C:\dev\nishiki` に統合する際、既存の型やテストとどこが衝突するか？ 最も手戻りのない統合手順は何か？
3. **【Fable 5.2 特務トラック（STAGE-F1〜STAGE-F6）の査定と優先順位】**:
   - 第5章の `STAGE-F1`（純Rust線形代数・球面幾何）、`STAGE-F2`（kasane統合）、`STAGE-F3`（日本語ゼロ代名詞・中心化理論ソルバー）、`STAGE-F4`（Ripser TDA未踏発見）、`STAGE-F5`（SMC高速デコーダ）、`STAGE-F6`（HSIC・繰り込み群）について、Fable 5.2（11/16までの開放枠＋1回無料リセット）で解くべき価値と優先順位を査定し、各モジュールの具体的なRust関数シグネチャ・ファイル構成を確定せよ。
4. **【Lane A（Sonnet 5.5 + Haiku 5.5）および Lane B（Antigravity 2.0）との分担境界】**:
   - どのファイルをどの順序で触れば Git 競合や手戻りがゼロになるかの厳密な実行順序（DAG）を策定せよ。

### 手順3：監査レポート兼・確定実装計画書の書き出し（2箇所への保存＋同期スクリプト実行）
監査結果と修正版実装計画（8節のHANDOFFテンプレート準拠）を、以下の **2つのパス両方** に Markdown ファイルとして書き出してください：
1. `C:\dev\nishiki\docs\handoff\2026-11-10_phase0_fable_critical_audit.md`
2. `F:\マイドライブ\Claude-Sync\inbox\2026-11-10_phase0_fable_critical_audit.md`

書き出し完了後、最後に以下のコマンドを1回実行して会話ログも Google Drive へ同期し、チャットには監査結論の要約（10〜15行）を出力して私の確認を待ってください：
`python C:\dev\claude-drive-sync\sync_claude_to_drive.py`
```

### 7.2 【Phase 0 監査完了後：Phase 2（Lane F：Fable 5.2 特務実装）起動プロンプト】
（※Phase 0 の監査レポートを確認した後、Fable 5.2 にそのまま続けて投入するプロンプト）

```markdown
/goal
Phase 0 で確定した監査レポート（`C:\dev\nishiki\docs\handoff\2026-11-10_phase0_fable_critical_audit.md`）の修正設計に従い、Phase 2【Lane F：Fable 5.2 特務実装】を開始してください。

1. 確定した優先順に従い、`STAGE-F1`（`crates/nishiki-linalg`：純Rust片側ヤコビSVD・対称固有値・超球面平行移動）から順に実装・単体テスト・退化境界値テストを作成し、各ステージごとに `cargo test --workspace` が 0 error / 0 warning で PASS することを確認してコミット＆Pushしてください。
2. 各ステージ（STAGE-F1, F2, F3...）が1つ完了するごとに、進捗と実測テスト結果を以下の2箇所に更新書き出ししてください：
   - `C:\dev\nishiki\docs\handoff\HANDOFF.md`
   - `F:\マイドライブ\Claude-Sync\inbox\HANDOFF_nishiki.md`
3. 全対象ステージ完了後、`python C:\dev\claude-drive-sync\sync_claude_to_drive.py` を実行して同期を完了してください。
```

### 7.3 【Lane B：Antigravity 2.0 事前監査＆青空文庫3,000場面サンプリング起動プロンプト】（`C:\dev\usir` で起動）

```markdown
あなたは `C:\dev\usir` 担当のリードエージェント（Antigravity 2.0）です。
まず `F:\マイドライブ\Claude-Sync\USIR_v5_Nishiki_Complete_Spec_and_Plan_20261110.md` を全文読み込み、特に第4章（青空文庫30作家×100場面サンプリング仕様＆GBNFスキーマ）と第5章（Phase 3 Lane B：STAGE-B1〜STAGE-B5）を確認してください。

【絶対遵守ルール】
- Google AI Studio 等の有料クラウドAPIは一切呼び出さないこと（課金 $0.00 厳守）。すべての処理はローカル環境（CPU + RTX 4070 12GB VRAM + 64GB RAM）で実行すること。

【実行ステップ】
1. **Step B0（`C:\dev\usir` 側の現状確認と監査）**:
   現在の `C:\dev\usir` 内の青空文庫コーパス配置状況とローカルLLM実行環境（Ollama / llama.cpp / HuggingFace Transformers + bitsandbytes 等で `Qwen3-8B` / `Qwen3-Swallow-8B` がロード可能か）を確認し、結果を `C:\dev\usir\docs\handoff\HANDOFF.md` および `F:\マイドライブ\Claude-Sync\inbox\HANDOFF_usir.md` に書き出せ。
2. **Step B1（`STAGE-B1`：青空文庫3,000場面の層化サンプリングと F/f ストリーム分離）**:
   `python/usir_py/v5/aozora_sampler.py` を実装し、既存コーパスおよびパブリックドメイン青空文庫データから序盤25%・中盤50%・終盤25%の場面（200〜1,600字）を切り出し、品詞ハードマスキングによる `F_stream` と `f_stream` を付与したJSONLデータセットと `pytest` 単体テストを作成せよ。
3. 完了ごとに `F:\マイドライブ\Claude-Sync\inbox\HANDOFF_usir.md` を更新し、コミット＆Pushせよ。
```
