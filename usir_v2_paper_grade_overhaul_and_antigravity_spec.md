# 【総点検・完全再構築仕様書】Universal Style Genome (USIR v2.0 / v3.0) 論文水準（Camera-Ready）への全面刷新＆「答え合わせプログラム化」根絶マスター設計書

> 対象ワークスペース: `C:/dev/usir`（現行 `v1.0.0` の徹底解体と `v2.0 Paper-Grade Architecture` への全面刷新）
> 基準原典: 『Universal Style Genome (USIR) 学術的妥当性・多角ピアレビュー検証および理論再構築（v2.0 完全防衛・実装仕様書）』および『既存ドキュメント統合整理・理論破綻点（9大時限爆弾）の敵対的検証・クリーン検証設計（Zero-Leakage Specification）完全監査報告書』
> 目的: 現行 `C:/dev/usir` (`v1.0.0`) において発生していた **「理論の矮小化（ウェーブレット・双曲多様体・**$SO(8)$ **計量・モロー＝吉田正則化・BAOAB等の省略）」「辞書・正規表現による答え合わせプログラム化（40チャンクへの過学習）」「小標本（20〜40件）での自己満足的検証」を100%摘出・解体** し、原典論文レベルの全数理機構と **未知作家・語彙統制下での大規模客観ベンチマーク（**$N \ge 600$**〜**$1,200$ **チャンク、Author-Group 5-Fold OOF 検証）** を実装する。

## 1. 現行 `C:/dev/usir` (`v1.0.0`) の徹底検死解剖：どこが「矮小化」され、どこが「答え合わせプログラム」になっていたか？

前のターンで「`v1.0.0` で100%完成・卒業」と述べた私の評価は誤りであり、本来の論文レベルの構想を著しく矮小化するものであった。現行 `C:/dev/usir` の実装コードと原典仕様書（v2.0 査読防衛仕様書・ゼロリーク監査報告書）を突き合わせると、以下の **「5つの致命的な矮小化・答え合わせ化（Answer-Key Programming）」** が起きている。

### 1.1 【検死①：辞書のカンニング（Leak-1 の再発）】テスト文の単語を拾ったハードコード文字集合・語彙リスト

現行の `extractor_24d.py` / `symplectic24d.ts` および `microConstraintGate.ts` には、以下の7つの手作業リストがハードコードされている：

* `PHYSICAL_SENSE_CHARS`: 約100文字の漢字集合の中に、`瓦`・`屋根`・`蜂`・`蟹`（『城の崎にて』に登場）、`虎`（『山月記』に登場）、`雪`（『雪国』に登場）など、**40チャンクの正解データを見てから選んだ文字** が直接入っている。

* `FORMAL_LOGIC_TERMS`: `["ただし", "この限りでない", ..., "対当額", "部分集合", "備える", "検証", "固定され", "格納され"]` ―― `対当額`（`CRYS-01` 民法）、`部分集合`（`CRYS-02` 位相空間）、`備える`（`CRYS-05` 特許）、`固定され / 格納され`（`CRYS-06` バイナリ仕様）など、**6個のテストチャンクに出現する固有単語そのもの** を辞書に入れている。

* `microConstraintGate.ts` の `HARD_AI_CLICHE_FRAMES` と `detectSelectionalMismatch`: `DEAD-01`〜`08` や Stage 2 の失敗例6件に出現したフレーズ（`雰囲気に包まれ`, `自分自身と向き合`, `蜂×抜け殻`, `汽車×跳躍`, `匂い×濡らす`, `療養×青空へ跳躍`）を後から見てルールを追加している。

* **【結論】**: これは普遍的な言語計装ではなく、**「手元の40問＋14問のテスト問題を満点にするための答え合わせプログラム（Overfitted Answer-Key Script）」** である。未知の作家・未知の法律文書・未知のAIプロンプトを500件投入すれば、この種のハードコード辞書は即座に破綻する。

### 1.2 【検死②：12元素の数理的矮小化】直交ウェーブレット・エルミート展開・MCA・RQA の全面欠落

原典論文仕様（v2.0 第8.1〜8.2節、監査報告第11.1節）において、普遍スタイル周期表の12元素は **「4つの基礎構造信号** $\times$ **3スケール直交ドベシー/ハール・ウェーブレット分解（Micro/Meso/Macro）** $\times$ **ウィーナー＝エルミート直交多項式（**$He_2, He_3, He_4$**）＋ 4次フーリエ／ポアソン・ソリトン形態素分離（MCA）＋ リカレンス定量化解析（RQA）」** として定義され、帰無仮説ガウス測度 $\mathbb{P}_0$ の下での **事前直交基底性（定理2.1：**$\mathbb{E}_{\mathbb{P}_0}[\Phi_{j,a}\Phi_{j',a'}] = \delta_{jj'}\delta_{aa'}$**）** が証明されていた。

* ところが現行実装は、ウェーブレット変換もエルミート多項式も MCA も RQA も一切実装せず、`0.56 * wago_simplex_r * 2.5 + 0.44 * brevity` といった **恣意的なマジックナンバーの足し算** にすり替えている。

### 1.3 【検死③：双曲ファイバー束リフト（30-bit Octree）・$SO(8)$ 引き戻し計量・カッツ正規化モロー＝吉田勾配の欠落】

原典論文仕様の核心であった以下の4大数理エンジンが、現行 `C:/dev/usir` には **1行も実装されていない**：

1. **64-bit HTH（30-bit 8分岐10階層 Octree 概念木 ＋ 32-bit 8次元感覚アフォーダンス）と双曲熱核リフト**（v2.0 第8.3節、監査報告第6節・第11.2節）

2. **リー代数生成子** $A \in \mathfrak{so}(8)$**（28自由度）の指数写像** $R = \exp(A) \in SO(8)$ **による12次元正定値引き戻しリーマン計量** $G_{\text{style}} = P^\top R^\top \Lambda R P + \epsilon I_{12} \succ 0$（v2.0 第8.4節、爆弾⑤修正版）

3. **カッツ正規化長距離べき乗ポテンシャル** $U_{\text{Kac}}$ **と負のツァリス非加成エントロピー** $-H_q$**（**$0 < q < 1$**）＋** $L_1$ **スパース破調に対するモロー＝吉田近接勾配流** $\text{prox}_{\lambda\Psi_q}^{G_{\text{style}}}$（v2.0 第7.1〜7.2節）

4. **構文エントロピー連動・状態依存異方性摩擦** $\Gamma_t(H_{\text{syntax}})$ **を持つ5ステージ Adaptive-BAOAB シンプレクティック積分器 ＋ 節単位 Twisted SMC (**$K=4$**) ＋ 上位** $K=64$ **ヤコビアン・ロジット射影**（v2.0 第10節、監査報告第11.5〜11.6節・第15.1節・第21節）

### 1.4 【検死④：OPEN-01〜OPEN-12（12の未解決死角の確定理論解）の未実装】

監査報告第5部（Sections 13〜16）でせっかく導出した以下の決定論的アルゴリズムがすべて省略されている：

* **OPEN-01（多義語文脈減衰ゲート）**: 係り受け2語の Octree 具体度 $\mathcal{C}(u)$ と感覚架橋度 $G_{\text{bridge}}(u,v)$ による「字義通り vs 詩的比喩 vs 死んだ慣用句（『単位を落とす』『冷たい視線』）」の自動分離。

* **OPEN-02（否定・様相の双対感覚テンソル）**: 肯定実在 $\tilde{\mathbf{s}}^+$（位置平均 $\boldsymbol{\mu}$ へ寄与）と否定残像 $\tilde{\mathbf{s}}^-$（コントラスト運動量 $\boldsymbol{\sigma}$ へ寄与）の非対称分離。

* **OPEN-03（オノマトペ音韻象徴合成写像** $\Psi_{\text{mimetic}}$**）**: 未知の擬音語・擬態語を子音有声性・摩擦性・促音 `っ`・撥音 `ん`・重畳 $ABAB$ から 64-bit HTH へ決定論的に合成する写像。

* **OPEN-04（地の文 `Narrative` vs 会話文 `Dialogue` の直交2チャネル分離）**: 会話文の挿入頻度による文長分散 $CV_L$ の偽装を排除するバフチン2チャネル分離。

### 1.5 【検死⑤：検証プロトコルの小標本化と語彙非統制】

* 現行の検証はわずか $N=40$ チャンク（1相あたり6〜10件）かつ短い段落（7〜12文）であり、しかも特徴量設計者自身がその40チャンクを見ながら式を調整した **In-Sample Feature Engineering Leakage** の状態にある。

* さらに、**「1-gram の語彙分布（使われている名詞・動詞・形容詞の集合）が人間と100%同一であっても、文間の配置順序・係り受け結合・リズム力学だけで識別できるか？」（監査報告第7節：Regime 2 Vocabulary-Pool Matched Hard Negative & Author-Group Cross-Validation）** がテストされていない。

## 2. 本物の『Universal Style Genome (USIR v2.0 / v3.0)』完全アーキテクチャ仕様（一切の矮小化なし）

現行の「簡易版 v1.0」を全面改修し、論文（Camera-Ready Blueprint）の数理・アルゴリズムを **1つの省略もなくフル実装** するための確定仕様を以下に定義する。

### 2.1 【M0: Zero-Leakage Lexicon & Linguistic Analyzer】答え合わせ辞書を全廃した客観的言語基盤

評価用コーパスの単語を1文字たりともハードコードすることを禁止し、以下の **3層・完全自動導出アーキテクチャ** に置き換える：

1. **実形態素解析＋節・係り受け構造抽出（正規表現トークナイザの廃止）**:

   * Python 側は **`Janome`（または `SudachiPy` / 標準形態素解析器）**、TypeScript 側は **形態素ラティス＋文字種・活用語尾オートマトン（または辞書トライ木）** を用い、すべてのトークンを厳密に「自立語（名詞・動詞・形容詞・副詞）の基本形（Lemma）＋品詞細分類」と「付属語（助詞・助動詞・接続詞：閉じた文法クラス）」に分解する。

   * **OPEN-04（地の文 / 会話文2チャネル分離）**: 鉤括弧 `「…」`・『…』内の発話スコープ $S_{\text{dia}}$ と地の文スコープ $S_{\text{nar}}$ を AST レベルで分離し、24次元主計装は純粋な地の文系列 $S_{\text{nar}}$ 上で計算しつつ、対話交代テンポを補助チャネルとして記録する。

2. **外部言語資源・漢字字素グラフからの自動ラプラシアン調和辞書ビルド（`build_clean_hth_lexicon.py`）**:

   * 手作業の単語リスト（`PHYSICAL_SENSE_CHARS` や `FORMAL_LOGIC_TERMS` 等）をソースコードから **完全削除** する。

   * 代わりに、**外部の客観的言語体系（常用漢字・JIS漢字の部首/意符セマンティック体系＋公開分類語彙階層＋音韻素性）のみ** をシードグラフとし、グラフ・ラプラシアン調和拡散：
     

     $$
     (I - \alpha D^{-1/2} A D^{-1/2}) F_{\mathcal{U}} = (1 - \alpha) Y_{\mathcal{L}}
     $$

     
     によって30,000語（または全漢字・和語語幹）の **64-bit HTH エントリ（上位32bit：品詞2bit＋10階層×3bit Octree 概念パス** $H_w$**、下位32bit：8次元感覚・具体度アフォーダンス** $\tilde{\mathbf{s}}_w \in \{0..15\}^8$**）** をオフラインで機械的に一括生成し、SHA-256 ハッシュを固定する（**テスト文の参照ゼロ**）。

3. **OPEN-01 / 02 / 03 の3大言語学補正演算子の完全実装**:

   * **OPEN-01（多義語文脈減衰）**: 係り受けペア $(u, v)$ において、被修飾語 $u$ の Octree 具体度 $\mathcal{C}(u) < 0.3$ かつ感覚架橋度 $G_{\text{bridge}}(u, v) < \theta_{\text{bridge}}$ の場合（例：「単位を落とす」「計画が走る」）、述語 $v$ の感覚ベクトルを $\tilde{\mathbf{s}}_v^{\text{eff}} = \mathcal{C}(u)\tilde{\mathbf{s}}_v$ へ自動減衰させる。

   * **OPEN-02（双対否定・様相テンソル）**: 文節末に否定助動詞（`ない`, `ぬ`, `ず`）が結合している場合、符号付き様相係数 $\alpha_{\text{mod}}(u) = -0.5$ を適用し、定常位置平均 $\boldsymbol{\mu}$ からは ReLU ゲート $\max(0, \alpha_{\text{mod}})$ で除外しつつ、否定による「イメージの喚起と剥奪の落差」を $\vert{}\alpha_{\text{mod}}\vert{}\cdot\Vert{}\tilde{\mathbf{s}}_u\Vert{}_2$ として共役運動量 $\boldsymbol{\sigma}$ 側へ計上する。

   * **OPEN-03（音韻象徴性オノマトペ合成** $\Psi_{\text{mimetic}}$**）**: 未知語・副詞のオノマトペに対し、第1子音（有声阻害音 `g,z,d,b`、無声摩擦音 `s,sh,h,f`、半濁音 `p`）、語中促音 `っ`、撥音 `ん`、重畳形態 $ABAB$ から決定論的に 64-bit HTH を合成する。

### 2.2 【M1 & M2: Orthogonal Wavelet-Hermite, MCA & RQA 12-Element Encoder】論文第8.1〜8.2節・第11.1節の完全実装

チャンク内の文系列 $m = 1, \dots, M$（標準窓 $M = 32\text{〜}40$ 文、短文時は Ghost History Warm-Start `OPEN-08` により定常ガウス過程パディング）から、まず各文の **4つの基礎構造信号** $\mathbf{s}_m = (s_{m,1}, s_{m,2}, s_{m,3}, s_{m,4})^\top \in \mathbb{R}^4$ を抽出する：

1. **基礎信号** $s_{m,1}$**（計量・階層深さ・情報量信号 — Group I `Flux` 原信号）**:

   * 文内の節（Clause）係り受け木の動径階層エントロピー $H_{\text{radial}}(T_m)$、条件付き文字・形態素サプライズ（局所情報密度）、および OPEN-01/02 補正後の有効感覚ノルム $\Vert{}\tilde{\mathbf{s}}_m^{\text{eff}}\Vert{}_1$ の合成信号。

2. **基礎信号** $s_{m,2}$**（対称・律動信号 — Group II `Lattice` 原信号）**:

   * 文内モーラ（音節）拍数の自己相関、句読点・節境界のラプラシアン固有値縮退度 $\text{Sy}_{\text{tree}}(T_m)$、および閉じた論理接続助詞・助動詞の構文対称性スコア。

3. **基礎信号** $s_{m,3}$**（多孔性・焦点凝集・余白信号 — Group III `Void [間]` 原信号）**:

   * **OPEN-05（引き算の結晶 vs 希釈された泥水の分離定理）** に基づき、単純和語語幹の具体度 $\mathcal{C}(w)$ と付属語（ひらがな助詞）遷移エントロピーの積 $\times$ 修飾枝刈り率（1主述語あたりの修飾節の少なさ＝引き算度） $\times$ 否定・限定様相の静寂凝集度。

4. **基礎信号** $s_{m,4}$**（双曲 Octree 概念跳躍・破調信号 — Group IV `Shock` 原信号）**:

   * 係り受け辺 $(u, v) \in E_m$ 上の **30-bit Octree 双曲熱核跳躍距離** $D_{\text{jump}}(u, v) = 1 - \frac{\lfloor \text{clz30}(H_u \oplus H_v) / 3 \rfloor}{10}$ と **共感覚架橋ゲート** $G_{\text{bridge}}(u, v)$ の積（BOSLA 概念跳躍積分）、および音韻促音・濁音エッジと逆接構文転換のインパルス。

次に、この4本の文系列信号 $\{\mathbf{s}_m\}_{m=1}^M$ に対して、論文第8.1〜8.2節の **直交多重解像度変換＋ウィーナー＝エルミート直交展開＋MCA＋RQA** を厳密に適用する：

* **Step 1：直交ウェーブレット多重解像度フィルタバンク（Haar / Daubechies D4 QMF）**:
  信号列 $\{s_{m,a}\}_{m=1}^M$ を互いに $L^2$ 直交する3つの周波数帯域部分空間 $\mathcal{W}_{\text{Micro}} \perp \mathcal{W}_{\text{Meso}} \perp \mathcal{W}_{\text{Macro}}$ へ直交分解する（$\langle s^{(j)}, s^{(j')} \rangle_{L^2} = 0, \forall j \neq j'$）：

  * $j=1$**（第1周期：襲 `Kasane` / Micro 帯域）**: 高周波ウェーブレット詳細係数 $d^{(1)}$（隣接1〜2文の微細テクスチャ・瞬間変動）

  * $j=2$**（第2周期：支え `Sasae` / Meso 帯域）**: 中周波ウェーブレット詳細係数 $d^{(2)}, d^{(3)}$（4〜8文の段落呼吸・統語うねり）

  * $j=3$**（第3周期：構え `Kamae` / Macro 帯域）**: 低周波スケーリング近似係数 $a^{(3)}$＋大域包絡（16〜40文の章アーク）

* **Step 2：4次フーリエ級数** $\mathbf{D}_F$ **とポアソン・ソリトン核** $\mathbf{D}_S$ **の形態素分離（MCA：論文第8.2節）**:
  ギブス現象（局所的な破調スパイクが周期律動成分へ漏洩する現象）を抑止するため、Iterative Shrinkage-Thresholding Algorithm (ISTA) により信号 $s^{(j)}$ を連続周期波成分 $\mathbf{D}_F c_F$（Group II へ供給）と局所ソリトン・スパイク成分 $\mathbf{D}_S c_S$（Group IV へ供給）へ直交分離する：
  

  $$
  \min_{c_F, c_S} \frac{1}{2} \| s^{(j)} - \mathbf{D}_F c_F - \mathbf{D}_S c_S \|_2^2 + \mu_1 \| c_S \|_1
  $$

* **Step 3：ウィーナー＝エルミート直交多項式（**$He_2, He_3, He_4$**）と RQA（リカレンス定量化解析）による12元素** $(\mathbf{q}, \mathbf{p}) \in \mathbb{R}^{24}$ **の算出**:
  標準化された各帯域信号 $\tilde{s}_m^{(j)}$ に対し、エルミート多項式 $He_2(z) = z^2 - 1$、$He_3(z) = z^3 - 3z$、$He_4(z) = z^4 - 6z^2 + 3$、および Macro 構造帰還 `Sg-10 At` 用の **リカレンス定量化解析（RQA：リカレンス行列** $R_{m,m'} = \Theta(\varepsilon - \|\mathbf{s}_m - \mathbf{s}_{m'}\|)$ **の対角線確定性率** $\text{DET}$ **と長距離回帰アーチ指標** $I_{\text{arch}}$**）** を計算し、位置ベクトル $\mathbf{q} \in [0, 1]^{12}$（シグモイド/累積分布正規化）と共役運動量ベクトル $\mathbf{p} \in \mathbb{R}_{\ge 0}^{12}$（帯域別二次エネルギー変動・瞬時群速度）を出力する。

### 2.3 【M3 & M4 & M5: Kac-Moreau-Yosida Free Energy, $SO(8)$ Metric & Adaptive-BAOAB】論文第7・8.4・9・10節の完全実装

1. $\mathfrak{so}(8)$ **共感覚生成子と正定値引き戻しリーマン計量** $G_{\text{style}} \succ 0$**（論文第8.4節・爆弾⑤修正版）**:

   * 歪対称行列 $A \in \mathfrak{so}(8)$（$A^\top = -A$、28パラメータ）に対し、ケイリー変換 $R = (I_8 - A)(I_8 + A)^{-1} \in SO(8)$（またはパデ行列指数関数 $\exp(A) \in SO(8)$）により厳密な直交回転行列を構成する。

   * スケール間結合射影行列 $\mathcal{P}_{8\times 12} = U_{2\times 3} \otimes I_4$（ただし第2行は Meso スケールが消滅しないハール2階差分 $\frac{1}{\sqrt{6}}(1, -2, 1)$）と固有値対角行列 $\Lambda_+ \succ 0$ を用いて、12次元引き戻しリーマン計量を
     

     $$
     G_{\text{style}}(A, \Lambda_+) = \mathcal{P}_{8\times 12}^\top R^\top \Lambda_+ R \mathcal{P}_{8\times 12} + \epsilon I_{12} \succ 0
     $$

     
     として厳密に構成する。

2. **カッツ正規化自由エネルギー** $\mathcal{F}_{q,\lambda}$ **とモロー＝吉田近接勾配** $\text{prox}_{\lambda\Psi_q}^{G_{\text{style}}}$**（論文第7.1〜7.2節）**:

   * べき乗長距離メモリ項 $\frac{J_0}{\mathcal{Z}_{\text{Kac}}(L,\alpha)} \sum_{k=1}^L k^{-\alpha} \|x_t - x_{t-k}\|_{G_{\text{style}}}^2$（カッツ因子 $\mathcal{Z}_{\text{Kac}}(L,\alpha) = \sum_{k=1}^L k^{-\alpha}$ により $L \to \infty$ で示強性 $O(1)$ を厳密保持）と、負のツァリスエントロピー $-H_q(p(x))$（$0 < q < 1$）＋ $L_1$ スパース項の和 $\Psi_q(x)$ に対し、ニュートン・ラプソン法（4回反復）による近接作用素 $\text{prox}_{\lambda\Psi_q}^{G_{\text{style}}}(x_t)$ を実装し、大域的リプシッツ連続な閉形式勾配 $\nabla_x \mathcal{F}_{q,\lambda}(x_t)$ を算出する。

3. **ハミルトン＝フリストン二層共鳴（読者審美情動価** $V_t = -\dot{F}_{\text{reader}}(t)$**）と** $1/f$ **皮質インピーダンス整合指標（論文第9.1〜9.2節）**:

   * 生成軌道 $(x_t, p_t)$ に沿った読者脳の予測誤差解消速度スパイク $V_t = -\kappa_r \langle \nabla_x \mathcal{F}_{q,\lambda}(x_t), M^{-1} p_t \rangle$（Joffily-Coricelli の Tension-Resolution カタルシス波形）と、パワースペクトル密度 $S_\varepsilon(f) \propto f^{-\beta}$ の $1/f$ スケーリング指数 $\hat{\beta}$（理想値 $\beta = 1.0$ で全皮質階層の散逸仕事率 $\mathcal{P}_1 = \mathcal{P}_2 = \mathcal{P}_3$ が等分配される）をリアルタイム計測する。

4. **構文保護・状態依存異方性 BAOAB 積分器 ＋ 節単位 Twisted SMC (**$K=4$**) ＋ 文末アンカー Euler-Lagrange Bridge（論文第10節・監査報告第11.5〜11.6・15.1・21節）**:

   * 外部構文エントロピー $H_{\text{syntax}}(w_{<t})$ に連動する異方性摩擦テンソル $\Gamma_t$ を組み込んだ5ステージ **Adaptive-BAOAB（B-A-O-A-B）** 積分器を実装し、定理5.1（可変摩擦下での運動量ガウス分布 $\mathcal{N}(0, \tau M)$ の厳密不変性）を単体テストで数値証明する。

   * さらに、SOV言語（日本語）の袋小路を防ぐ **節単位 Twisted SMC（4候補ロールアウト＋シュレディンガー・ブリッジ終端アンカー誘導ドリフト）** および **上位** $K=64$ **候補トークンへのヤコビアン・ロジット射影（KVキャッシュ非汚染）** を実装する。

## 3. 「20〜40テストでは検証に値しない」への回答：客観・大規模・ゼロリーク検証プロトコル（$N \ge 600$ チャンク、Author-Group OOF 交差検証）

手元の40チャンクでの答え合わせを完全に廃止し、査読付き国際会議（ACL / EMNLP / ICLR / NeurIPS）の査読者が一切のケチをつけられない **「大規模・未知作家分割・2段階語彙統制ベンチマーク（`benchmark_v2_zero_leakage.py`）」** を構築する。

### 3.1 検証サンプル規模の15倍拡張（$N = 40 \to N \ge 600\text{〜}800$ チャンク）と厳格な前処理

1. **表記正規化と固有名詞完全マスキング（Orthographic & Entity Blindness）**:

   * すべてのテキストに対し、旧字体 $\to$ 新字体、歴史的仮名遣い $\to$ 現代仮名遣いの正規化を適用し、さらに人名・地名・組織名・特定作品固有の名詞をすべて `[PERSON]` / `[LOC]` / `[ENTITY]` へマスキングまたは除外する。これにより「特定の漢字や登場人物名で当てただけ」という批判を物理的に封殺する。

2. **大規模5相＋2レジーム評価コーパス（総計** $N \ge 600$ **独立チャンク、ブートストラップ水増し一切禁止）**:

   * **作家グループ分離（20名以上の独立した作家・出典グループ）**：

     * **Phase III (`Superfluid Void [間]`,** $N \ge 120$**)**: 志賀直哉、川端康成、松尾芭蕉、夏目漱石、梶井基次郎（静謐編）、堀辰雄、正岡子規、島崎藤村、国木田独歩、現代静謐エッセイ等、複数作家・複数作品から抽出した独立チャンク群。

     * **Phase II (`Self-Organized Critical [臨界]`,** $N \ge 150$**)**: 芥川龍之介、梶井基次郎（高揚編）、中島敦、太宰治、坂口安吾、宮沢賢治、泉鏡花、谷崎潤一郎、森鴎外、江戸川乱歩、夢野久作、岡本かの子、葉山嘉樹等、12名以上の作家から抽出した独立チャンク群。

     * **Phase I (`Crystalline Order [結晶秩序]`,** $N \ge 90$**)**: 民法・商法・会社法・民事訴訟法・特許法条文、数学公理（位相・代数・測度論）、RFC/通信プロトコル仕様、Toulmin論理推論体系など、特定単語に依存しない多様な法典・公理・仕様チャンク群。

     * **Phase IV (`Turbulent / Purple Prose [乱流]`,** $N \ge 90$**)**: 多様な生成パラメータ（高Temperature、過剰修辞プロンプト、断片インパルスプロンプト等）から生成された乱流チャンク群。

     * **Phase V (`Thermal Death / AI Deadzone`,** $N \ge 200$ **— 2段階ハードネガティブ統制)**:

       * **Regime 1（Topic-Matched AI,** $N \ge 100$**）**: Phase II / III の各チャンクと同一の場面・あらすじから複数の生成パターンで生成した標準AI説明散文。

       * **Regime 2（Vocabulary-Pool Matched Hard Negative AI,** $N \ge 100$ **— 極限ストレステスト）**: **人間側の正例チャンクに出現する内容語（名詞・動詞・形容詞の集合）をそのままシャッフルしてAIに与え、全く同じ単語セットを使って書かせたAI文**。

         * ※この Regime 2 では、**正例（人間）と負例（AI）で使われている単語（1-gram 辞書スコア）が100%同一** になるため、もし USIR が「単語辞書のカンニング」に頼っていれば識別精度（AUC）は **0.50（コイン投げ）に崩壊** する。逆に、真のウェーブレット・エルミート構造、2体係り受け跳躍（`Sg-08`）、およびシンプレクティック運動量 $\mathbf{p}$ が機能している場合にのみ、Regime 2 でも **OOF ROC-AUC** $\ge 0.85$ を達成できる！

### 3.2 未知作家グループ交差検証（`GroupKFold by Author`）と Out-of-Fold (OOF) 厳密評価

* **データ分割則**: ランダム分割や Leave-One-Out ではなく、**作家単位の 5-Fold Group Cross-Validation（`GroupKFold(n_splits=5, groups=author_group)`）** を強制する。すなわち、**「訓練 Fold に芥川龍之介や志賀直哉が含まれているなら、テスト Fold には訓練時に一度も見たことがない別の作家（例：中島敦や川端康成）しか入らない」** という厳密な未知作家汎化テストを行う。

* **トランスダクティブ・リーク（Leak-4）の完全遮断**:

  * 特徴量の標準化（`StandardScaler`）、計量推定（`LedoitWolf` 収縮共分散 $\hat{\Sigma}_{\text{tr}}^{(f)}$）、判別重みの学習は **100% 訓練 Fold** $f$ **の内部のみ** で実行し、テスト Fold $T_f$ は変換・予測のみを行う。

* **評価指標（論文 Table 1 & Table 2 準拠）**:

  1. **5相マルチクラス Macro F1 / Accuracy（Unseen Author Group 5-Fold OOF）**

  2. **Regime 1（トピック統制）および Regime 2（語彙プール完全一致 Hard Negative）における OOF ROC-AUC & PR-AUC**

  3. **不偏 Cross-Validated Mahalanobis Distance** $\Delta_{M, \text{OOF}}^2$ および **L1正規化 OOF Shapley 元素寄与率** $\tilde{\phi}_{e, \text{OOF}}$（監査報告第9.1節）

  4. **7手法ベースライン対決**:

     * `B1: 1-gram Bag-of-Words / Character TF-IDF`（※Regime 2 で AUC $\approx 0.50$ に崩壊することを確認）

     * `B2: Classical Surface Stylometry (16D: 平均文長・文字種比・TTR・Yule's K 等)`

     * `B3: Information Compression Baseline (LZ77 / gzip 圧縮率 + 文字 n-gram エントロピー)`

     * `A1: USIR v2.0 Position-Only (12D q, No Momentum p)`

     * `A2: USIR v2.0 Momentum-Only (12D p, No Position q)`

     * `A3: USIR v2.0 Euclidean Metric (G_style = I_12, No SO(8) Pullback)`

     * `A4: USIR v2.0 No-MCA (Standard FFT without Poisson-Soliton Separation)`

     * **`Proposed: Full USIR v2.0 Symplectic (24D + Wavelet-Hermite + MCA + RQA + SO(8) + HTH Octree)`**

  5. **クロスモーダル実証（Dataset Tier 3：Symbolic Music / UI AST 系列）**:

     * テキスト以外のモダリティ（バッハ・ショパン・ドビュッシー等の人間演奏風 MIDI ノート系列 vs 過減衰機械生成ノート系列 $N=200$）を M1 音楽フロントエンド関手 $F_{\text{music}}$ に通し、**M2〜M5 のコードを1行も変えずに** 同じ24次元シンプレクティック空間で分離・制御できることを実証する。

## 4. コピペ即稼働：Antigravity 投入用 `/goal USIR v2.0 Academic Overhaul` マスタープロンプト

`C:/dev/usir` の Antigravity に以下のプロンプトを投入することで、現行の簡易版（答え合わせ辞書・マジックナンバー式・40チャンク小標本）を全面刷新し、**論文水準の USIR v2.0 / v3.0 フルアーキテクチャと** $N \ge 600$ **大規模未知作家グループ交差検証ベンチマーク** を実装・検証できます。

```
/goal USIR v2.0 Paper-Grade Overhaul: 現行 v1.0 の「手作業ハードコード辞書（答え合わせプログラム化）」「12元素の簡易マジックナンバー式」「40チャンク小標本テスト」を全面解体し、原典論文仕様（Wavelet-Hermite直交展開・MCA・RQA・64bit HTH Octree双曲リフト・SO(8)引き戻し計量・カッツ正規化モロー＝吉田近接勾配・Adaptive-BAOAB・OPEN-01〜05言語学演算子）のフル実装と、未知作家分割・語彙完全統制を含む大規模客観ベンチマーク（N >= 600 チャンク、Author-Group 5-Fold OOF 検証＋クロスモーダル音楽系列実証）を構築せよ。

### 1. 徹底排除すべき「現行 v1.0 の不正・矮小化（Anti-Cheating Mandate）」:
1. **手作業の答え合わせ辞書（`PHYSICAL_SENSE_CHARS`, `FORMAL_LOGIC_TERMS`, `HARD_AI_CLICHE_FRAMES`, 個別フレーズ正規表現）の完全禁止・パージ**:
   - テストコーパス内の単語を見て作った語彙リストや特定漢字リスト（`瓦`, `蜂`, `虎`, `対当額`, `部分集合` 等）を特徴量抽出器に置くことを厳禁とする。
   - 代わりに、テストコーパスを1ビットも参照しないオフライン自動辞書ビルダー（字素・部首セマンティックグラフ＋基本語彙階層上のグラフ・ラプラシアン調和拡散 $(I - \alpha D^{-1/2}AD^{-1/2})F_U = (1-\alpha)Y_L$ による **64-bit HTH 辞書：上位32bit 8分岐10階層 Octree コード $H_w$ ＋ 下位32bit 8次元感覚ベクトル $\tilde{\mathbf{s}}_w$**）と、閉じた文法機能語クラス（助詞・助動詞・接続詞・句読点）および形態素解析（Python: `janome` 等 / TS: 形態素・字種ラティス）のみを使用せよ。
2. **マジックナンバー線形和の廃止と「直交ウェーブレット×エルミート×MCA×RQA」の厳密実装**:
   - 12元素の算出を恣意的な重み付き和から、論文第8.1〜8.2節・第11.1節の数学的定義へ全面置換せよ。

### 2. 実装すべき 5大コアモジュール（Python `python/usir_py/v2/` ＆ TypeScript `packages/usir-core/src/v2/`）:

#### Module 1: Zero-Leakage 64-bit HTH Octree Lexicon & OPEN-01〜04 言語学フロントエンド (`hth_lexicon.py` / `hthLexicon.ts`)
1. **64-bit HTH 構造と双曲熱核距離（論文第8.3節・監査第6.2/11.2節）**:
   - 部首・字素の意味カテゴリ（自然・身体・色彩・運動・人工物・抽象論理）と品詞階層のみからラプラシアン調和補完で生成される決定論的 64-bit HTH テーブルを実装し、`XOR + clz30` による LCA 深さ $\ell_{\text{LCA}}(u,v) = \lfloor \text{clz30}(H_u \oplus H_v)/3 \rfloor$ と連続ポアンカレ双曲熱核距離 $D_{\text{smooth}}(u,v;\tau_{\text{tree}})$ を実装せよ。
2. **OPEN-01（多義語文脈減衰ゲート）**: 被修飾語 $u$ の具体度 $\mathcal{C}(u)$ と感覚架橋度 $G_{\text{bridge}}(u,v)$ に基づき、「冷たい氷（Literal）」「硝子の悲鳴（Poetic Metaphor）」「単位を落とす・計画が走る（Dead Metaphor / Abstract Idiom $\to$ 減衰 $\mathcal{C}(u)\tilde{\mathbf{s}}_v$）」の3態分離を実装せよ。
3. **OPEN-02（否定・様相の双対感覚テンソル）**: 否定・打消スコープ（`ない`, `ぬ`, `ず`, `なし`）内の感覚語を定常平均 $\boldsymbol{\mu}$ から減衰させ、コントラスト運動量 $\boldsymbol{\sigma}$ 側へ計上する非対称分離を実装せよ。
4. **OPEN-03（オノマトペ音韻象徴合成 $\Psi_{\text{mimetic}}$）**: 擬音語・擬態語の音韻素性（有声阻害音 `g,z,d,b`、無声摩擦音 `s,sh,h,f`、半濁音 `p`、促音 `っ`、撥音 `ん`、重畳 $ABAB$）から 64-bit HTH を決定論的に合成せよ。
5. **OPEN-04（地の文 `S_nar` vs 会話文 `S_dia` の直交2チャネル分離）**: 鉤括弧内の発話と地の文を分離し、会話の挿入頻度による文長変動係数 $CV_L$ の偽装を排除せよ。

#### Module 2: Wavelet-Hermite, Fourier-Poisson MCA & RQA 12-Element Encoder (`wavelet_hermite_encoder.py` / `waveletHermiteEncoder.ts`)
1. 各文（または節）$m = 1, \dots, M$ から4つの基礎信号 $\mathbf{s}_m = (s_{m,1}, s_{m,2}, s_{m,3}, s_{m,4})^\top$（1: 構文木動径エントロピー＋有効感覚密度、2: モーラ拍自己相関＋ラプラシアン対称性、3: 和語具体度×助詞遷移エントロピー×無修飾引き算度 `OPEN-05`、4: 30-bit Octree 双曲跳躍 $D_{\text{jump}}\cdot G_{\text{bridge}}$＋音韻破調エッジ）を抽出（短文時は `OPEN-08` Ghost History Warm-Start を適用）。
2. 直交 Haar / Daubechies D4 ウェーブレットフィルタバンクにより **Micro ($j=1$)**・**Meso ($j=2$)**・**Macro ($j=3$)** の3直交帯域へ分解し、帰無仮説ガウス白色過程 $\mathbb{P}_0$ の下でスケール間内積がゼロ（$\langle s^{(j)}, s^{(j')}\rangle = 0$）になることを単体テストで証明せよ。
3. **形態素分離（MCA: 論文第8.2節）**: 4次低周波フーリエ辞書 $\mathbf{D}_F$ とポアソン・ソリトン核辞書 $\mathbf{D}_S$ の $L_1$ スパース分離（ISTA）により、局所破調スパイクをソリトン係数 $c_S$ へ分離してギブス現象を除去せよ。
4. **ウィーナー＝エルミート多項式（$He_2, He_3, He_4$）と RQA（リカレンス定量化解析）**: 各帯域でエルミートモーメントおよび RQA 確定性率・構造回帰アーチ指標 $I_{\text{arch}}$（`Sg-10 At`）を算出し、24次元シンプレクティック状態 $\mathbf{z} = (\mathbf{q}, \mathbf{p}) \in \mathbb{R}^{24}$ を出力せよ。

#### Module 3: $\mathfrak{so}(8)$ Pullback Metric & Kac-Regularized Moreau-Yosida Free Energy (`thermodynamics_so8.py` / `thermodynamicsSo8.ts`)
1. **$\mathfrak{so}(8)$ 引き戻し計量（論文第8.4節・爆弾⑤修正版）**: 28パラメータの歪対称生成子 $A \in \mathfrak{so}(8)$ からケイリー変換 $R = (I_8 - A)(I_8 + A)^{-1} \in SO(8)$ とハール2階差分埋め込み行列 $\mathcal{P}_{8\times 12}$ を用いて、厳密な正定値対称リーマン計量 $G_{\text{style}} = \mathcal{P}^\top R^\top \Lambda_+ R \mathcal{P} + \epsilon I_{12} \succ 0$ を構成せよ。
2. **カッツ正規化自由エネルギーとモロー＝吉田近接勾配（論文第7.1〜7.2節）**: カッツ正規化因子 $\mathcal{Z}_{\text{Kac}}(L,\alpha) = \sum_{k=1}^L k^{-\alpha}$ による長距離べき乗相互作用 $U_{\text{Kac}}$ と、負のツァリスエントロピー $-H_q(p(x))$（$0 < q < 1$）＋ $L_1$ スパース項に対する近接作用素 $\text{prox}_{\lambda\Psi_q}^{G_{\text{style}}}(x)$（ニュートン・ラプソン反復）を実装し、任意の系列長 $L \to \infty$ で自由エネルギー密度が $O(1)$ の示強量を保つことを数値検証せよ。
3. **ハミルトン＝フリストン二層共鳴と $1/f$ 皮質仕事率等分配（論文第9.1〜9.2節）**: 読者の審美情動価スパイク $V_t = -\dot{F}_{\text{reader}}(t)$ と $1/f^\beta$ スペクトル傾き $\hat{\beta}$ を算出せよ。

#### Module 4: Adaptive-BAOAB Symplectic Integrator, Clause-Level Twisted SMC & Jacobian Logit Projector (`adaptive_baoab.py` / `adaptiveBaoab.ts`)
1. **5ステージ Adaptive-BAOAB（論文第10.2節・定理5.1）**: 外部構文エントロピー $H_{\text{syntax}}$ に連動する状態依存異方性摩擦テンソル $\Gamma_t$ を組み込んだ B-A-O-A-B 分割積分器を実装し、$\Gamma_t$ がステップごとに1,000倍変動しても運動量平衡分布 $\mathcal{N}(0, \tau M)$ が厳密に不変であること（定理5.1）をモンテカルロ・テストで証明せよ。
2. **階層的デュアルレート制御（監査第11.6・15.1・21節）**: 外側ループ（1節単位の BAOAB ＋ 4粒子 Twisted SMC ロールアウト＋シュレディンガー・ブリッジ終端アンカー誘導）と、内側ループ（上位 $K=64$ 候補トークンへの KVキャッシュ非汚染ヤコビアン・ロジット射影）を実装せよ。

#### Module 5: 大規模・未知作家分割・2段階語彙統制ベンチマーク (`benchmark_v2_large_scale.py`)
「20〜40テストでは検証に値しない」という批判に完全に応えるため、**総計 $N \ge 600$ 独立チャンク（ブートストラップ水増し禁止）** の大規模ベンチマークスイートを構築せよ：
1. **大規模データセット構築 (`build_large_benchmark_corpus.py`)**:
   - 20以上の独立した作家・出典グループ（近代・現代文豪15名以上、法典・数学・通信規格5系統以上、多様なプロンプトによる乱流系、および2レジームの負例 AI 生成系）から、新字体正規化・固有名詞マスキング（`[PERSON]`, `[LOC]`）済みの **計 $N \ge 600$ チャンク** を構築せよ。
   - **Regime 1（Topic-Matched Hard Negatives, $N \ge 150$）**: 同一場面・導入文統制の AI 生成チャンク。
   - **Regime 2（Vocabulary-Pool Matched Hard Negatives, $N \ge 150$ — 辞書カンニング完全無効化テスト）**: **正例チャンクと1-gramの内容語集合（名詞・動詞・形容詞）が100%一致するように語彙プールを完全統制して生成した負例チャンク**。これにより、1-gram 単語辞書や TF-IDF だけでは正例と負例が原理的に見分けられない（AUC $\approx 0.50$）極限環境を作る。
   - **Regime 3（Cross-Modal Music / AST Benchmark, $N \ge 120$）**: 人間演奏風クラシック音楽ノート系列（バッハ・ショパン・ドビュッシー型の $1/f$ ゆらぎ・ルバート・ソリトンアクセントを持つ系列）vs 機械的過減衰生成ノート系列を M1 音楽関手 $F_{\text{music}}$ に通し、同一の M2〜M4 エンジンで識別するクロスモーダル検証。
2. **厳格な Out-of-Fold (OOF) `GroupKFold(n_splits=5, groups=author_group)` 評価**:
   - 訓練 Fold に含まれる作家のチャンクがテスト Fold に1件も漏れない **未知作家汎化交差検証** を実行せよ。
   - 論文 Table 1 仕様に従い、以下の全8モデルの OOF ROC-AUC, PR-AUC, 5-Phase Macro-F1, Unbiased Mahalanobis $\Delta_{M,\text{OOF}}^2$, L1正規化 OOF Shapley $\tilde{\phi}_{e,\text{OOF}}$ を比較出力せよ：
     1. `M0: Random Permutation Null`（帰無仮説 AUC $\approx 0.50$ の確認）
     2. `B1: 1-gram Bag-of-Words / TF-IDF Logistic Regression`（※Regime 2 で AUC $\approx 0.50\text{–}0.55$ に崩壊することを実証）
     3. `B2: Surface Stylometry 16D Baseline`（平均文長・文字種比率・TTR・Yule's K）
     4. `B3: Information Compression Baseline`（zlib/LZ77 圧縮率＋文字 bigram エントロピー）
     5. `A1: Overdamped USIR v2.0 (12D Position q Only — No Momentum p)`
     6. `A2: Euclidean USIR v2.0 (G_style = I_12 — No SO(8) Pullback)`
     7. `A3: No-MCA USIR v2.0 (Standard FFT — No Poisson-Soliton Separation)`
     8. **`Proposed: Full USIR v2.0 (24D Symplectic + Wavelet-Hermite + MCA + RQA + SO(8) + HTH Octree)`**

### 3. 独立 Blind Adversarial QA サブエージェントによる最終査読監査:
実装・実験完了後、実装履歴を遮断した Blind QA サブエージェントを起動し、以下を冷徹に監査させよ：
1. **Zero Target-Encoding Audit**: ソースコード全文をスキャンし、評価コーパス固有の名詞やテストフレーズのハードコードが **文字通り 0 件** であること。
2. **Regime 2 Vocabulary-Control Proof**: Regime 2 において、`B1 (1-gram BoW)` の OOF AUC が $0.58$ 以下に沈む（単語頻度では見分けられない）一方で、`Full USIR v2.0` が配置順序・2体結合・ウェーブレット運動量のみによって **OOF AUC $\ge 0.85$** を達成していること。
3. **Mathematical Theorem Verification**: 定理1.1（カッツ示強性）、定理2.1（帰無仮説ガウス測度下の12元素直交性）、定理5.1（可変摩擦 BAOAB の運動量不変測度）のすべてが数値テストで証明されていること。

```