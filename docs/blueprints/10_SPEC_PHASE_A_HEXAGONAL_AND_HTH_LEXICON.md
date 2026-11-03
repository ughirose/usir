# Phase A 詳細技術仕様書：128-bit 多義辞書ビルド、ヘキサゴナル構文インターフェース、OPEN-01〜05、および多重解像度計装コア

> **対象ファイル**:
> - `python/scripts/compile_hth_clean_poly.py`（新規: Step 0 辞書ビルドスクリプト）
> - `python/usir_py/data/hth_poly_30k.bin`（新規: 480,000 Bytes 固定バイナリ）
> - `python/usir_py/syntax_frame.py` / `packages/usir-core/src/schema/syntaxFrame.ts`（新規: ヘキサゴナル純粋データ構造）
> - `python/usir_py/hth_lexicon.py` / `packages/usir-core/src/sensor/hthLexicon.ts`（新規: 480KB バイナリ O(1) 索引）
> - `python/usir_py/linguistic_gates.py`（新規: OPEN-01〜04 本実装）
> - `python/usir_py/extractor_24d.py` / `packages/usir-core/src/sensor/symplectic24d.ts`
> - `python/usir_py/manifold_geodesic.py` / `packages/usir-core/src/dynamics/manifoldGeodesic.ts`

---

## 1. 【Step 0】128-bit 多義対応辞書 `hth_poly_30k.bin`（厳密に 480,000 Bytes）ビルド仕様

原典第6節および第24.2節に基づき、1単語あたり 16 Bytes（128 bits）を割り当てた多義対応バイナリ辞書を構築する。

### 1.1 エントリ構造（1語 = 16 Bytes / uint64 × 2）
```text
[Word Entry: 128 bits (16 Bytes, Little-Endian)]
├── High 64 bits: Primary Physical Sense (第一語義：字義通りの身体・物理感覚)
│   ├── [bit 63..34 (30b)]: Octree Node ID 1 (物理空間クラスタ階層 10階層×3bit)
│   ├── [bit 33..32 (2b)] : POS Tag (0b00:名詞, 0b01:動詞, 0b10:形容詞, 0b11:副詞・その他)
│   └── [bit 31..00 (32b)]: 8次元感覚アフォーダンス q_1..q_8 (各4bit: 0x0..0xF)
└── Low 64 bits: Secondary Abstract / Figurative Sense (第二語義：抽象転用・制度・心理)
    ├── [bit 63..34 (30b)]: Octree Node ID 2 (抽象概念クラスタ階層 10階層×3bit)
    ├── [bit 33..32 (2b)] : Contextual Flag (0b00:多義なし, 0b01:感情転用, 0b10:制度転用, 0b11:機能語化)
    └── [bit 31..00 (32b)]: 抽象感覚アフォーダンス (通常は低ノルム、心理情動のみ保持)
```

### 1.2 外部公開言語資源とビルドパイプライン (`compile_hth_clean_poly.py`)
1. **データソース**:
   - 日本語WordNet（`wnjpn.db`）の上位・下位有向辺 ＋ Lancaster Sensorimotor Norms（英語39,707語評定値の WordNet ILI 共通 Synset 射影）＋ 国立国語研究所『分類語彙表』。
2. **グラフ・ラプラシアン調和補完**:
   - 既知シード語集合 $\mathcal{L}$ から未定義語 $\mathcal{U}$ へ正規化ラプラシアン $(I - \alpha S_{\mathcal{U}\mathcal{U}}) F_{\mathcal{U}} = \alpha S_{\mathcal{U}\mathcal{L}} Y_{\mathcal{L}} + (1 - \alpha) \bar{Y}_{\mathcal{L}}$（$\alpha = 0.85$）を解いて全 30,000 語の 8次元感覚ベクトルを算出する。
3. **分位点4bit量子化**:
   - 各列を等分位ランク変換し、`0x0`〜`0xF` の各ビンに厳密に 1,875 語ずつ割り当てて最大エントロピー 4.0 bits を達成する。
4. **決定論的シンセティック・モード（`--synthetic`）**:
   - 外部辞書ファイルがないテスト環境でも、乱数ではなく決定論的ハッシュ展開によって厳密に 480,000 Bytes の同一バイナリを出力できるようにする。

---

## 2. ヘキサゴナル・アーキテクチャ（依存性の逆転）による構文インターフェース

USIR コア内にオモチャの正規表現パーサを埋め込むことを禁じ、コアは純粋なデータ構造のみを扱う。

### 2.1 純粋構文データ構造 (`syntax_frame.py` / `syntaxFrame.ts`)
```python
@dataclass
class SyntaxNode:
    node_id: int                 # 文節 / トークンID (0-indexed)
    surface: str                 # 表層文字列
    lemma: str                   # 見出し語・基本形（HTH 辞書引きキー）
    pos: str                     # 品詞大分類 ("名詞", "動詞", "形容詞", "副詞", "助詞", "助動詞", "記号")
    pos_detail: str              # 品詞詳細 ("格助詞", "係助詞", "接続助詞", "終助詞", "非自立", etc.)
    hth_primary: int             # High 64bit: 物理語義 uint64
    hth_secondary: int           # Low 64bit: 抽象語義 uint64
    is_negation_scope: bool      # 否定助動詞のスコープ内か
    is_speculative_scope: bool   # 推量・比況助動詞のスコープ内か

@dataclass
class SyntaxEdge:
    source_id: int               # 係り元 node_id
    target_id: int               # 係り先 node_id
    relation: str                # "subj", "obj", "mod", "conj", "terminal"
    jump_distance: float         # HTH Octree XOR+LZCNT による双曲跳躍距離 D_jump in [0, 1]

@dataclass
class SyntaxTreeFrame:
    sentence_id: int
    text: str
    nodes: List[SyntaxNode]
    edges: List[SyntaxEdge]
    root_id: int                 # 文末主述語の node_id
    max_depth: int               # 木の最大深さ
    depth_distribution: List[float] # 各深さ層のノード存在確率 p_d
```

### 2.2 外部構文ドライバ (`SyntaxDriver`) の責務分離
- **Python 側 (`drivers/syntax_driver.py`)**:
  利用可能なパーサ（`janome` または `sudachipy`、係り受けはスタック型ルールまたは係り受けパーサ）を用いて入力文から `SyntaxTreeFrame` を構築し、USIR コア関数へ渡す。
- **TypeScript / CI 側**:
  クリーンルーム構文フィクスチャ（または軽量トークナイザ）から同一スキーマの `SyntaxTreeFrame` を生成して渡す。
- **USIR コアはパーサの実装詳細を一切知らず、`SyntaxTreeFrame` から数学的に 12元素を計算することだけに集中する。**

---

## 3. 言語・文芸死角解消モジュール（`OPEN-01` 〜 `OPEN-04`）の本実装

### 3.1 【OPEN-01】480KB 多義辞書と具体度 $\mathcal{C}(u)$ による文脈多義減衰ゲート
1. **語義スロットの動的選択**:
   被修飾語 $u$ と述語 $v$ の係り受け辺 $(u \to v)$ において：
   - もし $u$ の概念具体度 $\mathcal{C}(u) \ge 0.45$（物理物：例「氷」「硝子」）なら、述語 $v$（「冷たい」「落とす」）は **High 64bit（第一語義：物理感覚）** を選択する。
   - もし $u$ の概念具体度 $\mathcal{C}(u) < 0.30$（抽象概念：例「視線」「態度」「単位」）なら、述語 $v$ は **Low 64bit（第二語義：抽象転用）** を選択する。
2. **多義減衰積**:
   第二語義の感覚ベクトルに対し、具体度減衰因子 $\mathcal{C}(u)$ を乗じることで、「単位を落とす」の重力・運動覚誤爆をゼロ近くへ遮断する。

### 3.2 【OPEN-02】否定・反語・様相スコープに対する双対感覚テンソル $(\tilde{\mathbf{s}}^+, \tilde{\mathbf{s}}^-)$
係り受けパス上の助動詞（`ない/ぬ/ず` $\to \alpha_{\text{mod}} = -0.5$、`だろう/かのよう` $\to \alpha_{\text{mod}} = +0.6$、肯定 $\to +1.0$）に基づき：
- **知覚実在テンソル $\tilde{\mathbf{s}}^+ = \max(0, \alpha_{\text{mod}}) \cdot \tilde{\mathbf{s}}^{\text{eff}}$**（定常平均 $\boldsymbol{\mu}$ の `Sg-01, 04` へ）
- **認知残像コントラスト $\tilde{\mathbf{s}}^- = |\alpha_{\text{mod}}| \cdot \|\tilde{\mathbf{s}}^{\text{eff}}\|_2 \cdot \mathbf{1}_{\{\alpha_{\text{mod}} < 0\}}$**（文間運動量 $\boldsymbol{\sigma}$ の `Sg-08, 12` へ）

### 3.3 【OPEN-03】オノマトペ音韻象徴合成 $\Psi_{\text{mimetic}}(w)$ と Mimetic Immunity Protocol
未知オノマトペに対して音韻（第1子音の有声性、促音、撥音、重畳）から 8次元感覚ベクトルを決定論的合成し、かつ係り受け先が抽象語であっても減衰を免除する（第25節 Mimetic Immunity Protocol）。

### 3.4 【OPEN-04】地の文と会話文のバフチン2チャネル直交分離
鉤括弧 `「…」` を抽出して地の文系列 $S_{\text{nar}}$（$\mathbf{z}_{\text{nar}} \in \mathbb{R}^{24}$）と登場人物別発話 $S_{\text{dia}}^{(c)}$（$\mathbf{z}_{\text{dia}}^{(c)} \in \mathbb{R}^{24}$）を直交計算し、話者間マハラノビス距離 $\Delta_{\text{voice}} < 0.15$ の Dialogic Collapse を検知する。

---

## 4. Stage 1 $\to$ Stage 2 多重解像度ウェーブレット × ISTA × RQA ＆ Contrastive Fisher Distillation

1. **Stage 1（4基礎構造信号 $\mathbf{s}_m \in \mathbb{R}^4$）**：
   - $s_{m,1}$（Flux）：構文木動径エントロピー $H_{\text{radial}}(T_m) + \frac{1}{|V_m|}\sum_u \|\tilde{\mathbf{s}}_u^+\|_1 \cdot \mathcal{C}(u)$
   - $s_{m,2}$（Lattice）：モーラ長自己相関 ＋ 構文木分岐均整度
   - $s_{m,3}$（Void）：直交指示関数 $\mathbf{1}_{\Omega_0}$（修飾贅肉ゼロ・短文言い切り・余白率）
   - $s_{m,4}$（Shock）：係り受け辺上の双曲LCA跳躍 $D_{\text{jump}}(u,v) + \tilde{\mathbf{s}}_u^-$
2. **Stage 2（対数3等分ウェーブレット × ISTA 形態素分離 × RQA）**：
   - 文数 $M$ に対し $\ln M$ を対数3等分した Micro / Meso / Macro 帯域直交ハール展開と ISTA ソリトン分離により、12元素の位置 $\mathbf{q}$ と共役運動量 $\mathbf{p}$ を確定する。
3. **Contrastive Fisher Distillation (`fit_contrastive_fisher_metric`)**：
   - クラス間分散 $S_B$ とクラス内共分散 $S_W + \lambda I_{24}$ から、正定値引き戻しリーマン計量 $G_{\text{style}} = (S_W + \lambda I)^{-1}$ を対照学習で推定する。
