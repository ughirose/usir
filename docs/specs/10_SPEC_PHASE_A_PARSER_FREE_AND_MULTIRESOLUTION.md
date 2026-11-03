# Phase A 詳細技術仕様書：外部依存ゼロ・文節係り受けオートマトン、OPEN-01〜05、および多重解像度計装コア

> **対象ファイル**:
> - `python/usir_py/bunketsu_automaton.py`（新規） / `packages/usir-core/src/sensor/bunketsuAutomaton.ts`（新規）
> - `python/usir_py/extractor_24d.py` / `packages/usir-core/src/sensor/symplectic24d.ts`
> - `python/usir_py/manifold_geodesic.py` / `packages/usir-core/src/dynamics/manifoldGeodesic.ts`
> - `python/tests/test_phase_a_linguistic_core.py` / `packages/usir-core/tests/phase-a-linguistic-core.test.ts`

## 1. 外部依存ゼロ・決定論的「文節（Bunketsu）＆係り受け木オートマトン」仕様

外部形態素解析器（Janome / Sudachi 等）や数十MBの外部辞書に一切依存せず、Python と TypeScript/Wasm で **100% 同一の出力** を保証するため、日本語の文字種遷移と助詞・助動詞文法に基づく以下の決定論的オートマトンを実装する。

### 1.1 文節（Bunketsu）データ構造
```python
@dataclass
class BunketsuNode:
    index: int
    surface: str                 # 文節表層文字列（例：「冷たい」「氷を」「見つめていたが、」）
    stem: str                    # 自立語幹（例：「冷」「氷」「見」）
    tail: str                    # 付属語テイル（例：「たい」「を」「つめていたが」）
    pos_type: str                # "noun" | "verb_adj" | "adverb_mimetic" | "conj"
    case_particle: str           # 格助詞・係助詞（"が","を","に","へ","で","と","から","より","は","も","の",""）
    modality_alpha: float        # OPEN-02 様相係数: 肯定=+1.0, 推量/比況=+0.6, 否定=-0.5
    concreteness: float          # OPEN-01 概念具体度 C(u) in [0.0, 1.0]
    sensory_vec: np.ndarray      # shape (8,), [0.0, 1.0] に正規化された8次元感覚ベクトル
    is_mimetic: bool             # OPEN-03 オノマトペ判定フラグ
    head_index: int              # 係り先（親）文節の index（根である文末述語は -1）
    tree_depth: int              # 根（文末主述語）からの係り受け深さ d >= 0
```

### 1.2 決定論的・文節チャンキングアルゴリズム (`segment_into_bunketsu(sentence: str)`)
1. **オノマトペ先行保護抽出（`OPEN-03`）**：
   文中の以下のパターンを単一の `adverb_mimetic` 文節として先行マークする：
   - 2モーラ完全重畳形 $ABAB$（ひらがな/カタカナ4文字：$c_1 c_2 c_1 c_2$、例：`きらきら`、`ざわざわ`、`ゆらゆら`、`ギラギラ`）＋任意の後続 `と / に / した`
   - 促音・り形 $A\text{っ}B\text{り}$ または $AB\text{っ}と$（例：`ぷつり`、`ぎゅっと`、`はっと`、`どっしり`、`ぴたり`）
   - 撥音・長音余韻形 $A\text{ん}(B)$（例：`しんと`、`じんわり`、`ぼんやり`、`すーっと`）
2. **文字種境界による自立語＋付属語分割**：
   残りの文字列を以下の正規表現トークナイザで文節列に分割する：
   ```
   BUNKETSU_PATTERN = r'(?:[一-龯々ヶ]+|[ァ-ヴー]+|[a-zA-Z0-9]+|[ぁ-ん]{1,4}?)(?:ないで|なかった|なければ|ない|ぬ|ず|まい|だろう|でしょう|かもしれない|かのよう|みたい|らしい|べき|ため|ように|から|より|けれど|のに|ながら|たり|ても|では|には|とは|へは|まで|だけ|ほど|くらい|など|って|という|として|[がをにへでとはもやの、，]+|[aiueoぁ-ん]{0,4}(?:[、，]|$))'
   ```
   - 各チャンクの先頭の漢字列・カタカナ列（または先頭ひらがな語幹）を `stem`、後半の助詞・助動詞・活用語尾列を `tail` とする。

### 1.3 Head-Final 係り受け木 $T_m = (V_m, E_m)$ の決定論的構築規則 (`build_dependency_tree(nodes)`)
日本語の右側主要部（Head-Final）原則と非交差条件に基づき、第 $k$ 文節（$0 \le k < K-1$）の係り先 `head_index` ($> k$) を以下の優先順位で決定する（最後の文節 $K-1$ は根 `head_index = -1, tree_depth = 0`）：
1. **連体修飾（`case_particle == "の"` または `tail` が `な / た / る / い` で終わり直後が名詞文節）**：
   $\to$ 直後の名詞文節 $k+1$ に係る（`head_index = k + 1`）。
2. **副詞・オノマトペ（`is_mimetic == True` または `tail` が `と / に / く` の副詞形）**：
   $\to$ 後方で最も近い用言文節（`pos_type == "verb_adj"`）を探し、存在すればそこへ、なければ文末 $K-1$ に係る。
3. **格助詞項（`が / を / に / へ / で / から / より`）**：
   $\to$ 読点「、」を越えない範囲で後方直近の用言文節へ係る。該当がなければ文末主述語 $K-1$ に係る。
4. **主題・接続節（`は / も` または読点 `、` 終端の接続助詞 `て / で / ば / ながら / けれど / のに`）**：
   $\to$ 文末主述語 $K-1$（または次の節の主述語）へ係る。
5. **木の深さ `tree_depth` と動径階層エントロピー $H_{\text{radial}}(T_m)$（`Sg-05 Cp`）の算出**：
   - 根 $K-1$ から逆向きに辿り、各ノードの深さ $d(v)$ を確定する。
   - 最大深さ $D_{\max} = \max_v d(v)$ に対し、各深さ層 $d \in \{0, \dots, D_{\max}\}$ のノード数比率 $p_d = |\{v : d(v)=d\}| / K$ から動径エントロピーを算出する：
     $$H_{\text{radial}}(T_m) = -\sum_{d=0}^{D_{\max}} p_d \ln(p_d + 10^{-9})$$

## 2. 言語・文芸死角解消モジュール（`OPEN-01` 〜 `OPEN-04`）の数理仕様

### 2.1 【OPEN-01】概念具体度 $\mathcal{C}(u)$ と文脈多義減衰ゲート
1. **概念具体度 $\mathcal{C}(u) \in [0, 1]$ の決定論的算出**：
   - 文節 $u$ の `stem` に含まれる漢字群の 256部首HTH感覚ノルム $\|\tilde{\mathbf{s}}_u\|_\infty$ と、抽象接尾辞・制度語フラグ（`性 / 化 / 的 / 論 / 主義 / 構造 / 概念 / 制度 / 単位 / 課題 / 側面 / 関係` 等が含まれる場合は `is_abstract_stem = True`）から次式で定義する：
     $$\mathcal{C}(u) = \begin{cases} 0.15 \cdot \|\tilde{\mathbf{s}}_u\|_1 & (\text{if } \text{is\_abstract\_stem}) \\ \text{clip}\left(0.35 + 0.65 \|\tilde{\mathbf{s}}_u\|_\infty, \, 0.0, \, 1.0\right) & (\text{otherwise}) \end{cases}$$
2. **係り受け辺 $(u \to v)$ 上の3態分離と多義減衰**：
   被修飾語 $u$（例：「氷」vs「視線」vs「単位」）が述語・形容詞 $v$（例：「冷たい」「落とす」「走る」）に係る際：
   - **Mimetic Immunity Protocol（第25節）**：もし $v$ がオノマトペ（`v.is_mimetic == True`）であれば、絶対に減衰させない（$\tilde{\mathbf{s}}_v^{\text{eff}} = \tilde{\mathbf{s}}_v$）。
   - **Case 1（物理描写：$\mathcal{C}(u) \ge 0.45$、「冷たい氷」「硝子を落とす」）**：減衰なし（$\tilde{\mathbf{s}}_v^{\text{eff}} = \tilde{\mathbf{s}}_v$）。
   - **Case 2（共感覚メタファー：$\mathcal{C}(u) < 0.45$ だが $\langle \tilde{\mathbf{s}}_u, \tilde{\mathbf{s}}_v \rangle > 0.25$、「硝子の悲鳴」「青い憂鬱」）**：減衰なし、かつ `Sg-08 Lp`（比喩跳躍）に $+0.35$ ボーナス加算。
   - **Case 3（抽象転用・慣用句：$\mathcal{C}(u) < 0.30$ かつ感覚共有なし、「単位を落とす」「冷たい態度」「計画が走る」）**：
     $$\tilde{\mathbf{s}}_v^{\text{eff}} = \mathcal{C}(u) \cdot \tilde{\mathbf{s}}_v$$
     これにより、外部WSDモデル無しで「冷たい視線 / 単位を落とす」の物理感覚誤爆を劇的に減衰させる。

### 2.2 【OPEN-02】否定・反語・様相スコープに対する双対感覚テンソル $(\tilde{\mathbf{s}}^+, \tilde{\mathbf{s}}^-)$
1. **様相係数 $\alpha_{\text{mod}}(u) \in [-1, 1]$ の判定**：
   文節 $u$ 自身の `tail`、または $u$ から根（文末主述語 $B_{K-1}$）に至る係り受けパス上の文節に以下のマーカーが存在するかを検査する：
   - **明示的否定・不在**（`ない / ぬ / ず / なかった / まい / なし / 消え / 失わ`）：$\alpha_{\text{mod}}(u) = -0.5$
   - **推量・伝聞・比況**（`だろう / でしょう / かもしれない / かのよう / みたい / らしい / 思われ`）：$\alpha_{\text{mod}}(u) = +0.6$
   - **確定肯定（上記以外）**：$\alpha_{\text{mod}}(u) = +1.0$
2. **非対称寄与則（定常位置 $\boldsymbol{\mu}$ vs コントラスト運動量 $\boldsymbol{\sigma}$）**：
   - **知覚実在テンソル $\tilde{\mathbf{s}}_u^+$（チャンク平均 $\boldsymbol{\mu}$ の `Sg-01, Sg-04` へ寄与）**：
     $$\tilde{\mathbf{s}}_u^+ = \max\big(0.0, \, \alpha_{\text{mod}}(u)\big) \cdot \tilde{\mathbf{s}}_u^{\text{eff}}$$
     （※否定された「深紅の炎も凍てつく風もなかった」は $\tilde{\mathbf{s}}_u^+ = \mathbf{0}$ となり、場面の物理温度平均 $\mu$ を誤って上昇させない）
   - **認知残像・剥奪コントラスト $\tilde{\mathbf{s}}_u^-$（文間運動量 $\boldsymbol{\sigma}$ の `Sg-08, Sg-12` へ寄与）**：
     $$\tilde{\mathbf{s}}_u^- = \big|\alpha_{\text{mod}}(u)\big| \cdot \|\tilde{\mathbf{s}}_u^{\text{eff}}\|_2 \cdot \mathbf{1}_{\{\alpha_{\text{mod}}(u) < 0\}}$$
     （※否定による「イメージの喚起と打ち消しの落差（Negative Contrast Spike）」は運動量 $\boldsymbol{\sigma}$ 側を励起する）

### 2.3 【OPEN-03】オノマトペ音韻象徴性（Sound Symbolism）決定論的合成関数 $\Psi_{\text{mimetic}}(w)$
`is_mimetic == True` と判定された語（または4〜6文字の副詞的ひらがな/カタカナ語）に対し、ローマ字音素変換または五十音行判定により 8次元感覚ベクトル $\tilde{\mathbf{s}} = (d_1 \text{視覚}, d_2 \text{聴覚}, d_3 \text{触覚}, d_4 \text{味嗅覚}, d_5 \text{運動重量}, d_6 \text{温度}, d_7 \text{情動覚醒}, d_8 \text{内受容余韻}) \in [0, 1]^8$ を合成する：
- **第1文字が濁音（が・ざ・だ・ば行）**（例：「ぎらぎら」「ざわざわ」「どろり」）：
  $d_5 (\text{運動重量}) += 0.45$, $d_3 (\text{触覚粗さ}) += 0.35$, $d_7 (\text{情動覚醒}) += 0.35$
- **第1文字が無声摩擦・破擦音（さ・しゃ・は・ふ・ち・つ行）**（例：「しんと」「さらさら」「ふわり」）：
  $d_2 (\text{聴覚静寂/摩擦}) += 0.40$, $d_3 (\text{触覚滑らかさ}) += 0.30$, $d_6 (\text{冷涼温度}) += 0.30$
- **第1文字が半濁音（ぱ・ぴ・ぷ・ぺ・ぽ）**（例：「ぱちり」「ぷつり」「ぴかぴか」）：
  $d_1 (\text{視覚輝度}) += 0.40$, $d_5 (\text{運動瞬間性}) += 0.45$
- **語中・語末に促音「っ / ッ」または語末「り」を含む**（例：「ぎゅっと」「ぷつり」「はっと」）：
  $d_5 (\text{運動急峻度}) += 0.35$, 破調インパルスボーナス $+0.30$
- **語中・語末に撥音「ん / ン」または長音「ー」を含む**（例：「しんと」「じんわり」「ぼーっと」）：
  $d_8 (\text{内受容・余韻}) += 0.45$, $d_2 (\text{聴覚共鳴}) += 0.25$
- **完全重畳形態 $ABAB$**（例：「きらきら」「ざわざわ」）：
  Micro 律動ボーナス $+0.35$

### 2.4 【OPEN-04】地の文と会話文のバフチン2チャネル直交分離 (`extract_dual_channel_bakhtinian`)
1. テキストから鉤括弧 `「([^」]+)」`（および `『([^』]+)』`）で囲まれた発話スパンを抽出し、直前または直後の地の文節から話者ラベル $c$（例：「太郎は『…』と言った」$\to c=\text{"太郎"}$、明示話者なしの場合は交互話者 `Speaker_A / Speaker_B`）を付与する。
2. **Primary Channel（地の文 $\mathbf{z}_{\text{nar}} \in \mathbb{R}^{24}$）**：発話スパンを除去した純粋な地の文系列 $S_{\text{nar}}$ から 24D ベクトルを算出する（会話の挿入頻度による文長変動係数 $CV_L$ の偽陽性汚染を防止）。
3. **Secondary Channel（発話ポリフォニー $\mathbf{z}_{\text{dia}}^{(c)} \in \mathbb{R}^{24}$）と Dialogic Collapse 検知**：
   各話者 $c$ の発話集合から特徴ベクトルを計算し、2話者間のマハラノビス/ユークリッド距離 $\Delta_{\text{voice}}(c_1, c_2) = \|\mathbf{z}_{\text{dia}}^{(c_1)} - \mathbf{z}_{\text{dia}}^{(c_2)}\|_2$ を算出する。$\Delta_{\text{voice}} < 0.15$ の場合、AI特有の「全キャラクターの喋り方同化（Dialogic Collapse）」として警告フラグ `dialogic_collapse_detected = True` を返す。

## 3. Stage 1 ($\mathbf{s}_m \in \mathbb{R}^4$) $\to$ Stage 2（多重解像度ウェーブレット・ISTA・RQA）＆ Contrastive Fisher Distillation

1. **Stage 1（各文 $m=1,\dots,M$ の4基礎信号 $\mathbf{s}_m = (s_{m,1}, s_{m,2}, s_{m,3}, s_{m,4})^\top \in \mathbb{R}^4$）**：
   - $s_{m,1}$（Flux：密度・具象深さ）：文節木の動径エントロピー $H_{\text{radial}}(T_m)$ ＋ 知覚実在ノルム $\frac{1}{|V_m|}\sum_u \|\tilde{\mathbf{s}}_u^+\|_1 \cdot \mathcal{C}(u)$
   - $s_{m,2}$（Lattice：構文対称・律動）：モーラ/文節長の自己相関 ＋ 木分岐均整度 $\text{Sy}_{\text{tree}}(T_m)$
   - $s_{m,3}$（Void：余白・引き算）：直交指示関数 $\mathbf{1}_{\Omega_0}$（修飾贅肉ゼロ・短文言い切り・引き算の余白率）
   - $s_{m,4}$（Shock：跳躍・コントラスト）：係り受け辺上の双曲LCA跳躍 $D_{\text{jump}}(u,v)$ ＋ 否定剥奪コントラスト $\tilde{\mathbf{s}}_u^-$ ＋ オノマトペ促音インパルス
2. **Stage 2（対数3等分自己相似ウェーブレット × ISTA ソリトン分離 × RQA）**：
   - 文数 $M$ に応じて固定窓（8文・40文）ではなく、$\ln M$ を対数3等分したスケール（Micro: 隣接差分 $j=1$、Meso: $\lfloor M^{1/3}\rfloor \sim \lfloor M^{2/3}\rfloor$ 移動窓 $j=2$、Macro: 大域トレンド＋RQAリカレンス率 $I_{\text{arch}}$ $j=3$）へ直交ハール変換で分解し、12元素の位置 $\mathbf{q} \in \mathbb{R}^{12}$ と共役運動量 $\mathbf{p} \in \mathbb{R}^{12}$ を出力する。
3. **Grey-Box Contrastive Fisher Distillation (`fit_contrastive_fisher_metric(pos_z, neg_z)`)**：
   - 正例クラスタ（Critical / Void）と負例クラスタ（Deadzone / Chaos）の24次元状態集合から、クラス間分散 $S_B$ とクラス内正則化共分散 $S_W + \lambda I_{24}$ の一般化レイリー商 $(S_W + \lambda I)^{-1} S_B$ により、正定値引き戻し計量行列 $G_{\text{style}} = (S_W + \lambda I)^{-1}$ と最適判別射影をデータから直接推定する関数を `manifold_geodesic.py` / `manifoldGeodesic.ts` に実装する。
