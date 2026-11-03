# Phase B 詳細技術仕様書：階層的デュアルレート 5ステージ Adaptive-BAOAB、節単位 Twisted SMC ($P=4$)、および 3,000字 3モードHSMM長尺生成

> **対象ファイル**:
> - `python/usir_py/realtime_decoder.py` / `packages/usir-core/src/dynamics/realtimeDecoder.ts`（または `bidirectionalBridge.ts` 連携）
> - `python/usir_py/hsmm_longform_director.py`（新規） / `packages/usir-core/src/dynamics/hsmmLongformDirector.ts`（新規）
> - `python/colab/run_v4_bidirectional_experiment.py`
> - `python/tests/test_realtime_decoder.py` & `python/tests/test_hsmm_longform.py`（新規）

## 1. 階層的デュアルレート 5ステージ Adaptive-BAOAB 積分器の厳密実装

従来の「固定ベクトルとのコサイン類似度加算」を廃止し、原典第11.5節（定理5.1）および第11.6節のデュアルレート制御を `realtime_decoder.py` に完全実装する。

### 1.1 `OPEN-08` Ghost History プリロードとカルマン・シュリンケージ
1. デコーダ初期化時（$m = 0$）、長さ $L = 16$ の仮想履歴リングバッファを空にするのではなく、目標平均 $\boldsymbol{\mu}^* \in \mathbb{R}^{12}$ の周りに $1/f$ 相関ガウス揺らぎ $\boldsymbol{\eta}_k \sim \mathcal{N}(\mathbf{0}, 0.02 I_{12})$ を加えた **Ghost History $\{\tilde{\mathbf{q}}_{-L+1}, \dots, \tilde{\mathbf{q}}_0\}$** でプリロードする。
2. 第 $m$ 節（$m = 1, 2, \dots$）が確定して実測12元素 $\hat{\mathbf{q}}_m$ が得られた際、半減期 $m_0 = 5$ のベイズ収縮ゲイン $\alpha_m = \frac{m}{m + 5}$ を用いてコールドスタート発散を抑止する：
   $$\hat{\mathbf{q}}_m^{\text{shrunk}} = \alpha_m \hat{\mathbf{q}}_m + (1 - \alpha_m) \boldsymbol{\mu}^*$$

### 1.2 Outer Loop：節境界（Clause Boundary）における 5ステージ Adaptive-BAOAB 更新 (`step_baoab_clause`)
読点「、」または句点「。」で1節が確定するごとに、現在の位置 $\mathbf{q}_m \in \mathbb{R}^{12}$、共役運動量 $\mathbf{p}_m \in \mathbb{R}^{12}$、目標アトラクタ $\boldsymbol{\mu}^*$、計量行列 $G_{\text{style}}$、構文エントロピー $H_{\text{syntax}}$ に対して以下の 5ステージ（時間刻み $\Delta t = 0.25$、質量 $M = I_{12}$、熱浴温度 $\tau = 0.04$）を実行する：

1. **ポテンシャル勾配（Moreau-Yosida 近接勾配＋終点ブリッジ引力 $\overleftarrow{\boldsymbol{\phi}}$）**：
   $$\nabla_{\mathbf{q}} V(\mathbf{q}) = G_{\text{style}} (\mathbf{q} - \boldsymbol{\mu}^*) - \lambda_{\text{bridge}} \overleftarrow{\boldsymbol{\phi}}$$
2. **Stage B（前半運動量半ステップキック）**：
   $$\mathbf{p}^{(1)} = \mathbf{p}_m - \frac{\Delta t}{2} \nabla_{\mathbf{q}} V(\mathbf{q}_m)$$
3. **Stage A（前半位置半ステップドリフト）**：
   $$\mathbf{q}^{(1)} = \mathbf{q}_m + \frac{\Delta t}{2} M^{-1} \mathbf{p}^{(1)}$$
4. **Stage O（構文エントロピー連動・可変摩擦 $\Gamma_t$ 下での厳密ガウス熱浴更新 — 定理5.1）**：
   - 直近トークン群の平均予測エントロピー $H_{\text{syntax}}$ が閾値 $\epsilon_{\text{syn}} = 0.6$ 未満（助詞・括弧閉じ等の構文拘束局面）のとき摩擦 $\gamma_t = 50.0$（強ブレーキ）、自由選択局面のとき $\gamma_t = 1.2$（不足減衰振動）とおく。
   - 減衰係数 $\theta_t = \exp(-\Delta t \cdot \gamma_t)$、標準正規乱数（または決定論的シードガウス）$\boldsymbol{\xi} \sim \mathcal{N}(\mathbf{0}, I_{12})$ を用いて運動量を更新する：
     $$\mathbf{p}^{(2)} = \theta_t \mathbf{p}^{(1)} + \sqrt{\tau (1 - \theta_t^2)} \, \boldsymbol{\xi}$$
5. **Stage A（後半位置半ステップドリフト）**：
   $$\mathbf{q}_{m+1}^* = \mathbf{q}^{(1)} + \frac{\Delta t}{2} M^{-1} \mathbf{p}^{(2)}$$
6. **Stage B（後半運動量半ステップキック）**：
   $$\mathbf{p}_{m+1}^* = \mathbf{p}^{(2)} - \frac{\Delta t}{2} \nabla_{\mathbf{q}} V(\mathbf{q}_{m+1}^*)$$
7. **次節の目標変位ベクトル $\Delta \mathbf{q}_{m+1}^* = \mathbf{q}_{m+1}^* - \hat{\mathbf{q}}_m^{\text{shrunk}}$** を Inner Loop および Twisted SMC へ渡す。

## 2. 節単位・軽量粒子フィルタ（Clause-Level Twisted SMC, $P=4$ 粒子、`OPEN-07` & 第21.2節）

日本語 SOV 構文の袋小路（Garden-Path）を根絶し、文末アンカー $W_{\text{end}}^*$ への自然な着地を保証するため、`ClauseTwistedSMCDecoder` クラスを実装する。

### 2.1 アルゴリズム手順 (`generate_with_clause_twisted_smc`)
各節ステップ $m = 1, \dots, M_{\text{clauses}}$ において：
1. **並列ショートロールアウト（$P = 4$ 粒子）**：
   現在の確定プレフィックス $w_{\le m}$ から、次の句読点（「、」または「。」、最大 20 BPE トークン）が出現するまで、温度 $T \in \{0.65, 0.72, 0.80, 0.88\}$ で $P = 4$ 本の候補節 $\{c_{m+1}^{(1)}, c_{m+1}^{(2)}, c_{m+1}^{(3)}, c_{m+1}^{(4)}\}$ をバッチ並列生成する。
2. **Twisted Importance Weight（捻り重み $w^{(p)}$）の算出**：
   各候補節 $c^{(p)}$ に対し、以下の3項からなる捻り重み $w^{(p)}$ を計算する：
   $$w^{(p)} = \underbrace{\ln P_{\text{LLM}}\big(c^{(p)} \mid w_{\le m}\big)}_{\text{① 局所構文自然度 (平均トークン対数尤度)}} + \gamma_{\text{reach}} \underbrace{\ln P_{\text{LLM}}\big(W_{\text{end}}^* \mid w_{\le m} \circ c^{(p)}\big)}_{\text{② Terminal Reachability (未来の着地点への接続尤度)}} - \frac{\beta_{\text{steer}}}{2} \underbrace{\left\| \hat{\mathbf{q}}\big(w_{\le m} \circ c^{(p)}\big) - \mathbf{q}_{m+1}^* \right\|_{G_{\text{style}}}^2}_{\text{③ BAOAB目標軌道への追従誤差}}$$
   - GPU 推論時は `model(input_ids)` の 1 回の forward パス（バッチサイズ $P=4$）で `[w_{<=m} + c^{(p)} + W_end^*]` のロジットから ① と ② の正確な対数尤度を $O(1)$ バッチ計算する。
3. **リサンプリングと BAOAB 状態更新**：
   $\max_p w^{(p)}$ を達成した最良候補節 $c^{(p^*)}$ を採択して $w_{\le m+1} \leftarrow w_{\le m} \circ c^{(p^*)}$ とし、`step_baoab_clause` を呼び出して次節へ進む。

## 3. 4KB 3モード HSMM 多井戸スイッチング ＋ Momentum Banking による 3,000字長尺生成 (`hsmm_longform_director.py`)

1段落（500字）の短尺実験を超え、**全 6〜8 段落（3,000〜4,000字）の長尺生成**において AI Deadzone への熱的死を完全に防ぐ `HSMMLongformDirector` を実装する。

1. **3つの相井戸（Wells）とマルコフ遷移行列 $T_{ij} \in \mathbb{R}^{3\times 3}$**：
   - **Well 0 (`Narrative / Crystal-Void`)**：静謐な情景・事実描写（目標 $CV_L = 0.48, \phi_{\text{void}} = 0.65$）
   - **Well 1 (`Dialogue / Polyphony`)**：対話または多角的論点交錯（目標 $CV_L = 0.55$, 話者間距離 $\Delta_{\text{voice}} \ge 0.25$）
   - **Well 2 (`Climax / Critical-Shock`)**：核心の破調・カタルシス解放（目標 $CV_L = 0.65, \kappa_{\text{kin}} = 0.70$）
2. **`OPEN-09` Momentum Banking（運動量エネルギー蓄電機構）**：
   - 事実伝達や前提定義のために破調が抑制された段落 $k$ では、未解放の運動量エネルギー $\Delta \mathbf{p}_{\text{banked}} \leftarrow \Delta \mathbf{p}_{\text{banked}} + \max(\mathbf{0}, \mathbf{p}_k^* - \hat{\mathbf{p}}_k)$ を蓄電し、`Climax` 井戸に入った瞬間に $\mathbf{p}_{\text{climax}}^* \leftarrow \mathbf{p}_{\text{climax}}^* + \Delta \mathbf{p}_{\text{banked}}$ として一気に解放する。
3. **`OPEN-10` Wundt-Berlyne 逆U字審美効用と不応期（Refractory Period）**：
   - `Climax` 井戸で大きな運動量解放 $\|\dot{\mathbf{p}}\|_2^2$ が発生した直後の段落では、審美疲労（Purple Prose 化）を防ぐため、強制的に `Void`（静寂な余韻・残心）井戸へ遷移させる不応期ゲートを稼働させる。
