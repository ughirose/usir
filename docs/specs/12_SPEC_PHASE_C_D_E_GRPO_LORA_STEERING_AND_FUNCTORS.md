# Phase C・D・E 詳細技術仕様書：実モデル直結 USIR-GRPO QLoRA 鍛造、中間層12元素 Activation Steering、OSWE `usir-forge`、および普遍クロスモーダル関手

> **対象ファイル**:
> - **Phase C**: `python/usir_py/lora_grpo_forge.py`（新規）、`python/usir_py/activation_steerer.py`（新規）、`python/colab/run_v4_bidirectional_experiment.py`
> - **Phase D**: `python/usir_py/usir_forge.py`（新規）、`packages/usir-core/src/schema/usirForge.ts`（新規）、`packages/usir-wasm/`
> - **Phase E**: `python/usir_py/nishiki_unified_field.py`（新規）、`packages/usir-bridges/src/crossModalFunctors.ts`（拡張）

## 1. 【Phase C】実モデル直結 USIR-GRPO / Style-LoRA 鍛造パイプライン（ダミー実装の完全撤去）

### 1.1 参照モデル PPL バリア $\mathcal{M}_{\text{semantic}}$ と KL 正則化を含む完全版 4重積報酬（原典第22.2節）
`compute_multiplicative_grpo_reward` に、単語スタッフィング（意味破綻ハック）を100%遮断する参照パープレキシティ関門を追加する：
$$\mathcal{M}_{\text{semantic}}(y \mid x) = \frac{1}{1 + \exp\big(\alpha_{\text{ppl}} (\text{PPL}_{\text{ref}}(y \mid x) - \text{PPL}_{\text{thresh}})\big)}$$
- GPU 実行時はベースモデル（LoRA 無効状態の参照分布 $\pi_{\text{ref}}$）によるクロスエントロピー損失 $\mathcal{L}_{\text{CE}}(y \mid x)$ から $\text{PPL}_{\text{ref}} = \exp(\mathcal{L}_{\text{CE}})$ を実測する（閾値 $\text{PPL}_{\text{thresh}} = 45.0$）。

### 1.2 実 LLM の Attention 層へ直結する `AttachedLoRALinear` と本物の GRPO 政策勾配更新 (`lora_grpo_forge.py`)
`08_run_v4_bidirectional_experiment.py` にあった単体 `LinearLoRAAdapter(64, 64)`（`torch.randn` を入れていたダミー）を廃止し、実モデル `model`（`Qwen2.5-1.5B/3B/7B`）の最終4層の `self_attn.q_proj` および `self_attn.v_proj` を以下の `AttachedLoRALinear` でラップする：

```python
class AttachedLoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False
        in_f = base_layer.in_features
        out_f = base_layer.out_features
        dtype = base_layer.weight.dtype
        device = base_layer.weight.device
        self.lora_A = nn.Parameter(torch.randn(r, in_f, device=device, dtype=torch.float32) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_f, r, device=device, dtype=torch.float32))
        self.scaling = alpha / float(r)
        self.lora_enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if not self.lora_enabled:
            return base_out
        x_fp32 = x.to(torch.float32)
        lora_out = (x_fp32 @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out.to(base_out.dtype)
```

#### 本物の GRPO 学習ループ (`run_authentic_grpo_lora_forge`)：
1. プロンプト $x$ に対し、現在のポリシー $\pi_\theta$ で $G = 4$ 本のロールアウト $\{y_1, \dots, y_G\}$ を生成する。
2. 各生成文 $y_i$ に対して `compute_multiplicative_grpo_reward` で物理報酬 $R_i$ を計算し、グループ相対アドバンテージ $\hat{A}_i = (R_i - \mu_R) / (\sigma_R + 10^{-6})$ を算出する。
3. **実トークン対数確率と KL ダイバージェンスの計算**：
   - `lora_enabled = False`（参照モデル $\pi_{\text{ref}}$）で `logits_ref = model(input_ids).logits` を取得（`torch.no_grad()`）。
   - `lora_enabled = True`（学習対象 $\pi_\theta$）で `logits_theta = model(input_ids).logits` を取得。
   - 生成トークン部分の対数確率 $\ln \pi_\theta(y_{i,t} \mid x, y_{i,<t})$ とトークン単位 KL 項 $D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) = \exp(\ln \pi_{\text{ref}} - \ln \pi_\theta) - (\ln \pi_{\text{ref}} - \ln \pi_\theta) - 1$ を計算し、目的関数：
     $$\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G}\sum_{i=1}^G \left( \hat{A}_i \cdot \frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \ln \pi_\theta(y_{i,t} \mid x, y_{i,<t}) - \beta_{\text{KL}} D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) \right)$$
     に対して `loss.backward()` と `optimizer.step()` を実行する。
4. 学習完了後、**外部 LogitsProcessor を一切使わない素の `model.generate()`（推論オーバーヘッド `0.00 ms`）** を実行し、学習前（Base Unsteered）と比較して $CV_L$・$\mathcal{D}_{\text{3-phase}}$・$R_{\text{GRPO}}$ が劇的に改善することを実証する。

## 2. 【Phase C】Level 2：潜在空間への12元素神経回路直結 (`SymplecticActivationSteerer`、原典第54節)

プロンプトを介さず、LLMの中間層（Residual Stream）のニューロン電位を直接 12元素（`Sg-01..12`）に沿って物理制御する。

1. **12元素ステアリング軸の較正 (`calibrate_12_element_axes(model, tokenizer, corpus)`)**：
   - パブリックドメイン検証コーパスの各文をモデルに入力し、中間層（第 $L_{\text{mid}} = \lfloor 0.65 N_{\text{layers}} \rfloor$ 層）の文末トークン隠れ状態 $\mathbf{h}_m \in \mathbb{R}^{d_{\text{model}}}$ と 12元素実測値 $\mathbf{q}_m \in \mathbb{R}^{12}$ を収集する。
   - リッジ回帰（または特異値分解直交化 Gram-Schmidt）により、各元素 $k \in \{1,\dots,12\}$ を励起する正規直交ステアリング行列 $V_{\text{steer}} = [\mathbf{v}_1, \dots, \mathbf{v}_{12}] \in \mathbb{R}^{d_{\text{model}} \times 12}$（$V_{\text{steer}}^\top V_{\text{steer}} = I_{12}$）を抽出する。
2. **推論時フォワードフック介入 (`register_forward_hook`)**：
   - 目標変位 $\Delta \mathbf{q}^* \in \mathbb{R}^{12}$（例：第III族 `Void [間]` を $+0.5$、または第IV族 `Shock [破調]` を $+0.5$）に対し、中間層出力 $\mathbf{h}_t$ に次式を直接注入する：
     $$\mathbf{h}_t^{\text{steered}} = \mathbf{h}_t + \alpha_{\text{act}} \cdot \|\mathbf{h}_t\|_2 \cdot (V_{\text{steer}} \Delta \mathbf{q}^*)$$

## 3. 【Phase D】OSWE ストリーミング・極座標フーリエ位相アライメント蒸留器 (`usir-forge`)

複数の作品チャンクを単純平均すると、クライマックス位置 $t_{\text{peak}}$ のズレによって波形が相殺され Deadzone 化する（原典爆弾⑥）。これを防ぐため `usir_forge.py` / `usirForge.ts` に以下を実装する：
1. **極座標フーリエ位相アライメント**：
   各チャンクのエネルギー包絡線 $E(t)$ の基本波フーリエ係数 $c_1 = \frac{1}{M}\sum_{m=0}^{M-1} E(m) e^{-i 2\pi m / M}$ からクライマックス位相角 $\hat{\theta}_{\text{peak}} = \arg(c_1)$ を求め、全チャンクを基準位相 $\theta_0 = \frac{3\pi}{4}$（黄金分割クライマックス位置 $t/M \approx 0.62\text{〜}0.75$）へ巡回シフト整列してから Welford 統計量を累積する。
2. **4,096 Byte（4KB）3モード GMM 完全バイナリ出力**：
   地の文（Mode 1）・会話（Mode 2）・クライマックス（Mode 3）の各 24次元平均 $\boldsymbol{\mu}_k \in \mathbb{R}^{24}$、正則化精度行列の下三角コレスキー因子 $L_k \in \mathbb{R}^{300}$（$\Sigma_k^{-1} = L_k L_k^\top$）、混合重み $\boldsymbol{\pi} \in \mathbb{R}^3$、遷移行列 $T \in \mathbb{R}^{3\times 3}$、および SHA-256 署名を原典第24.1節のメモリマップ通り厳密に 4,096 Bytes の `ArrayBuffer` / `bytes` にパックする。

## 4. 【Phase E】Nishiki 統一場 $\tau_{\text{genre}}$・WorldCraft ヌル空間調停・6大モダリティ普遍関手

1. **Nishiki-USIR 統一場（`OPEN-06` ジャンル温度 $\tau_{\text{genre}} \in [0, 1]$）**：
   - Toulmin 論理ポテンシャル $U_{\text{Toulmin}}(x)$：Claim（主張文末）・Data（数値/根拠/引用）・Warrant（因果接続詞「なぜなら/したがって/ゆえに」）の三要素充足度、1文あたりの係り受け最大深さ超過ペナルティ（$D_{\max} > 6$）、主語と文末述語のねじれペナルティを算出する。
   - 統一エネルギー $\mathcal{F}_{\text{total}}(x) = (1 - \tau_{\text{genre}}) U_{\text{Toulmin}}(x) + \tau_{\text{genre}} \mathcal{F}_{\text{USIR}}(x)$ により、仕様書（$\tau=0.1$）・note記事（$\tau=0.5$）・小説（$\tau=0.9$）を単一の数式で連続制御する。
2. **WorldCraft オントロジー制約との辞書式ヌル空間射影調停（`OPEN-09`）**：
   - 禁止語彙・死亡キャラ・未解放知識のハードマスク（Tier 1）に加え、オントロジー制約勾配行列 $J_C \in \mathbb{R}^{k \times 12}$ の零空間射影作用素 $P_{\text{null}} = I_{12} - J_C^\top (J_C J_C^\top + \epsilon I)^{-1} J_C$ を用いて、世界観制約を1ミリも侵害しない方向 $\Delta \mathbf{q}_{\text{safe}} = P_{\text{null}} \Delta \mathbf{q}^*$ のみへ文体運動量を射影し、直交カットされた残余運動量を Momentum Bank に蓄電する。
3. **視覚関手 $\mathcal{F}_{\text{visual}}$ ＆ 音楽関手 $\mathcal{F}_{\text{music}}$ の双方向トランスパイル（第27・28節）**：
   - $\mathcal{F}_{\text{visual}}$：2D画像配列 $I(x,y)$ から ①4方向 Gabor フィルタ空間周波数エネルギー、②Lab/RGB 色空間エントロピー、③局所コントラスト（キアロスクーロ勾配）、④Spatial Quadtree 深さエントロピー、⑤顕著性極大点を結ぶ Saliency Scanpath $p_1 \to \dots \to p_M$ を実測して 24D `.usir` へ写像し、逆写像 $\mathcal{F}_{\text{visual}}^{-1}$ で Diffusion 用の CFG スケール・高周波ノイズゲイン・構図余白比を出力する。
   - $\mathcal{F}_{\text{music}}$：ノート/和音系列 $\{(p_t, d_t, v_t)\}$ から五度圏トーラス距離、不協和度、シンコペーション、休符比（Void）を実測して 24D `.usir` へ写像し、逆写像 $\mathcal{F}_{\text{music}}^{-1}$ で小説の 24D 軌道からリアルタイムに BPM・和声緊張度・休符間隔・リバーブ減衰を制御した WebAudio / MIDI シーケンスを生成する。
