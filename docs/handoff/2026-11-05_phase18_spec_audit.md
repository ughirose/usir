# Phase 18 原典仕様 監査表 (2026-11-05, 記入者: Claude Code)

## 決定事項
- 監査対象: `docs/specs/09-12`（`docs/blueprints/` は 09 のみ内容差あり、10 に HEXAGONAL 版が別に存在）、`docs/handoff/13`・`14`・`2026-11-04_handoff13-audit.md`。実装は `python/usir_py/**` と `packages/usir-core|usir-mcp|usir-bridges/src/**`。コードは変更していない。
- 判定は本監査でコードを読んで確認した範囲に限る。テストの実行・数値の再測定はしていない。引用する数値は既存 handoff の記録値。
- 区分の基準: 実装済み = 該当コードを特定し、仕様の数式・意味と一致。簡略化 = 手設定定数、近似、片言語のみ、別定義、結線なし等で仕様どおりではない。未実装 = 該当コードなし。意図的不採用 = 検証のうえ採用を見送った記録がある。
- 仕様が Python・TS の両方を要求し片方が欠落または別定義の場合は「簡略化」とし、備考に記す。LoRA・Activation Steering・GRPO は性質上 Python のみ（TS 欠落は減点しない）。
- 範囲外（本ラウンドで判定対象としない）: 多重解像度抽出器の較正、Table 1 / Table 2、GRPO の完走と効果実証。該当行は「範囲外」と備考に明記し、構造上の事実のみ記す。
- 仕様の表記: S09/S10/S11/S12 = `docs/specs/` の各ファイル、B10 = `docs/blueprints/10_SPEC_PHASE_A_HEXAGONAL_AND_HTH_LEXICON.md`、H13/H14 = `docs/handoff/13_*`/`14_*`、A13 = `2026-11-04_handoff13-audit.md`。数字は行番号。

### 区分ごとの件数（全 53 行）
- 実装済み: 12
- 簡略化: 36
- 未実装: 3
- 意図的不採用: 2

### 主な所見
1. 仕様が 2 系統ある（S10 = 外部依存ゼロ・自前オートマトン、B10 = Janome/Sudachi を外部ドライバで注入）。実装は B10 の `SyntaxTreeFrame`/`SyntaxDriver` 構成に寄り、`bunketsu_automaton.py` / `bunketsuAutomaton.ts` は存在しない。Python は janome が入っていれば janome を優先する（`syntax_driver.py:57-62`）ため、S10 の「Python/TS で 100% 同一出力」は成立していない。
2. 下流の主要入力が手設定のままのものがある: QDA の article / cot_reasoning 定数、`momentum_from_summary` の固定 Λ、HSMM 遷移行列・Well 目標、Stage 1/2 の係数、H_syntax の代理値。
3. 4KB Tier-M バイナリ（`usir_forge`）は生成・読込できるが、本番経路（Director・QDA・G_style）の消費者がない（`load_tier_m_binary` は `__init__` の export とテストのみ）。`fit_contrastive_fisher_metric` も同様に本番から呼ばれず、デコーダの `G_style` は単位行列。
4. TS 側は Phase A の言語ゲート（OPEN-01/02/04）、Toulmin/統一場、Ghost History/ベイズ収縮、LLM 経路が未移植または別定義。

## 監査表

| # | 仕様項目 | 出典 | Py | TS | 区分 | 備考 |
|---|---|---|---|---|---|---|
| 1 | 外部依存ゼロの文節＆係り受けオートマトン（新規ファイル） | S10:1-5, S09:11 | `drivers/syntax_driver.py:31`(`BUNKETSU_CHUNK_RE`), `:188`(`_parse_cleanroom`)。`bunketsu_automaton.py` なし | なし。フレームは外部注入（`sensor/symplectic24d.ts:70`） | 簡略化 | 仕様のファイル名・`BunketsuNode` 構造（S10:8-28）は未作成。janome があれば janome 優先（`syntax_driver.py:57-62`）で S10 の外部依存ゼロ・100% 同一出力に反する。TS に構文ドライバがなく Py/TS で同一入力が作れない |
| 2 | Phase A 仕様の二重化（S10 自前オートマトン vs B10 外部パーサ+ヘキサゴナル） | S10:1-5, B10:43-84 | `syntax_frame.py`, `drivers/syntax_driver.py`（B10 構成に準拠） | `schema/syntaxFrame.ts` | 簡略化 | どちらを正本とするかを記録した文書が見当たらない。実装は B10 + S10 の cleanroom 規則の折衷。判断が必要 |
| 3 | Head-Final 係り受け木の決定論的構築（規則 1〜4） | S10:44-53 | `syntax_driver.py:250`(`_build_tree_frame`) | なし | 簡略化 | 連体節規則と HTHPolyLexicon 共有キャッシュは A13 で追加済み。規則 1〜4 の逐条照合は本監査では未実施。主述の入れ子は深さ 10 まで取れるが平文は 1〜3（A13 ③） |
| 4 | 動径階層エントロピー H_radial | S10:54-57 | `syntax_driver.py:353` | `schema/syntaxFrame.ts:38` | 実装済み | 式 −Σ p_d ln(p_d+1e-9) と一致。TS 側は同一式（数値一致テストは未確認） |
| 5 | 128-bit 多義辞書 `hth_poly_30k.bin`（480,000 B、16 B/語、4bit 分位量子化） | B10:14-39, S09:9 | `hth_lexicon.py:169`, `python/scripts/compile_hth_clean_poly.py:314`。実体 480,000 B | `sensor/hthLexicon.ts:130` | 簡略化 | 構造とサイズは仕様どおり。内容は `hth_poly_30k_meta.json` の `build_mode: "synthetic"`（部首+SHA-256 展開、語に「物物」「物心」等の生成語を含む）。実データ版 `derive_features_from_wnjpn_lancaster`(`compile_hth_clean_poly.py:368`) は未実行 |
| 6 | 実言語資源（日本語 WordNet / Lancaster / 分類語彙表）でのビルド | B10:31-37, S09:10 | コードのみ | なし | 未実装 | S09:10 で最終フェーズへ意図的に後回しと明記。Phase E 完了後も synthetic のまま |
| 7 | HTH 跳躍距離 D_jump（Octree XOR+LZCNT） | B10:66, S10:115 | `hth_lexicon.py:131` | `sensor/hthLexicon.ts:95` | 実装済み | 離散木距離 1−LCA/10。仕様の「双曲」は離散近似 |
| 8 | OPEN-01 概念具体度 C(u) | S10:61-64 | `linguistic_gates.py:35` | `symplectic24d.ts` 内の場当たり値(0.05/0.90/max(0.65,maxS)/max(0.30,maxS)) | 簡略化 | Py: 抽象語リストが手書きで仕様例語（視線・態度・単位等）を含む。抽象側は ‖s‖₁/8 で正規化し [0.02,0.28] にクリップ（仕様は 0.15·‖s‖₁）。TS は仕様式を使わない（Py/TS 不一致） |
| 9 | OPEN-01 3 態分離と多義減衰（Case1/2/3、Mimetic Immunity） | S10:65-72, B10:90-96 | `linguistic_gates.py:83`。閾値 0.45/0.30/0.25/+0.35 は仕様どおり、遷移帯 [0.30,0.45) の補間は仕様外の追加 | extractor に減衰処理なし（`v2/hthLexicon.ts:728` にも該当定数なし） | 簡略化 | Py は一致。TS は未移植 |
| 10 | OPEN-02 様相係数 α と双対感覚テンソル (s⁺, s⁻) | S10:74-86 | `linguistic_gates.py:184`, `:212`。スコープ伝播 `syntax_driver.py:319-333` | `symplectic24d.ts` 内で表層正規表現により α を決定 | 簡略化 | Py は α=−0.5/+0.6/+1.0、s⁺・s⁻ の式とも仕様一致（実装済み相当）。TS は係り受けパスではなく表層マッチ。s⁻ は s4 の重み 0.40 経由でのみ Shock 側へ入る |
| 11 | OPEN-03 オノマトペ音韻象徴 Ψ_mimetic と先行保護抽出 | S10:32-36, 88-101 | `linguistic_gates.py:266`(判定), `:318`(合成) | `v2/hthLexicon.ts:443`(`synthesizeMimeticHTH`) | 簡略化 | Py の加算値（0.45/0.35/0.40/0.30…）は仕様一覧と一致。TS は別実装で、Py との値一致テストを確認していない。除外語リストは手書き（`MIMETIC_EXCLUSIONS`） |
| 12 | OPEN-04 バフチン 2 チャネル分離と Dialogic Collapse | S10:103-107, B10:106-107 | `linguistic_gates.py:529` | なし | 簡略化 | 話者ベクトルは 12 次元の文体統計（`:468`）で、仕様の 24D z_dia ではない。Δ_voice<0.15 の判定は実装。extractor・Director から未接続（呼び出しはテストのみ）。Well 1 の Δ_voice≥0.25 は目標値のフィールド(`hsmm_longform_director.py:84`)に留まり計測も強制もしない |
| 13 | Stage 1 の 4 基礎信号 s_m（Flux/Lattice/Void/Shock） | S10:111-115 | `extractor_24d.py:241` | `symplectic24d.ts:70` | 簡略化 | 構造は仕様に沿う（H_radial+知覚実在、モーラ自己相関+木対称、直交指示関数、D_jump+s⁻+促音）。結合係数は手設定（0.35/0.65、`31.0`、`2.2`、0.45/0.40/0.35 等）。範囲外（多重解像度較正） |
| 14 | Stage 2 対数 3 等分ウェーブレット（Micro/Meso/Macro） | S10:116-117 | `extractor_24d.py:326`。`v2/wavelet_hermite_encoder.py:87` は固定 3 レベル Haar(support 2/4/8)、長さを 2 冪へパディング | `symplectic24d.ts:216` | 簡略化 | 仕様は ln M の 3 等分でスケールを決める。`compute_logarithmic_scales`(`wavelet_hermite_encoder.py:295`) は定義済みだが Stage 2 から未使用。範囲外（較正）だが構造差として記録 |
| 15 | ISTA ソリトン分離（MCA）と RQA I_arch | S10:117 | `wavelet_hermite_encoder.py:162`, `:243` | `v2/waveletHermiteEncoder.ts` | 実装済み | Fourier 4 次+sech 辞書の ISTA を実装。Stage 2 での使用は q3, q7, q9, q11 のみで限定的 |
| 16 | 共役運動量 p は時間微分量のみ（文間 RMS 速度・CV_L、q のコピー禁止） | `.antigravity/rules.md` §1.2、S10:117 | `extractor_24d.py:416-419`: p = std(ウェーブレット詳細)×2.5×cadence_alive | `symplectic24d.ts:216` 以降 | 簡略化 | q の複製ではないが、規約の「文間 RMS 速度と CV_L」とは別定義。係数 2.5 は手設定 |
| 17 | Contrastive Fisher 蒸留 `fit_contrastive_fisher_metric` | S10:118-119, B10:120-121 | `manifold_geodesic.py:624` | `dynamics/manifoldGeodesic.ts:826` | 実装済み | G=(S_W+λI)⁻¹ と判別軸を算出。2 クラス閉形式で、仕様の一般化レイリー商 (S_W+λI)⁻¹S_B と同じ解空間 |
| 18 | G_style の本番結線（BAOAB 勾配・SMC 項③の計量） | S11:21, H13:3 柱 | `realtime_decoder.py:1612-1615`: 既定は単位行列。関数 17 の呼び出しはテストのみ | `adaptiveBaoab.ts:643` は受け取るのみ | 未実装 | 次元も不整合（Fisher は 24D、デコーダは 12D）。Phase 18 の候補 |
| 19 | 新特徴空間（多重解像度）を既定パスにする | H13 欠陥①, A13 実測 | `extractor_24d.py:466`: `use_multiresolution=False` 既定 | `symplectic24d.ts:368` 同様 | 意図的不採用 | A13: 旧 39/40(97.5%)、新 27/40(67.5%)、連結 48D 35/40(87.5%) で基準未達のため見送り。旧パスの 97.5% は手調整特徴で過学習の可能性があり未検証 |
| 20 | 3 相 QDA 距離 D_3phase（novel） | H13:1 柱 2、欠陥② | `manifold_geodesic.py:81` | `dynamics/manifoldGeodesic.ts:126` | 簡略化 | Mahalanobis 項は対角 Σ(`_FEATURE_STD`) を 24 で割った形で、完全共分散の QDA ではない。novel の σ 境界 1.40/3.90 と shock 0.60 は 40 チャンクから較正、独立データでの検証なし（A13） |
| 21 | QDA ドメイン定数 article / cot_reasoning | H13 欠陥②, S09:1 | `manifold_geodesic.py:43-74`（`_MU_ARTICLE`, `_MU_COT`, σ 境界） | `manifoldGeodesic.ts:86` | 簡略化 | 実データなしの推測値。Phase 18 で較正予定。現状は novel の μ を一部上書きしただけ |
| 22 | 5 熱力学相の分類（結晶/臨界/Void/乱流/Deadzone） | H13:1 柱 2 | `cleanroom_motifs.py`(重心)、`corpus_40_benchmark.py` | `dynamics/landauPhase.ts` | 簡略化 | 重心・境界は 40 チャンクの実測。LOOCV 結果はあるが独立コーパスでの検証なし |
| 23 | 意味測地線曲率・Brownian ゲート M_geodesic | H13:1 柱 2、S09:59 | `manifold_geodesic.py:304` | `manifoldGeodesic.ts:412` | 簡略化 | `_embed_sentence_unit`(`:195`) は hidden_states または frame が無いとき文字ハッシュ近似へ戻る。S09:59 は文字ハッシュ近似の撤去を指示 |
| 24 | 4 重積報酬 + 参照 PPL バリア M_semantic | S12:10-13 | `manifold_geodesic.py:467`、PPL 実測 `lora_grpo_forge.py:428-439`(LoRA 無効で CE) | `manifoldGeodesic.ts:606` | 実装済み | 式・閾値 45.0 は仕様一致。α_ppl=0.1 は仕様に値なし。`ppl_ref=None` のときバリア 1.0（CPU/模擬経路） |
| 25 | OPEN-08 Ghost History プリロード（1/f ゆらぎ, L=16, 分散 0.02）とベイズ収縮 α=m/(m+5) | S11:13-16 | `realtime_decoder.py:669`, `:437`, `:481-537` | `v2/adaptiveBaoab.ts` に該当処理なし | 簡略化 | Py は仕様一致。TS の BAOAB は Ghost History・収縮を持たない（Py/TS 不一致） |
| 26 | 5 ステージ Adaptive-BAOAB（B-A-O-A-B、γ=50/1.2、ε_syn=0.6） | S11:18-35 | `realtime_decoder.py:481`(`step_baoab_clause`) | `v2/adaptiveBaoab.ts:83` | 簡略化 | Py は仕様どおり 2 値摩擦。TS は連続摩擦（γ_base=0.1, γ_wall=100.0, 二乗ランプ）と質量スケール θ で定義が異なる（`:19-35`, `:61-71`） |
| 27 | Stage O の入力 H_syntax = 直近トークンの平均予測エントロピー | S11:28 | `realtime_decoder.py:2459-2463`: 「短い『。』終わりなら 0.3、それ以外 1.2」の代理値 | 同様の代理 | 簡略化 | トークン単位の Shannon エントロピーは LogitsProcessor 側(`:160`, `:401`)にあるが `step_clause` に渡されない |
| 28 | 節単位 Twisted SMC（P=4, T∈{0.65,0.72,0.80,0.88}, 3 項重み, 最良粒子採択） | S11:37-50 | `realtime_decoder.py:1369`(既定 `:1385-1392`)、重み `:2183-2260`、既定 `resample_method="map"` | `v2/adaptiveBaoab.ts:308`(LLM なし) | 実装済み | 式・温度・argmax は仕様一致。② 到達性は候補ごとに別 forward（`_compute_hf_logprob:1643`）で、仕様の 1 回バッチ O(1)（S11:48）ではない（計算量のみの差）。N=8、温度スプレッド、各種 logit バイアス等の仕様外拡張が多数ある |
| 29 | ② Terminal Reachability の着地アンカー W_end* | S11:47 | `realtime_decoder.py:2243-2260`: `target_end` 指定時のみ。未指定は 0 | なし | 簡略化 | 本番ランナーでアンカーを常時与えているかは未確認 |
| 30 | HSMM 3 Well（Narrative/Dialogue/Climax）とマルコフ遷移 | S11:56-59 | `hsmm_longform_director.py:89`、遷移行列 `:63-70` 付近 | `dynamics/hsmmLongformDirector.ts:69` | 簡略化 | Well 目標（CV_L 0.48/0.55/0.65、φ_void 0.65、κ 0.70）は仕様一致。遷移行列・Well の 12D 中心(`:34-57`)・基底運動量は手設定で、Tier-M バイナリ（データ推定）から読む経路がない |
| 31 | OPEN-09 Momentum Banking（蓄電と Climax での放出） | S11:60-61 | `hsmm_longform_director.py:256-325`（基底運動量・部分放出は仕様外の拡張） | `hsmmLongformDirector.ts`, `v2/adaptiveBaoab.ts:29`(MomentumBank) | 実装済み | 式 Δp_banked += max(0, p*−p̂) は実装 |
| 32 | OPEN-10 Wundt-Berlyne 逆 U 効用と不応期ゲート | S11:62-63 | `hsmm_longform_director.py:133`(効用は診断出力のみ `:656`)、`:232-247`(不応期) | `hsmmLongformDirector.ts:112` | 簡略化 | 仕様は「大きな ‖ṗ‖² 放出の直後」に条件付き。実装は Climax 直後に無条件で Well 0 へ戻す固定規則。a*=0.60 は手設定で、効用は選択に使われない |
| 33 | 3,000 字・6〜8 段落の長尺生成で Deadzone 回避・文途中切断 0 | S11:52-54, S09:87 | `run_v4_bidirectional_experiment.py:1326`(`run_longform_director`) | なし | 簡略化 | H14:18-19 の記録は 2,000〜2,350 字・文末「、」切断 0。3,000 字到達の記録は未確認。seed・サンプル数が少ない |
| 34 | HSMM Director の本番ランナー結線と WorldCraft 調停の統合 | H13 欠陥⑤⑦ | `run_v4_bidirectional_experiment.py:1326`, CLI `--longform`(`:1577`) | なし | 実装済み | A13 に Colab L4 での LLM 経路実測あり（Qwen2.5-1.5B）。制約あり/なしの本文差は 1 回目では出ず、2 回目の補正後に分岐を確認 |
| 35 | AttachedLoRALinear と実 Attention 層（q_proj/v_proj 最終 4 層）への直結、ダミー `LinearLoRAAdapter` 撤去 | S12:15-41, S09:17 | `lora_grpo_forge.py:42`、`attach_lora_to_model:177`、ランナー `target_layers=4`(`run_v4:1068`) | 対象外（Python のみ） | 実装済み | `LinearLoRAAdapter` は本体コードから消えている。関数既定は全層（`target_layers=None`）でランナーが 4 を指定 |
| 36 | 本物の GRPO 学習ループ（G ロールアウト、群相対アドバンテージ、トークン KL、素の generate で評価） | S12:43-52 | `lora_grpo_forge.py:347` | 対象外 | 簡略化 | 損失は仕様式（Â·平均 log π − β·KL, β=0.04）。CPU 模擬経路は擬似値（`sim_ppl=25+4i`、`:582`）。完走・CV_L/D/R 改善の実証は範囲外で未検証 |
| 37 | 独立 LLM Judge（自前特徴量での自己採点の撤去） | S09:105, H13 欠陥⑦ | `run_v4_bidirectional_experiment.py:287`(`evaluate_llm_judge`)。`judge_fn` 未指定時は offline ルーブリック | 対象外 | 簡略化 | A13: 既定は文長・接続詞・bigram 多様性・定型句のルーブリックと手設定ベースライン定数で LLM 審査ではない |
| 38 | SymplecticActivationSteerer（12 軸較正 `calibrate_12_element_axes`、forward hook 注入、L_mid=⌊0.65N⌋） | S12:54-63 | `activation_steerer.py:33`, `:95`, `:243` | 対象外 | 簡略化 | 式・層位置は仕様一致（リッジ+QR 直交化、h+α‖h‖V Δq）。既定の較正コーパスは自作 12 文（`:112-125`）で d_model≫12 のため軸が 12 サンプルの張る空間に限られる。仕様はパブリックドメインコーパス。本番ランナーからの呼び出しなし（テストと export のみ）。効果の記録なし |
| 39 | OSWE 極座標フーリエ位相アライメント（基準位相 3π/4） | S12:67-69 | `usir_forge.py:37-60` | `schema/usirForge.ts:55`, `:85` | 実装済み | c₁ の式・θ₀=3π/4 とも一致 |
| 40 | 4,096 B 3 モード GMM バイナリ（μ_k, Cholesky 300, π, T, 署名） | S12:70-71 | `usir_forge.py:174`(レイアウト `:30`), `OSWEStreamingCompiler:318`。遷移行列・混合重みはデータから推定 | `usirForge.ts:173`, `OSWEStreamingCompiler:224` | 簡略化 | レイアウトとサイズは仕様どおり。署名は SHA-256 ではなく CRC32（`:4092-4095`）。Σ=a·S+(1−a)·tr(S)/24·I, a=0.75 は手設定（A13）。蒸留物を使う下流の測定は未実施。下流消費者なし（所見 3） |
| 41 | Wasm SIMD 距離・内積カーネル | S09:118, S12:7 | 対象外 | `packages/usir-wasm/cpp/usir_simd.cpp`, `src/wasmLoader.ts:154-170` | 簡略化 | `.wasm` バイナリは未ビルド（`find` で 0 件）。ローダは Float32Array の JS フォールバックで動作 |
| 42 | Toulmin 論理ポテンシャル U_Toulmin（Claim/Data/Warrant、深さ D_max>6、主述ねじれ、用語一義性） | S12:76 | `nishiki_unified_field.py:144` | なし | 簡略化 | Claim/Data/Warrant は正規表現・語彙カウント。トートロジー検出は文字 bigram Jaccard>0.70 かつ新規概念≤10%（A13 ④、粗い）。主述ねじれは固定語のみ。用語一義性は未実装。TS なし |
| 43 | 主述ねじれの SyntaxTreeFrame ベース一般化 | H13 欠陥④ | 試作後に撤回 | なし | 意図的不採用 | A13: 「彼は走ったが、私は疲れた」を誤検出し、真陽性も作れない設計だったため撤回 |
| 44 | 統一場 F_total=(1−τ)U_Toulmin+τF_USIR、τ_genre の連続制御 | S12:77 | `nishiki_unified_field.py:276` | なし | 簡略化 | 式は一致。F_USIR(`:261`) の重みは手設定。τ（0.1/0.5/0.9）は呼び出し側指定でジャンルからの導出はない。TS なし |
| 45 | 零空間射影 P_null=I−J⁺J（SVD、一般線形結合、Tikhonov 版） | S12:79, H13 欠陥⑤ | `nishiki_unified_field.py:439-533` | `dynamics/worldcraftArbitrator.ts:134`, `:182`, `:200` | 実装済み | Py/TS 両方あり、`test_worldcraft_parity.py` / `worldcraft-parity.test.ts` で整合を固定。Tier 1 ハードマスクは Py のみ(`:367-412`) |
| 46 | オントロジー制約から J_C を導出するコンパイラ | S12:79, H13 欠陥⑤ | `compile_constraint_jacobian` (`nishiki_unified_field.py:413`) | `worldcraftArbitrator.ts:206` | 簡略化 | モード(evidence/logic_voice/stillness)→凍結要素の対応は設計上の仮定で、データによる検証なし。文面キーワード選択は手書き(`_CONSTRAINT_MODE_CUES`)。k=0 のとき P=I |
| 47 | 視覚関手 F_visual（Gabor 4 方向、色エントロピー、キアロスクーロ、Quadtree、Saliency）と逆写像 | S12:81 | `visual_functor_forward_24d:698`、`_inverse:722`（特徴 dict を受ける） | `bridges/crossModalFunctors.ts:253`(画像からの抽出), `:400`, `:420` | 簡略化 | 画像→特徴の抽出は TS のみ。Py は特徴 dict 入力。写像係数は手設定。Lab/Saliency Scanpath の抽出の詳細は本監査で未精査 |
| 48 | 音楽関手 F_music と逆写像（BPM・音列生成） | S12:82, H13 欠陥⑥ | `music_functor_forward_24d:771`, `_synthesize_music_sequence:806` | `crossModalFunctors.ts:688`, `:722`, `:752`, `:816` | 簡略化 | 固定ペンタトニックは撤廃済み（16 スロット格子、黄金比による決定論的選択）。ノート列からの特徴抽出は TS のみ。写像係数は手設定。Py/TS で 12 音の一致をテストで固定（A13） |
| 49 | 共役運動量 p の全 12 次元（窓系列入力） | H13 欠陥⑥ | `nishiki_unified_field.py:641`(`momentum_from_window_series`) | `crossModalFunctors.ts:291` | 実装済み | 窓間 RMS 変動速度。視覚は g×g ブロック列、音楽は 4 拍小節窓（A13） |
| 50 | 共役運動量 p（要約 q のみの入力） | S12:81-82, H13 欠陥⑥ | `nishiki_unified_field.py:680`(`momentum_from_summary`)、定数 `:660-665` | `crossModalFunctors.ts:319` | 簡略化 | 固定結合 Λ（`EQ_W_WITHIN=0.3`, `EQ_W_CROSS=0.1`）、`EQ_SCALE=2.2`（51 本の合成小節系列で較正）。時間情報がなく、q からの再構成値。`EQ_GROUPS=((0,3,7),(1,4,8),(2,5,9),(6,10,11))` は規約の 4 族（Flux 0,4,8 / Lattice 1,5,9 / Void 2,6,10 / Shock 3,7,11）と一致しない。意図は未確認。Phase 18 で Tier-M ゲノム経路を追加予定 |
| 51 | 6 大モダリティ普遍関手（視覚・音楽以外） | S12 表題, S09:36 | なし | `bridges/multimodalAudioVideo.ts` 等は存在 | 未実装 | S12 本文に明記された関手は視覚と音楽の 2 つのみ。残り 4 つの仕様記述が `docs/specs/` に見当たらず、対応関係を特定できていない |
| 52 | Py/TS 数値パリティ（誤差<1e-5）と Phase A 専用テスト | S09:64-65 | `tests/test_phase_a_linguistic_gates.py`, `test_phase_a_multiresolution.py` | `tests/phase-a-multiresolution.test.ts`（`phase-a-linguistic-core.test.ts` は無し） | 簡略化 | 仕様の `test_phase_a_linguistic_core.py` / `phase-a-linguistic-core.test.ts` は未作成。言語ゲートの Py/TS 値一致テストなし（行 8〜12 の TS 欠落と整合） |
| 53 | 終点ブリッジ引力 φ（Schrödinger bridge）の BAOAB への注入 | S11:21 | `bidirectional_bridge.py:115`、`realtime_decoder.py:710`, `:773`（`phi_bridge`） | `dynamics/bidirectionalBridge.ts` | 簡略化 | 関数と注入点は両言語にある。ランナーが φ を常に与えているかは未確認 |

## 未完了（優先順）
1. **QDA ドメイン定数の較正（行 21）**: article / cot_reasoning を実データで較正し、独立データで検証する。Phase 18 で対応中。
2. **Tier-M ゲノムの消費者を作る（行 18, 30, 40, 50）**: Director の遷移行列・Well 中心、QDA の μ/Σ、`G_style`、`momentum_from_summary` の Λ を `usir_forge` の出力から読む経路。現在はすべて手設定または単位行列。`momentum_from_summary` は Phase 18 で着手。
3. **Fisher 計量の本番結線（行 17-18）**: 24D と 12D の次元整合を決めたうえで `G_style` に使う。結線しても生成品質が改善するかは未測定。
4. **TS 側の欠落（行 8-12, 25-26, 42, 44）**: OPEN-01/02/04、Ghost History とベイズ収縮、BAOAB 摩擦定義、Toulmin/統一場。Py を正本にして移植するか、TS 対象外と文書化するか決める。
5. **Phase A 仕様の正本決定（行 1-3, 52）**: S10 と B10 のどちらかを正本として文書に記す。janome 優先の挙動を cleanroom に揃えるかも決める。
6. **H_syntax の実エントロピー化（行 27）**: LogitsProcessor のトークンエントロピーを `step_clause` に渡す。影響は未測定。
7. **不応期を ‖ṗ‖² 条件付きにする（行 32）**、Wundt-Berlyne 効用を選択に使うかの判断。
8. **Activation Steering の較正コーパスとランナー結線（行 38）**: 12 文の自作コーパスの置換。効果測定が先に必要。
9. **OPEN-04 を extractor/Director に接続（行 12）**: 話者ベクトルを 24D にするか、仕様側を 12D に改めるか。
10. **細部**: Tier-M 署名 SHA-256（行 40）、Wasm ビルド（行 41）、Judge の LLM 化（行 37）、`EQ_GROUPS` の意図確認（行 50）。

範囲外（再掲）: 多重解像度抽出器の較正（行 13-14, 19）、Table 1 / Table 2、GRPO の完走と効果実証（行 36）。

## 次の一手
- 行 21 の QDA 定数較正（Phase 18）を進める。並行して、行 2 の正本決定（S10 か B10）をユーザーに確認する。
- 進捗: 監査表作成済み（53 行）。コード変更・コミットなし。未検証: 行 3, 47, 53 は逐条または実走の照合が未実施。
