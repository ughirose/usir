# USIR v5 × Nishiki Stage P-4 設計書: Plotailor「離見の見」診断器

- 版: v1（2026-11-18）。設計のみ。コード・計測スクリプトは追加していない。
- ブランチ: `stage/p4-design`（基点 `feat/fable-v5-shootout` `3c23915`）
- 状態: 完了（判断待ちの項目あり。§6）
- 根拠の表記: 「確認」= 本セッションでファイルを読んだ事実。「未確認」= 読んでいない。「推測」= 根拠が間接的。

## 0. 前提と確認済みの事実（食い違いを含む）

| # | 事実 | 出所 |
|---|---|---|
| F1 | 本流 `3c23915` には `scripts/detector_v3/`、`scripts/detector_v4/`、`scripts/p2b/p2b_analysis.py`、`scripts/p2c/p2c_features.py`、`data/detv4_scene_signals.npz`、`data/p2c_scene_profiles.npz`、`data/aozora_p2{b,c}_manifest.json` が無い。これらは未マージの `stage/detector-v4`（`8ad54fa`）にだけある。本書はそれらを `git show stage/detector-v4:<path>` で読んだ。 | `git ls-tree`、`git merge-base --is-ancestor`（not ancestor） |
| F2 | `data/aozora_cache/`（原文）は gitignore 済みでローカルにある。 | `git check-ignore -v` |
| F3 | 検出器 v4 の登録済み p_OP は 97.0。探索格子 80.0〜97.0（0.5 刻み）の最大値で、「最大の p」規則は常に格子上限に張り付く。最小 p 規則で同じ行を読むと、v4 は 92.5（平均 FPR 0.0853、SE 0.0094、上界 0.0973）、abl_ii は 93.0（0.0845、0.0101、0.0974）。 | `stage/detector-v4:scripts/detector_v4/pop_registered.json` の `rows` |
| F4 | v4 構成の合算 AUC は 0.8603、作品クラスタ bootstrap 95% CI は [0.8461, 0.8726]（下限 < 0.85）。abl_ii（q3' を v3 感覚語率に置換）の合算 AUC は 0.8822 で、**CI は未計測**（`measure.py` は `c.name == "v4"` のときだけ `ci=True`）。abl_ii の型別 AUC は ending 0.9779、comma 0.9267、sensory 0.7420。 | `detv4_results.json`、`measure.py:main` |
| F5 | AUC の陽性例は人為摂動（語尾置換・読点削除・感覚語削除）の合成データ。人間の違和感への感度は測っていない。盲検評価（§4）が初めて測る。 | `p2b_analysis.py:perturb` |
| F6 | v3 の感覚語は「語」ではなく文字集合。`PHYSICAL_SENSE_INTERVALS` は 82 区間・201 コードポイント（`python/usir_py/extractor_24d.py:40`）。 | 確認 |
| F7 | nishiki に `sadame-core` / `nagare-core` / `awase-core` は存在しない。仕様書 §2.2 の担当クレート名と実体が違う。実体は §1.1 の表。 | `Cargo.toml` workspace members |
| F8 | Sadame / Nagare / Awase の Rust 実装（`usir-metrics`）は注釈済み `SceneAnnotation` を入力にする。生テキストから注釈を作る Rust コードは無い。注釈を作る決定論ヒューリスティクスは Python の `aozora_scene_splitter.py`（`compute_heuristic_signals:695`、`build_scene_annotation:978`、`extract_scene_characters_and_edges:852`）にある。 | 確認 |
| F9 | 場面分割 `split_work_into_scenes`（`aozora_scene_splitter.py:466`、min 200 / max 1600 / target 750）の Rust 版は無い。分割後の場面文字列は行を strip して `"\n"` で連結するため、元入力の部分文字列とは限らない。長さの判定は読点を除いた `effective_len` を使う。 | 確認 |
| F10 | nishiki の Wasm は `audit_markdown` / `link_markdown_files` / `audit_dialogue_turns` の 3 関数のみ。`diagnose` は無い。CLI サブコマンドにも `diagnose` は無い。 | `crates/nishiki-wasm/src/lib.rs`、`crates/nishiki-cli/src/main.rs` |
| F11 | Gate 3b は `GATE3B_FROZEN`（B-9、B-13 報告）。盲検設計は引継ぎの次マイルストーンとして残っている（`docs/handoff/HANDOFF.md:94`: 評価者の初稿マークと Cohen's κ）。 | 確認 |

## 1. 「離見の見」6 レンズの同定（タスク 2.1）

### 1.1 同定結果

仕様書（`usir_v5_nishiki_complete_spec_and_plan_20261110.md` §2.2）は、6 レンズを 6 極と 1 対 1 に対応させている。4 ミラー案は旧案の不備として撤回されている。したがって 6 レンズ = Kasane / Sasae / Kamae / Sadame / Nagare / Awase。持ち越し異常検出器は 6 レンズに入らない。完了条件 1 の「6 極＋持ち越し異常検出器」は、**6 レンズ + 検出器チャネル 1 本 = 出力 7 系統**と読む（判断待ち J2）。

| # | レンズ（仕様書の名称） | 実装の所在（確認済み） | 入力単位 | 出力（スコア） | 出力（スパン） |
|---|---|---|---|---|---|
| 1 | Kasane 呼吸・五感偏り | `kasane-core`、`nishiki-cli/src/kasane_adapter.rs:evaluate_kasane` | 文字窓（250 字、刻み 125）を原稿全体に適用 | `dc_6d`、`semantic_tension`、`ac_4d`、`attractor_kind`、`deadzone_count` | `stagnation_spans`（文字区間）。既存の出力をそのまま使える |
| 2 | Sasae 設定・時系列・視点汚染 | `sasae-core`、`nishiki-cli` の `run_full_audit` | 文・段落（未確認） | 未確認（`AuditReport` の構造は未読） | 未確認。P-4b の棚卸しで確定 |
| 3 | Kamae 説明過多・秘する花・迎合 | A: `kamae-core`（auditor / transaction）。B: `usir-metrics/src/kamae.rs`（hook survival） | A: 文・段落（未確認）。B: 作品全体の場面列 | A: 未確認。B: `hook_survival` 等のスカラー | A: 未確認。B: 宣言台帳が無い無ラベル原稿では作品単位のスカラーのみ |
| 4 | Sadame 人物の操り人形化・公理ブレ | `usir-metrics/src/sadame.rs`（`polyphony_inter`、`intra_variance`、`sadame_total`） | 作品（`CharacterStalk` と `DialogueEdge` の列） | `E_inter`、`E_intra`、`sadame_total` | 人物単位。文字区間は未定義（注釈が人物名しか持たない） |
| 5 | Nagare 序破急・展開停滞 | `usir-metrics/src/nagare.rs:analyze_work` | 作品（場面列の `event_logp` / `mood_logp`） | `phi_zeami`、事象・風情別の運動学量 | `still_runs`（遷移番号の範囲）→ 場面範囲 → 文字区間に写像 |
| 6 | Awase 感情ベタ張り・対位法 | `usir-metrics/src/awase.rs`（`irony`、`detachment`）、`usir-schema` の `PolarityDivergenceSpan` | 場面（`AwaseAnn`）。極性乖離は文 | `irony`、`detachment`（場面ごと） | `PolarityDivergenceSpan.{dialogue_span, narration_span}`（`CharSpan`、Unicode スカラー値、半開区間） |
| — | 持ち越し異常検出器（レンズではない） | 未実装（§3）。Python 正本は `stage/detector-v4` | 場面 | 場面ごとの `u_ending` / `u_comma` / `u_sensory` と max スコア | 場面全体の文字区間（場面内の局在は出さない） |

### 1.2 同定が曖昧な点（判断待ちへ）

- J3: Kamae レンズの実体が 2 通りある。仕様書の役割記述（説明過多・秘する花・迎合 `E005`）は A（`kamae-core`）に対応する。`usir-metrics` 側の Kamae は伏線の生存率（B）で別物。
- J4: レンズ 3（B）・4・5・6 は注釈（F/f 価数、事象・風情コード、人物の stalk、フック）を要する。生テキストから作る Rust コードが無い（F8）。
- 仕様書のレンズ 2（Sasae）の「ゼロ代名詞・知識境界」が `sasae-core` のどの関数に対応するかは、`sasae-core/tests/zero_anaphora.rs` の存在以外は読んでいない。P-4b 冒頭の棚卸しで確定する。

## 2. `nishiki diagnose` の I/O 契約（タスク 2.2）

### 2.1 呼び出し

```text
CLI : nishiki diagnose <FILE|-> [--format json] [--scene-split auto|none] [--artifact <detector.json>]
Wasm: diagnose(text: &str, options_json: &str) -> String   // 戻り値は JSON 文字列。失敗は {"error": "..."}
Core: nishiki_cli::diagnose::run_diagnose(text: &str, opts: &DiagnoseOptions) -> DiagnoseReport
```

CLI と Wasm は `run_diagnose` を呼ぶだけの薄い皮にする。`nishiki-wasm` は既に `nishiki_cli` に依存しており（`lib.rs:8`）、同じ構成で足せる。

### 2.2 入力規則

- 入力は UTF-8 テキスト。BOM は除去する。改行は `\n` に正規化するが、正規化前後の文字オフセット対応表を保持する（出力のオフセットは**呼び出し側が渡した元テキスト**基準、Unicode スカラー値、0 起算、半開区間 `[start, end)`。`usir-schema::CharSpan` と同じ）。
- 場面分割: `--scene-split auto` は Python `split_work_into_scenes`（min 200 / max 1600 / target 750、`effective_len` は読点除去後の長さ）の Rust 移植を使う。移植は場面ごとに元オフセットの区間を返す（F9 のとおり場面文字列は部分文字列と限らないため、検出器は場面文字列で特徴を計算し、スパンは元オフセットの区間で報告する）。`none` は全体を 1 場面とする。
- 移植の注意（照合対象）: Python の `str.strip()` / `\s` と Rust `trim()` / regex `\s` の空白集合は一致しない可能性がある（推測: Python の `isspace()` は U+001C〜U+001F を含む）。境界フィクスチャで確認する。分割側の正規表現（`RE_HEADING`、`RE_TIME_MARKER`、`RE_SYMBOL_DIVIDER`）は先読み・後読みを含まず `regex` クレートで書ける。`SENT_SPLIT`（`(?<=...)`）は注釈ヒューリスティクス側で使われ、Rust 移植では後読みなしの手書き分割にする。

### 2.3 出力スキーマ `nishiki.diagnose/1`

```json
{
  "schema": "nishiki.diagnose/1",
  "input": {"chars": 12345, "scenes": 9, "scene_split": "usir_v5_splitter@1"},
  "scenes": [{"index": 0, "span": {"start": 0, "end": 812}}],
  "lenses": [
    {"id": "kasane", "pole": "Kasane", "status": "ok",
     "scores": {"deadzone_count": 2, "semantic_tension": 0.41},
     "spans": [{"start": 120, "end": 560, "scene": 0, "score": 0.0,
                "rule": "stagnation", "basis": {"window": 250, "hop": 125}}]},
    {"id": "sasae",  "...": "..."},
    {"id": "kamae",  "...": "..."},
    {"id": "sadame", "...": "..."},
    {"id": "nagare", "...": "..."},
    {"id": "awase",  "...": "..."}
  ],
  "anomaly": {
    "detector": "carryover@p4-1",
    "artifact_sha256": "<hex>",
    "p_op": 92.5, "threshold": 0.0,
    "reliability": "ok|low",
    "scenes": [{"scene": 0, "u": [0.0, 0.0, 0.0], "score": 0.0, "flag": false,
                "dominant": "ending|comma|sensory"}],
    "spans": [{"start": 0, "end": 812, "scene": 0, "score": 0.0,
               "rule": "carryover_max", "basis": {"dominant": "ending"}}]
  },
  "warnings": []
}
```

- `lenses` は常に 6 要素を `kasane, sasae, kamae, sadame, nagare, awase` の順で出す。実装できていない、または入力が足りないレンズは `"status": "unavailable"` と `"reason"` を持ち、`scores` と `spans` は空にする（欠落させない。パリティ比較の形を固定するため）。
- `spans[].score` は各レンズの定義による根拠スコア。「根拠」は `rule` と `basis` に機械可読で持たせ、評価者向けの文章は盲検レポート側で作る。
- 数値は有限値のみ。非有限値が出た場合は `warnings` に記録し、その項目を `unavailable` にする。
- 数値の丸め: 出力時に絶対 `1e-9` 単位で丸める（§2.4 の理由による）。スパンの位置・フラグ・件数は整数・真偽値で、丸めの対象外。

### 2.4 CLI/Wasm パリティの担保

1. **同一コア関数**: 両方が `run_diagnose` を呼ぶ。出力 JSON の組み立て（`serde_json`）も共有する。
2. **ゴールデン照合**: `tests/fixtures/diagnose/*.json`（入力テキストと期待 JSON）を置く。ネイティブ側は `cargo test`、Wasm 側は `wasm-bindgen-test`（Node）で同じフィクスチャを読み、**JSON 文字列のバイト一致**を要求する。フィクスチャは、短文・空入力・サロゲートペア（`nishiki-wasm` の既存テストにある `𠮷`）・絵文字・場面 1 個・場面多数・全角空白・CRLF・極端な長行を含める。
3. **数値の再現性**: ネイティブと wasm32 で `exp` / `ln` の最下位ビットがずれる可能性がある（推測。未検証）。対策として、検出器コアの超越関数は `libm` クレートを両ターゲットで使う（判断待ち J10）。それでも差が出る場合に備え、丸め（§2.3）を入れる。閾値直近で丸めがフラグを割る可能性は残るので、フィクスチャに閾値近傍の場面を意図的に含める。
4. **判定基準**: スパンの位置・フラグ・件数は完全一致。連続値は丸め後に完全一致。丸め前の差の実測値は報告に記録する（B-18 の前例: 標準条件で `1e-8` 以下でも、小スケール条件で `6.7e-2` の乖離が出た。仕様の曖昧さが原因だった）。

## 3. 持ち越し異常検出器の Rust 移植仕様（タスク 2.3）

正本は `stage/detector-v4:scripts/detector_v4/detector_v4.py`（156 行）と `scripts/detector_v4/signals.py`、`scripts/detector_v3/signals.py`、`scripts/p2c/p2c_features.py:surface`、`p2b_analysis.py:_SENSE`。ただし v13 で確定した最終構成は正本と異なる点がある（§3.2 の「変更点」）。**移植対象は v13 構成**で、正本の v4 そのままではない。

### 3.1 最終構成（v13）

- 3 量（場面単位。向きは「下側が異常」。番号は v4 の信号名で、v3 の npz 列名とは異なる: v4 q1 = v3 q3（語尾）、v4 q2 = v3 q1（読点）、v4 q3 = v3 q2（感覚語率））
  1. q1 語尾多様性: 文を `_SENT = [^。！？]+[。！？]+[」』）)]*` で切り、末尾の `」』）)` と `。！？` を除いた末尾 3 文字の異なり数 / 文数。文数 0 なら 0。`sgn = −1`。
  2. q2 読点密度: `count("、") / len`（`len` は場面文字列の長さ、最低 1）。`sgn = −1`。
  3. q3 場面全体の感覚語率: 感覚文字（82 区間・201 コードポイント、凍結）の個数 / `len`。`sgn = −1`。v13 で分散安定化を適用（§3.3）。
- 語リスト（文字集合）は凍結。DETV3 / DETV4 と同一。追加・削除は P-4 の範囲外。
- 作品内 LOO 経験ベイズ正規化（m0 = 8、`FLOOR = 1e-6`）と、校正分布による片側分位スコア、K = 3 の max 集約。
- 移植から**外す**もの: Hurdle 重み、A/B の 2 部複合、地の文限定の感覚語率、スパン長（`detector_v4.py` の `hurdle_weight`、`composite_r`、`IA`/`IB`）。v13 で不採用が確定している。

### 3.2 関数ごとの仕様（正本の行為を数式でなく手順で固定する）

| 関数（Python） | 仕様 | Rust 側の注意 |
|---|---|---|
| `derive` | 生行列から [q1, q2, q3] を取り出す | 列順は正本の `COLS` ではなく本書の順に固定し、フィクスチャに列名を持たせる |
| `fit` | `mu_g` = 列平均。`sig_w` = 作品ごとの不偏分散（ddof=1、場面 2 以上の作品のみ）を自由度 `n−1` で加重平均し平方根（下限 `FLOOR²`）。各作品を LOO 正規化した Z の列を昇順ソートして校正分布 `cal` とする | 浮動小数点の加算順序を Python（numpy）の合計順と一致させる必要はないが、許容誤差の測定（§3.4）で差を実測する |
| `loo_normalize` | 作品内の場面 i について、他場面 k = n−1 個で `m` = 平均、`s2` = 不偏分散（k≥2 のみ、他は 0）。`mu = (k·m + m0·mu_g)/(k + m0)`。`w = max(k−1, 0)`、`var = (w·s2 + m0·sig_w²)/(w + m0)`。`z = (x − mu)/max(√var, FLOOR) · sgn`。k = 0 のときは `mu_g`、`sig_w²` をそのまま使う | 正本は `s2 = (Σx² − x_i² − k·m²)/(k−1)` の 1 パス式で、桁落ちしうる。**同じ式で移植する**（Welford 等に変えない）。変えると正本と数値が合わなくなる。差は §3.4 で測る |
| `tail_p` | `p = (1 + cnt)/(n_cal + 1)`、`cnt` = `cal` のうち `z` 以上の個数（`searchsorted(left)`） | 非有限 `z` は `u = 0` とせずエラー（§2.3 の `unavailable`） |
| スコア | `u_j = −ln(tail_p_j)`、`score = max(u_1, u_2, u_3)` | `ln` は `libm` |
| `p_op_procedure` | 格子 80.0〜97.0（0.5 刻み）の各 p で、作品ごとの leave-one-work-out 閾値 = 他作品の場面スコアの p パーセンタイル（numpy 既定の線形補間）。`FPR_w` = 作品 w の場面でスコア ≥ 閾値の割合。`mean_w FPR_w + 1.28 × SE_w ≤ 0.10`（SE は作品間の不偏標準偏差 / √作品数）を満たす**最小の p**を `p_OP` とする | 変更点: 正本は「最大の p」で格子上限に張り付く（F3）。最小 p に変える。満たす p が無い場合の挙動は判断待ち J6（推奨: 失敗として扱い、floor に黙って落とさない） |

百分位は numpy の `method="linear"`（位置 `(n−1)·p/100` の線形補間）で実装する。AUC は `scipy.stats.rankdata`（平均順位、同点は平均）の式 `(Σ rank_pos − n1(n1+1)/2)/(n1·n0)` と一致させる。

### 3.3 感覚語の分散安定化: ポアソン下側片側 p 値を採る（暫定）

v13 の候補は「Anscombe 平方根変換」か「ポアソン下側片側 p 値」。根拠となる実測（`data/detv4_scene_signals.npz` の `orig`、1,877 場面、**地の文のみ**の感覚語個数 = `S2a × L_narr`。q3 は場面全体の率なので近似であり、全体個数ではない）:

| 量 | 値 |
|---|---|
| 場面あたり個数の平均 | 13.07 |
| 5 / 25 / 50 / 75 / 95 パーセンタイル | 2 / 6 / 12 / 18 / 30 |
| 個数 0 の場面 | 1.6% |
| 個数 < 4 の場面 | 11.4% |
| 個数 < 10 の場面 | 41.3% |
| `L_narr`（地の文長）の 5 / 50 / 95 パーセンタイル | 124 / 528 / 1037 字 |
| 作品内の 分散/平均 比（場面 ≥ 5 の作品）の中央値、四分位 | 4.44、[3.16, 5.79] |

判断:

- 場面長が 8 倍以上ばらつくため、率の分散は場面ごとに異なる。LOO 正規化は作品内で 1 つの分散を使うので、この不均一を除けない。ポアソン下側 p 値は期待個数 `λ = μ̂ × 場面長` を場面ごとに作る（露出を内包する）ため、不均一を直接扱える。v4 の `e_count = μ_LOO × L_narr / 1000` も露出を個数に直す同じ考え方。
- Anscombe は小さい個数で近似が悪い（一般に λ が数個以下で不正確。個数 < 4 が 11.4%）。ポアソン p は離散の厳密な下側確率で、小個数でも定義が崩れない。
- 欠点: 分散/平均比が 4.4 で過分散（ただし場面長の変動を含む値で、純粋な過分散ではない）。ポアソン p は過信になる。校正分布による `tail_p` がその後に単調写像として吸収するので、順位ベースの AUC への影響は小さいと予想するが、**未測定**。
- Rust 実装: `P(X ≤ k | λ) = Σ_{i≤k} exp(−λ + i ln λ − lnΓ(i+1))` を直接和で計算する（k は最大でも数百）。外部クレートを増やさず、`scipy.stats.poisson.cdf` との差を §3.4 で測る。
- 率の推定 `μ̂`: v4 の `loo_mean_rate`（他場面の平均率、場面 1 つなら全体平均）は縮小推定を使わない。正規化 q1・q2 は m0 = 8 の縮小を使う。ポアソン側も同じ縮小 `μ̂ = (k·m + m0·μ_g)/(k + m0)` にするのが整合的だが、**未測定の提案**。P-4a で縮小あり/なしを事前登録のうえ比較する。

結論: 暫定でポアソン下側片側 p を採る。Anscombe は対照として P-4a で同一プロトコルに並べて測り、事前登録した規則（合算 AUC の 95% CI 下限が高い方、ただし FPR 制約を満たすこと）で最終確定する（判断待ち J5）。

### 3.4 Python との数値照合計画

1. **フィクスチャ**（`stage/detector-v4` の評価データから、数値のみで生テキストを含まない）
   - 入力: 場面ごとの生量 `[q1, q2, q3, 場面長 L_total]`（現行の npz に `L_total` は無く `L_narr` のみ。P-4a の `signals` 拡張で追加する）を、`orig` と摂動 3 種（主）・用量 5 種（副）で JSON 化。作品 ID・fold 割当（5 分割、seed 11）付き。
   - 中間量: `derive` 出力、`mu_g`、`sig_w`、LOO 正規化後の Z、`cal` 配列、`tail_p`、`u`、`score`、各 fold の閾値、`p_op_procedure` の `rows`（格子の全行）。
   - 最終量: 合算 AUC と型別 AUC、bootstrap CI。
2. **テキスト層**（生量を原文から作る部分）の照合は別フィクスチャ。原文は著作権・クリーンルーム規約により**リポジトリに置かない**（`data/aozora_cache/` はローカルのみ）。そのため、原文由来の照合は「Python が出した生量」と「Rust が同じ原文から出した生量」をローカルで突き合わせ、コミットするのは不可逆な数値の最大差だけにする。合成テキスト（リポジトリ内で生成）の小フィクスチャは、サロゲートペア・全角・空文・句点無しなどの境界用にコミットする。
3. **許容誤差の決め方**（事前登録）: まず Python 同士の再現性の底値を測る。同じ入力を ①加算順序を変えた、②`np.longdouble` で計算した、の 2 通りで走らせ、各中間量の最大絶対差 `ε_py` を求める。Rust の許容誤差は `max(100 × ε_py, 1e-12)` とする。判定を割りうる量（閾値との比較、`≤ 0.10` の境界）は、差の有無でなく**判定の完全一致**を条件にする。一致しない場合は、B-18 の前例どおり原因（仕様の曖昧さか実装差か）を切り分けて報告する。
4. 正本が変更点（最小 p 規則、Hurdle 等の除去、ポアソン変換）を含むため、Rust との照合の前に **P-4a で v13 構成の Python 参照実装を作る**（§5）。v4 の `detector_v4.py` に Rust を合わせない。

### 3.5 CI 下限の付随確認の手順

確認する量: 同一プロトコル（56 作品・1,877 場面、作品単位 5 分割、fold seed 11、内側 seed 11+fold、bootstrap B = 2000・seed 7）で、v13 構成の合算 AUC と作品クラスタ 95% CI。合格線は CI 下限 ≥ 0.85（F4 のとおり、現存の測定は v4 構成で 0.8461、abl_ii は CI 未計測）。

**手順は Python 先行**を推奨する（判断待ち J1 と連動）:

1. P-4a: v13 構成を Python で実装し、上記プロトコルで測る（AUC・CI・FPR）。p_OP は最小 p 規則で再登録する。この段で CI 下限が 0.85 を下回れば、Rust 移植に着手せず設計に戻す（移植コストの空費を避ける）。
2. P-4b: Rust 移植後、同じ fold・同じ入力で AUC と CI を再計算し、Python の値と照合する（Rust 側の bootstrap は同じ乱数列を再現できないため、**CI の一致は求めず、AUC 点推定の一致のみを条件**にする。CI は Python 値を正本とし、Rust 側の bootstrap は参考値）。
3. 付随確認の結果は、合格・不合格にかかわらず数値を全件報告する。

## 4. 盲検ハーネスの仕様（タスク 2.4）

目的: 評価者（原稿の書き手本人または協力者）が、システムの出力を見ずに自分の違和感マークを付け、後でシステム出力と突き合わせて一致度を測れる形にする。完了条件 2「無ラベル原稿の投入・違和感スパン出力」を満たす。

### 4.1 入力

- `manuscripts/<id>.txt|md`: ラベルの無い原稿。ID は `blind-run` が原稿ごとにランダムな不透明 ID（例: `m-7f3a`）へ置換する。元ファイル名との対応は `key.json` にだけ持つ。
- 原稿は `.gitignore` 対象のローカルディレクトリに置く。コミットするのは不可逆な数値（件数、κ、フラグ率）だけ。原文・スパン内容・場面テキストは残さない（B-9 / クリーンルーム規約）。

### 4.2 コマンド

```text
nishiki blind-run   --in <dir> --out <dir> [--seed S] [--artifact <detector.json>]
nishiki blind-score --out <dir> --marks <rater_marks.json>
```

実装は `nishiki-cli` のサブコマンドとし、内部で `run_diagnose` を使う（Python への依存を作らない）。

### 4.3 出力（`--out`）

| ファイル | 内容 | 評価者に見せるか |
|---|---|---|
| `run.json` | 全原稿の `nishiki.diagnose/1`（レンズ名・スコア・根拠つき） | **見せない**。`run.sha256` を先にコミットメントとして記録する |
| `rater_sheet.md` | 場面ごとに本文を表示し、違和感を感じた箇所にマークを付ける欄。レンズ名・スコア・検出器の有無は載せない。場面の提示順は seed で固定した乱数順（同一原稿の場面順は保つが、原稿の提示順と、システムが旗を立てた場面の位置は無関係） | 見せる |
| `blind_spans.json` | システムが旗を立てた場面・スパンの一覧（不透明な ID と文字区間だけ。理由・レンズ・スコアなし） | 比較時まで見せない |
| `key.json` | 不透明 ID ↔ 元ファイル名 | 見せない |

### 4.4 評価手順

1. `blind-run` を実行し、`run.json` の SHA-256 を記録する（以後 `run.json` を変えられない）。
2. 評価者は `rater_sheet.md` だけを見て、自分の違和感マークを `rater_marks.json`（場面 ID と文字区間）に付ける。システム出力は見ない（ステップ 3 まで）。
3. `blind-score` を実行する。`run.json` のハッシュが記録と一致することを確認してから、場面単位の一致を計算する: 評価者の旗 vs システムの旗の 2×2 表、Cohen's κ、感度、特異度。スパン単位は文字区間の重なり（IoU）を補助指標として併記する。
4. 対照: 原稿のうち 1 本について、既知の摂動（語尾置換・読点削除・感覚語削除）を人為的に入れた版を混ぜ、システムが摂動場面を拾うかを別集計する（評価者には摂動の有無を伏せる）。摂動の位置は `key.json` 側に持つ。
5. 判定の出力: 数値を報告するだけ。Gate 3b の解凍判断は本ステージの範囲外（F11、判断待ち J11）。

### 4.5 限界（先に記録する）

- F5 のとおり、検出器の AUC は合成摂動に対する値で、人間の違和感への感度は未測定。盲検評価の結果がこの AUC と大きく乖離する可能性がある。
- 現代散文では Gate 3a のレジスター問題（B-9）が残る。検出器は作品内 LOO 正規化を使うので作品間のレジスター差は直接は効かないが、校正分布は青空文庫の 56 作品に由来する。現代文の原稿での挙動は未測定。
- 評価者人数・原稿数・κ の合格水準は未定（判断待ち J8）。本設計は水準を決めない。

## 5. 実装ステージの分割案（タスク 2.5）

| ステージ | リポジトリ | 範囲 | ゲート | 依存 | 満たす完了条件 |
|---|---|---|---|---|---|
| P-4a 検出器 Python 参照実装と先行計測 | usir | v13 構成の Python 参照実装（3 量、LOO EB、分位スコア、max、ポアソン / Anscombe の A/B、p_OP 最小 p 規則）。`signals` に `L_total` を追加。同一プロトコルで AUC・CI・FPR を測定。フィクスチャ出力 | CI 下限 ≥ 0.85 と FPR ≤ 0.10（P 動作点）。未達なら P-4b に進まず設計へ戻る | `stage/detector-v4` の取り込み（J1） | 条件 1 の前提（正本の確定） |
| P-4b 検出器 Rust 移植 | nishiki | 検出器コア（新クレート案 `nishiki-detector`、`no_std` ではなく `libm` 使用）。`fit` / `score` / `p_op_procedure`。成果物 `detector.json`（校正配列と定数。数値のみ）の読み書き。§3.4 の数値照合。場面分割の Rust 移植（オフセット対応つき） | フィクスチャ照合が許容誤差内、`cargo test --workspace` PASS | P-4a | 条件 1（検出器部分） |
| P-4c `diagnose` コアと 6 レンズ配線 | nishiki | スキーマ `nishiki.diagnose/1` の凍結。6 レンズの配線。注釈生成の Rust 移植（J4 の決定に従う）。CLI `diagnose` と Wasm `diagnose`。ゴールデン照合（ネイティブ・wasm32） | CLI / Wasm 出力の JSON バイト一致、レンズ 6 + 検出器の全出力でパリティ一致 | P-4b（スキーマ凍結のみなら P-4a と並行可） | 条件 1（完成） |
| P-4d 盲検ハーネス | nishiki（+ usir に手順書） | `blind-run` / `blind-score`、`rater_sheet.md` 生成、コミットメント（ハッシュ）、不可逆な集計のみのコミット規則。小さな合成原稿で動作確認 | 合成原稿で一連が通り、`run.json` 改変がハッシュ不一致で検出される | P-4c の出力スキーマ | 条件 2 |
| P-4e 統合ゲート | usir + nishiki | 全テスト一括: usir（`pnpm test:all`、`uv run pytest`）、nishiki（`cargo test --workspace`、wasm テスト）。Python/Rust フィクスチャの再生成と一致確認。報告書 | 全テスト PASS | P-4a〜d | 条件 3 |

- 各ステージで `docs/handoff/` の STATUS を更新し、節目で push する。
- P-4a の結果が不合格の場合、P-4b 以降を止めて設計を見直す（検出器が条件 1 の対象なので、ここが分岐点）。
- 6 レンズのうち、注釈を要するレンズ（3B・4・5・6）の移植範囲は J4 の決定でステージ P-4c の規模が変わる。J4 が未決のまま P-4c に入らない。

## 6. 判断待ち一覧

| ID | 内容 | 候補と根拠 | 推奨 |
|---|---|---|---|
| J1 | 検出器 Python 正本が本流に無い（F1）。P-4a の基点をどうするか | (a) `stage/detector-v4` を本流へマージしてから P-4a。(b) 必要ファイルだけ cherry-pick。(c) P-4a を `stage/detector-v4` から切る。同ブランチの STATUS は「done」だが、本セッションではそのブランチのテストを再実行していない | (a)。マージ前にそのブランチのテストを実行して確認する |
| J2 | 検出器をレンズと別チャネルにするか、Kasane レンズに含めるか | 検出器の 3 量は表層（語尾・読点・感覚語率）で、Kasane の領域に近い。ただし出力が場面単位の旗で、Kasane の窓スパンと粒度が違う | 別チャネル `anomaly`（§2.3 のとおり） |
| J3 | Kamae レンズの実体: A（説明過多・秘する花・迎合、`kamae-core`）か B（伏線生存率、`usir-metrics`） | 仕様書の役割記述は A。B は宣言台帳のある入力で意味を持ち、無ラベル原稿では作品スカラーのみ | A をスパン出力、B を作品スカラーとして併記。A の出力構造は P-4c 冒頭で棚卸し |
| J4 | Sadame / Nagare / Awase（と Kamae B）の注釈の出所 | (1) Python ヒューリスティクス（`aozora_scene_splitter.py` の `compute_heuristic_signals` 等、約 1.4k 行のファイル内）を Rust に移植。(2) 外部の SLM 注釈を入力に取る（決定論でなく、Wasm 向きでない）。(3) 当面 `unavailable` | 条件 1 を文字どおり満たすには 6 極すべてでパリティが要る。(3) は一致するが実質空。(1) を推奨するが規模が大きい。Python 側 `compute_heuristic_signals` の「未較正」定数の扱いを含め、ユーザー判断を要する |
| J5 | 分散安定化の最終確定 | ポアソン下側片側 p（暫定採用、§3.3）。Anscombe は対照 | P-4a の A/B を事前登録規則で確定 |
| J6 | 最小 p 規則で満たす p が無い場合の挙動 | 正本は格子下限に落とす（最大 p 規則の名残）。最小 p 規則で下限（80）に落とすのは偽陽性を最大にする側 | 失敗として扱い、診断器を `unavailable` にする |
| J7 | 違和感スパンの粒度 | 現行検出器は場面単位。場面は 200〜1600 字で、書き手が直す単位として粗い可能性がある。文単位の局在は検出器の設計外で、未検証 | P-4 では場面単位。文単位は別ステージ |
| J8 | 盲検の合格水準（κ の値、評価者数、原稿数） | 引継ぎ文書に数値の記載なし（F11） | ユーザー指定を待つ。本設計は決めない |
| J9 | 場面数が少ない原稿の `reliability: low` の閾値 | LOO は場面 1 つなら全体平均にフォールバックし、少数場面では正規化が不安定。閾値は未測定 | P-4a で場面数別の挙動を測って決める |
| J10 | 数値実装（`libm` の使用、丸め桁） | ネイティブと wasm32 の `exp` / `ln` 差は未検証（推測） | `libm` を既定にし、差の実測を P-4c で報告 |
| J11 | Gate 3b の解凍判断 | 本ステージ範囲外。完了条件の達成と解凍判断は別 | 本ステージでは判定しない |

## 7. 本ステージで未検証の事項

- `stage/detector-v4` のテスト（`python/tests/test_detector_v4.py` ほか）は実行していない。
- `sasae-core` / `kamae-core` の出力構造（スパンの有無）は読んでいない。
- ネイティブと wasm32 の数値差、`regex` クレートと Python `re` の空白・Unicode 扱いの差は実測していない。
- abl_ii と v13 構成の CI は未計測（F4）。p_OP の「最小 p」値（92.5 / 93.0）は v4・abl_ii 構成の既存 `rows` から読んだ値で、v13 構成（ポアソン変換あり）では再計算が必要。
- §3.3 の個数統計は地の文のみで、q3（場面全体）の個数ではない。
