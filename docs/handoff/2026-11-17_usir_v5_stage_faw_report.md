# USIR v5 Stage FAW 完了報告書：Awase／Nagare 連続補助指標の仕様設計

- 作業ブランチ: `stage/faw-design`（起点 `f008df5`）。コード変更なし。仕様書・討議パックは未編集。本流へマージしていない。
- 設計書: `docs/handoff/2026-11-17_usir_v5_faw_design.md`（A〜D の全内容）。STATUS: `docs/handoff/STATUS_faw.md`。
- 計測スクリプトと結果: `scripts/faw/`（`faw_measure.py`、`faw_pilot.py`、各 `*_results.json`、`logs/`）。

## 1. 結論

1. Awase の場面単位ゼロ率は irony 89.8%、detachment 82.95〜83.0% で、2026-11-13 の値を再現した。ゼロの内訳は入力側の 0（V_F=0 または V_f=0）が 76.5pt、ReLU による符号切り捨てが 13.3pt、符号一致して正になるものが 10.2pt。主因は注釈器の語ヒットのスパース性（style valence の 66.8%、量子化前でも厳密 0）で、連続化では解消しない。
2. 推奨は、Awase は符号付き対位平均 AW-1（Tier 1）と傾き AW-4、Nagare は滑らかな静止質量 NG-1 と速度傾き NG-2（phi_zeami と mood_speed_mean は変更なし）。いずれも退化は (ii-a) 型で `None`、既定値代入と ε を廃止する。再現性の改善は未実証。
3. 作品分割の再現性パイロット（作家数 19／10）は、2 コホート間で符号が反転する指標が多く、候補の順位付けに使えない。合格ラインの数値は根拠がないため置かず、登録すべき形のみ提示した。

## 2. 主要数値（`scripts/faw/faw_measure_results.json`、`faw_pilot_results.json`）

| 量 | AOZORA106 (4,576 場面) | POSTWAR (1,877 場面) |
|---|---|---|
| irony ゼロ率（場面） | 89.79% | 89.98% |
| detachment ゼロ率（場面） | 82.95%（凍結 jsonl 直読 83.02%） | 82.26% |
| 入力 0 ／ 符号反転(irony>0) ／ 同符号(ReLU で 0) | 76.46／10.21／13.33% | 78.16／10.02／11.83% |
| 作品単位 irony_grad ゼロ率 | 53.77% | 48.21% |
| 作品単位 still_density ゼロ率 | 92.45% | 96.43% |
| 作品単位 irony_mean ゼロ率／異なる値 | 16.04%／61 | 14.29%／41 |

## 3. A〜D の結果

- A: 設計書 §A。定義の構造から、離散性の発生源を 7 点特定した（語ヒット、ReLU、量子化、irony_active の二値、irony_grad の 3 分岐、still の 3 重離散化、speed_ratio の定数 1.0 と +1e-4）。
- B: 設計書 §B。Awase は AW-1〜AW-4、Nagare は NG-0〜NG-4 を候補化し、各案に定義式・値域・退化契約・旧指標との関係・Tier・Scheme 3 整合・計算量（全て O(T)）を付した。S_haz の踏襲点（legacy 併走、無重み正本、感度対照の分離、(ii-a)、Tier 1／2）と変更点（次元別に規則を書く）を明記した。
- C: 設計書 §C。確認データの分離、R_xx の作家間相関版、ノイズ床、摂動感度、nHSIC を含む事前登録案。合格ラインの形（作家単位ブートストラップの下限）のみ提案。
- D: 設計書 §D。推奨と反証条件、仕様書の反映点（§3.2、§3.3、§3.4.3 形式）、Gemini v10 で諮る論点 5 件。

## 4. advisor 相談の記録

- 経路: `/advisor fable` はこのセッションから実行できるツールが無く（ToolSearch で該当なし）、Fable のサブエージェント（Agent の model=fable）経由で 1 回相談した。事前登録の「最大あと 2 回」は、付帯終了（ユーザー通知）のため実施していない。以後は Sonnet 単独。
- 聞いたこと: 候補案の軸、退化ケース契約の罠、S_haz の踏襲範囲と Awase／Nagare の設計分離。
- 採用したもの: 候補の 4 軸（クリップ位置、閾値の扱い、基準、集約）、既定値代入・場面数分岐・ε を避ける契約、次元別に仕様を書く方針。
- 変わったこと／採らなかったもの:
  - 助言は「Awase の張り付きは ReLU 由来」としたが、実測では ReLU は 13.3pt で、主因は入力側の 0（76.5pt）だった。設計はこの実測に合わせた。
  - 助言は still の閾値を 0.2 rad としたが、コードは 0.25（`six_pole_atlas.py:521`）で、コメントとコードが不一致。設計は 0.25 を基準にした。
  - 助言側が確認できなかった (ii-a)／(ii-g) は、B-22 報告（`stage/b22-kamae-haz`）で定義を確認した。
  - 助言の「Awase の低再現性が張り付きの帰結か」は、本計測でも切り分けられていない。

## 5. 検証ゲート

| ゲート | 結果 |
|---|---|
| `python scripts/harness.py test` | 終了コード 0。PASSED 672 件、FAILED/ERROR 0 件（`scripts/faw/logs/harness_test.log`）。実行前にコミット済み |
| パリティ（実推敲ペア v1.1、439 件） | `python scripts/p2c/p2c_parity_sd_audit.py` で F17／C216 とも不一致 0（`scripts/faw/logs/parity.log`）。追跡対象の JSON に差分なし |
| 計測の再現性 | `python -I scripts/faw/faw_measure.py`、`python -I scripts/faw/faw_pilot.py`。対象コミットは `1e11ff5` ＋ 本ステージ最終コミット。乱数・シードなし（決定論）。データ入力は `data/aozora_cache`（gitignored）。再実行は他環境では未確認 |

コード変更が無いことは `git diff f008df5 -- python crates packages` 相当で、`scripts/faw/` と `docs/handoff/` の追加のみであることを最終コミット時に確認する。

## 6. 前提との差異

- 前提の「P-2c の戦後小説コーパス 1,877 場面」は 32D のテキスト特徴のみで、Awase／Nagare 注釈を持たない。注釈器 `compute_heuristic_signals` を同一テキストへ決定論的に適用して導出した（量子化前の値を複製して再量子化し、0 場面不一致を確認）。
- 凍結 jsonl 直読の detachment ゼロ率 83.02% と再導出値 82.95% の差（約 3 場面相当）は原因未特定。
- Python 評価側と Rust 側で still 判定（固定 0.25 対 適応閾値）と phi の空入力値（0.5 対 0.0）が異なる。

## 7. 未実施の検証と理由

- 確認データでの R_xx、ノイズ床、摂動感度、nHSIC: 実装後の検証であり、設計のみの本ステージの範囲外。
- 確認用データの選定: `p2cA2`／`p2cE` が P-2c で既に使われているかを未確認。
- Rust 側での連続指標の実装可否・コスト: 未検討（設計のみ）。
- arousal の量子化不感帯の寄与: 未計測。
- パイロットの作家数（19／10）が小さく、候補の順位付けは未実施。

## 8. 判断待ち事項（Orion／Gemini）

1. 合格ライン（§C.4）の数値。
2. Awase の次元定義（4D 維持か 3D か）。irony_active の後継なしの是非。
3. still の正本（固定 θ か適応閾値か）。Python／Rust の統一方針。
4. 確認データの指定。
5. 注釈器の語リスト（オープンクラス）と憲章の整合（v5.1+）。
6. 仕様書への反映（設計書 §D.2）と、討議パック該当節の所在確認。

進捗: 設計書・レポート作成済み。NEXT: コミット・push、レポートを `F:\マイドライブ\Claude-Sync\inbox\` へ複製。
