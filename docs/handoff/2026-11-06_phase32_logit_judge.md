# Phase 32: ロジット脱バイアス・ペアワイズ Judge（2026-11-06）

## 方式（commit 0899737）
- 1 回の forward で、プロンプト末尾「より優れている作品の番号（1 または 2）は、「」の次トークンの対数確率を読む。
- s(A,B) = lnP("1") − lnP("2")、Score = (s(A,B) − s(B,A)) / 2。順序に依存しない定数の位置バイアスは厳密に相殺される。内容 × 位置の交互作用は相殺されない。
- position_bias = (s(A,B) + s(B,A)) / 2 も記録する。
- 全文モードと段落対応モード（同じ段落番号どうしを比較）。
- 交差採点: Swallow-8B が Qwen3 の出力を、Qwen3-8B が Swallow の出力を採点。比較対象は Phase 31 の new_ctrl / gA / gB / gC と Phase 27 の min（seed 3, 42 のみなので n=4）。
- A100、218.9 秒、各 Judge 392 回の forward。

## 結果（`docs/handoff/v10/p32j_cross.json`）
Preference > 0 は前者（a）が優位。単位は対数オッズ。

| 生成モデル（Judge） | 比較 | n | 全文 preference 平均 | 全文で a 勝ち | 位置バイアス平均 | 段落 preference 平均 | 段落で a 勝ち |
|---|---|---|---|---|---|---|---|
| Qwen3（Swallow） | new_ctrl vs min | 4 | +0.188 | 3/4 | 2.00 | +0.596 | 18/24 |
| Qwen3（Swallow） | gA vs new_ctrl | 6 | +0.068 | 4/6 | 2.05 | +0.051 | 22/36 |
| Qwen3（Swallow） | gB vs new_ctrl | 6 | −0.005 | 3/6 | 2.02 | −0.050 | 20/36 |
| Qwen3（Swallow） | gC vs new_ctrl | 6 | −0.104 | 3/6 | 2.25 | −0.477 | 13/36 |
| Qwen3（Swallow） | gA vs gB | 6 | +0.115 | 3/6 | 2.24 | +0.074 | 15/36 |
| Swallow（Qwen3） | new_ctrl vs min | 4 | −0.437 | 2/4 | 6.38 | −1.121 | 8/24 |
| Swallow（Qwen3） | gA vs new_ctrl | 6 | −0.115 | 3/6 | 6.49 | +0.189 | 19/36 |
| Swallow（Qwen3） | gB vs new_ctrl | 6 | −0.177 | 1/6 | 6.70 | +0.448 | 21/36 |
| Swallow（Qwen3） | gC vs new_ctrl | 6 | −0.328 | 1/6 | 6.49 | +0.363 | 19/36 |
| Swallow（Qwen3） | gA vs gB | 6 | +0.224 | 3/6 | 6.47 | −0.338 | 14/36 |

- 「1」「2」への確率質量: Swallow Judge 平均 0.621（最小 0.558）。**Qwen3 Judge は平均 0.017（最小 0.0046）**で、次トークンの大半が番号以外に流れている。Qwen3 Judge の値は信頼できない。

## 判定
- 位置バイアス（2.0〜6.7）は、脱バイアス後の preference（絶対値 0.6 以下）より 1 桁近く大きい。脱バイアスで定数分は除けるが、残る差は小さく、符号は全文と段落、Judge 間で一致しない。
- 一致した傾向は Swallow Judge の new_ctrl vs min（全文 3/4、段落 18/24 で new_ctrl 優位）だけ。
- ゲノムステア（gA/gB/gC）が new_ctrl より「読んで優れている」とも「劣る」とも言えない。Phase 28 の生成型 Judge（ステア条件が負ける傾向）は、位置バイアスの影響を受けた結果だった可能性がある。
- 8B Judge は条件差の測定器として不十分という Phase 28 の結論は変わらない。より大きい Judge か人手評価が要る。
