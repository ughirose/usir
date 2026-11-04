# ローカル CPU `--quick` の結果 (2026-11-04 実行, Claude モデルによる課題洗い出し時)

`python colab/run_v4_bidirectional_experiment.py --quick`（Colab GPU ではなくローカル CPU）の出力を退避したもの。
同名のコミット済みファイル（Colab T4 実機結果）を上書きしていたため、元の場所は T4 版に戻してある。

## 読むときの注意（実行時の報告より）
- 実 LLM の出力ではない。文字数が毎回 268 / 220 / 405 と一致し、事前に用意した候補テキストを評価しているだけ。GPU 生成の性能は示さない。
- Judge は決定論ルーブリックで、独立した LLM Judge の結果ではない。
- article で v4 の R_GRPO が vanilla の約 1/9 まで下がり相が Chaos になった一方、Judge は 83.8 と高く、報酬と Judge が逆向き。どちらを信じるかは未解決。
- Qwen2.5 を使う 3,000 字生成と GRPO は未実施。
- `table2_progress.json` はテスト実行で再生成される副産物（この版は参考用）。
