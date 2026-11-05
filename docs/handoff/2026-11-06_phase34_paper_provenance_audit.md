# Phase 34 事前監査: 論文数値の出典（2026-11-06）

- 対象: `usir_v3_camera_ready_paper.md`、`usir_v3_camera_ready_paper_ja.md`、`scripts/paper_sections/*.md`。
- 方法: 読み取りのみ（Sonnet サブエージェント）。下表の主要 2 点は Opus が元ファイルを直接確認した（「確認」欄）。
- 組み立て: `scripts/update_papers_v3.py` が v2 の本文（Table 1・2 を含む）に `paper_sections` を差し込む。数値は手書きで、JSON からの自動生成ではない。

| 項目 | 論文の値 | 出所 | 判定 | 確認 |
|---|---|---|---|---|
| Table 1（N=890、ROC-AUC 0.934、R2 0.851） | JSON と一致 | `python/usir_py/v2/benchmark_v2_large_scale.py` / `benchmark_v2_oof_report.json`。コーパスは `build_large_benchmark_corpus.py` | **合成コーパス上の計算値**。語彙バンクから rng で文を組み立てている。docstring（L5-6）に「Regime 1 を AUC 0.62-0.78、Regime 2 を Proposed で 0.86-0.95 に較正」とある | Opus が L1-16 を確認 |
| Table 1 の音楽（AUC 0.936） | — | 同コーパス L261-340 | **合成**。rng で作った 32 音列。作曲家名はラベルだけ | — |
| Table 2（N=300、T4、D2 4.69、\|β-1\| 0.013、MAUVE 0.987、U 0.270） | JSON の丸め値と一致 | `python/colab/run_experiment_b_batch.py` / `artifacts/table2_generation_control_results.json` | **mock**。`metadata.model="Mock-Model-Simulator"`、`mock_mode=true`。β は手法ごとの目標値（v3 は 0.995）をスペクトルに直接指定している（L1034-1042） | Opus が JSON と L1030-1045 を確認 |
| 「Colab Tesla T4 で検証」 | — | Table 2 の JSON に GPU 情報なし | **出所なし**（T4 の記録は別実験の `v4_bidirectional_experiment_results.json`、n=30） | — |
| Table 3（4.17 ms、0.00 ms、escape 100%、MAUVE 0.745） | — | `run_experiment_c_lora_forge.py` | **mock**（Phase 28 監査で確認済み） | 既報 |
| Table 4（N=400 英語、ROC-AUC 0.934、Δ² 41.28） | JSON と一致 | `benchmark_v3_english_cross_linguistic.py`。コーパスは `build_english_benchmark_corpus.py` | **合成コーパス上の計算値**。手書きの文バンクを再結合したもので、400 件中、一意なテキストは 227 件。著者ラベルは架空 | — |
| §5.2「8/8 REJECT、20/20 PASS」 | — | TS の固定フィクスチャ | **フィクスチャ上の合否**（保留データでの汎化率ではない） | — |

## 結論（Phase 34 の方針）
- 論文本文のうち、実モデル・実データで測った数値は、Table 1〜4 のどれにもない。
- 改稿では、実測 JSON だけを引用する:
  - `docs/handoff/v9`、`v10`（Phase 24〜33）
  - `artifacts/style_atlas/*.json`
- Table 1・4 は「合成コーパス上の自己整合テスト」と明記して付録へ移すか、本文から外す。Table 2・3 の mock 値は削除する。
- 自動照合 `scripts/verify_paper_numbers.py` で、論文の数値と JSON の値（`mock_used` が false のもの）の一致を検査する。
