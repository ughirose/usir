# USIR — 計測規格・216作品ノイズ床・ハイブリッド注釈センサー

> **2026-11-23 現行ステータス**:
> - アクティブ作業線: **`feat/fable-v5-shootout`**（先端 `eac3027`+, `main` `1a30ec2` は 2026-11-11 時点で停止中）。
> - **稼働中の主パイプライン**: ハイブリッド注釈センサー `scripts/sensor_runner/`（ローカル `qwen2.5:14b` による Step 1 発話・話者抽出 ＋ Policy X 口調 ＋ 強モデルセッションバッチ `sensor.strong/1` → `sensor.v2` 生成 → `nishiki diagnose /3` 接続）。
> - ドキュメント全体の索引は [docs/INDEX.md](docs/INDEX.md) を参照してください。

---

## 1. 主な構成要素

- `scripts/sensor_runner/`: ハイブリッド注釈実行系（`cli.py`, `local_from_step1_nogold.py`, `merge.py`, `SESSION.md`）。公開 2 作品（森鷗外『山椒大夫』7場面、夏目漱石『二百十日』40場面・904発話）で `nishiki diagnose /3.2` 全 6 レンズ `ok` 完走。Step 1 改良（`46518c0`）により全発話話者 `null` 場面を `2 -> 0` へ解消。
- `python/usir_py/sensor_eval/`: Step 1 発話評価・Policy X 口調分類・センサー精度測定基盤。
- `schemas/`: `sensor.v2` / `sensor.strong/1` スキーマ定義。*(注意: Nishiki `/3` 凍結形 `trait_variations` / `STATE_CHANGED.{target,category}` への追従更新は次期タスク)*。
- `data/aozora_cache/` & `experiments/`: 青空文庫 216 作品ノイズ床較正および Fable v5 比較実験資産。

---

## 2. 検証テスト実行例

```bash
# ハイブリッドセンサー実行系＆Step 1 改良テスト (70 passed)
python -m pytest python/tests/test_sensor_runner.py python/tests/test_step1_sensor_eval.py -rs
```

---

## 3. 関連 SSOT リンク

- **4 プロジェクト横断の統合設計・開発計画・用語正本**: Google Drive `05_Nishiki_錦/00_INDEX_AND_SSOT/`
- **ステージ完了レポート・HANDOFF 正本**: Google Drive `Claude-Sync/inbox/`（リポジトリ内に `Claude-Sync/` を作成しないこと）
- **診断コンパイラ本体**: `ughirose/nishiki`（`main`）
