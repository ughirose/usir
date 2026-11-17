# STATUS_faw_impl（USIR v5 Stage FAW-IMPL）

- 開始: 2026-11-17
- 作業ブランチ: `stage/faw-impl`（起点 `stage/faw-design` 先端 `628161f`）
- 作業ディレクトリ: C:\dev\usir
- 制約: 本流へマージしない。旧指標・注釈器・仕様書・討議パックを変更しない。Gate 判定しない。

## タスク 0：確認データの確定 → 停止

結果: `p2cA2`（84 件）、`p2cE`（67 件）の両方が P-2c で使用済み。指示（両方使用済みなら停止して報告）に従い、実装・計測には進まない。

根拠（Grep、`scripts/` と `docs/handoff/` を対象）:
- `p2cA2`: `scripts/p2c/p2c_essay_select_profile.py:83,118`。docstring「P-2c Task A2: essay register trial（NDC 914-916）」。`STATUS_p2c.md:105` に、原文キャッシュ `data/aozora_cache/p2cA2_*.txt` から 32D プロファイル（`extract_ortho24_kasane8`、`split_work_into_scenes`）を作り `data/p2c_essay_scene_profiles.npz` に保存したと記載。
- `p2cE`: `scripts/p2c/p2c_taskE_level3b.py:98,128`。docstring「P-2c Task E: Level 3b」。重心基準コーパスを v1.1 の 15 作家へ拡張した計測で、`STATUS_p2c.md:85` に登録 E。

未確認:
- 上記 2 件の Grep は `scripts/` と `docs/handoff/` のみ。リポジトリ全体の検索は 120 秒でタイムアウトし、未完了。
- 「84 件」「67 件」の件数は設計書の記述であり、キャッシュ実ファイルの件数は未照合。
- 両群のうち、P-2c の重心計算・Level 3b の判定に実際に寄与した作品の範囲は未確認。部分的に未使用の作品が残るかは、確認していない。

## 判断待ち（Orion）
- 確認データの代替選定（指示書により Orion が行う）。

## 進捗
- [x] ブランチ作成（`stage/faw-impl`）
- [x] タスク 0（両方使用済み、停止）
- [ ] タスク A／B／C は未着手（確認データ確定待ち）
