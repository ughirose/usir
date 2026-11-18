# STATUS: Stage P-4 設計（Plotailor「離見の見」診断器）

- 開始: 2026-11-18
- ブランチ: stage/p4-design（基点 feat/fable-v5-shootout 3c23915）
- 状態: 設計書完成。レポートの Drive 配信待ち
- 5 時間枠使用率（着手時）: 8%

## 確認済み事実
- 本流 3c23915 に scripts/detector_v3/・detector_v4/ は無い。stage/detector-v4（8ad54fa）にのみ存在、未マージ。
  → 検出器 Python 正本は `git show stage/detector-v4:<path>` で参照する。

## 進捗
- [x] 6 レンズ同定（仕様書・nishiki 調査）
- [x] 検出器 Python 正本の読解
- [x] 設計書 docs/handoff/2026-11-18_usir_v5_p4_design.md
- [ ] レポートを Drive Claude-Sync inbox へアップロード・到達確認

## 判断待ち
- (1) 検出器 Python 正本が本流に無い（上記）。
- (2)〜(11) は設計書 §6（J1〜J11）を参照。
