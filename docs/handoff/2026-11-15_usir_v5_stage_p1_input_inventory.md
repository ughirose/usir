# USIR v5 Stage P-1 入力棚卸し（2026-11-15）

| 入力 | 状態 | 確認内容 |
|---|---|---|
| usir ブランチ | OK | 開始時 `stage/b22-kamae-haz`（作業ツリー clean）→ `feat/fable-v5-shootout`（85c618b）へ切替。B-22 の 5 コミットは b22 ブランチ側に残り、本ブランチには含まれない |
| nishiki クローン | OK | `C:\dev\nishiki` `feat/fable-hardcore-v5` HEAD `64236bde58d45e34b69aeee6a57a0e6c926ac733`。未追跡 `scripts/web_auditor.py`・`start_web_auditor.bat` のみ（他作業、触らない）。ブランチ切替・マージなし |
| 青空文庫キャッシュ | OK | `data/aozora_cache` 369 ファイル（gitignore 資産） |
| 実推敲ペア v1.1 | OK | SHA 55D2DDCF…、439 件、同一正書法 Primary 405 件 |
| 実推敲ペア v1 | OK | SHA 62DA7527… |
| F-17 床 / 106候補 / 216候補 / 216正本 | OK | SHA 0575B64A… / 39DA6CC8… / 07E43FDF… / 2FC18E67…（`docs/handoff/v10/`、git 管理下で一致） |
| Rust 実行体 | OK | `C:\dev\nishiki\target\f14\release\nishiki.exe` |
| Google Drive inbox | OK | `F:\マイドライブ\Claude-Sync\inbox\` |

必須入力の欠落なし。ブロック報告は不要。
