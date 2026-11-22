# sensor_runner（強いモデル注釈の実行系） (2026-11-22, 記入者: Claude Code)

- ブランチ: `feat/sensor-runner`（基点 `feat/fable-v5-shootout` d8bf733）、worktree `C:\dev\usir-sensor-runner`。既存ブランチへのマージなし。
- 種: `phase2/hybrid-full-run` @ 6043212 の注釈（読み取りのみ。マージはしていない）。
- nishiki: `segment` / `diagnose` を持つビルドが必要。`NISHIKI_BIN` 未設定時の既定は `C:/dev/nishiki-hybrid/target/debug/nishiki.exe`（`C:/dev/nishiki/target/release/nishiki.exe` は `segment` / `diagnose` が無い旧ビルド）。

## 決定事項
- 配置: `scripts/sensor_runner/{common,cache,strong,prepare,ingest,merge,diagnose,seed_from_hybrid,cli}.py`、手順書 `SESSION.md`、テスト `python/tests/test_sensor_runner.py`。
- 強いモデルの注釈形式は中間形式 `sensor.strong/1`（sensor.v2 そのものではない）。merge で場面ごとの sensor.v2 記録に合成する。sensor.v2 スキーマは無変更。
- STATE_CHANGED は半構造化（target / category / text）。sensor.v2 の object には `[category] target: text` で入る（seed の `unclassified` は text のみ）。
- 特性は台帳（`cache/ledger/<work>.traits.json`）に基準値、場面側は変動 (axis, delta) のみ。値 = clamp(baseline + delta)。
- merge は同席を自動補完しない（ローカル発話の話者・宛先が同席に無い場合は件数 `local_ids_absent_from_present` で報告）。自動追加すると Sasae の off_stage_speaker が消えるため。
- キャッシュキー = 場面ハッシュ。`INSTRUCTION_VERSION` が一致するエントリのみ cached。

## 未完了
- seed の発話（話者・宛先・evidence）は強いモデルが注釈したもので、ローカルモデルの出力ではない。`local/*.utterances.json` の provenance に明記。実ローカル出力（三段構成）への差し替え未実施。
- 新規場面での session 注釈の実走は未実施（seed は全 35 場面が既知）。パックの実用性（トークン量、prior の有効性）は未測定。
- 名簿・台帳変更時のキャッシュ無効化は手動（エントリ削除）。

## 次の一手
- 未知の1作品を `works/` に登録し、SESSION.md の手順で 1 バッチ（5 場面）を実走して、ingest の拒否率と所要を測る。
