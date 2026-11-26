# Phase 1.5 Stage 2 Step 7 レポート（2026-11-27）

計測のみ。実装（strong.py、INSTRUCTION_VERSION、物差し、出力スキーマ、鍵の仕様）は変更していない。main へのマージなし。

## 1. 旧指示文の復元と確認
- 旧文 = `c4d367f` の親 `ac3d1a65a11de627915d7e5f2bdc61d5e04a094c` の strong.py の INSTRUCTIONS。
- 復元先は作業コピー（`step2_work/s7a/`）のみ。run ディレクトリの `instructions_commit.txt` にリビジョンを置くと、`step2.py` が `git show <rev>:scripts/sensor_runner/strong.py` から INSTRUCTIONS を取り出し、prep / ingest / final の各プロセス内で `strong.INSTRUCTIONS` を差し替える（`_apply_instruction_override`）。strong.py 本体は不変。
- 確認: 復元した旧 strong.py と現行 strong.py の diff は「毎回 ledger_additions に入れること…」の1行の追加のみ。s7a の pack に当該文言は含まれず、pack の指示文ハッシュは 31875cee7bb6。

## 2. 注釈の実施
- 11場面（000035_000301: 7、001798_056779: 4）を、新規コンテキストの Sonnet サブエージェントで1 pack 1 agent、場面順、1ラウンド最大2並列で注釈。
- 注釈呼び出し: 11 回（上限15）。全件 初回 ACCEPT（試行 1 回）。再試行・失敗なし。
- 所要: 11 pack 合計 168.0 秒（各 14〜17 秒）。
- 注釈者 ID: `claude-code-subagent:alias=sonnet(exact-model-id-unverified)`（SENSOR_ANNOTATOR_MODEL_ID）。サブエージェントの実モデル ID は取得できず、alias=sonnet 指定までが確認範囲。Step 6 と同じ宣言。
- s7a の ledger_additions は 11 場面すべて空。

## 3. 突合結果（結果: `scripts/sensor_runner/results/phase15_stage2_step7_compare.json`）
バイト一致 = annotator / provenance を除く正規化 JSON が完全一致。意味一致の項目: present の id 集合、pov、location、event / mood の code 集合と top1、trait の (人物, 軸, delta) 集合、facts 件数、awase 4値、time.relative_order。

### ① s7a（旧指示）vs s6a（新指示）= 指示文1行の効果
- バイト一致: 0/11。
- フィールド別の差分場面数（/11）: present 8、pov 0、location 3、facts 11、trait_changes 3、ledger_additions 0、time 7、event_top3 11、mood_top3 9、awase 9。
- 意味項目の差分場面数: present_ids 0、pov 0、location 3、event_codes 5、event_top1 0、mood_codes 3、mood_top1 3、trait_set 2、facts_count 4、awase4 7、relative_order 2。全意味項目一致の場面: 1/11（001798 場面0）。
- 例（000035 場面0）: location「…」→「…の（その）部屋」の文字列差、event_codes [E02,E10,E14] → [E08,E09,E14]、relative_order continuous → unknown。

### ② s7a（旧指示・新方式）vs seed（旧指示・旧方式）= 方式・形式の効果
- バイト一致: 0/11。
- フィールド別: present 11、pov 0、location 3、facts 11、trait_changes 9、ledger_additions 0、time 8、event_top3 11、mood_top3 11、awase 11。
- 意味項目: present_ids 1、pov 0、location 3、event_codes 9、event_top1 7、mood_codes 6、mood_top1 2、trait_set 8、facts_count 9、awase4 8、relative_order 1。全意味項目一致: 0/11。
- 例（000035 場面0）: present_ids [yozo] → [horiki, yozo]、event_codes [E06,E09,E14] → [E08,E09,E14]。

## 4. 読み取り（事実と、そこからの推測）
- ①と②で、バイト一致はどちらも 0/11。文字列を含むフィールドは、指示文が同じでも注釈ごとに揺れる。
- 意味項目の差分場面数は、全体として ② が ① より多い（例: event_top1 は ① 0 に対し ② 7、trait_set は 2 に対し 8、facts_count は 4 に対し 9）。pov は両方 0、present_ids は ① 0・② 1。同じ新方式の注釈同士（①）の方が、旧方式の seed との比較（②）より近い。
- 推測（未検証）: Step 6 で見えた seed との差の多くは、指示文1行より注釈方式（seed は null 基底の手動注釈）に由来する。ただし ① でも意味項目に差があるので、同一指示文でどの程度揺れるかの基準は取れていない。
- ledger_additions はどの比較でも差 0 だが、s7a の ledger_additions は全場面で空であり、指示文1行の効果（台帳に無い人物を毎回追加させる）がこの 11 場面では観測条件に出ていない。この点でこの計測は指示文の効果を検出できていない。

## 5. テスト
- 既存 86 件（python/tests の test_sensor_runner*.py 6 ファイル）: 86 passed。コミット `068828a`（クリーンツリー）で実行。

## 6. 変更ファイル
- `scripts/sensor_runner/step2.py`（旧指示文の差し替え `_apply_instruction_override` を追加。測定ツールのみ）
- `scripts/sensor_runner/step7_compare.py`（新規）
- `scripts/sensor_runner/results/phase15_stage2_step7_compare.json`（新規）
- `docs/handoff/STATUS_phase15_stage1.md`、本レポート
- git 管理外: `step2_work/s7a/`、`step2_work/annotator_prompt.txt`、`step2_work/strong_old.py`

## 7. 残課題
- 同一指示文・同一方式の注釈を複数回取り、場面ごとの注釈揺れの基準値を得ていない（本計測では ① の差が揺れか指示文効果かを分離できない）。
- 指示文1行の効果を見るには、台帳に無い人物が複数場面にまたがる条件（ledger_additions が非空になる場面）が必要。
- サブエージェントの実モデル ID は未確認。
- s7a / seed の方式差は、seed が手動注釈であることに由来する交絡を含む。
