# sensor_runner 実走その2 引き継ぎ（完了）

- 作業ツリー: C:\dev\usir-merge（feat/fable-v5-shootout）。手順0は完了: stage/sensor-trust-results = 9fa6413（origin に push 済み）。
- 作品: 000148_000751（夏目漱石「二百十日」）。nishiki segment: 40 場面・発話 904 件。F9 Dev/Holdout/現代文3 のいずれでもない。
- 結果: 40/40 ACCEPT。初回 REJECT 3 件（s10 basis 誤写、s25 location が本文に無い、s38 basis が前場面の台詞）。`merge --diagnose`: merged=40/40 errors=0 missing_local_utt=0、6 レンズ ok。
- findings: Awase 8 件（すべて 0.25 相当。判定 妥当4/保留3/誤検出1）、sasae 142 件（presence_unconfirmed 129、move_destination_mismatch 10、place_conflict 3）。判定の根拠は F:\マイドライブ\Claude-Sync\inbox\2026-11-22_sensor_live_run2_report.md。
- pytest tests/test_sensor_runner.py: 35 passed / 1 failed。失敗は `test_seed_all_35_scenes_cached_in_manifests`（期待 35、実測 82）。base 50b2fb1 の manifests が既に 42 場面で、実走その2より前から成立している。テストは未修正。
- ローカル発話: `local_from_step1_nogold.py`（gold 不要、場面境界のチャンク分割）。Step 1 実時間 4,150.7 秒、話者 null 70/904（s31 と s39 の全発話。stage 1 が parse_error 無しで全 null を返した）。

## 未解決・次の候補
- Step 1 が全発話 null を返す条件（s31・s39）の調査。入力は 761・837 字、28〜40 発話。再現は未確認。
- 2 者対話の話者の交互反転（s32・s34・s35 で目視確認）。全数の正答率は未計測（正解データなし）。
- sasae の move_destination_mismatch: 包含関係・言い換え・人物相対表記を同一場所と扱う規則が無い（誤検出 4 件）。
- Awase の保留 3 件（s12 irony、s21 detachment、s26 detachment）は文体値 1 段階の差に依存。
- テストの期待場面数 35 の更新要否（依頼範囲外のため未対応）。

## 再現
- `export NISHIKI_BIN=C:/dev/nishiki-hybrid/target/debug/nishiki.exe`; `cd python; uv run python ../scripts/sensor_runner/cli.py status`
- diagnose JSON: `scripts/sensor_runner/results/000148_000751.diagnose.json`（`nishiki diagnose --format json --characters merged/... --annotations merged/...`）。
- 本文 `data/aozora_cache/heldout_000148_000751.txt` は gitignore 対象。
