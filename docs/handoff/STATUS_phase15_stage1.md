# STATUS: Phase 1.5 Stage 1 (sensor_runner L1/L2 cache)

- ブランチ: feat/phase15-stage1-cache（worktree: C:\dev\usir-phase15）、ベース feat/sensor-runner @ b3da0e4（現在ブランチに sensor_runner が無いため）
- 状態: 実装・テスト・実測・レポート完了。main へのマージは Orion 判断待ち。
- レポート: docs/handoff/2026-11-26_phase15_stage1_report.md
- 検証: sensor_runner 59 テスト合格、パリティ（4ケース + 3作品×5回）全一致、harness 全体の失敗 7 件はベースでも同一。

## NEXT
Orion が report §6 の判断点（ledger_hash の範囲、Early Cutoff の健全性、40場面作品での L3 要否測定）を決める。

## Stage 2 Step 0（2026-11-26 開始、計測のみ・キャッシュ実装は変更しない）
- 指示: Drive inbox の 2026-11-26_phase15_stage2_step0_prompt_v1.md / consult_synthesis_v1.md
- 計測A（繰り返し注釈の揺れ）、計測B（prior 感度）。注釈呼び出し上限 合計300回。
- 結果: scripts/sensor_runner/results/phase15_stage2_step0_*.json、レポート docs/handoff/2026-11-26_phase15_stage2_step0_report.md
- 状態: 計測完了（注釈148件、試行151）、レポート作成済み。59 テスト合格、実装変更なし。main 未マージ。
- NEXT: Orion が Step 0 レポート §4・§5 を読み、Step 1 の方針（location ID 化、facts の型付け、前提変化表示方式）を決める。

## Stage 2 Step 0.5（2026-11-26 開始、計測のみ・キャッシュ実装/本番スキーマ/物差しは変更しない）
- 指示: Drive inbox の 2026-11-26_phase15_stage2_step05_prompt_v1.md
- Q-A 型付き facts の一致率・語彙被覆、Q-B location 候補リスト選択の一致率。20場面×5回、注釈上限150回。
- 結果: scripts/sensor_runner/results/phase15_stage2_step05_*.json、レポート docs/handoff/2026-11-26_phase15_stage2_step05_report.md
- 状態: 計測完了（受理100件）、レポート作成済み。59 テスト合格、実装変更なし。main 未マージ。
- 結果要約: 型付き facts 不一致率 T3 1.00 / T0 0.88（鍵に不可）、location 閉じた語彙一致 0.90（楽観的条件）、prior ハッシュ 0.74→0.835（再計算値）。
- NEXT: Orion が Step 0.5 レポート §3・§4 を読み、Step 1 の方針を決める。

## Stage 2 Step 1（2026-11-26 開始、実装）
- 指示: Drive inbox の 2026-11-26_phase15_stage2_step1_prompt_v1.md
- 確定設計: pack バイト列ハッシュ鍵 / facts は鍵に入れず「前提変化あり」を記録 / prior_state 核=present+pov、location は手確定の場所リストがある作品のみ / 台帳エポック固定（最大2） / 台帳の pack 載せ方=大域+表示人物のみ。
- 状態: 開始（プロンプト全文読み込み中）

## Step 1 完了（2026-11-26）
実装コミット eb031f4。72 passed（既存59+新規13）、parity 4条件一致、epoch 凍結シナリオ 6→4 call。レポート: docs/handoff/2026-11-26_phase15_stage2_step1_report.md

## Step 1.5 開始（2026-11-26）
roster_all_ids を L2 鍵から外す（pack 内容は不変）。roster_all_digest を L2 メタに保存し roster_changed を stale 記録。鍵スキーマ版上げ（pack_bytes/2, l2/3）。状態: 実装中。

## Step 1.5 完了（2026-11-26）
実装コミット 0321953。80 passed（既存72のうち2件は新仕様に合わせ期待値更新 + 新規8）、パリティ4条件一致、人物追加シナリオ 4→0/1/2 call。レポート: docs/handoff/2026-11-26_phase15_stage2_step15_report.md

## 枝の統合（2026-11-26 22:40 +0900）
統合元 feat/phase15-stage1-cache を統合先 feat/sensor-runner（統合前の先端 b3da0e4）へ fast-forward で進める。事前確認: b3da0e4 は統合元の祖先（git merge-base --is-ancestor 成立）。main には触れない。統合元の枝は削除しない。

## Step 2 開始（2026-11-26）
実LLM通し測定（計測のみ）。ブランチ feat/phase15-stage2-step2（土台 feat/sensor-runner 33bf69c）。計測道具 scripts/sensor_runner/step2.py（実装コード無変更）。サンドボックス scripts/sensor_runner/step2_work/<run>/（seed キャッシュは非接触）。上限: 注釈呼び出し 合計150回。状態: 実装前準備完了、cold 実行前。

## Step 2 完了（2026-11-26）
A–E 実測完了。cold 35 call、warm 0 call、編集 1/2 call、location 65.7%。詳細: `2026-11-26_phase15_stage2_step2_report.md`、データ: `scripts/sensor_runner/results/phase15_stage2_step2_results.json`。

## Step 3 開始（2026-11-27）
不具合A（ingest 検証基準を凍結スナップショットへ）・B（timing 並列書込）修正、001875 location 調査。土台は feat/phase15-stage2-step2（33bf69c を含み、step2.py を持つ）。

## Step 4 開始（2026-11-27）
(a) 先勝ちで捨てた2回目ベクトルの診断記録、(b) ep条件の実LLM確認、(c) roster_changed 実測、(d) 001875 location 再注釈。上限30 call。土台 feat/phase15-stage2-step3 (caa65c5)。

## Step 5 開始（2026-11-27）
(a) 注釈者指示に「台帳に無い人物は毎回 ledger_additions へ」を明示、ep 条件で実LLM再確認（上限10 call）。(b) feat/sensor-runner へ fast-forward 統合。土台 feat/phase15-stage2-step4 (7ce179c)。

## Step 8 開始（2026-11-27）
seed 7作品・全35場面を現行指示文・現行手順で cold 再注釈し、全場面そろってから seed を差し替える。上限50 call。差し替え後に warm 0回・パリティ4条件・86テスト、feat/sensor-runner へ fast-forward 統合。土台 feat/sensor-runner（22b2e26）。

## Step 8 完了（2026-11-27）
全35場面を現行指示文・現行手順で再注釈（35 call、全件初回 ACCEPT）し、seed を差し替え（cd02ac0、旧 seed は 22b2e26 に残存）。warm 0 call・バイト一致、パリティ4条件合格、テスト86件合格（3fc4e79、SENSOR_ANNOTATOR_MODEL_ID 設定時）。未設定だと 6 件失敗（seed の鍵に model_id が入るため）。詳細: docs/handoff/2026-11-27_phase15_stage2_step8_report.md

## Step 9 開始（2026-11-27）
新 seed の L2 鍵に入る model_id を、環境変数未設定でも一致させる（案 a: CLI 既定値とテスト側で固定）。鍵の仕様・unspecified への偽装は行わない。seed の再注釈なし。土台 feat/sensor-runner（76d1e45）。

## Step 9 完了（2026-11-27）
common.py の model_id 既定値を新 seed の宣言値に変更（鍵の仕様は不変、env 明示指定は優先）。環境変数なしで、テスト 86 件合格、warm 0 call・バイト一致、パリティ 4 条件合格、回帰テスト 4 件追加（90 件）。実行コミット b8afa9e。詳細: docs/handoff/2026-11-27_phase15_stage2_step9_report.md

## Step 10 開始（2026-11-27）
(a) feat/phase15-stage2-step9（8708b5d）を feat/sensor-runner へ fast-forward 統合（環境変数未設定で90件・パリティ4条件を統合先端で確認）。(b) SENSOR_ANNOTATOR_MODEL_ID の運用ルールを SESSION.md に明記。seed の再注釈なし。土台 feat/sensor-runner（76d1e45）。

## Step 10 完了（2026-11-27）
8708b5d を feat/sensor-runner へ fast-forward 統合（環境変数未設定で90件・パリティ4条件を統合先端で確認）。SESSION.md に SENSOR_ANNOTATOR_MODEL_ID の運用ルールを明記。注釈呼び出し 0 回。詳細: docs/handoff/2026-11-27_phase15_stage2_step10_report.md
