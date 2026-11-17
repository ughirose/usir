# STATUS: USIR v5 Stage P-2（Gate 3a 通過基準の再定義と層 C 64 件素性監査）

作業ブランチ: `stage/p2-gate3a`（開始 HEAD `3c23915`、`feat/fable-v5-shootout` 由来）
開始日: 2026-11-17

## 進捗
- [x] ブランチ作成、前提資料の読み込み（P-1 レポート、§25 戦略 `2026-11-14_gemini_s24_strategy.md`）
- [ ] タスク A 素性監査
- [ ] タスク B R_xx
- [ ] タスク C 摂動 ROC-AUC
- [ ] 付随 D / E（A〜C 完了後のみ）
- [ ] 検証ゲート（harness、パリティ、SHA 19/19）
- [ ] レポート作成・inbox 複製

## 初期所見（タスク A、未検証の観察）
- `data/usir_v5_eval/tier_c_modern_benchmark_expanded.jsonl` は 64 件（R1〜R4 各 16）。JSONL に原文はなく、非可逆プロファイルのみ。
- 初版 `tier_c_modern_benchmark.jsonl` は 32 件（各レジスター 8）。追加 32 件のうち R1〜R3 の 24 件が §16.9 の「追加 24 件」に当たる（推測。§16.9 の原文は未確認）。
- R1〜R3 の原文は `scripts/build_tier_c_modern_benchmark_expanded.py` 内に直書き。R1 の出典欄は場面説明（例: 「早朝の駅ホームで…の会話」）で、実在作品の出典ではない。R3 は実在機関名を冠するが、本文が実在文書の逐語引用かは未確認。
- R4 は青空文庫の実テキスト（manifest の work_id と開始位置で再現可能）。

## 事前登録
（結果を見る前に、各タスクの方法・検定・多重性補正・摂動パラメータをここへ記録する。未記録の間は該当タスクの結果計算をしない。）

## 判断待ち・未定義事項
（なし）
