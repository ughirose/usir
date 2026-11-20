# STATUS: phase2/stage4-fact-sentence-index (usir)
- updated: 2026-11-20
- base: phase2/sensor-full-annotation @ 7f75069（既存ブランチへのマージなし）
- state: 完了
- done: scene_annotator.resolve_sentence_index、scripts/phase2_stage4/regen_annotations.py、scripts/phase2_stage4/annotations/（7作品、fact に sentence_index、characters.json に default_politeness）
- 要点: 事実 92→62（30件は根拠が1文に収まらず除外）。LLM は再実行せず生応答を再検証。default_politeness は25人に付与
- report: F:\マイドライブ\Claude-Sync\inbox\2026-11-20_phase2_stage4_report.md
