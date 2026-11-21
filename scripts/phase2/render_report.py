"""Render the Markdown report from phase2_step1_results.json + register_audit.json (values copied from artifacts)."""
import json, sys
from pathlib import Path
R = Path(__file__).resolve().parent
d = json.loads((R / "results/phase2_step1_results.json").read_text(encoding="utf-8"))
a = json.loads((R / "annotations/register_audit.json").read_text(encoding="utf-8"))
s = json.loads((R / "results/sensor_run/qwen2.5_14b_dev.json").read_text(encoding="utf-8"))
titles = {"001798_056779": "ボロ家の春秋", "001801_001786": "他人の夏", "001875_059897": "雑居家族"}
W = list(titles)
L = []
p = L.append
p("# Phase 2 第1段 実センサー出力による縦貫通 レポート\n")
p("- 日付: 2026-11-20 / 対象: 青空文庫3作品（公開テキストのみ。妻の原稿は不使用）")
p(f"- usir ブランチ: `phase2/step1-real-annotation`（ワークツリー C:/dev/usir-phase2、基点 stage/step1-eval 51854f1）")
p("- nishiki ブランチ: `phase2/vertical-real-data`（ワークツリー C:/dev/nishiki-phase2、基点 stage/vertical-slice 2129618）。既存ブランチへのマージなし。")
p("- 成果物: `scripts/phase2/annotations/*.annotations.json`（11場面）、`*.characters.json`、`register_audit.json`、`results/diagnose/*.diagnose.json`、`results/phase2_step1_results.json`、`results/sensor_run/qwen2.5_14b_dev.json`\n")
p("## 1. センサー実行（凍結構成）")
rt = s["runtime_totals"]
p(f"- 構成: prompt `{s['prompt_version']}`（hash `{s['prompt_hash'][:16]}…`）＋第2段後処理＋第3段話者再確認、`{s['model_spec']}`、seed {s['inference_options']['seed']}、temperature 0。設定は `scripts/phase2/phase2_sensor_config.json`（out_dir のみ差し替え）。")
p(f"- 実行時間 {rt['elapsed_sec']} 秒（第1段 {rt['stage1_elapsed_sec']} / 第3段 {rt['stage3_elapsed_sec']}）、context window 警告: {rt['any_context_window_warning']}。")
p("- 既存の凍結結果 `results/modern_v5_threestage/qwen2.5_14b_dev.json` と3作品すべて parsed_utterances が完全一致（再現確認、別途比較）。")
p("- センサー実行の単位は凍結評価と同じ「作品全体」。得られた発話注釈を nishiki segment の場面へ、引用テキスト一致を検査しつつ順序対応で割り当てた（検査は assert、全発話一致）。場面ごとの再推論は行っていない。\n")
p("## 2. 注釈の組み立て")
p("| 作品 | 場面数 | 場面別発話数 | pov_character |\n|---|---:|---|---|")
for w in W:
    x = a["works"][w]
    p(f"| {titles[w]} (`{w}`) | {x['scene_count']} | {x['scene_utterance_counts']} | {x['pov_character']} |")
p("")
p("- 全11場面の注釈を `schemas/sensor_v2.schema.json` で検証し、エラー 0。nishiki 側パーサ（deny_unknown_fields＋値検査）も通過、diagnose の warnings は3作品とも空（scene_hash 照合で stale 0）。")
p("- センサーが出さない項目の扱い（実データではない箇所）:")
p("  - `scene_meta.pov_character`: 一人称作品（ボロ家の春秋）は gold の narrator（boku）、他2作品は null。gold 由来であり、センサー出力ではない。")
p("  - `scene_meta.present_character_ids`: 場面内のセンサー予測の speaker／addressee の和集合（＋一人称の語り手）。センサー自身の出力から作ったため、`off_stage_speaker` 判定は構造的に発火しない（循環）。")
p("  - `character_states`（values_6d）、`fact_updates`: 凍結センサーは出力しないため空配列。`time` は unknown、`location_id` は null。")
p("  - 名簿: gold の characters。代名詞別名（彼・彼女・私・俺）と複数人物で重複する別名を除外、gender は gold の値を使用、honored は false。")
p("  - `register.first_person` はモデル出力のまま。`register.politeness` のみ Policy X で付け直した。\n")
rows = [r for w in W for r in a["works"][w]["register_rows"]]
import collections
c = collections.Counter((r["model_politeness"], r["policy_x"]) for r in rows)
p("## 3. 口調（Policy X による付け直し）")
p(f"- 対象発話 {len(rows)} 件。Policy X 出力 → sensor.v2 の politeness 値: polite→polite、plain→plain、ambiguous→unknown（スキーマに ambiguous が無いため）。")
p("- モデル出力 → Policy X のクロス集計:")
p("\n| モデル出力 | Policy X | 件数 |\n|---|---|---:|")
for (m, x), n in sorted(c.items()):
    p(f"| {m} | {x} | {n} |")
p("")
g = json.loads(Path("F:/マイドライブ/Claude-Sync/inbox/2026-11-20_modern_gold_frozen.json").read_text(encoding="utf-8"))
gp = {(w['id'], u['n']): u.get('politeness') for w in g['works'] for u in w['utterances']}
diff = [(r['work_id'], r['global_utterance'], gp[(r['work_id'], r['global_utterance'])], r['policy_x']) for r in rows if gp[(r['work_id'], r['global_utterance'])] != r['policy_x']]
p(f"- 凍結ゴールドとの一致: {len(rows)-len(diff)}/{len(rows)}。不一致 {len(diff)} 件: " + "; ".join(f"{w} U{k}（gold={gv}、Policy X={px}）" for w, k, gv, px in diff) + "。前者は gold で politeness=null（非発話）の1件、後者は Orion 凍結時に plain→ambiguous へ手動変更された n=18（規則出力は plain）。\n")
p("## 4. diagnose 出力（レンズ別）")
for w in W:
    x = d["works"][w]
    p(f"### {titles[w]} (`{w}`)  — schema `{x['schema']}`、{x['input']['chars']} 文字、{x['input']['scenes']} 場面、warnings {len(x['warnings'])} 件")
    p("| レンズ | status | findings(spans) | 内訳 |\n|---|---|---:|---|")
    for lid in ["kasane", "sasae", "sadame", "kamae", "nagare", "awase"]:
        l = x["lenses"][lid]
        st = l["status"] + (f"（{l['reason']}）" if l["reason"] else "")
        p(f"| {lid} | {st} | {l['span_count']} | {l['span_by_rule'] or '-'} |")
    for lid in ["kasane", "sasae", "sadame", "kamae"]:
        p(f"- {lid} scores: `{json.dumps(x['lenses'][lid]['scores'], ensure_ascii=False)}`")
    p("")
p("- findings（spans）は Sasae の pov_contamination のみ。`speaker_mismatch_count`（センサー話者とソルバー話者の不一致）は件数のみで位置は出力されないため、findings には含まれない。")
p("- Kasane は本文のみから計算され、注釈を使わない。Sadame は status=ok だが `character_states` が空のため edges_eff=0、全スコア0（意味のある値ではない）。\n")
p("## 5. findings 全件")
p(f"全 {len(d['findings'])} 件（実用性の判定は Orion）。\n")
p("| # | 作品 | レンズ | rule | 場面 | score | 範囲 | basis | 該当テキスト |\n|---:|---|---|---|---:|---:|---|---|---|")
for i, f in enumerate(d["findings"], 1):
    p(f"| {i} | {f['work_id']} | {f['lens']} | {f['rule']} | {f['scene']} | {f['score']} | {f['start']}–{f['end']} | {json.dumps(f['basis'], ensure_ascii=False)} | {f['text']} |")
p("")
p("## 6. 縦貫通の数え方")
p("| レンズ | status が ok（3作品とも） | 注釈→射影→出力で実注釈が結果に寄与 | 備考 |\n|---|---|---|---|")
p("| Kasane | ○ | ×（注釈を使わない） | 本文のみ。注釈経路は通っていない |")
p("| Sasae 第1段 | ○ | ○ | 実発話注釈＋名簿が結果に反映（speaker_mismatch 計上、pov_contamination 2件）。present／pov は上記のとおり一部が派生値 |")
p("| Sadame | ○ | ×（空入力） | character_states が空のため射影後の辺が0（edges_eff=0）。数値は空疎 |")
p("- 厳密な数え方（実注釈が出力に寄与）: **1/6**（Sasae 第1段）。")
p("- status=ok の数え方: 3/6（Kasane、Sasae、Sadame）。ただし Kasane は注釈非依存、Sadame は入力が空。")
p("- Nagare / Awase: unavailable、Kamae: Kamae A のみ（辞書走査、注釈非依存）で Kamae B は未接続。\n")
p("## 7. 限界・未検証")
p("- Sadame を実データで通すには values_6d を出すセンサー（Step 3 相当）が必要。凍結構成は出さない。今回は値を捏造していない。")
p("- present_character_ids の循環により off_stage_speaker は検証できていない。pov は gold 由来。")
p("- 話者精度は本ステップの測定対象外（凍結評価の数値は別レポート）。findings の正誤（真陽性率）は未判定。")
p("- 3作品・11場面・77引用の小標本。")
Path(sys.argv[1]).write_text("\n".join(L) + "\n", encoding="utf-8")
