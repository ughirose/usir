"""Render the stage-2 report; every number is read from the artifacts (stageA summaries, stageB metrics, stageC results/audit)."""
import json
import sys
from pathlib import Path

U = Path(__file__).resolve().parent
N = Path("C:/dev/nishiki-phase2b/scripts/phase2b/results")
A = json.loads((U / "annotations/stageA_summary_all.json").read_text(encoding="utf-8"))
B = json.loads((U / "results/stageB_metrics.json").read_text(encoding="utf-8"))
BR = json.loads((U / "results/stageB_metrics_rerun_modern.json").read_text(encoding="utf-8"))["modern_11"]
C = json.loads((N / "stageC_results.json").read_text(encoding="utf-8"))
CA = json.loads((N / "stageC_findings_audit.json").read_text(encoding="utf-8"))


def f(x):
    return "-" if x is None else f"{x:.3f}"


def prf(r):
    return f"{r['tp']} / {r['fp']} / {r['fn']} | {f(r['precision'])} | {f(r['recall'])} | {f(r['f1'])}"


L = []
p = L.append
m, f9, al = B["modern_11"], B["f9dev_24"], B["all"]
SETS = (("現代文", m), ("F9 Dev", f9), ("全体", al))

p("# Phase 2 第2段 センサー注釈範囲の拡張・精度測定・再貫通 レポート\n")
p("- 日付: 2026-11-20 / 対象: 公開テキストのみ（現代文3作品11場面＋F9 Dev 4作品24場面。妻の原稿・F9 holdout は不使用）")
p("- usir: ブランチ `phase2/sensor-full-annotation`（ワークツリー C:/dev/usir-sensor-full、基点 phase2/step1-real-annotation 592dca0）／ nishiki: `phase2/vertical-full-annotation`（C:/dev/nishiki-phase2b、基点 phase2/vertical-real-data f4e3eba）。既存ブランチへのマージなし。")
p("- モデル: qwen2.5:14b（凍結構成と同じ）、seed 42、temperature 0。\n")

p("## 1. 結論（数値は後節の表から転記）")
p(f"- 同席者: 現代文11場面で precision {f(m['present']['precision'])} / recall {f(m['present']['recall'])}。発話の話者・宛先と一人称語り手から機械的に作った同席者（比較用）は precision {f(m['present_mechanical_baseline']['precision'])} / recall {f(m['present_mechanical_baseline']['recall'])}。センサーの上積みは TP {m['present']['tp']} 対 {m['present_mechanical_baseline']['tp']}。F9 Dev では機械的方式の recall が上回る（{f(f9['present_mechanical_baseline']['recall'])} 対 {f(f9['present']['recall'])}）。")
p(f"- 時間（直前場面との関係）: 現代文 {f(m['time_relative_order']['rate'])}、F9 Dev {f(f9['time_relative_order']['rate'])}。場所: 現代文 {f(m['location']['rate'])}、F9 Dev {f(f9['location']['rate'])}。")
p(f"- 事実更新: 現代文 precision {f(m['fact_updates_pred_obj']['precision'])} / recall {f(m['fact_updates_pred_obj']['recall'])}、F9 Dev precision {f(f9['fact_updates_pred_obj']['precision'])} / recall {f(f9['fact_updates_pred_obj']['recall'])}。実用水準に達していない。")
p(f"- values_6d: 6軸完全一致は現代文 {m['values_6d']['vector_exact']['match']}/{m['values_6d']['vector_exact']['n']}、F9 Dev {f9['values_6d']['vector_exact']['match']}/{f9['values_6d']['vector_exact']['n']}。軸ごとの一致率 {f(m['values_6d']['axis_exact']['rate'])}（現代文）は全軸0固定の基準 {f(m['values_6d']['all_zero_baseline']['axis_exact_rate'])} を下回る。ゴールドで非0の軸のうち符号まで一致した割合は現代文 {f(m['values_6d']['gold_nonzero_axes_same_sign']['rate'])}、F9 Dev {f(f9['values_6d']['gold_nonzero_axes_same_sign']['rate'])}。")
tot = len(C["works"])
sad = [(w, x["results"]["full"]["sadame"]["scores"]) for w, x in C["works"].items()]
nz = [w for w, s in sad if s["total"] > 0]
ed = [w for w, s in sad if s["edges_eff"] > 0]
p(f"- Sadame: {tot} 作品中 edges_eff > 0 が {len(ed)} 作品、total > 0 が {len(nz)} 作品。第1段（edges_eff = 0）からは前進したが、入力の values_6d が上記の精度であるため、スコアの意味は未検証。")
c = CA["counts"]
p(f"- Sasae findings 全 {len(CA['findings'])} 件: off_stage_speaker は {c.get('off_stage_speaker:false_positive', 0)} 件が誤検出（ゴールド上は在場する話者を、センサーの同席者が落としたため）、{c.get('off_stage_speaker:plausible_true', 0)} 件が妥当。pov_contamination は {c.get('pov_contamination:false_positive', 0)} 件が誤検出、{c.get('pov_contamination:plausible_true', 0)} 件が妥当の可能性。quote_bracket {c.get('quote_bracket:true_structural', 0)} 件は場面境界が引用の途中に落ちる構造的事実。")
p("")

p("## 2. Stage A: センサー拡張\n")
p("- 実装: `python/usir_py/sensor_eval/scene_annotator.py`（場面プロンプト scene_v1、構造化出力スキーマ、決定論バリデータ）、`scripts/phase2_stage2/run_full_annotation.py`（実行）。")
p("- 構成: 発話（話者・宛先・evidence・一人称）は凍結構成（v5＋第2段後処理＋第3段話者再確認）の既存出力をそのまま使用（再推論なし。現代文は phase2 第1段で凍結結果との完全一致を確認済み、F9 Dev は凍結結果 `scripts/step1/results/v5_threestage/qwen2.5_14b_dev.json`）。口調は Policy X で付け直し。scene_meta／character_states／fact_updates は場面ごとに qwen2.5:14b を1回呼んで出力。")
p("- 同席者は本文の根拠（basis 語句）付きで出力させ、発話者からの機械的な派生は行っていない。basis は注釈ファイルには入れず `*.trace.json` に保存。")
p("- 決定論バリデータ: 名簿外 ID の拒否、values_6d の範囲（6要素・{-1,0,+1}）、evidence 必須、述語 enum、basis／場所語句／時刻語句が場面本文の部分文字列であること、sensor.v2 JSON Schema 検証。不正項目は修復せず除外して件数を記録。")
t = A["total"]
p(f"- 結果（{t['scenes']}場面）: 採用 同席者 {t['present']} 件／人物状態 {t['states']} 件／事実更新 {t['facts']} 件、除外 {t['rejected']} 件、スキーマ違反 {t['schema_errors']} 件、JSON 解析失敗 {t['parse_failures']} 件、LLM 時間合計 {t['llm_sec']} 秒。")
p("- 除外の内訳: " + "、".join(f"{k} {v}" for k, v in t["reasons"].items()) + "。")
p("\n| 作品 | 場面 | 同席者 | 状態 | 事実更新 | 除外 | LLM 秒 |\n|---|---:|---:|---:|---:|---:|---:|")
for w, x in A["works"].items():
    p(f"| `{w}` | {x['scenes']} | {x['present']} | {x['states']} | {x['facts']} | {x['rejected']} | {x['llm_sec']} |")
p("")

p("## 3. Stage B: 精度測定\n")
p("- ゴールド: `scripts/phase2_stage2/gold/stage_b_gold.json`（35場面）。セッションのモデルが場面本文を直接読んで作成。作成時点でセンサーの場面注釈出力は開いていない（ゴールドのコミット ac9c5ea はセンサー注釈のコミットより前）。限界: 作成者1名、二重注釈なし、価値観・事実更新は主観を含む。")
p("- 照合規則は測定前に固定（`measure_stage_b.py` 冒頭）。事実更新は述語一致＋対象語句の一致（包含、または文字 bigram の重なり係数 0.5 以上）で1対1対応、`述語＋対象＋主語` は主語も要求。")
p("\n### 同席者・時間・場所\n")
p("| 集合 | 場面 | 同席者 TP/FP/FN | P | R | F1 | 時間(順序) | 時間(表現) | 場所 |\n|---|---:|---|---:|---:|---:|---:|---:|---:|")
for name, r in SETS:
    pr = r["present"]
    tr, te, lo = r["time_relative_order"], r["time_explicit"], r["location"]
    p(f"| {name} | {r['scenes']} | {pr['tp']} / {pr['fp']} / {pr['fn']} | {f(pr['precision'])} | {f(pr['recall'])} | {f(pr['f1'])} | {tr['match']}/{tr['n']} = {f(tr['rate'])} | {te['match']}/{te['n']} = {f(te['rate'])} | {lo['match']}/{lo['n']} = {f(lo['rate'])} |")
p("\n比較用（発話の話者・宛先＋一人称語り手の和集合）:\n")
p("| 集合 | TP/FP/FN | P | R | F1 |\n|---|---|---:|---:|---:|")
for name, r in SETS:
    pr = r["present_mechanical_baseline"]
    p(f"| {name} | {pr['tp']} / {pr['fp']} / {pr['fn']} | {f(pr['precision'])} | {f(pr['recall'])} | {f(pr['f1'])} |")
p("\n### 事実更新（述語＋対象）\n")
p("| 集合 | 方式 | TP / FP / FN | P | R | F1 |\n|---|---|---|---:|---:|---:|")
for name, r in SETS:
    p(f"| {name} | 述語＋対象 | {prf(r['fact_updates_pred_obj'])} |")
    p(f"| {name} | 述語＋対象＋主語 | {prf(r['fact_updates_strict_subject'])} |")
p("\n### values_6d\n")
p("| 集合 | 照合できた人物×場面 | 6軸完全一致 | 軸ごと一致 | 両方非0の軸の符号一致 | ゴールド非0軸のうち符号一致 | 全0固定の軸一致（基準） | 人物集合 P / R |\n|---|---:|---:|---:|---:|---:|---:|---|")
for name, r in SETS:
    v = r["values_6d"]
    p(f"| {name} | {v['matched_character_scenes']} | {v['vector_exact']['match']}/{v['vector_exact']['n']} | {f(v['axis_exact']['rate'])} | {v['polarity_both_nonzero']['match']}/{v['polarity_both_nonzero']['n']} | {v['gold_nonzero_axes_same_sign']['match']}/{v['gold_nonzero_axes_same_sign']['n']} | {f(v['all_zero_baseline']['axis_exact_rate'])} | {f(r['states_character_set']['precision'])} / {f(r['states_character_set']['recall'])} |")
p("\n- 符号の扱い: 値が {-1,0,+1} のため、符号一致は値一致と同じ意味になる。そこで「両方非0の軸での極性一致」と「ゴールドが非0の軸で符号まで当てた割合」を別に数えた。")
p("\n### 実行間のばらつき（現代文11場面を同じ設定で再実行）\n")


def rr(r):
    return (f(r['present']['precision']), f(r['present']['recall']), f(r['location']['rate']), f(r['fact_updates_pred_obj']['precision']), f(r['fact_updates_pred_obj']['recall']), f(r['values_6d']['axis_exact']['rate']), f(r['values_6d']['gold_nonzero_axes_same_sign']['rate']))


p("| 実行 | 同席者 P | 同席者 R | 場所 | 事実 P | 事実 R | 軸一致 | 非0軸の符号一致 |\n|---|---:|---:|---:|---:|---:|---:|---:|")
p("| 1回目 | " + " | ".join(rr(m)) + " |")
p("| 再実行 | " + " | ".join(rr(BR)) + " |")
p("- seed 固定・temperature 0 でも出力が一致しない場面がある（再実行で 11 場面中 8 場面が scene_meta／character_states／fact_updates まで一致、3 場面が不一致。比較は別途実行し、結果は本レポートに転記）。上の差は実行間のばらつきの目安。")

p("\n## 4. Stage C: 再貫通（nishiki 側）\n")
p("- 入力: 7作品35場面の拡張注釈と名簿（`scripts/phase2b/annotations/`）。`nishiki diagnose --characters --annotations`、出力 schema は `nishiki.diagnose/3`（候補版）。Rust コード無変更（バイナリは f4e3eba と同一コード）。warnings は全作品で 0。")
p("\n| 作品 | 場面 | Sadame edges_eff | e_inter | e_intra | total | intra_characters | Sasae off_stage | Sasae pov_contam | speaker_mismatch |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for w, x in C["works"].items():
    s = x["results"]["full"]["sadame"]["scores"]
    q = x["results"]["full"]["sasae"]["scores"]
    p(f"| {x['title']} (`{w}`) | {x['scenes']} | {s['edges_eff']} | {f(s['e_inter'])} | {f(s['e_intra'])} | {f(s['total'])} | {s['intra_characters']} | {q['off_stage_utterance_count']} | {q['fid_pov_contamination_count']} | {q['speaker_mismatch_count']} |")
p("\n### 縦貫通の数え直し（実注釈が出力に寄与したレンズ数）\n")
p("レンズ6個（kasane, sasae, sadame, kamae, nagare, awase）。`no_annotations`（注釈なし）と `full` の出力を比較。nagare／awase は注釈生成が未移植のため unavailable。")
cnt = {}
for w, x in C["works"].items():
    for lens, v in x["contrib_full_vs_none"].items():
        cnt[lens] = cnt.get(lens, 0) + int(v)
p(f"- 注釈の有無で出力が変わったレンズ（作品数／{tot}）: " + "、".join(f"{k} {v}" for k, v in cnt.items()) + "。寄与したレンズは 6 個中 " + str(sum(1 for v in cnt.values() if v > 0)) + " 個。第1段は sasae のみ寄与（sadame は空配列で edges_eff=0）。")
cs = {"sadame": 0, "sasae": 0}
for w, x in C["works"].items():
    r = x["results"]
    cs["sadame"] += int(r["full"]["sadame"] != r["no_character_states"]["sadame"])
    cs["sasae"] += int(r["full"]["sasae"] != r["no_present"]["sasae"])
p(f"- 新項目の寄与（アブレーション）: character_states を空にすると sadame の出力が変わる作品 {cs['sadame']}/{tot}。present_character_ids を空にすると sasae の出力が変わる作品 {cs['sasae']}/{tot}。")
p("- fact_updates と location／time は nishiki 側のレンズが未消費（Sasae 第2段が Rust 側に未実装）。diagnose 出力には寄与していない。")
p("\n### Sasae 第2段の探索的 probe（注釈のみから計算する Python、nishiki 本体の出力ではない）\n")
pr = {}
for w, x in C["works"].items():
    for q in x["sasae_stage2_probe"]:
        pr[q["rule"]] = pr.get(q["rule"], 0) + 1
p("- 検出件数: " + "、".join(f"{k} {v}" for k, v in pr.items()) + "。内訳は `stageC_results.json` の `sasae_stage2_probe`。妥当性は未判定。")
p(f"\n### findings 全件（{len(CA['findings'])}件）\n")
p("位置・基準・本文・判定理由の全件は `scripts/phase2b/results/stageC_findings_audit.json`。判定者は1名（セッションのモデル）で盲検ではない。")
p("\n| 規則 | 判定 | 件数 |\n|---|---|---:|")
for k, v in sorted(CA["counts"].items()):
    a, b = k.split(":")
    p(f"| {a} | {b} | {v} |")
p("\n| # | 作品 | 規則 | 場面 | 本文（先頭40字） | 判定 |\n|---:|---|---|---:|---|---|")
for x in CA["findings"]:
    p(f"| {x['index']} | `{x['work_id']}` | {x['rule']} | {x['scene']} | {x['text'][:40].replace(chr(10), ' ')} | {x['verdict']} |")

p("\n## 5. 限界と未検証\n")
p("- ゴールドは作成者1名で、二重注釈・評価者間一致なし。values_6d と fact_updates のゴールドは特に主観を含む。")
p("- present_character_ids の誤りの一部は、basis 部分文字列検査で正しい人物が除外された結果（例: 他人の夏 s1 は同席者が空）。検査を緩めた場合の精度は未測定。")
p("- 場面プロンプトは1版のみ。ゴールド作成後に調整していない。")
p("- 発話部分は再推論していない。F9 Dev の凍結結果は本セッションで再実行していない。")
p("- location_id は本文中の語句をそのまま入れており、台帳IDではない。pov_character は gold の narrator（一人称作品）または null で、センサー出力ではない。")
p("- Sadame のスコアは kappa 未較正で、入力の values_6d の精度も低いため、意味づけは未検証。")
p("- 出力は実行間で完全には再現しない（上記の再実行）。")
p("\n## 6. 成果物\n")
p("- usir `phase2/sensor-full-annotation`: `python/usir_py/sensor_eval/scene_annotator.py`、`scripts/phase2_stage2/`（run_full_annotation.py、measure_stage_b.py、summarize_stage_a.py、render_report.py、gold/、annotations/、annotations_rerun/、results/stageB_metrics*.json）。")
p("- nishiki `phase2/vertical-full-annotation`: `scripts/phase2b/`（run_stage_c.py、audit_findings.py、annotations/、results/）、`docs/handoff/STATUS_phase2_stage2.md`。")
out = Path(sys.argv[1])
out.write_text("\n".join(L) + "\n", encoding="utf-8")
print("written", out)
