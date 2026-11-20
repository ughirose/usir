"""Comparison table (hybrid = strong model, all items) vs local (qwen2.5:14b), every number read from result files.
Usage: python build_comparison.py  (run from repo root of usir-hybrid; needs nishiki hybrid_diagnose_results.json copy)"""
import json
from pathlib import Path
R = Path("scripts/phase2_hybrid/results")
H = json.loads((R/"stageB_metrics_hybrid.json").read_text(encoding="utf-8"))["all"]
L = json.loads(Path("scripts/phase2_stage2/results/stageB_metrics.json").read_text(encoding="utf-8"))["all"]
UH = json.loads((R/"utterances_hybrid.json").read_text(encoding="utf-8")); UL = json.loads((R/"utterances_local.json").read_text(encoding="utf-8"))
EH = json.loads((R/"event_overlap_hybrid.json").read_text(encoding="utf-8"))
EL = json.loads(Path("C:/dev/usir-nagare-awase/scripts/phase2_nagare_awase/results_overlap_out.json").read_text(encoding="utf-8"))
C = json.loads(Path("scripts/phase2_ceiling/results/ceiling_summary.json").read_text(encoding="utf-8"))
ND = json.loads(Path("C:/dev/nishiki-hybrid/docs/handoff/hybrid_full_run_results/hybrid_diagnose_results.json").read_text(encoding="utf-8"))
f3 = lambda x: f"{x:.3f}"
rate = lambda d: f"{f3(d['rate'])} ({d['match']}/{d['n']})"
pr = lambda d: f"{f3(d['precision'])} / {f3(d['recall'])} (TP {d['tp']}, FP {d['fp']}, FN {d['fn']})"
rows = []
a = rows.append
ua, ul = UH["all"], UL["all"]
a(("発話の話者（正解が attributed の発話）", f"{ua['attr_ok']}/{ua['attr_n']} = {f3(ua['attr_ok']/ua['attr_n'])}", f"{ul['attr_ok']}/{ul['attr_n']} = {f3(ul['attr_ok']/ul['attr_n'])}"))
a(("not_speech に対し話者 null で棄権", f"{ua['not_speech_abstain']}/{ua['not_speech_n']}", f"{ul['not_speech_abstain']}/{ul['not_speech_n']}"))
a(("undetermined に対し話者 null で棄権", f"{ua['undetermined_abstain']}/{ua['undetermined_n']}", f"{ul['undetermined_abstain']}/{ul['undetermined_n']}"))
a(("同席 P / R", pr(H["present"]), pr(L["present"])))
a(("時間順序の一致", rate(H["time_relative_order"]), rate(L["time_relative_order"])))
a(("明示時刻の一致", rate(H["time_explicit"]), rate(L["time_explicit"])))
a(("場所の一致", rate(H["location"]), rate(L["location"])))
a(("事実 P / R（述語＋対象）", pr(H["fact_updates_pred_obj"]), pr(L["fact_updates_pred_obj"])))
a(("事実の出力件数（ゴールは 74）", str(sum(H["fact_pred_predicate_counts"].values())), str(sum(L["fact_pred_predicate_counts"].values()))))
a(("特性・場面単位 非0軸の同符号（ゴール非0軸 全 97）", C["trait_scene_unit_like_for_like"]["strong_on_all_97_gold_nonzero"], C["trait_scene_unit_like_for_like"]["local_on_all_97"]))
a(("特性・ベクトル完全一致", rate(H["values_6d"]["vector_exact"]), rate(L["values_6d"]["vector_exact"])))
a(("特性・人物単位（47 軸に限定）", C["trait_person_unit"]["strong_47_subset(excl yozo,hagiwara: inferred from 54-7=47)"], C["trait_person_unit"]["local_v3_per_person_reported"] + "（v3 報告）"))
a(("出来事 top3 の重なり（スロットあり場面）", f"{EH['mean_overlap_filled']:.4f}（{EH['n_filled']}/35 場面）", f"{EL['mean_overlap_filled']:.4f}（{EL['n_filled']}/35 場面）"))
a(("出来事 top3 の重なり（35 場面、null=0）", f"{EH['mean_overlap_all35_null_as_0']:.4f}", f"{EL['mean_overlap_all35_null_as_0']:.4f}"))
a(("出来事 top3 一致数の分布 0/1/2/3", "/".join(str(EH['distribution_filled'].get(k, 0)) for k in "0123"), "/".join(str(EL['distribution_filled'].get(k, 0)) for k in "0123")))
a(("出来事 1 位一致", f"{EH['top1_match_filled']}/{EH['n_filled']}", f"{EL['top1_match_filled']}/{EL['n_filled']}"))
a(("参考: 定数 top3 {E03,E02,E10}（自己データ）", f"{EH['baseline_constant_top3']['mean_overlap_all35_in_sample']:.4f}", f"{EL['baseline_constant_top3']['mean_overlap_all35_in_sample']:.4f}"))
a(("参考: ランダム期待値", f"{EH['baseline_random_expected']:.4f}", f"{EL['baseline_random_expected']:.4f}"))
W = sorted(ND["works"])
cnt = lambda s, f: sum(f(ND["works"][w][s]) for w in W)
a(("diagnose: 6 レンズ ok の作品数", " ".join(f"{lid}:{cnt('hybrid', lambda x, lid=lid: x['lenses'][lid]['status']=='ok')}" for lid in ("kasane","sasae","kamae","sadame","nagare","awase")),
   " ".join(f"{lid}:{cnt('local', lambda x, lid=lid: x['lenses'][lid]['status']=='ok')}" for lid in ("kasane","sasae","kamae","sadame","nagare","awase"))))
a(("Nagare フルカーネル（annotated_slot）の作品数", f"{cnt('hybrid', lambda x: x['lenses']['nagare']['labels'].get('event_axis_source')=='annotated_slot')}/7", f"{cnt('local', lambda x: x['lenses']['nagare']['labels'].get('event_axis_source')=='annotated_slot')}/7"))
a(("Awase scenes_scored 合計", str(cnt('hybrid', lambda x: x['lenses']['awase']['scores']['scenes_scored'])), str(cnt('local', lambda x: x['lenses']['awase']['scores']['scenes_scored']))))
a(("Awase irony_mean が正の作品数", f"{cnt('hybrid', lambda x: x['lenses']['awase']['scores']['irony_mean']>0)}/7", f"{cnt('local', lambda x: x['lenses']['awase']['scores']['irony_mean']>0)}/7"))
a(("Awase detachment_mean が正の作品数", f"{cnt('hybrid', lambda x: x['lenses']['awase']['scores']['detachment_mean']>0)}/7", f"{cnt('local', lambda x: x['lenses']['awase']['scores']['detachment_mean']>0)}/7"))
a(("Sadame edges_eff > 0 の作品数", f"{cnt('hybrid', lambda x: x['lenses']['sadame']['scores']['edges_eff']>0)}/7", f"{cnt('local', lambda x: x['lenses']['sadame']['scores']['edges_eff']>0)}/7"))
a(("Sadame edges_eff 合計", str(cnt('hybrid', lambda x: x['lenses']['sadame']['scores']['edges_eff'])), str(cnt('local', lambda x: x['lenses']['sadame']['scores']['edges_eff']))))
for rule in ("off_stage_speaker","presence_unconfirmed","move_destination_mismatch","place_conflict","stay_location_conflict","double_location","pov_contamination","quote_bracket"):
    a((f"Sasae findings: {rule}", str(cnt('hybrid', lambda x: x['sasae_rule_counts'].get(rule,0))), str(cnt('local', lambda x: x['sasae_rule_counts'].get(rule,0)))))
a(("Sasae findings 合計", str(cnt('hybrid', lambda x: sum(x['sasae_rule_counts'].values()))), str(cnt('local', lambda x: sum(x['sasae_rule_counts'].values())))))
md = ["| 項目 | 強いモデル（hybrid） | ローカル qwen2.5:14b |", "|---|---|---|"] + [f"| {k} | {x} | {y} |" for k, x, y in rows]
(R/"comparison.md").write_text("\n".join(md)+"\n", encoding="utf-8")
(R/"comparison.json").write_text(json.dumps([{"item": k, "hybrid": x, "local": y} for k, x, y in rows], ensure_ascii=False, indent=1)+"\n", encoding="utf-8")
print("\n".join(md))
