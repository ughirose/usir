import json
from pathlib import Path
R = Path("scripts/phase2_ceiling/results")
s = json.loads((R/"stageB_metrics_strong.json").read_text(encoding="utf-8"))["all"]
l = json.loads(Path("scripts/phase2_stage2/results/stageB_metrics.json").read_text(encoding="utf-8"))["all"]
e = json.loads((R/"extra_metrics_strong.json").read_text(encoding="utf-8"))
pick = lambda a: {"present_P": a["present"]["precision"], "present_R": a["present"]["recall"], "order": a["time_relative_order"]["rate"],
  "explicit_time": a["time_explicit"]["rate"], "location": a["location"]["rate"],
  "fact": a["fact_updates_pred_obj"], "fact_strict_subject": a["fact_updates_strict_subject"],
  "fact_pred_count": sum(a["fact_pred_predicate_counts"].values()),
  "trait_scene_unit_gold_nonzero_same_sign_on_matched_char_scenes": a["values_6d"]["gold_nonzero_axes_same_sign"],
  "vector_exact": a["values_6d"]["vector_exact"], "polarity_both_nonzero": a["values_6d"]["polarity_both_nonzero"], "states_character_set": a["states_character_set"]}
out = {"strong_model": pick(s), "local_qwen2.5_14b_stage2": pick(l),
  "trait_scene_unit_like_for_like": {"strong_on_all_97_gold_nonzero": "48/97", "strong_on_local_matched_66": "31/66", "local_on_all_97": "17/97", "local_on_matched_66": "17/66"},
  "trait_person_unit": {"rule": "gold trait = majority sign over gold scenes (ties->0, excluded); pred trait aggregated by the same rule from scene-level states",
    "strong_all_54": "30 same / 1 opposite / 23 zero", "strong_47_subset(excl yozo,hagiwara: inferred from 54-7=47)": "24 same / 0 opposite",
    "local_stage2_scene_states_aggregated_all_54": "12 same / 4 opposite", "local_v3_per_person_reported": "5/47"},
  "fact_manual_semantic_match_adjustment": {"note": "6 pairs judged by the annotator (same model) as semantically equivalent but not matched by the script rule", "tp": 63, "precision": 63/109, "recall": 63/74}}
(R/"ceiling_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1)+"\n", encoding="utf-8")
print(round(63/109,3), round(63/74,3))
