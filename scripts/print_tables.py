import json

with open("artifacts/benchmark_v3_english_oof_report.json", "r", encoding="utf-8") as f:
    t4 = json.load(f)

print("=== ALL MODELS IN TABLE 4 ===")
for model_key, m in t4["table4_models"].items():
    print(f"{model_key} | ROC-AUC={m['overall_roc_auc']:.3f} | PR-AUC={m['overall_pr_auc']:.3f} | Macro-F1={m['macro_f1']:.3f} | R1-AUC={m['regime1_roc_auc']:.3f} | R2-AUC={m['regime2_roc_auc']:.3f} | D2={m['unbiased_mahalanobis_d2']:.2f}")

print("\nMarkdown from report:")
print(t4.get("table4_markdown", ""))
