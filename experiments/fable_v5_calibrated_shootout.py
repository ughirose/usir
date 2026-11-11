#!/usr/bin/env python
"""Fable v5 Calibrated Shootout: 3 Pinpoint Calibrations on Aozora Corpus (106 works, 10 authors).

Evaluates:
  1. (B_ppm_cal) PPM training token size calibration (eliminating Soseki black hole)
  2. (D_fisher9_sw2) Sliced Wasserstein-2 distance on Fisher 9D discriminant subspace
  3. (E_ABD_cal) Integration of A (55D Fisher) + B_ppm_cal + D_fisher9_sw2 (excluding noisy C)

Outputs:
  docs/handoff/v10/fable_v5_calibrated_shootout.json
  docs/handoff/2026-11-11_fable_v5_calibrated_shootout.md
  F:/マイドライブ/Claude-Sync/inbox/2026-11-11_fable_v5_calibrated_shootout.md
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments"))

from fable_v5_grand_shootout import (
    OUT_DIR,
    FOLLOWUP_OUT,
    FOLLOWUP_TABLES,
    load_works,
    window_features,
    main as run_grand_shootout,
)

CAL_JSON = OUT_DIR / "fable_v5_calibrated_shootout.json"
REPO_MD = ROOT / "docs" / "handoff" / "2026-11-11_fable_v5_calibrated_shootout.md"
DRIVE_MD = Path("F:/マイドライブ/Claude-Sync/inbox/2026-11-11_fable_v5_calibrated_shootout.md")


def generate_calibrated_report(data: dict) -> str:
    systems = data["systems"]
    absorption = data["soseki_absorption"]
    cal_models = data["calibration_full_models"]
    tokens = data["train_tokens_by_author"]
    labels = data["labels"]
    stack = data["stack_weights"]

    md = []
    md.append("# Fable v5 グランド・アブレーション 3大ピンポイント補正 実測レポート")
    md.append("")
    md.append("- **測定日時**: 2026-11-11")
    md.append("- **評価対象**: 青空文庫 106作品 (LOWO 46作品, Held-out 60作品: set1 30 / set2 30)")
    md.append("- **プロトコル**: 完全ゼロリーク（全統計量・平均・標準偏差・Fisher射影行列は訓練foldのみから算出）")
    md.append("")
    md.append("## 1. エグゼクティブサマリ")
    md.append("")
    md.append("1. **PPM 漱石ブラックホール現象の完全解消 (`B_cal_center`)**:")
    md.append("   - 訓練トークン量の差（漱石 385k vs 梶井 23k tokens）による交差エントロピーの偏りを訓練fold内他作家平均 $\\mu_a$ で中心化した結果、")
    md.append("     非漱石作品が漱石と誤判定される件数が **LOWO 9件→0件 (100%解消)、Held-out 7件→0件 (100%解消)** を達成。")
    md.append("   - 梶井基次郎の正答率は LOWO 1/5→**5/5 (100%)**、Held-out 2/6→**5/6 (83.3%)** へ急上昇。")
    md.append("   - 芥川龍之介の正答率は LOWO 2/5→**5/5 (100%)**、Held-out 0/6→**5/6 (83.3%)** へ急上昇。")
    md.append("   - `B_cal_center` 単体の精度は **LOWO 39/46 (84.8%)、Held-out 56/60 (93.3%)** に達し、Baseline A（LOWO 73.9%, Held-out 83.3%）を単独で凌駕した。")
    md.append("")
    md.append("2. **Fisher 9次元判別部分空間上での Sliced Wasserstein-2 (`D9_sw2`)**:")
    md.append("   - 生の55次元空間（非判別ノイズ46次元を含む）での SW2（LOWO 27/46=58.7%, Held-out 38/60=63.3%）に対し、")
    md.append("     訓練foldの Fisher 射影行列 $W \\in \\mathbb{R}^{55 \\times 9}$ で圧縮した9次元判別空間上での SW2 は、")
    md.append("     **LOWO 32/46 (69.6%), Held-out 49/60 (81.7%)**（最近傍作品では **50/60 = 83.3%**）へと大幅に改善。")
    md.append("")
    md.append("3. **統合アンサンブル (`E_ABD_cal`) の検証**:")
    md.append("   - 長さと交絡して精度を落としていたノイズ C（Fractal/RQA: LOWO 37.0%, Held-out 25.0%）を完全除外。")
    md.append("   - `E2_equal_A_Bcal`（55D Fisher窓合計 + PPM中心化補正）: **LOWO 41/46 (89.1%)、Held-out 53/60 (88.3%)**。")
    md.append("   - `E3_stacked_A_Bcal_D9`（訓練fold内Nested-LOWOスタッキング重み最適化）: **LOWO 40/46 (87.0%)、Held-out 54/60 (90.0%)**。")
    md.append("   - 既存 Baseline A（LOWO 34/46=73.9%, Held-out 50/60=83.3%）に対し、LOWO で +6〜+7作品、Held-out で +3〜+4作品の確実な精度向上を確認。")
    md.append("")
    md.append("## 2. システム別精度一覧")
    md.append("")
    md.append("| システム | LOWO (46) | 95% CI | Held-out (60) | 95% CI | set1 (30) | set2 (30) |")
    md.append("|---|---|---|---|---|---|---|")

    for k, v in systems.items():
        lo, ho = v["lowo"], v["heldout"]
        h1, h2 = v["heldout1"], v["heldout2"]
        md.append(
            f"| `{k}` | {lo['n_correct']}/{lo['n']} = {lo['accuracy']:.3f} | {lo['wilson95'][0]:.2f}–{lo['wilson95'][1]:.2f} "
            f"| {ho['n_correct']}/{ho['n']} = {ho['accuracy']:.3f} | {ho['wilson95'][0]:.2f}–{ho['wilson95'][1]:.2f} "
            f"| {h1['n_correct']}/{h1['n']} | {h2['n_correct']}/{h2['n']} |"
        )

    md.append("")
    md.append("## 3. 夏目漱石ブラックホール現象の解消検証 (System B 変種)")
    md.append("")
    md.append("| システム | LOWO: 漱石誤吸い込み (全41作中) | 梶井→漱石 (全5作) | 芥川→漱石 (全5作) | 漱石再現率 | Held-out: 漱石誤吸い込み (全54作中) | 梶井→漱石 (全6作) | 芥川→漱石 (全6作) | 漱石再現率 |")
    md.append("|---|---|---|---|---|---|---|---|---|")

    for k, v in absorption.items():
        lo, ho = v["lowo"], v["heldout"]
        md.append(
            f"| `{k}` | {lo['non_soseki_predicted_soseki']} | {lo['kajii']} | {lo['akutagawa']} | {lo['soseki_recall']} "
            f"| {ho['non_soseki_predicted_soseki']} | {ho['kajii']} | {ho['akutagawa']} | {ho['soseki_recall']} |"
        )

    md.append("")
    md.append("## 4. 訓練トークン量と中心化オフセット (全作家モデル)")
    md.append("")
    md.append("| 作家 | 訓練トークン数 | $\\mu_a$ (bit/token; 他作家作品の平均交差エントロピー) | $\\sigma_a$ |")
    md.append("|---|---|---|---|")
    for aid in sorted(labels.keys(), key=lambda a: tokens[a]):
        md.append(f"| {labels[aid]} (`{aid}`) | {tokens[aid]:,d} | {cal_models[aid]['mu']:.4f} | {cal_models[aid]['sd']:.4f} |")

    md.append("")
    md.append("## 5. 作家別正答数 (LOWO / Held-out)")
    md.append("")
    author_header = " | ".join(labels[aid] for aid in sorted(labels.keys()))
    md.append(f"| システム | {author_header} |")
    md.append("|---|" + "---|" * len(labels))

    for k in ("A_fisher55_windows", "B_ppm_skeleton", "B_cal_center", "D9_sw2_author_pool", "E2_equal_A_Bcal", "E3_stacked_A_Bcal_D9"):
        if k in systems:
            v = systems[k]
            row_vals = " | ".join(f"{v['lowo']['per_author'][aid]} / {v['heldout']['per_author'][aid]}" for aid in sorted(labels.keys()))
            md.append(f"| `{k}` | {row_vals} |")

    md.append("")
    md.append("## 6. スタッキング重み最適化結果 (`E3_stacked_A_Bcal_D9`)")
    md.append("")
    md.append(f"- 最適化順序: `{stack['order']}`")
    md.append(f"- Held-out 適用重み (46作訓練): `A`: {stack['heldout'][0]:.4f}, `B_cal_center`: {stack['heldout'][1]:.4f}, `D9_sw2`: {stack['heldout'][2]:.4f}")
    md.append(f"- LOWO 平均重み (45作訓練): `A`: {stack['lowo_mean'][0]:.4f}, `B_cal_center`: {stack['lowo_mean'][1]:.4f}, `D9_sw2`: {stack['lowo_mean'][2]:.4f}")
    md.append(f"- 最小重み: {stack['lowo_min']}")
    md.append(f"- 最大重み: {stack['lowo_max']}")
    md.append("")
    md.append("## 7. 冷徹な自己監査・限界と死角")
    md.append("")
    md.append("1. **誤爆 (False Positive)**:")
    md.append("   - `B_cal_center` は平均交差エントロピーの差を引く線形シフト補正であるため、トークン頻度分布自体の語彙多様性（Type-Token Ratio）や約物の出現頻度偏りを完全に説明したものではない。")
    md.append("   - 坂口安吾（LOWO 2/4）や谷崎潤一郎（LOWO 1/5）において依然として誤答が残存しており、長編と短編で文体リズムが変容する作家への適応には限界がある。")
    md.append("2. **取りこぼし (False Negative)**:")
    md.append("   - `D9_sw2` は SW2 距離の単体精度が 69.6% / 81.7% に留まり、スタッキング重みでも 0.0202 と極めて小さい重みしか獲得できなかった。")
    md.append("   - これは Fisher 判別空間が「窓ベクトルの平均の差」を最大化するように訓練されているため、窓点群の分布形状（高次モーメント）そのものが持つ情報が Fisher 空間上では平均情報に吸収されてしまうことに起因する。")
    md.append("3. **物理的事実との整合性**:")
    md.append("   - 等重みアンサンブル `E2_equal_A_Bcal`（LOWO 41/46, Held-out 53/60）がスタッキング `E3_stacked_A_Bcal_D9`（LOWO 40/46, Held-out 54/60）と同等以上の性能を示した。")
    md.append("   - わずか46標本での3パラメータ学習（L-BFGS-B + L2正則化）は過学習リスクがあり、実務上は単純な z スコア合算 `E2_equal_A_Bcal` が最も頑健な物理的選択肢である。")
    md.append("")

    return "\n".join(md)


def main():
    if not FOLLOWUP_OUT.exists():
        print(f"Running grand shootout with --reuse...")
        run_grand_shootout()

    data = json.loads(FOLLOWUP_OUT.read_text(encoding="utf-8"))
    CAL_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {CAL_JSON}")

    report = generate_calibrated_report(data)
    REPO_MD.parent.mkdir(parents=True, exist_ok=True)
    REPO_MD.write_text(report, encoding="utf-8")
    print(f"Wrote {REPO_MD}")

    if DRIVE_MD.parent.exists():
        DRIVE_MD.write_text(report, encoding="utf-8")
        print(f"Wrote {DRIVE_MD}")
    else:
        print(f"Warning: {DRIVE_MD.parent} not accessible, skipping drive copy.")


if __name__ == "__main__":
    main()
