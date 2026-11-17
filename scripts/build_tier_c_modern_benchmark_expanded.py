"""Build Tier-C Modern Multi-Genre Benchmark Expanded (64 samples: 4 registers x 16 samples).

Registers:
- R1: Modern Colloquial Fiction (現代口語小説, 16 samples, 400-800 chars)
- R2: Modern Essay & Criticism (現代エッセイ・評論, 16 samples, 400-800 chars)
- R3: Modern Technical & Business Prose (現代技術・ビジネス散文, 16 samples, 400-800 chars)
- R4: Aozora Bunko Colloquial Baseline (青空文庫口語ベースライン, 16 samples, 400-800 chars)

Tier-C Clean-Room Rule:
Raw text is loaded in memory, irreversible profile is extracted via extract_tier_c_irreversible_profile,
and only irreversible numerical features and metadata are written to the JSONL benchmark.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

# Ensure python/ package root is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python"))

from usir_py.v5.aozora_scene_splitter import AozoraCorpusSplitter
from usir_py.v5.modern_text_normalizer import extract_tier_c_irreversible_profile

OUTPUT_FILE = Path("data/usir_v5_eval/tier_c_modern_benchmark_expanded.jsonl")


def get_aozora_r4_texts() -> List[Dict[str, Any]]:
    """Retrieve 16 authentic Aozora Bunko colloquial texts (400-800 chars each)."""
    splitter = AozoraCorpusSplitter(Path("data"))
    manifests = splitter.load_manifests()
    man_map = {m["work_id"]: m for m in manifests}

    target_works = [
        # Original 8 works
        ("000035", "000301", "太宰治『人間失格』", 0.2, 600),
        ("000035", "001567", "太宰治『走れメロス』", 0.2, 600),
        ("000148", "000752", "夏目漱石『坊っちゃん』", 0.2, 600),
        ("000148", "000773", "夏目漱石『こころ』", 0.2, 600),
        ("000879", "000127", "芥川龍之介『羅生門』", 0.2, 600),
        ("000879", "000179", "芥川龍之介『藪の中』", 0.2, 600),
        ("000074", "000424", "梶井基次郎『檸檬』", 0.2, 600),
        ("000119", "000624", "中島敦『山月記』", 0.2, 600),
        # Extra 8 works
        ("000081", "000456", "宮沢賢治『銀河鉄道の夜』", 5000, 700),
        ("000129", "045245", "森鴎外『高瀬舟』", 1500, 700),
        ("000035", "001565", "太宰治『斜陽』", 10000, 700),
        ("000148", "000794", "夏目漱石『三四郎』", 15000, 700),
        ("000879", "000060", "芥川龍之介『地獄変』", 4000, 700),
        ("000074", "000427", "梶井基次郎『桜の樹の下には』", 200, 700),
        ("000081", "000470", "宮沢賢治『セロ弾きのゴーシュ』", 2000, 700),
        ("000129", "000689", "森鴎外『山椒大夫』", 3000, 700),
    ]

    r4_samples = []
    for idx, (auth_id, work_id, title_desc, pos_spec, win_len) in enumerate(target_works, start=1):
        m = man_map.get(work_id)
        assert m is not None, f"Missing manifest for {work_id}"
        full_text = splitter.get_cached_text(m)

        if isinstance(pos_spec, float):
            start_pos = int(len(full_text) * pos_spec)
        else:
            start_pos = pos_spec

        cand = full_text[start_pos:start_pos + win_len]
        s_idx = cand.find("。")
        if s_idx != -1 and s_idx < 100:
            cand = cand[s_idx + 1:]
        e_idx = cand.rfind("。")
        if e_idx != -1 and e_idx > 400:
            cand = cand[:e_idx + 1]

        cand = cand.strip().replace("\r\n", "\n")
        assert 400 <= len(cand) <= 800, f"Sample R4_{idx:02d} length {len(cand)} not in [400, 800]"

        r4_samples.append({
            "sample_id": f"R4_{idx:02d}",
            "register": "R4",
            "register_name": "R4_aozora_colloquial_baseline",
            "source_type": "aozora",
            "source_description": f"青空文庫口語ベースライン：{title_desc}",
            "text": cand,
        })
    return r4_samples


def get_r3_technical_texts():
    """P-2b: 原文の直書きを除去した（クリーンルーム規約 D3）。置換後の層 C は
    scripts/p2b/p2b_build_layer_c_v2.py が青空文庫キャッシュから生成する。旧原文は git 履歴に残る。"""
    raise RuntimeError("R1-R3 literal text removed in Stage P-2b; use scripts/p2b/p2b_build_layer_c_v2.py")



def get_r2_essay_texts():
    """P-2b: 原文の直書きを除去した（クリーンルーム規約 D3）。置換後の層 C は
    scripts/p2b/p2b_build_layer_c_v2.py が青空文庫キャッシュから生成する。旧原文は git 履歴に残る。"""
    raise RuntimeError("R1-R3 literal text removed in Stage P-2b; use scripts/p2b/p2b_build_layer_c_v2.py")



def get_r1_fiction_texts():
    """P-2b: 原文の直書きを除去した（クリーンルーム規約 D3）。置換後の層 C は
    scripts/p2b/p2b_build_layer_c_v2.py が青空文庫キャッシュから生成する。旧原文は git 履歴に残る。"""
    raise RuntimeError("R1-R3 literal text removed in Stage P-2b; use scripts/p2b/p2b_build_layer_c_v2.py")



def main() -> None:
    print("Collecting 64 samples across 4 registers (16 samples each)...")
    r4_samples = get_aozora_r4_texts()
    r3_samples = get_r3_technical_texts()
    r2_samples = get_r2_essay_texts()
    r1_samples = get_r1_fiction_texts()

    all_raw = r1_samples + r2_samples + r3_samples + r4_samples
    assert len(all_raw) == 64, f"Expected 64 samples, got {len(all_raw)}"

    print("Extracting Tier-C irreversible profiles (strictly numbers, no raw text in output)...")
    records = []
    for s in all_raw:
        char_len = len(s["text"])
        # Extract irreversible profile
        profile = extract_tier_c_irreversible_profile(s["text"])

        # Construct benchmark record
        rec = {
            "sample_id": s["sample_id"],
            "register": s["register"],
            "register_name": s["register_name"],
            "char_len": char_len,
            "source_type": s["source_type"],
            "source_description": s["source_description"],
            "profile": profile,
        }
        records.append(rec)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Successfully wrote {len(records)} benchmark records to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
