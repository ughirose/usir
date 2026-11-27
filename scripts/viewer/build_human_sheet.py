"""H4: build the blind human-judgement sheet (markdown) from the S1 flags of the 9 public works (measure/) and 『弟子』 (measure_E/),
plus controls from unflagged scenes. Rule names / scores / machine verdicts are NOT shown. The key goes to measure_H/key.json (repo only).
usage: python build_human_sheet.py -> measure_H/human_judge_sheet.md, measure_H/key.json"""
import json
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "measure_H"
SEED = 20261128
QUOTA = {"pov_contamination": 7, "move_destination_mismatch": 5, "awase_detachment": 3, "quote_bracket": 2, "place_conflict": 2, "off_stage_speaker": 1}
N_CONTROL = 5

TITLES = {
    "000148_000751": "夏目漱石『二百十日』", "000129_000689": "森鴎外『山椒大夫』", "000035_000301": "太宰治『人間失格』",
    "000050_003586": "泉鏡花『草迷宮』", "000081_000456": "宮沢賢治『銀河鉄道の夜』", "001383_056643": "谷崎潤一郎『少年』",
    "001798_056779": "梅崎春生『ボロ家の春秋』", "001801_001786": "山川方夫『他人の夏』", "001875_059897": "壺井栄『雑居家族』",
    "000119_001738": "中島敦『弟子』",
}


def main():
    pool = []
    for src in ("measure", "measure_E"):
        pool += json.loads((HERE / src / "items.json").read_text(encoding="utf-8"))["items"]
    rnd = random.Random(SEED)
    chosen = []
    for rule, n in QUOTA.items():
        cand = [i for i in pool if i["stratum"] == "S1" and i["rule"] == rule]
        # one per (work, scene) at most, to avoid showing the same passage twice
        seen, uniq = set(), []
        for i in rnd.sample(cand, len(cand)):
            k = (i["work"], i["scene"])
            if k not in seen:
                seen.add(k)
                uniq.append(i)
        chosen += uniq[:n]
    ctrl = [i for i in pool if i["stratum"] == "C"]
    used = {(i["work"], i["scene"]) for i in chosen}
    ctrl = [i for i in rnd.sample(ctrl, len(ctrl)) if (i["work"], i["scene"]) not in used][:N_CONTROL]
    items = chosen + ctrl
    rnd.shuffle(items)
    OUT.mkdir(exist_ok=True)
    key, lines = [], []
    lines.append("# 判定シート（盲検）— 読み返しビューアの自動指摘")
    lines.append("")
    lines.append("判定者: Yuji さん（作者ではなく、注意深い一般読者として）。所要の目安: 1 項目 2〜3 分、全 %d 項目。" % len(items))
    lines.append("")
    lines.append("## やり方")
    lines.append("- 各項目に、機械が付けた「指摘文」と本文の抜粋があります。**指摘は誤りであることも多い**ので、正しいものとして扱わないでください。")
    lines.append("- 本文は青空文庫の公開作品です。作者の意図的な表現（文体・比喩・省略）を「問題」にしないでください。")
    lines.append("- 判定は、その抜粋だけを根拠にしてください（作品の記憶や外部情報は使わない）。")
    lines.append("- 「この場面に違和感の箇所がある可能性」という一般的な文言の項目は、その場面のどこにも違和感・矛盾が無ければ not_real です。")
    lines.append("- 【】で囲まれた部分が、機械が指摘した範囲です（無い項目もあります）。")
    lines.append("")
    lines.append("## 判定の基準")
    lines.append("- **real**: 注意深い読者が実際に引っかかる、または直すべきと考える矛盾・不自然さが、指摘の内容に対応して本文中に確認できる。")
    lines.append("- **not_real**: 言い換え・比喩・省略・語りの技法・意図的な文体として自然に読める、または指摘の前提（人物の居場所・話者・視点など）が本文と合っていない。")
    lines.append("- **unclear**: 抜粋だけでは決められない、または解釈が真に割れる。")
    lines.append("")
    lines.append("各項目の「判定」欄の [ ] のどれか 1 つに x を付け、「理由」欄に一言書いてください。")
    lines.append("")
    for n, it in enumerate(items, 1):
        hid = f"H{n:02d}"
        key.append({"sheet_id": hid, "item_id": it["id"], "source": "measure_E" if it["work"] == "000119_001738" else "measure", "work": it["work"], "stratum": it["stratum"], "rule": it["rule"], "scene": it["scene"]})
        lines.append(f"---\n\n### {hid}（{TITLES.get(it['work'], it['work'])}）\n")
        lines.append(f"**指摘:** {it['claim']}\n")
        for c in it["context"]:
            lines.append(f"**{c['label']}**\n")
            lines.append("> " + c["text"].replace("\n", "\n> ") + "\n")
        lines.append("**判定:** [ ] real　[ ] not_real　[ ] unclear\n")
        lines.append("**理由（一言）:** \n")
    (OUT / "human_judge_sheet.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "key.json").write_text(json.dumps({"seed": SEED, "quota": QUOTA, "n_control": N_CONTROL, "items": key}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(items), dict(Counter(k["rule"] for k in key)), "sheet chars", sum(len(x) for x in lines))


if __name__ == "__main__":
    main()
