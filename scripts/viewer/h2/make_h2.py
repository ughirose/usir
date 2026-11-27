"""H2 setup: build a copy of public text 『山椒大夫』(data/000129_000689) with inserted contradictions (one or more per rule),
plus an unmodified baseline copy, and the sandbox work registry entries for both. Public text only.
usage: python h2/make_h2.py   (run from scripts/viewer)"""
import json
import shutil
from pathlib import Path

H = Path(__file__).resolve().parent
V = H.parent
t = (V / "data/000129_000689/text.txt").read_text(encoding="utf-8")
ins = []


def rep(old, new, kind, rule):
    global t
    assert t.count(old) == 1, (old, t.count(old))
    t = t.replace(old, new)
    ins.append({"rule": rule, "kind": kind, "old": old, "new": new})


# I1 move_destination_mismatch: scene 3 sends 厨子王 to the mountain; scene 4 puts him at the shore
rep("厨子王は気を取り直して、ようよう午までに一荷苅り、午からまた一荷苅った。", "厨子王は浜の汐汲み場の砂の上に立って、気を取り直して、ようよう午までに一荷苅り、午からまた一荷苅った。", "場所の食い違い（山へ向かったのに浜にいる）", "move_destination_mismatch")
# I2 place_conflict: same explicit time, two places (scene 5 end vs scene 6 head)
rep("二郎は三の木戸に小屋を掛けさせて、姉と弟とを一しょに置いた。", "二郎は三の木戸に小屋を掛けさせて、姉と弟とを一しょに置いた。十日目の夕暮れ、安寿は浜辺の岩の上に立って、沖を眺めていた。", "同時刻の場所の食い違い（前半：浜辺）", "place_conflict")
rep("ある日の暮れに二人の子供は、いつものように父母のことを言っていた。", "十日目の夕暮れに安寿は山の雑木林の中で、弟と、いつものように父母のことを言っていた。", "同時刻の場所の食い違い（後半：山）", "place_conflict")
# I3 pov_contamination: interiority of minor characters inside 厨子王's / narrator's frame
rep("奴頭が出て行くころには、もうあたりが暗くなった。", "奴頭は、心の中でこの子供らを不憫に思い、自分も昔売られた身であったことを思い出していた。奴頭が出て行くころには、もうあたりが暗くなった。", "視点の滑り（奴頭の内面）", "pov_contamination")
rep("樵は我が荷をおろして置いて、すぐに一荷苅ってくれた。", "樵は我が荷をおろして置いて、すぐに一荷苅ってくれた。樵は、亡くなった自分の子をこの童に重ねて、胸が痛んで仕方がなかった。", "視点の滑り（樵の内面）", "pov_contamination")
# I4 quote_bracket
rep("その通りにさせなされい」", "その通りにさせなされい", "閉じ忘れた括弧", "quote_bracket")
rep("目を見合わせて泣いた。夜は", "目を見合わせて泣いた。」夜は", "余分な閉じ括弧", "quote_bracket")
# I5 off_stage_speaker: 母親 is on another boat in scene 1
rep("「もう呼ぶな」と宮崎が叱った。", "「もう呼ぶな」と宮崎が叱った。「姉さま、泣くでない」と母親が言った。", "不在の人物の台詞（母親は別の舟）", "off_stage_speaker")
# I6 awase_detachment: strong event narrated flatly (scene 5)
rep("二郎は三の木戸に小屋を掛けさせて、姉と弟とを一しょに置いた。十日目", "二郎は三の木戸に小屋を掛けさせて、姉と弟とを一しょに置いた。その夜、安寿は小屋で母が死ぬ夢を見て、胸を裂かれる思いで一晩じゅう泣きじゃくった。厨子王もそれを聞いて、絶望のあまり声をあげて泣いた。しかしこれは、さしたる事ではなく、あたりはいたって静かであった。十日目", "語りの温度差（強い出来事を平坦に語る）", "awase_detachment")

(H / "works").mkdir(exist_ok=True)
(H / "local").mkdir(exist_ok=True)
(H / "text_h2.txt").write_text(t, encoding="utf-8", newline="")
(H / "text_base.txt").write_text((V / "data/000129_000689/text.txt").read_text(encoding="utf-8"), encoding="utf-8", newline="")
(H / "insertions.json").write_text(json.dumps(ins, ensure_ascii=False, indent=1), encoding="utf-8")
for tag, wid in (("h2", "000129_000689_h2"), ("base", "000129_000689_h2b")):
    desc = "with inserted contradictions" if tag == "h2" else "unmodified"
    (H / "works" / f"{wid}.work.json").write_text(json.dumps({
        "narration": "third", "narrator": None, "pov": None,
        "source": f"aozora 森鴎外『山椒大夫』(public text) H2 true-positive check; {desc}",
        "text_path": (H / f"text_{tag}.txt").resolve().as_posix(), "work_id": wid}, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.copy(V / "data/000129_000689/characters.json", H / "works" / f"{wid}.characters.json")
    (H / "local" / f"{wid}.utterances.json").write_text(json.dumps({
        "schema": "sensor.local_utterances/1", "work_id": wid,
        "provenance": "none: no local utterance attribution (H2)", "scenes": {}}, ensure_ascii=False, indent=1), encoding="utf-8")
print("insertions", len(ins))
