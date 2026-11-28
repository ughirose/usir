"""Pack4 J3: condition-B measurement. Build text_j3.txt from public text 『山椒大夫』 (text_base.txt) by inserting errors in the forms
described in docs/viewer/rule_conditions_memo_v1.md §4 (2 per rule, quote_bracket 3), plus work registry for sandbox annotation.
usage: python h2/make_j3.py   (from scripts/viewer). Public text only."""
import json
from pathlib import Path

H = Path(__file__).resolve().parent
t = (H / "text_base.txt").read_text(encoding="utf-8")
ins = []


def rep(old, new, rule, kind):
    global t
    assert t.count(old) == 1, (old, t.count(old))
    t = t.replace(old, new)
    ins.append({"rule": rule, "kind": kind, "old": old, "new": new})


# --- move_destination_mismatch x2: person goes to an explicit place other than the scene's place; next scene has the person elsewhere
rep("荷は軽うして取らせる」\n", "荷は軽うして取らせる」\n厨子王は、その日の暮れに、大夫のもとを出て、ひとり山の寺へ向かった。\n",
    "move_destination_mismatch", "M1 scene2->3: 厨子王 山の寺へ向かう → 次の場面は新参小屋にいる")
rep("日の暮れまでに首尾よく調った。\n", "日の暮れまでに首尾よく調った。\n安寿は、その夕べ、ひとり由良の港へ向かって歩き出した。\n",
    "move_destination_mismatch", "M2 scene4->5: 安寿 由良の港へ向かう → 次の場面は三の木戸にいる")

# --- place_conflict x2: same time expression as an independent sentence at the head of two adjacent scenes
rep("三郎が言った。「過分のいたわりようじゃ。", "三日目の夕暮れ。\n三郎が言った。「過分のいたわりようじゃ。",
    "place_conflict", "P1 scenes 3&4 head: 三日目の夕暮れ。(場所は新参小屋/浜辺)")
rep("日がよほど昇ってから、柴を背負って", "三日目の夕暮れ。\n日がよほど昇ってから、柴を背負って",
    "place_conflict", "P1 scenes 3&4 head: 三日目の夕暮れ。")
rep("姉は潮を汲み、弟は柴を苅って、一日一日と暮らして行った。", "十日目の夕暮れ。\n姉は潮を汲み、弟は柴を苅って、一日一日と暮らして行った。",
    "place_conflict", "P2 scenes 5&6 head: 十日目の夕暮れ。(三の木戸/小屋)")
rep("ある日の暮れに二人の子供は、いつものように父母のことを言っていた。", "十日目の夕暮れ。\nある日の暮れに二人の子供は、いつものように父母のことを言っていた。",
    "place_conflict", "P2 scenes 5&6 head: 十日目の夕暮れ。")

# --- pov_contamination x2: scene pov (focal character established) + FID-form sentences (T3/T4/T5) attributed to another character
rep("樵は我が荷をおろして置いて、すぐに一荷苅ってくれた。", "厨子王は、ただ樵のすることを見ていた。樵は我が荷をおろして置いて、すぐに一荷苅ってくれた。樵は厨子王の青ざめた顔をじっと見た。ああ、何という哀れな童であろう。",
    "pov_contamination", "V1 scene 4 (pov 厨子王): FID attributed to 樵 (perception verb + 詠嘆 + 主観形容詞)")
rep("こう言って出て行った。\n", "こう言って出て行った。安寿は弟の手を握って、じっと二郎の顔を見上げていた。二郎は戸口に立って、二人の後ろ姿をじっと見た。ああ、何という不憫な子供らであろう。\n",
    "pov_contamination", "V2 scene 6 (pov 安寿): FID attributed to 二郎")

# --- quote_bracket x3 (inside a scene)
rep("安寿は汐を汲み覚えた。", "安寿は汐を汲み覚えた。」", "quote_bracket", "Q1 scene 4: 余分な閉じ括弧")
rep("どうにでも勝手にしておけ」大夫", "どうにでも勝手にしておけ大夫", "quote_bracket", "Q2 scene 5: 閉じ忘れ")
rep("それを二郎が通りかかって聞いた。二郎は邸を見廻って", "それを二郎が通りかかって聞いた。」二郎は邸を見廻って", "quote_bracket", "Q3 scene 6: 余分な閉じ括弧")

(H / "text_j3.txt").write_text(t, encoding="utf-8", newline="")
(H / "insertions_j3.json").write_text(json.dumps(ins, ensure_ascii=False, indent=1), encoding="utf-8")
wd = H / "works_j3"
wd.mkdir(exist_ok=True)
wid = "000129_000689_h2b"
(wd / f"{wid}.work.json").write_text(json.dumps({
    "narration": "third", "narrator": None, "pov": None,
    "source": "aozora 森鴎外『山椒大夫』(public text) pack4 J3 condition-B measurement; defined-form insertions",
    "text_path": (H / "text_j3.txt").resolve().as_posix(), "work_id": wid}, ensure_ascii=False, indent=1), encoding="utf-8")
import shutil
shutil.copy(H / "works" / f"{wid}.characters.json", wd / f"{wid}.characters.json")
(H / "local_j3").mkdir(exist_ok=True)
(H / "local_j3" / f"{wid}.utterances.json").write_text(json.dumps({
    "schema": "sensor.local_utterances/1", "work_id": wid,
    "provenance": "none: no local utterance attribution (J3)", "scenes": {}}, ensure_ascii=False, indent=1), encoding="utf-8")
print("insertions", len(ins))
for i in ins:
    print(i["rule"], "|", i["kind"])
