"""Pack4 J3 round 2 (K2): same rules/counts as round 1 (make_j3.py), re-designed after round-1 analysis:
 - time-expression sentences are placed behind a blank line so that the scene splitter (strong boundary on blank line) makes them the first
   sentence of two adjacent scenes; a competing in-scene time phrase is removed;
 - the scene before a move insertion ends with the move sentence;
 - focal-character sentences are added before the free-indirect sentences so a scene pov can be annotated.
usage: python h2/make_k2.py (from scripts/viewer). Public text only."""
import json
import shutil
from pathlib import Path

H = Path(__file__).resolve().parent
t = (H / "text_base.txt").read_text(encoding="utf-8")
ins = []


def rep(old, new, rule, kind):
    global t
    assert t.count(old) == 1, (old, t.count(old))
    t = t.replace(old, new)
    ins.append({"rule": rule, "kind": kind, "old": old, "new": new})


def edit(old, new):  # supporting edit (not an insertion under test): removes a competing time phrase
    global t
    assert t.count(old) == 1, (old, t.count(old))
    t = t.replace(old, new)


# P1: heads of scenes 3 and 4 (blank line before => scene boundary)
rep("荷は軽うして取らせる」\n三郎が言った。", "荷は軽うして取らせる」\n\n三日目の夕暮れ。\n三郎が言った。", "place_conflict", "P1 scene 3 head: 三日目の夕暮れ。(新参小屋)")
edit("翌日の朝はひどく寒かった。", "ひどく寒い朝であった。")
rep("ひとり涙をこぼしていた。\n日がよほど昇ってから、柴を背負って", "ひとり涙をこぼしていた。厨子王は、その夕べ、ひとり由良の港へ向かって歩き出した。\n\n三日目の夕暮れ。\n柴を背負って",
    "place_conflict+move_destination_mismatch", "P1 scene 4 head: 三日目の夕暮れ。 / M1 scene 3 end: 厨子王 由良の港へ向かう")
# V1: focal 厨子王 + FID attributed to 樵 in scene 4
rep("樵は我が荷をおろして置いて、すぐに一荷苅ってくれた。", "厨子王は、ただ樵のすることを見ていた。厨子王は、寒さに震えながら、姉のことを思っていた。樵は我が荷をおろして置いて、すぐに一荷苅ってくれた。樵は厨子王の青ざめた顔をじっと見た。ああ、何という哀れな童であろう。",
    "pov_contamination", "V1 scene 4 (pov 厨子王): FID attributed to 樵")
# P2 head of scene 5 (+ M2 at the end of scene 5)
rep("姉は潮を汲み、弟は柴を苅って", "十日目の夕暮れ。\n姉は潮を汲み、弟は柴を苅って", "place_conflict", "P2 scene 5 head: 十日目の夕暮れ。(三の木戸; a divider line precedes it in the base text)")
rep("姉と弟とを一しょに置いた。\nある日の暮れに二人の子供は、", "姉と弟とを一しょに置いた。安寿は、その夕べ、ひとり山の寺へ向かって歩き出した。\n\n十日目の夕暮れ。\n二人の子供は、",
    "place_conflict+move_destination_mismatch", "P2 scene 6 head: 十日目の夕暮れ。(小屋) / M2 scene 5 end: 安寿 山の寺へ向かう")
edit("ほど経てまたある日の暮れに、二人の子供は", "ほど経てまた、二人の子供は")
# V2: focal 安寿 + FID attributed to 二郎 in scene 6
rep("こう言って出て行った。\n", "こう言って出て行った。安寿は弟の手を握って、じっと二郎の顔を見上げていた。安寿は、二郎の足音が遠ざかるのを聞いていた。二郎は戸口に立って、二人の後ろ姿をじっと見た。ああ、何という不憫な子供らであろう。\n",
    "pov_contamination", "V2 scene 6 (pov 安寿): FID attributed to 二郎")
# quote_bracket x3
rep("安寿は汐を汲み覚えた。", "安寿は汐を汲み覚えた。」", "quote_bracket", "Q1 scene 4: 余分な閉じ括弧")
rep("どうにでも勝手にしておけ」大夫", "どうにでも勝手にしておけ大夫", "quote_bracket", "Q2 scene 5: 閉じ忘れ")
rep("それを二郎が通りかかって聞いた。二郎は邸を見廻って", "それを二郎が通りかかって聞いた。」二郎は邸を見廻って", "quote_bracket", "Q3 scene 6: 余分な閉じ括弧")

(H / "text_k2.txt").write_text(t, encoding="utf-8", newline="")
(H / "insertions_k2.json").write_text(json.dumps(ins, ensure_ascii=False, indent=1), encoding="utf-8")
wd = H / "works_k2"
wd.mkdir(exist_ok=True)
wid = "000129_000689_h2b"
(wd / f"{wid}.work.json").write_text(json.dumps({
    "narration": "third", "narrator": None, "pov": None,
    "source": "aozora 森鴎外『山椒大夫』(public text) pack4 J3 round 2 (K2); defined-form insertions re-designed after round 1",
    "text_path": (H / "text_k2.txt").resolve().as_posix(), "work_id": wid}, ensure_ascii=False, indent=1), encoding="utf-8")
shutil.copy(H / "works" / f"{wid}.characters.json", wd / f"{wid}.characters.json")
(H / "local_k2").mkdir(exist_ok=True)
shutil.copy(H / "local_j3" / f"{wid}.utterances.json", H / "local_k2" / f"{wid}.utterances.json")
print("insertions", len(ins))
