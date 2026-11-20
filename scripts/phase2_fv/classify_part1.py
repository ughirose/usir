# Hand classification (reviewer read each scene text; see review dump). Part 1: modern works 001798 / 001801.
# FP classes: A=fabricated(no textual basis) B=granularity C=predicate D=subject/object E=gold-omission-suspected
# FN classes: M=missed(no counterpart prediction) B/C/D = a counterpart prediction exists in the scene but differs by granularity/predicate/subject-object
# note keys: surf=semantically same fact but object-surface mismatch in metric; unc=borderline label
PART = {
 "001798_056779:s0": {"FP": {1:("C","世帯道具を運ぶだけで取得ではない"),2:("E","契約締結は根拠明示。ゴールドに無い",'unc'),3:("D","四万円を渡す側(boku)が取得者になっている"),4:("C","祝宴の開催は取得ではない"),5:("E","fuwa が四万円を受け取るのは本文通り。ゴールドに無い"),6:("C","祝宴の開催は取得ではない"),7:("D","夫人が参加しただけ。fuwa が夫人を取得は取り違え"),8:("D","酒豪は夫人の性質。fuwa の状態変化ではない")}, "FN":{}},
 "001798_056779:s1": {"FP": {0:("B","酔っているは根拠あり。細かすぎる"),2:("B","赤穂行の計画。ゴールドの hook(借金の事情)とは別の粒度",'unc'),3:("E","野呂を留守番に置く申し出は根拠あり。ゴールドに無い")}, "FN":{0:("D","pred は fuwa STATE_CHANGED『酔っている』。ゴールドは『失踪』"),2:("M","")}},
 "001798_056779:s2": {"FP": {0:("D","到着は noro の MOVED_TO。boku の LEARNED にしている"),1:("B","不快は根拠ありだが細かすぎる"),2:("B","口調の変化。細かすぎる")}, "FN":{0:("D","pred#0 が到着を boku LEARNED として出力(主語・述語違い)"),1:("M",""),2:("M","")}},
 "001798_056779:s3": {"FP": {0:("E","野呂の主張を boku が知る。s2 で既出の内容でゴールドに無い",'unc'),1:("B","誤解の hook。ゴールドの『家の二重契約の誤解』と意味は同じ、object 表層が不一致",'surf')}, "FN":{0:("M",""),1:("B","pred#1 と意味が同じ。object 表層が不一致",'surf')}},
 "001801_001786:s0": {"FP": {0:("E","女の発見は根拠あり。ゴールドは『今日の女と気づく』のみ",'unc'),1:("B","疲れているは細かすぎる"),2:("C","『知る』は LEARNED のはずで ACQUIRED は述語違い"),3:("C","『顔を覚えていた』は LEARNED(ゴールドの気づき)に当たる。述語違い")}, "FN":{0:("C","pred#3 が ACQUIRED で出力"),1:("M","")}},
 "001801_001786:s1": {"FP": {0:("D","父の教えを語る側(shinichi)が LEARNED になっている。受け手は onna"),1:("B","無言は細かすぎる")}, "FN":{0:("M",""),1:("M",""),2:("D","pred#0 は shinichi LEARNED。受け手 onna の LEARNED が正")}},
 "001801_001786:s2": {"FP": {0:("D","『怒ったような目つき』は onna の状態。shinichi LEARNED は主語・述語違い")}, "FN":{0:("M",""),1:("M","")}},
}
PART.update({
 "001875_059897:s0": {"FP": {1:("D","同一事実(兵六の来訪)の重複。lenient 照合は subject を見ず yasue 版(#0)を TP にし、正しい subject の #1 が FP になる",'dup'),2:("C","『三度とも来る回りあわせ』は会話中の感想で、未解決の懸案(hook)ではない",'unc')}, "FN":{0:("M","")}},
 "001875_059897:s1": {"FP": {0:("C","ミルク運搬は本文にあるが、根拠句は会話で『安江が知る』更新ではない"),1:("D","兵六が来なくなったのは兵六側の変化。yasue LEARNED にしている")}, "FN":{0:("M",""),1:("M",""),2:("M","")}},
 "001875_059897:s2": {"FP": {}, "FN":{0:("M",""),1:("M",""),2:("M","")}},
 "001875_059897:s3": {"FP": {0:("D","気持ちが動いたのは安江。susumu STATE_CHANGED は主語違い、object『気持ち』も粗い")}, "FN":{0:("M",""),1:("M","")}},
})
