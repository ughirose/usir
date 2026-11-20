| 項目 | 強いモデル（hybrid） | ローカル qwen2.5:14b |
|---|---|---|
| 発話の話者（正解が attributed の発話） | 165/166 = 0.994 | 157/166 = 0.946 |
| not_speech に対し話者 null で棄権 | 7/9 | 8/9 |
| undetermined に対し話者 null で棄権 | 2/9 | 0/9 |
| 同席 P / R | 1.000 / 0.989 (TP 90, FP 0, FN 1) | 0.962 / 0.824 (TP 75, FP 3, FN 16) |
| 時間順序の一致 | 0.971 (34/35) | 0.914 (32/35) |
| 明示時刻の一致 | 0.943 (33/35) | 0.629 (22/35) |
| 場所の一致 | 0.914 (32/35) | 0.771 (27/35) |
| 事実 P / R（述語＋対象） | 0.523 / 0.770 (TP 57, FP 52, FN 17) | 0.152 / 0.189 (TP 14, FP 78, FN 60) |
| 事実の出力件数（ゴールは 74） | 109 | 92 |
| 特性・場面単位 非0軸の同符号（ゴール非0軸 全 97） | 48/97 | 17/97 |
| 特性・ベクトル完全一致 | 0.282 (20/71) | 0.022 (1/46) |
| 特性・人物単位（47 軸に限定） | 24 same / 0 opposite | 5/47（v3 報告） |
| 出来事 top3 の重なり（スロットあり場面） | 0.7905（35/35 場面） | 0.5208（32/35 場面） |
| 出来事 top3 の重なり（35 場面、null=0） | 0.7905 | 0.4762 |
| 出来事 top3 一致数の分布 0/1/2/3 | 0/1/20/14 | 1/15/13/3 |
| 出来事 1 位一致 | 27/35 | 13/32 |
| 参考: 定数 top3 {E03,E02,E10}（自己データ） | 0.4667 | 0.4667 |
| 参考: ランダム期待値 | 0.1875 | 0.1875 |
| diagnose: 6 レンズ ok の作品数 | kasane:7 sasae:7 kamae:7 sadame:7 nagare:7 awase:7 | kasane:7 sasae:7 kamae:7 sadame:7 nagare:7 awase:7 |
| Nagare フルカーネル（annotated_slot）の作品数 | 7/7 | 4/7 |
| Awase scenes_scored 合計 | 35 | 35 |
| Awase irony_mean が正の作品数 | 5/7 | 0/7 |
| Awase detachment_mean が正の作品数 | 7/7 | 2/7 |
| Sadame edges_eff > 0 の作品数 | 7/7 | 2/7 |
| Sadame edges_eff 合計 | 65 | 6 |
| Sasae findings: off_stage_speaker | 1 | 24 |
| Sasae findings: presence_unconfirmed | 3 | 4 |
| Sasae findings: move_destination_mismatch | 3 | 2 |
| Sasae findings: place_conflict | 2 | 0 |
| Sasae findings: stay_location_conflict | 0 | 0 |
| Sasae findings: double_location | 0 | 0 |
| Sasae findings: pov_contamination | 12 | 12 |
| Sasae findings: quote_bracket | 4 | 4 |
| Sasae findings 合計 | 25 | 46 |
