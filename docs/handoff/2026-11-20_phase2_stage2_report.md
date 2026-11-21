# Phase 2 第2段 センサー注釈範囲の拡張・精度測定・再貫通 レポート

- 日付: 2026-11-20 / 対象: 公開テキストのみ（現代文3作品11場面＋F9 Dev 4作品24場面。妻の原稿・F9 holdout は不使用）
- usir: ブランチ `phase2/sensor-full-annotation`（ワークツリー C:/dev/usir-sensor-full、基点 phase2/step1-real-annotation 592dca0）／ nishiki: `phase2/vertical-full-annotation`（C:/dev/nishiki-phase2b、基点 phase2/vertical-real-data f4e3eba）。既存ブランチへのマージなし。
- モデル: qwen2.5:14b（凍結構成と同じ）、seed 42、temperature 0。

## 1. 結論（数値は後節の表から転記）
- 同席者: 現代文11場面で precision 1.000 / recall 0.889。発話の話者・宛先と一人称語り手から機械的に作った同席者（比較用）は precision 0.958 / recall 0.852。センサーの上積みは TP 24 対 23。F9 Dev では機械的方式の recall が上回る（0.812 対 0.797）。
- 時間（直前場面との関係）: 現代文 1.000、F9 Dev 0.875。場所: 現代文 0.909、F9 Dev 0.708。
- 事実更新: 現代文 precision 0.097 / recall 0.115、F9 Dev precision 0.180 / recall 0.229。実用水準に達していない。
- values_6d: 6軸完全一致は現代文 0/15、F9 Dev 1/31。軸ごとの一致率 0.678（現代文）は全軸0固定の基準 0.799 を下回る。ゴールドで非0の軸のうち符号まで一致した割合は現代文 0.167、F9 Dev 0.292。
- Sadame: 7 作品中 edges_eff > 0 が 2 作品、total > 0 が 4 作品。第1段（edges_eff = 0）からは前進したが、入力の values_6d が上記の精度であるため、スコアの意味は未検証。
- Sasae findings 全 40 件: off_stage_speaker は 23 件が誤検出（ゴールド上は在場する話者を、センサーの同席者が落としたため）、1 件が妥当。pov_contamination は 11 件が誤検出、1 件が妥当の可能性。quote_bracket 4 件は場面境界が引用の途中に落ちる構造的事実。

## 2. Stage A: センサー拡張

- 実装: `python/usir_py/sensor_eval/scene_annotator.py`（場面プロンプト scene_v1、構造化出力スキーマ、決定論バリデータ）、`scripts/phase2_stage2/run_full_annotation.py`（実行）。
- 構成: 発話（話者・宛先・evidence・一人称）は凍結構成（v5＋第2段後処理＋第3段話者再確認）の既存出力をそのまま使用（再推論なし。現代文は phase2 第1段で凍結結果との完全一致を確認済み、F9 Dev は凍結結果 `scripts/step1/results/v5_threestage/qwen2.5_14b_dev.json`）。口調は Policy X で付け直し。scene_meta／character_states／fact_updates は場面ごとに qwen2.5:14b を1回呼んで出力。
- 同席者は本文の根拠（basis 語句）付きで出力させ、発話者からの機械的な派生は行っていない。basis は注釈ファイルには入れず `*.trace.json` に保存。
- 決定論バリデータ: 名簿外 ID の拒否、values_6d の範囲（6要素・{-1,0,+1}）、evidence 必須、述語 enum、basis／場所語句／時刻語句が場面本文の部分文字列であること、sensor.v2 JSON Schema 検証。不正項目は修復せず除外して件数を記録。
- 結果（35場面）: 採用 同席者 78 件／人物状態 54 件／事実更新 92 件、除外 59 件、スキーマ違反 0 件、JSON 解析失敗 0 件、LLM 時間合計 596.3 秒。
- 除外の内訳: fact:basis_not_in_text 13、state:basis_not_in_text 24、present:basis_not_in_text 10、location:surface_not_in_text 11、time:explicit_time_not_in_text 1。

| 作品 | 場面 | 同席者 | 状態 | 事実更新 | 除外 | LLM 秒 |
|---|---:|---:|---:|---:|---:|---:|
| `000035_000301` | 7 | 13 | 6 | 17 | 11 | 131.0 |
| `000050_003586` | 5 | 11 | 10 | 14 | 3 | 61.1 |
| `000081_000456` | 6 | 14 | 12 | 15 | 6 | 78.8 |
| `001383_056643` | 6 | 16 | 10 | 15 | 12 | 103.3 |
| `001798_056779` | 4 | 8 | 8 | 18 | 6 | 80.6 |
| `001801_001786` | 3 | 4 | 3 | 7 | 6 | 46.7 |
| `001875_059897` | 4 | 12 | 5 | 6 | 15 | 94.8 |

## 3. Stage B: 精度測定

- ゴールド: `scripts/phase2_stage2/gold/stage_b_gold.json`（35場面）。セッションのモデルが場面本文を直接読んで作成。作成時点でセンサーの場面注釈出力は開いていない（ゴールドのコミット ac9c5ea はセンサー注釈のコミットより前）。限界: 作成者1名、二重注釈なし、価値観・事実更新は主観を含む。
- 照合規則は測定前に固定（`measure_stage_b.py` 冒頭）。事実更新は述語一致＋対象語句の一致（包含、または文字 bigram の重なり係数 0.5 以上）で1対1対応、`述語＋対象＋主語` は主語も要求。

### 同席者・時間・場所

| 集合 | 場面 | 同席者 TP/FP/FN | P | R | F1 | 時間(順序) | 時間(表現) | 場所 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 現代文 | 11 | 24 / 0 / 3 | 1.000 | 0.889 | 0.941 | 11/11 = 1.000 | 8/11 = 0.727 | 10/11 = 0.909 |
| F9 Dev | 24 | 51 / 3 / 13 | 0.944 | 0.797 | 0.864 | 21/24 = 0.875 | 14/24 = 0.583 | 17/24 = 0.708 |
| 全体 | 35 | 75 / 3 / 16 | 0.962 | 0.824 | 0.888 | 32/35 = 0.914 | 22/35 = 0.629 | 27/35 = 0.771 |

比較用（発話の話者・宛先＋一人称語り手の和集合）:

| 集合 | TP/FP/FN | P | R | F1 |
|---|---|---:|---:|---:|
| 現代文 | 23 / 1 / 4 | 0.958 | 0.852 | 0.902 |
| F9 Dev | 52 / 0 / 12 | 1.000 | 0.812 | 0.897 |
| 全体 | 75 / 1 / 16 | 0.987 | 0.824 | 0.898 |

### 事実更新（述語＋対象）

| 集合 | 方式 | TP / FP / FN | P | R | F1 |
|---|---|---|---:|---:|---:|
| 現代文 | 述語＋対象 | 3 / 28 / 23 | 0.097 | 0.115 | 0.105 |
| 現代文 | 述語＋対象＋主語 | 3 / 28 / 23 | 0.097 | 0.115 | 0.105 |
| F9 Dev | 述語＋対象 | 11 / 50 / 37 | 0.180 | 0.229 | 0.202 |
| F9 Dev | 述語＋対象＋主語 | 8 / 53 / 40 | 0.131 | 0.167 | 0.147 |
| 全体 | 述語＋対象 | 14 / 78 / 60 | 0.152 | 0.189 | 0.169 |
| 全体 | 述語＋対象＋主語 | 11 / 81 / 63 | 0.120 | 0.149 | 0.133 |

### values_6d

| 集合 | 照合できた人物×場面 | 6軸完全一致 | 軸ごと一致 | 両方非0の軸の符号一致 | ゴールド非0軸のうち符号一致 | 全0固定の軸一致（基準） | 人物集合 P / R |
|---|---:|---:|---:|---:|---:|---:|---|
| 現代文 | 15 | 0/15 | 0.678 | 3/5 | 3/18 | 0.799 | 0.938 / 0.625 |
| F9 Dev | 31 | 1/31 | 0.704 | 14/17 | 14/48 | 0.759 | 0.816 / 0.660 |
| 全体 | 46 | 1/46 | 0.696 | 17/22 | 17/66 | 0.772 | 0.852 / 0.648 |

- 符号の扱い: 値が {-1,0,+1} のため、符号一致は値一致と同じ意味になる。そこで「両方非0の軸での極性一致」と「ゴールドが非0の軸で符号まで当てた割合」を別に数えた。

### 実行間のばらつき（現代文11場面を同じ設定で再実行）

| 実行 | 同席者 P | 同席者 R | 場所 | 事実 P | 事実 R | 軸一致 | 非0軸の符号一致 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1回目 | 1.000 | 0.889 | 0.909 | 0.097 | 0.115 | 0.678 | 0.167 |
| 再実行 | 1.000 | 0.889 | 0.818 | 0.133 | 0.154 | 0.689 | 0.111 |
- seed 固定・temperature 0 でも出力が一致しない場面がある（再実行で 11 場面中 8 場面が scene_meta／character_states／fact_updates まで一致、3 場面が不一致。比較は別途実行し、結果は本レポートに転記）。上の差は実行間のばらつきの目安。

## 4. Stage C: 再貫通（nishiki 側）

- 入力: 7作品35場面の拡張注釈と名簿（`scripts/phase2b/annotations/`）。`nishiki diagnose --characters --annotations`、出力 schema は `nishiki.diagnose/3`（候補版）。Rust コード無変更（バイナリは f4e3eba と同一コード）。warnings は全作品で 0。

| 作品 | 場面 | Sadame edges_eff | e_inter | e_intra | total | intra_characters | Sasae off_stage | Sasae pov_contam | speaker_mismatch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ボロ家の春秋 (`001798_056779`) | 4 | 4 | 0.167 | 0.167 | 0.167 | 2 | 1 | 2 | 9 |
| 他人の夏 (`001801_001786`) | 3 | 0 | 0.000 | 0.000 | 0.000 | 0 | 11 | 0 | 5 |
| 雑居家族 (`001875_059897`) | 4 | 0 | 0.000 | 0.000 | 0.000 | 0 | 1 | 0 | 9 |
| 人間失格(F9dev) (`000035_000301`) | 7 | 0 | 0.000 | 0.271 | 0.271 | 2 | 0 | 4 | 12 |
| F9dev 001383 (`001383_056643`) | 6 | 0 | 0.000 | 0.089 | 0.089 | 3 | 7 | 4 | 11 |
| F9dev 000050 (`000050_003586`) | 5 | 0 | 0.000 | 0.000 | 0.000 | 0 | 1 | 0 | 9 |
| 銀河鉄道の夜(F9dev) (`000081_000456`) | 6 | 2 | 0.000 | 0.062 | 0.037 | 2 | 3 | 2 | 9 |

### 縦貫通の数え直し（実注釈が出力に寄与したレンズ数）

レンズ6個（kasane, sasae, sadame, kamae, nagare, awase）。`no_annotations`（注釈なし）と `full` の出力を比較。nagare／awase は注釈生成が未移植のため unavailable。
- 注釈の有無で出力が変わったレンズ（作品数／7）: kasane 0、sasae 7、kamae 0、sadame 7、nagare 0、awase 0。寄与したレンズは 6 個中 2 個。第1段は sasae のみ寄与（sadame は空配列で edges_eff=0）。
- 新項目の寄与（アブレーション）: character_states を空にすると sadame の出力が変わる作品 7/7。present_character_ids を空にすると sasae の出力が変わる作品 7/7。
- fact_updates と location／time は nishiki 側のレンズが未消費（Sasae 第2段が Rust 側に未実装）。diagnose 出力には寄与していない。

### Sasae 第2段の探索的 probe（注釈のみから計算する Python、nishiki 本体の出力ではない）

- 検出件数: location_shift_without_moved_to 10、state_for_absent_character 2、fact_subject_not_present 8。内訳は `stageC_results.json` の `sasae_stage2_probe`。妥当性は未判定。

### findings 全件（40件）

位置・基準・本文・判定理由の全件は `scripts/phase2b/results/stageC_findings_audit.json`。判定者は1名（セッションのモデル）で盲検ではない。

| 規則 | 判定 | 件数 |
|---|---|---:|
| off_stage_speaker | false_positive | 23 |
| off_stage_speaker | plausible_true | 1 |
| pov_contamination | false_positive | 11 |
| pov_contamination | plausible_true | 1 |
| quote_bracket | true_structural | 4 |

| # | 作品 | 規則 | 場面 | 本文（先頭40字） | 判定 |
|---:|---|---|---:|---|---|
| 0 | `001798_056779` | pov_contamination | 1 | 不破の言によると、赤穂行は墓参のためだけでなく、あちらにある山林を処分する用も兼 | plausible_true |
| 1 | `001798_056779` | off_stage_speaker | 2 | 野呂君と仲良くしてやって呉れ給え。彼はしんからの好人物であります | plausible_true |
| 2 | `001798_056779` | pov_contamination | 2 | 庭に入って来た男は僕を見て、少しおどろいたらしく立ちすくんだ。 | false_positive |
| 3 | `001801_001786` | off_stage_speaker | 1 | そのときは、親父も生命からがらだったんです。牛みたいな大きなカジキを、ふらふらに | false_positive |
| 4 | `001801_001786` | off_stage_speaker | 1 | 銛で打ったの？ | false_positive |
| 5 | `001801_001786` | off_stage_speaker | 1 | そうです。とても重い銛なんです | false_positive |
| 6 | `001801_001786` | off_stage_speaker | 1 | ずいぶん、原始的ね | false_positive |
| 7 | `001801_001786` | off_stage_speaker | 1 | 死にました。去年 | false_positive |
| 8 | `001801_001786` | off_stage_speaker | 1 | ……あなたは、自殺するつもりですか？ | false_positive |
| 9 | `001801_001786` | off_stage_speaker | 1 | ほっといてよ。……あなたには、関係ないことだわ | false_positive |
| 10 | `001801_001786` | off_stage_speaker | 1 | べつに、やめなさい、っていうつもりじゃないんですよ | false_positive |
| 11 | `001801_001786` | off_stage_speaker | 1 | からかうの？　軽蔑しているのね、私を。子どものくせに | false_positive |
| 12 | `001801_001786` | off_stage_speaker | 1 | ちがいます。親父がぼくにいったんです。死のうとしている人間を、軽蔑しちゃいけない | false_positive |
| 13 | `001801_001786` | off_stage_speaker | 1 | 人間には、他の人間のこと、ことにその生きるか死ぬかっていう肝心のことなんかは、決 | false_positive |
| 14 | `001875_059897` | off_stage_speaker | 3 | そう思うていますけど、鷲さんになかなかあえんのです。ぼく、ここのところ、毎日のよ | false_positive |
| 15 | `000035_000301` | pov_contamination | 0 | しかも、Ｐ（党の事を、そういう隠語で呼んでいたと記憶していますが、或いは、違って | false_positive |
| 16 | `000035_000301` | pov_contamination | 2 | その頃になると、自分も、敢えて堀木の案内に頼らずとも、ひとりで電車にも乗れるし、 | false_positive |
| 17 | `000035_000301` | pov_contamination | 2 | 心では、相変らず、人間の自信と暴力とを怪しみ、恐れ、悩みながら、うわべだけは、少 | false_positive |
| 18 | `000035_000301` | pov_contamination | 6 | 十一月の末、自分は、堀木と神田の屋台で安酒を飲み、この悪友は、その屋台を出てから | false_positive |
| 19 | `001383_056643` | off_stage_speaker | 0 | まだ其の外にも人を殺したろう。よし、よし、云わないな。云わなければ拷問にかけてや | false_positive |
| 20 | `001383_056643` | off_stage_speaker | 0 | もう此れだけでございますから、堪忍しておくんなさい | false_positive |
| 21 | `001383_056643` | off_stage_speaker | 2 | あゝもうとても助からない | false_positive |
| 22 | `001383_056643` | pov_contamination | 3 | 今に私もあんな態をさせられるのだ。 | false_positive |
| 23 | `001383_056643` | pov_contamination | 3 | 私の耳には甲斐絹の羽織の裏のさや／＼とこすれて鳴るのが聞え、私の鼻は着物から放つ | false_positive |
| 24 | `001383_056643` | pov_contamination | 4 | 女中は恐ろしい眼つきをして叱りながら、泥の足型が印せられて居る仙吉の目鼻を、様子 | false_positive |
| 25 | `001383_056643` | pov_contamination | 4 | 私はまだ蹈みつけられた顔の痕がぴり／＼するのをじっと堪えて何か餘程の悪事でも働い | false_positive |
| 26 | `001383_056643` | off_stage_speaker | 5 | うゝん | false_positive |
| 27 | `001383_056643` | off_stage_speaker | 5 | 今ね、仙吉を白酒で酔っ拂わしてやるんだよ | false_positive |
| 28 | `001383_056643` | off_stage_speaker | 5 | おい仙吉、これから四人でお酒盛りをしようじゃないか | false_positive |
| 29 | `001383_056643` | off_stage_speaker | 5 | 此れはどうも結構な御酒でございますな | false_positive |
| 30 | `000050_003586` | quote_bracket | 0 | 「 | true_structural |
| 31 | `000050_003586` | quote_bracket | 1 | 」 | true_structural |
| 32 | `000050_003586` | quote_bracket | 2 | 「 | true_structural |
| 33 | `000050_003586` | quote_bracket | 3 | 」 | true_structural |
| 34 | `000050_003586` | off_stage_speaker | 4 | お前様もね、当前だあこれ、空を飛ぼうが、泳ごうが、活きた猫なら秋谷中私ら知己だ。 | false_positive |
| 35 | `000081_000456` | pov_contamination | 0 | 旅人たちはしずかに席に戻り、二人も胸いっぱいのかなしみに似た新らしい気持ちを、何 | false_positive |
| 36 | `000081_000456` | off_stage_speaker | 2 | おや、変なものがあるよ。 | false_positive |
| 37 | `000081_000456` | off_stage_speaker | 2 | くるみの実だよ。そら、沢山ある。流れて来たんじゃない。岩の中に入ってるんだ。 | false_positive |
| 38 | `000081_000456` | off_stage_speaker | 3 | 標本にするんですか。 | false_positive |
| 39 | `000081_000456` | pov_contamination | 5 | その一つのあかりに黒い甲虫がとまってその影が大きく天井にうつっていたのです。 | false_positive |

## 5. 限界と未検証

- ゴールドは作成者1名で、二重注釈・評価者間一致なし。values_6d と fact_updates のゴールドは特に主観を含む。
- present_character_ids の誤りの一部は、basis 部分文字列検査で正しい人物が除外された結果（例: 他人の夏 s1 は同席者が空）。検査を緩めた場合の精度は未測定。
- 場面プロンプトは1版のみ。ゴールド作成後に調整していない。
- 発話部分は再推論していない。F9 Dev の凍結結果は本セッションで再実行していない。
- location_id は本文中の語句をそのまま入れており、台帳IDではない。pov_character は gold の narrator（一人称作品）または null で、センサー出力ではない。
- Sadame のスコアは kappa 未較正で、入力の values_6d の精度も低いため、意味づけは未検証。
- 出力は実行間で完全には再現しない（上記の再実行）。

## 6. 成果物

- usir `phase2/sensor-full-annotation`: `python/usir_py/sensor_eval/scene_annotator.py`、`scripts/phase2_stage2/`（run_full_annotation.py、measure_stage_b.py、summarize_stage_a.py、render_report.py、gold/、annotations/、annotations_rerun/、results/stageB_metrics*.json）。
- nishiki `phase2/vertical-full-annotation`: `scripts/phase2b/`（run_stage_c.py、audit_findings.py、annotations/、results/）、`docs/handoff/STATUS_phase2_stage2.md`。
