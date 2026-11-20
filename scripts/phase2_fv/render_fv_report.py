"""Render the report from result JSONs (numbers are read from the artifacts, not typed)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE / "results"
J = lambda n: json.loads((R / n).read_text(encoding="utf-8"))  # noqa: E731
cls, mism, vcmp, vcomp = J("fact_error_classification.json"), J("values_mismatch_summary.json"), J("variant_comparison.json"), J("values_compare.json")
M = {v: J(f"metrics_{v}.json") for v in ("baseline", "control", "fact_v2", "values_ctx")}
logs = {v: J(f"run_{v}/run_log.json") for v in ("control", "fact_v2", "values_ctx")}
items = cls["items"]


def ex(kind, scene, idx):
    for it in items:
        if it["kind"] == kind and it["scene_id"] == scene and it["idx"] == idx:
            return it
    raise KeyError((kind, scene, idx))


def line(it):
    sub = it["subject"] or "null"
    base = f"`{it['scene_id']}` {sub} {it['predicate']}「{it['object']}」"
    return f"{base} — {it['reason']}" if it["reason"] else base


def pf(x):
    return "-" if x is None else f"{x:.3f}"


def frow(name, key="all"):
    f = M[name][key]["fact_updates_pred_obj"]
    s = M[name][key]["fact_updates_strict_subject"]
    return f"| {name} | {f['tp']+f['fp']} | {f['tp']} | {f['fp']} | {f['fn']} | {pf(f['precision'])} | {pf(f['recall'])} | {pf(f['f1'])} | {pf(s['precision'])} / {pf(s['recall'])} |"


def vrow(name):
    v = M[name]["all"]["values_6d"]
    c = vcomp[name]
    return (f"| {name} | {c['matched']}/{c['gold_char_scenes']} | {v['gold_nonzero_axes_same_sign']['match']}/{v['gold_nonzero_axes_same_sign']['n']} "
            f"({pf(v['gold_nonzero_axes_same_sign']['rate'])}) | {c['same']}/{c['gold_nz_all']} ({pf(c['same_rate_over_all_gold_nz'])}) | {c['pred0']} | {c['opp']} | "
            f"{c['pred_nz']} | {c['pred_nz_on_goldzero']} | {c['allzero_vectors']} |")


fp, fn = cls["FP_by_class"], cls["FN_by_class"]
fl = cls["flags"]
L = []
a = L.append
a("# センサー fact_updates / values_6d の失敗分析と改善試行（1回）\n")
a("- 日付: 2026-11-20 / ブランチ: usir `phase2/sensor-fact-values-analysis`（基点 `phase2/sensor-full-annotation` 7f75069、ワークツリー `C:\\dev\\usir-sensor-fact-values`）。既存ブランチへのマージはしていない。")
a("- 対象: 公開テキストのみ（現代文3作品11場面＋F9 Dev 4作品24場面＝35場面）。妻の原稿は使っていない。")
a("- モデル・設定: qwen2.5:14b、seed 42、temperature 0（ベースラインと同一）。ローカル Ollama で実行。")
a("- 数値の出所: `scripts/phase2_fv/results/*.json`（本文は `render_fv_report.py` が JSON から転記）。\n")

a("## 1. 結論\n")
fv, fc, fb = M["fact_v2"]["all"]["fact_updates_pred_obj"], M["control"]["all"]["fact_updates_pred_obj"], M["baseline"]["all"]["fact_updates_pred_obj"]
vb, vc, vx = (M[k]["all"]["values_6d"]["gold_nonzero_axes_same_sign"] for k in ("baseline", "control", "values_ctx"))
a(f"- **fact_updates: 改善しなかった。** 定義を締めた版（fact_v2）の P/R は {pf(fv['precision'])}/{pf(fv['recall'])}。同一プロンプトの再実行（control）は {pf(fc['precision'])}/{pf(fc['recall'])}、元のベースラインは {pf(fb['precision'])}/{pf(fb['recall'])}。"
  f"出力件数は {fc['tp']+fc['fp']} → {fv['tp']+fv['fp']} に減り、作り話・述語取り違え・主語取り違えは減ったが、正解も同時に減って precision は横ばい、recall は下がった。")
a(f"- **values_6d: 改善しなかった。** 既知情報を足した条件（values_ctx）の非0軸の符号一致は {vx['match']}/{vx['n']}（{pf(vx['rate'])}）。ベースライン {vb['match']}/{vb['n']}（{pf(vb['rate'])}）、control {vc['match']}/{vc['n']}（{pf(vc['rate'])}）を下回った。")
a(f"- **測定そのものの揺れが大きい。** 同じプロンプト・seed・temperature の control がベースラインと異なる出力になり、fact P/R が {pf(fb['precision'])}/{pf(fb['recall'])} から {pf(fc['precision'])}/{pf(fc['recall'])} に動いた（正解として一致した金事実はベースライン {vcmp['baseline']['gold_matched_any']} 件、control {vcmp['control']['gold_matched_any']} 件、両方で一致 {vcmp['gold_matched_overlap']['baseline&control']} 件）。"
  "35場面・金事実74件では、この揺れより小さい差は判定できない。\n")

a("## 2. 作業1: fact_updates の失敗分析（ベースライン、全件手分類）\n")
a(f"対象: ベースライン（`scripts/phase2_stage2/annotations`）の FP {cls['FP_total']} 件・FN {cls['FN_total']} 件。件数は `stageB_metrics.json`（all: fp 78 / fn 60）と一致。分類は私（1名）が本文を読んで付けた。第二評価者はいない。\n")
a("### 2.1 FP の分類\n")
a("| 分類 | 件数 | 定義 |\n|---|---|---|")
for k, name in (("A", "本文に根拠がない作り話"), ("B", "根拠はあるが粒度が違う（細かすぎ・粗すぎ）"), ("C", "述語の取り違え"), ("D", "主語・対象の取り違え"), ("E", "ゴールド側の漏れの疑い")):
    a(f"| {k} {name} | {fp.get(k, 0)} | {cls['class_legend'][k]} |")
a(f"| 計 | {cls['FP_total']} | |\n")
a("代表例:\n")
for lab, picks in (("A", [("FP", "001383_056643:s2", 0), ("FP", "000050_003586:s2", 0)]), ("B", [("FP", "001798_056779:s2", 1), ("FP", "000081_000456:s3", 0), ("FP", "001383_056643:s0", 0)]),
                   ("C", [("FP", "000035_000301:s6", 1), ("FP", "000050_003586:s0", 0)]), ("D", [("FP", "000050_003586:s3", 0), ("FP", "001798_056779:s0", 7)]),
                   ("E", [("FP", "001798_056779:s0", 5), ("FP", "001798_056779:s1", 3)])):
    a(f"- **{lab}**")
    for p in picks:
        a(f"  - {line(ex(*p))}")
a("")
a("### 2.2 FN の分類\n")
a("FN は「作り話」「ゴールドの漏れ」に当たらない。次の4区分で付けた: M=場面内に対応する予測が無い（取りこぼし）／B・C・D=対応する予測が場面内にあるが、粒度（B）・述語（C）・主語や対象（D）が違う。\n")
a("| 分類 | 件数 |\n|---|---|")
for k in ("M", "B", "C", "D"):
    a(f"| {k} | {fn.get(k, 0)} |")
a(f"| 計 | {cls['FN_total']} |\n")
for p in (("FN", "001875_059897:s2", 0), ("FN", "000035_000301:s4", 1), ("FN", "001801_001786:s0", 0)):
    a(f"- {line(ex(*p))}")
a("")
a("### 2.3 読み取れること\n")
a(f"- FP の {fp.get('C', 0)+fp.get('D', 0)}/{cls['FP_total']} 件は述語または主語・対象の取り違えで、定義の曖昧さ（LEARNED を「話した側」に付ける、ACQUIRED を「渡した側・見るだけ」に付ける、HOOK_OPENED を地名や感想に付ける等）に由来する。作り話（A）は {fp.get('A', 0)} 件で少ない。")
a(f"- B（粒度）{fp.get('B', 0)} 件は、感情・口調・動作の叙述を STATE_CHANGED にしたもの。本文に根拠はあるが、ゴールドは人物の知識・所有・移動・大きな状態変化に絞っている。")
a(f"- E（ゴールドの漏れの疑い）は {fp.get('E', 0)} 件。うち境界例（`unc` 付き）を含む。ゴールドは1名がテキストから作成したもので、網羅性は保証されない。E を TP とみなせば precision は上がるが、その場合も {fp.get('A',0)+fp.get('B',0)+fp.get('C',0)+fp.get('D',0)} 件は誤り側に残る。")
a(f"- FN の {fn.get('M', 0)}/{cls['FN_total']} 件は対応する予測が場面内に無い取りこぼし。FP を減らすだけでは recall は上がらない。")
a(f"- 照合規則の影響: 意味は対応するが object の表層が一致せず FP＋FN の対になった例が {fl.get('surf', 0)} 組（`surf`）、同一事実の重複で subject 違いの版が TP になった例が {fl.get('dup', 0)} 件（`dup`）。lenient 照合は subject を見ないため、strict（subject 一致必須）では P/R が下がる（baseline: {pf(M['baseline']['all']['fact_updates_strict_subject']['precision'])}/{pf(M['baseline']['all']['fact_updates_strict_subject']['recall'])}）。")
a(f"- 分類の不確実さ: 境界例として `unc` を付けた件が {fl.get('unc', 0)} 件ある。\n")
a("述語別（FP / FN）:\n")
a("| 述語 | FP | FN |\n|---|---|---|")
for p in sorted(set(cls["FP_by_predicate"]) | set(cls["FN_by_predicate"])):
    a(f"| {p} | {cls['FP_by_predicate'].get(p, 0)} | {cls['FN_by_predicate'].get(p, 0)} |")
a("")

a("## 3. 作業1: values_6d のズレ分析（ベースライン）\n")
c = mism["gold_nonzero_axes_category"]
a(f"- ゴールドの人物×場面 {mism['gold_char_scenes']} 件のうち、センサーが状態を出した人物は {mism['predicted_char_scenes']} 件、出していない人物が {mism['not_predicted']} 件。")
a(f"- ゴールドの非0軸 {mism['gold_nonzero_axes']} 個の内訳: 人物ごと未出力 {c['character_not_predicted']} / 全軸0のベクトル {c['pred_zero_on_axis(all-zero vector)']} / 他の軸は非0だがその軸が0 {c['pred_zero_on_axis']} / 同符号 {c['same_sign']} / 逆符号 {c['opposite_sign']}。")
a(f"- 既存の指標（人物が両方にある46件、金非0軸66）では 同符号 {c['same_sign']}、予測0 {c['pred_zero_on_axis(all-zero vector)']+c['pred_zero_on_axis']}、逆符号 {c['opposite_sign']}。")
a(f"- センサーの非0軸 {mism['pred_nonzero_axes']} 個のうち、ゴールドが0の軸に付いたものが {mism['pred_nonzero_on_gold_zero_axes']} 個。出力した全ベクトルのうち全軸0が {mism['pred_vectors_all_zero']} 件、5軸以上が非0の過密ベクトルが {len(mism['pred_dense_vectors_(>=5 nonzero axes)'])} 件。")
pers = mism["gold_nonzero_axes_persisting_in_other_scenes_of_same_character"]
a(f"- ゴールドの非0軸 {pers['of']} 個のうち {pers['n']} 個は、同じ人物の別の場面でも同じ符号で付いている（人物に付く恒常的な傾向）。\n")
a("ズレの型（人物×場面を読んだ結果。`values_mismatch.py` の出力に基づく）:\n")
a("1. **場面だけでは決まらない人物の恒常的特性を読もうとしている／読めない。** 例: `001798_056779` の boku はゴールドが4場面とも秩序軸 +1（契約・権利を律儀に主張する人物）だが、s0・s1 でセンサーは全軸0。場面内の1文では裏付けが無く0にしている。")
a("2. **感情・口調を価値軸に変換している。** 例: `001798_056779:s2/s3` boku の根拠は「むっとして言い返しました」等で、センサーは利己・情念軸に ±1 を付け、ゴールド（秩序+1）とは別の軸になる。")
a("3. **引用された他人の価値観・台詞の内容を話者の価値観にしている。** 例: `001801_001786:s1` shinichi は父の教えを語る場面で、センサーは6軸すべて非0（ゴールドは利己・献身 +1 の1軸）。`000050_003586:s1` tabiso も6軸非0（ゴールドは神秘 +1・献身 +1）。")
a("4. **気分・体調の描写を複数軸の負に広げている。** 例: `000035_000301:s4` yozo はゴールドが全軸0だが、センサーは5軸を -1 にしている（根拠は「陰鬱な心を隠さず…歯痛」）。")
a("5. **遊戯・役柄の言動を価値観にしている。** 例: `001383_056643:s3` shinichi はゴールドの秩序 -1 に対し、センサーは秩序 +1。")
a("6. **台詞や動作が少ない人物は状態自体が出ない。** 例: jochu（女中）、tsuneko、doshi、horiki、kundo、saihachi 等はゴールドに状態があるが、予測に人物が出ない／全軸0。\n")
a("軸別の集計は `results/values_mismatch_summary.json` の `by_axis`。\n")

a("## 4. 作業2: 改善を1回だけ試した結果\n")
a("### 4.1 実施内容\n")
a("- 共通: 同じ 35 場面、qwen2.5:14b、seed 42、temperature 0。スキーマ・決定論バリデータ（basis は本文の部分文字列）・照合規則は変更なし。発話部分（話者・宛先）はベースラインのものを流用。")
a("- control: プロンプト無変更の再実行。ベースラインとの差（実行間の揺れ）を測る対照。")
a("- fact_v2: システムプロンプトの項目5のみを差し替え（項目1〜4は同一）。内容: 本文に明示の根拠がある状態変化だけ／根拠語句を変化そのものの語句にする／主体が特定できなければ出さない（hook のみ subject=null 可）／聞き手・受け取り側が subject／迷ったら出さない／1場面最大3件／述語7種の定義を厳格化し、気分・口調・内心・反復動作・情景・配役・地名の言及を除外。")
a("- values_ctx: システムプロンプトに補足1段落＋利用者メッセージに【既知情報】を追加。既知情報は、名簿の説明（`role`。現代文3作品にのみ存在し、F9 Dev 4作品には無い）と、**同じ実行内で前の場面までにセンサーが出力しバリデータを通った explicit の価値観・事実の要約**（人物ごと最新6件）。ゴールドは使っていない。センサー自身の出力が全0に近いため、既知情報は薄い。")
a(f"- 実行時間（ローカル）: control {logs['control']['total_elapsed_sec']}s、fact_v2 {logs['fact_v2']['total_elapsed_sec']}s、values_ctx {logs['values_ctx']['total_elapsed_sec']}s（values_ctx は履歴のため全場面を実行）。\n")
a("### 4.2 fact_updates（全35場面、lenient 照合）\n")
a("| 条件 | 出力件数 | TP | FP | FN | precision | recall | F1 | strict P / R |\n|---|---|---|---|---|---|---|---|---|")
for n in ("baseline", "control", "fact_v2"):
    a(frow(n))
a("")
a("グループ別:\n")
a("| 条件 | グループ | precision | recall |\n|---|---|---|---|")
for n in ("baseline", "control", "fact_v2"):
    for g in ("modern_11", "f9dev_24"):
        f = M[n][g]["fact_updates_pred_obj"]
        a(f"| {n} | {g} | {pf(f['precision'])} | {pf(f['recall'])} |")
a("")
a("ベースラインで手分類した FP が、各条件で（subject・述語・object 一致で）再び出力された件数:\n")
a("| 分類 | control | fact_v2 |\n|---|---|---|")
for k in ("A", "B", "C", "D", "E"):
    a(f"| {k} | {vcmp['control']['baseline_FP_still_emitted_by_class'][k]} | {vcmp['fact_v2']['baseline_FP_still_emitted_by_class'][k]} |")
a(f"| 計 | {vcmp['control']['total']} | {vcmp['fact_v2']['total']} |\n")
a(f"- 1場面あたり件数: control 平均 {vcmp['control']['facts_per_scene']['mean']}（最大 {vcmp['control']['facts_per_scene']['max']}、空 {vcmp['control']['facts_per_scene']['empty_scenes']} 場面）、fact_v2 平均 {vcmp['fact_v2']['facts_per_scene']['mean']}（最大 {vcmp['fact_v2']['facts_per_scene']['max']}、空 {vcmp['fact_v2']['facts_per_scene']['empty_scenes']} 場面）。subject=null の件数は control {vcmp['control']['null_subject']} → fact_v2 {vcmp['fact_v2']['null_subject']}。HOOK_OPENED は {vcmp['control']['pred_by_predicate'].get('HOOK_OPENED', 0)} → {vcmp['fact_v2']['pred_by_predicate'].get('HOOK_OPENED', 0)} 件。")
a(f"- ベースラインで一致していた金事実のうち fact_v2 でも一致したのは {vcmp['gold_matched_overlap']['baseline&fact_v2']} 件、一致しなくなったのは {vcmp['gold_matched_overlap']['baseline_only_vs_fact_v2']} 件、新たに一致したのは {vcmp['gold_matched_overlap']['fact_v2_only_vs_baseline']} 件。")
a("- 解釈: 厳格化は狙った型（作り話A、述語取り違えC、主語取り違えD、hook の乱用）の出力を大きく減らした。一方で、当たっていた出力も減り（金事実の一致数 control "
  f"{vcmp['control']['gold_matched_any']} → fact_v2 {vcmp['fact_v2']['gold_matched_any']}）、残った FP は粒度違い（B）と、新たに出た誤り（例: 遊戯中の STATE_CHANGED、主語を取り違えた MOVED_TO）が中心。precision は control と同水準で、recall が下がった。")
a("- 誤りの型を絞っても精度が上がらなかった理由（推測、未検証）: ①モデルが「明示の根拠がある変化」を判別する力が弱く、抑制が正解・不正解を区別せず一律に効いた可能性。②ゴールドの粒度（人物の知識・所有・移動・大きな状態変化）と、プロンプトの定義が一致していない箇所が残る。③FN の大半（M）は取りこぼしで、抑制では解決しない。\n")
a("### 4.3 values_6d（全35場面）\n")
a("| 条件 | 状態を出した人物/ゴールド人物場面 | 非0軸の符号一致（人物一致分） | 符号一致/ゴールド非0軸97全体 | 予測0 | 逆符号 | 予測の非0軸 | うちゴールド0の軸 | 全軸0ベクトル |\n|---|---|---|---|---|---|---|---|---|")
for n in ("baseline", "control", "values_ctx"):
    a(vrow(n))
a("")
a("- 各条件で分母（人物が両方にある場合の金非0軸数）が異なるため、全体97軸に対する符号一致数も併記した。")
a(f"- values_ctx は非0の出力が減り（{vcomp['control']['pred_nz']} → {vcomp['values_ctx']['pred_nz']}）、誤りは減ったが（ゴールド0の軸への付与 {vcomp['control']['pred_nz_on_goldzero']} → {vcomp['values_ctx']['pred_nz_on_goldzero']}）、当たりも減った（{vcomp['control']['same']} → {vcomp['values_ctx']['same']}）。状態を出した人物も減った（{vcomp['control']['matched']} → {vcomp['values_ctx']['matched']}）。軸単位の完全一致率はゼロが増えた分だけ上がるが（{pf(M['control']['all']['values_6d']['axis_exact']['rate'])} → {pf(M['values_ctx']['all']['values_6d']['axis_exact']['rate'])}）、ゼロ固定の基準との比較では意味が薄い。")
a("- 解釈（推測、未検証）: 既知情報はセンサー自身の過去出力が主で、全0が多く情報量が乏しい。名簿の説明（role）は現代文3作品にしか無い。さらに「恒常的特性は場面外の文脈から決まる」というズレの型に対し、既知情報を足してもモデルは非0を出す方向に動かず、むしろ保守的になった。作業1の型1（恒常的特性）を狙うなら、人物の恒常的な価値観を別の入力（作品全体の読解結果など）として与える必要があるが、本試行はそこまで行っていない。\n")

a("## 5. 限界・未検証\n")
a("- ゴールドは1名（モデル）が本文から作成した35場面。価値観・事実更新は主観を含み、評価者間一致は無い。FP/FN の手分類も1名で、第二評価者による確認は無い。")
a("- 同一条件の再実行でも結果が揺れた（control とベースライン）。各条件1回ずつの測定なので、条件間の差のうち揺れの範囲内のものは有意と言えない。複数回の反復は、依頼どおり行っていない。")
a("- fact_v2 は複数の変更（主体必須、件数上限3、述語定義、除外規則）を同時に入れた1版であり、どの変更が効いた／効かなかったかは切り分けていない。")
a("- values_ctx の既知情報は、センサー自身の出力と名簿の説明のみ。ゴールド由来の情報は使っていない。F9 Dev 4作品には名簿の説明が無い。")
a("- 照合規則（ベースラインと同じ）は object の部分一致またはbigram重なり。表層が違う同義の事実は不一致になる。")
a("- 成果物は `scripts/phase2_fv/results/`。GPU はローカル（RTX 4070）で実行した。\n")
a("## 6. 成果物\n")
a("- `scripts/phase2_fv/results/fact_error_classification.json` — FP/FN 全件の分類（理由・フラグ付き）")
a("- `scripts/phase2_fv/results/values_mismatch_summary.json` — values_6d のズレの集計")
a("- `scripts/phase2_fv/results/metrics_{baseline,control,fact_v2,values_ctx}.json` — 再測定の指標")
a("- `scripts/phase2_fv/results/variant_comparison.json`, `values_compare.json` — 条件間の比較")
a("- `scripts/phase2_fv/results/run_*/` — 各条件のアノテーション・生出力・ログ")
a("- `scripts/phase2_fv/prompts_fv.py`, `run_variant.py` — プロンプトと実行スクリプト\n")
a("進捗: 作業1・作業2とも完了。NEXT: 結果の扱い（fact は recall 側の取りこぼし対策、values は恒常的特性の入力設計）をユーザーが決める。")
text = "\n".join(L) + "\n"
(R / "report.md").write_text(text, encoding="utf-8")
print(len(text))
