"""Viewer production version: one self-contained HTML page per work (no external files, no network).

Tabs: 物語構造 (scene list + scene text) / 設定台帳 (per-character facts registered by the annotations) / 自動診断の指摘.
Automatic-diagnosis flags are HIDDEN by default (tab present but muted; an explicit switch shows them).
Display only: reads nishiki diagnose JSON and merged annotations as they are; no judging logic here.

usage: python build_viewer.py <work_id> <out.html>   (inputs under scripts/viewer/data/<work_id>/)
"""
import html
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_items  # noqa: E402

E = html.escape
TITLES = {
    "000148_000751": "夏目漱石『二百十日』", "000129_000689": "森鴎外『山椒大夫』（抜粋）", "000035_000301": "太宰治『人間失格』（抜粋）",
    "000050_003586": "泉鏡花『草迷宮』（抜粋）", "000081_000456": "宮沢賢治『銀河鉄道の夜』（抜粋）", "001383_056643": "谷崎潤一郎『少年』（抜粋）",
    "001798_056779": "梅崎春生『ボロ家の春秋』（抜粋）", "001801_001786": "山川方夫『他人の夏』（抜粋）", "001875_059897": "壺井栄『雑居家族』（抜粋）",
    "000119_001738": "中島敦『弟子』",
}
RULE_LABEL = {
    "move_destination_mismatch": "場所のつながり（移動先）", "place_conflict": "場所のつながり（同時刻）", "pov_contamination": "視点の混ざり",
    "quote_bracket": "かぎ括弧", "off_stage_speaker": "話者の居場所", "awase_detachment": "語りの温度差", "awase_irony": "語りの温度差",
}


def para(t):
    return "".join(f"<p>{E(x)}</p>" for x in t.split("\n") if x.strip())


def build(work: str) -> str:
    d = HERE / "data" / work
    text = (d / "text.txt").read_text(encoding="utf-8")
    diag = json.loads((d / "diagnose.json").read_text(encoding="utf-8"))
    ann = json.loads((d / "annotations.json").read_text(encoding="utf-8"))
    chars = json.loads((d / "characters.json").read_text(encoding="utf-8"))
    nm = build_items.names(chars)
    scenes = diag["scenes"]
    title = TITLES.get(work, work)

    # --- 物語構造
    rows = []
    for i, sc in enumerate(scenes):
        a = ann[i] if i < len(ann) else {}
        m = a.get("scene_meta", {})
        present = "、".join(nm.get(c, c) for c in m.get("present_character_ids", []))
        moves = [f"{nm.get(f['subject'], f['subject'])}→{f['object']}" for f in a.get("fact_updates", []) if f.get("predicate") == "MOVED_TO"]
        ng = m.get("nagare", {})
        ev = " ".join(x["code"] for x in ng.get("event_top3", [])[:2])
        md = " ".join(x["code"] for x in ng.get("mood_top3", [])[:2])
        body = text[sc["span"]["start"]:sc["span"]["end"]]
        head = body.strip().replace("\n", " ")[:46]
        rows.append(
            f'<details class="scene"><summary><span class="no">場面{i + 1}</span><span class="head">{E(head)}…</span>'
            f'<span class="meta">{E(present) or "登場人物の登録なし"}</span></summary>'
            f'<div class="sceneinfo"><div>移動の登録：{E("、".join(moves)) if moves else "なし"}</div>'
            f'<div>出来事コード：{E(ev) or "-"}　気分コード：{E(md) or "-"}</div></div><div class="body">{para(body)}</div></details>'
        )
    structure = f'<p class="note">全{len(scenes)}場面。場面名をクリックすると本文を開きます。</p>' + "".join(rows)

    # --- 設定台帳
    per = defaultdict(list)
    for i, a in enumerate(ann):
        for f in a.get("fact_updates", []):
            per[f["subject"]].append((i, f))
    ledger = ['<p class="note">注釈として登録された事実だけを並べています（作者が登録した設定ではなく、機械が本文から読み取った記録です）。</p>']
    for c in chars:
        cid = c["id"]
        facts = per.get(cid, [])
        li = "".join(f"<li><span class=\"no\">場面{i + 1}</span>{E(f['predicate'])}：{E(str(f.get('object', '')))}<span class=\"ev\">{E(f.get('evidence', ''))}</span></li>" for i, f in facts)
        alias = "・".join(x for x in c["aliases"][1:]) or cid
        ledger.append(f'<details class="char"><summary><b>{E(nm.get(cid, cid))}</b><span class="meta">呼称：{E(alias)}／登録 {len(facts)} 件</span></summary><ul>{li or "<li>登録なし</li>"}</ul></details>')
    ledger_html = "".join(ledger)

    # --- 診断の指摘（既定で非表示）
    items, _pt, _cp = build_items.build()
    flags = [it for it in items if it["work"] == work and it["stratum"] == "S1"]
    cards = []
    for n, it in enumerate(flags, 1):
        ctx = "".join(f'<div class="ctxl">{E(c["label"])}</div><div class="quote">{para(c["text"]).replace("【", "<mark>").replace("】", "</mark>")}</div>' for c in it["context"])
        cards.append(f'<article class="flag"><div class="fh"><span class="no">#{n}</span><span class="tag">{E(RULE_LABEL.get(it["rule"], it["rule"]))}</span><span class="meta">場面{it["scene"] + 1}</span></div><p class="claim">{E(it["claim"])}</p>{ctx}</article>')
    flags_html = (f'<p class="note">自動診断が付けた確認事項です。機械の読み違いを含むため、既定では隠しています。（{len(flags)}件）</p>' + "".join(cards)) if flags else '<p class="note">この作品では確定した指摘はありません。</p>'

    css = """
body{margin:0;background:#f6f4ef;color:#2b2620;font:15px/1.8 "Hiragino Kaku Gothic ProN","Yu Gothic UI",Meiryo,sans-serif}
.wrap{max-width:900px;margin:0 auto;padding:20px 16px 64px}h1{font:600 1.35rem "Yu Mincho",serif;margin:0 0 4px}.sub{color:#6c6358;font-size:.88rem;margin:0 0 14px}
nav{display:flex;gap:6px;margin:10px 0 14px;flex-wrap:wrap}nav button{border:1px solid #dfd8cb;background:#fff;padding:6px 16px;border-radius:999px;cursor:pointer;font:inherit;font-size:.9rem}
nav button.on{background:#2b2620;color:#fff;border-color:#2b2620}.sw{margin-left:auto;font-size:.85rem;color:#6c6358;display:flex;align-items:center;gap:6px}
section{display:none}section.on{display:block}.note{color:#6c6358;font-size:.85rem}
details{background:#fffdf8;border:1px solid #dfd8cb;border-radius:8px;margin:8px 0;padding:8px 14px}summary{cursor:pointer;display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
.no{font-weight:700;color:#6c6358;font-size:.85rem}.head{font-family:"Yu Mincho",serif}.meta{color:#8a7e6e;font-size:.8rem;margin-left:auto}.sceneinfo{font-size:.82rem;color:#6c6358;margin:8px 0}
.body p,.quote p{margin:0 0 6px;font-family:"Yu Mincho",serif}.ev{margin-left:8px;color:#a39786;font-size:.75rem}ul{margin:8px 0;padding-left:18px}
.flag{background:#fffdf8;border:1px solid #dfd8cb;border-left:5px solid #c9a61a;border-radius:8px;padding:12px 16px;margin:12px 0}.fh{display:flex;gap:10px;align-items:center}
.tag{background:#f1ede4;border-radius:6px;padding:1px 9px;font-size:.8rem}.claim{margin:8px 0;font-size:.92rem}.ctxl{font-size:.78rem;color:#6c6358;margin-top:8px}
.quote{background:#fcfaf4;border:1px solid #e6dfd1;border-radius:6px;padding:8px 12px;font-size:.92rem}mark{background:#ffe58f}
.muted{border:1px dashed #c8bfb0;border-radius:8px;padding:14px;color:#6c6358;font-size:.9rem;background:#fbf9f3}footer{margin-top:28px;color:#8a7e6e;font-size:.78rem}
@media print{nav,.sw{display:none}section{display:block!important}body{background:#fff}.flag,details{break-inside:avoid}}
"""
    js = """
function tab(id,b){document.querySelectorAll('section').forEach(s=>s.classList.toggle('on',s.id===id));document.querySelectorAll('nav button').forEach(x=>x.classList.toggle('on',x===b));}
var _o=[];window.addEventListener('beforeprint',function(){_o=[];document.querySelectorAll('details').forEach(function(d){_o.push(d.open);d.open=true;});});window.addEventListener('afterprint',function(){document.querySelectorAll('details').forEach(function(d,i){d.open=_o[i];});});
function sw(c){document.getElementById('flagsbox').style.display=c.checked?'block':'none';document.getElementById('flagsmuted').style.display=c.checked?'none':'block';}
"""
    n_flags = len(flags)
    return f"""<!DOCTYPE html><html lang="ja"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>読み返しビューア — {E(title)}</title><style>{css}</style></head><body><div class="wrap">
<h1>{E(title)}</h1><p class="sub">対象：公開テキスト（青空文庫）／全{len(scenes)}場面／診断 {E(diag['schema'])}</p>
<nav><button class="on" onclick="tab('s1',this)">物語構造</button><button onclick="tab('s2',this)">設定台帳</button><button onclick="tab('s3',this)">自動診断の指摘</button>
<label class="sw"><input type="checkbox" onchange="sw(this)"> 自動診断の指摘を表示する（既定は非表示）</label></nav>
<section id="s1" class="on">{structure}</section>
<section id="s2">{ledger_html}</section>
<section id="s3"><div id="flagsmuted" class="muted">自動診断の指摘（{n_flags}件）は非表示にしています。右上のスイッチで表示できます。</div><div id="flagsbox" style="display:none">{flags_html}</div></section>
<footer>1枚で完結するHTMLです（外部ファイル・通信なし）。生成元：nishiki diagnose の出力と場面注釈。</footer></div><script>{js}</script></body></html>"""


if __name__ == "__main__":
    out = Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(sys.argv[1]), encoding="utf-8")
    print("wrote", out, out.stat().st_size, "bytes")
