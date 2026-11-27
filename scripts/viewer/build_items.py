"""Build the judgement items for the practical hit-rate measurement (display/measurement only; diagnose is not modified).

Strata (decided before judging, see the report section 1):
  S1 span            every confirmed `spans` entry of nishiki diagnose (all 9 works)                    -> primary denominator
  S2 cand_awase      every `span_candidates` entry with lens=awase (reason below_span_min)              -> secondary
  S3 cand_presence   fixed-seed sample of `presence_unconfirmed` (count_only) candidates                -> secondary
  C  control         random scenes WITHOUT any flag, shown with the same generic framing                  -> judge baseline
usage: python build_items.py  -> measure/items.json (key), measure/batches/batch_NN.json (shuffled, blind)
"""
import json
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "measure"
SEED = 20261127
N_PRESENCE_SAMPLE = 30
N_CONTROL = 14
BATCH = 6

VAL = {-2: "強く否定的", -1: "否定的", 0: "中立", 1: "肯定的", 2: "強く肯定的"}
ARO = {-2: "非常に静か", -1: "静か", 0: "平常", 1: "高揚", 2: "非常に高揚"}


def names(chars):
    return {c["id"]: (c["aliases"][1] if len(c["aliases"]) > 1 else c["id"]) for c in chars}


def scene_text(text, scenes, i):
    s = scenes[i]["span"]
    return text[s["start"]:s["end"]]


def trunc_head(t, n):
    return t if len(t) <= n else t[:n] + "…（以下略）"


def trunc_tail(t, n):
    return t if len(t) <= n else "（前略）…" + t[-n:]


def awase_claim(b, rule):
    ev = f"出来事の感情は{VAL.get(int(b['f_event_valence']), '?')}・{ARO.get(int(b['f_event_arousal']), '?')}"
    st = f"語り口の温度は{VAL.get(int(b['f_style_valence']), '?')}・{ARO.get(int(b['f_style_arousal']), '?')}"
    kind = "出来事に比べて語りが距離を置いて冷静すぎる" if rule == "awase_detachment" else "出来事と語り口の調子が逆向きである"
    return f"この場面は、{ev}と読めるが、{st}と読める。{kind}ため、読み手に温度差の違和感を与える可能性がある。"


def marked(text, s, e, before, after):
    a, z = max(0, s - before), min(len(text), e + after)
    return text[a:s] + "【" + text[s:e] + "】" + text[e:z]


def build():
    rnd = random.Random(SEED)
    items, flagged = [], {}
    pres_pool = []
    loaded = {}
    for wd in sorted(DATA.iterdir()):
        w = wd.name
        j = json.loads((wd / "diagnose.json").read_text(encoding="utf-8"))
        text = (wd / "text.txt").read_text(encoding="utf-8")
        nm = names(json.loads((wd / "characters.json").read_text(encoding="utf-8")))
        loaded[w] = (j, text)
        sc = j["scenes"]
        fl = flagged.setdefault(w, set())
        for l in j["lenses"]:
            for k, s in enumerate(l["spans"]):
                b, rule, si = s["basis"], s["rule"], s["scene"]
                fl.add(si)
                it = {"work": w, "stratum": "S1", "lens": l["id"], "rule": rule, "scene": si, "ref": f"{l['id']}#{k}", "score": s["score"]}
                if rule in ("move_destination_mismatch", "place_conflict"):
                    prev = int(b.get("known_scene", b.get("previous_scene")))
                    pl = b.get("known_location", b.get("previous_location"))
                    who = nm.get(b["character"], b["character"])
                    if rule == "move_destination_mismatch":
                        claim = f"場面{prev}で{who}は『{pl}』へ向かった／移ったと機械は読んだ。しかし場面{si}では{who}は『{b['location']}』にいる。場所のつながりが食い違っている可能性がある。"
                    else:
                        claim = f"場面{prev}で{who}は『{pl}』にいたが、同じ時刻と読める場面{si}では『{b['location']}』にいる。場所が食い違っている可能性がある。"
                    it["context"] = [
                        {"label": f"場面{prev}（直前）", "text": trunc_tail(scene_text(text, sc, prev), 800)},
                        {"label": f"場面{si}（指摘された場面）", "text": trunc_head(scene_text(text, sc, si), 1600)},
                    ]
                elif rule == "pov_contamination":
                    claim = f"この語りの視点人物は『{nm.get(b['pov'], b['pov'])}』だが、【】の箇所は『{nm.get(b['attributed_to'], b['attributed_to'])}』の内面や知覚として読める。視点が混ざっている可能性がある。"
                    it["context"] = [{"label": f"場面{si}の該当箇所（【】が指摘範囲）", "text": marked(text, s["start"], s["end"], 300, 300)}]
                elif rule == "quote_bracket":
                    claim = "かぎ括弧が閉じられないまま、本文（作品全体）が終わっている可能性がある。【】が閉じていない括弧の位置。"
                    tail = "（本文末）" if s["end"] + 500 >= len(text) else ""
                    it["context"] = [{"label": "括弧の位置から本文末まで（【】が括弧）", "text": marked(text, s["start"], s["end"], 200, 500) + tail}]
                elif rule == "off_stage_speaker":
                    claim = f"【】の台詞の話者は『{nm.get(b['speaker'], b['speaker'])}』と機械は読んだが、その人物はこの場面に居ない（登場していない）可能性がある。"
                    it["context"] = [{"label": f"場面{si}の該当箇所（【】が指摘範囲）", "text": marked(text, s["start"], s["end"], 500, 300)}]
                elif rule.startswith("awase_"):
                    claim = awase_claim(b, rule)
                    it["context"] = [{"label": f"場面{si}", "text": trunc_head(scene_text(text, sc, si), 1800)}]
                else:
                    raise ValueError(rule)
                it["claim"] = claim
                items.append(it)
        for c in j["span_candidates"]:
            si = c["scene"]
            if c["lens"] == "awase":
                fl.add(si)
                items.append({"work": w, "stratum": "S2", "lens": "awase", "rule": c["rule"], "scene": si, "ref": f"cand_awase#{si}", "score": c["score"],
                              "claim": awase_claim(c["basis"], c["rule"]), "context": [{"label": f"場面{si}", "text": trunc_head(scene_text(text, sc, si), 1800)}]})
            elif c["rule"] == "presence_unconfirmed":
                who = nm.get(c["basis"]["speaker"], c["basis"]["speaker"])
                pres_pool.append({"work": w, "stratum": "S3", "lens": "sasae", "rule": c["rule"], "scene": si, "ref": f"cand_presence#{si}:{c['utterance_index']}", "score": c["score"],
                                  "claim": f"場面{si}で『{who}』が台詞を話しているが、その人物がその場面に居ることが本文から確認できない可能性がある。",
                                  "context": [{"label": f"場面{si}", "text": trunc_head(scene_text(text, sc, si), 1600)}]})
                fl.add(si)
    presence_total = len(pres_pool)
    items += rnd.sample(pres_pool, min(N_PRESENCE_SAMPLE, presence_total))
    pool = [(w, i) for w, (j, _t) in sorted(loaded.items()) for i in range(len(j["scenes"])) if i not in flagged[w]]
    for n, (w, i) in enumerate(rnd.sample(pool, N_CONTROL)):
        j, text = loaded[w]
        items.append({"work": w, "stratum": "C", "lens": "-", "rule": "control", "scene": i, "ref": f"control#{n}", "score": None,
                      "claim": "この場面には、読み手が違和感を覚える箇所（場所・登場人物・視点・語り口のつながりの食い違い）がある可能性がある。",
                      "context": [{"label": f"場面{i}", "text": trunc_head(scene_text(text, j["scenes"], i), 1800)}]})
    for n, it in enumerate(items):
        it["id"] = f"I{n:03d}"
    return items, presence_total, len(pool)


if __name__ == "__main__":
    items, presence_total, pool = build()
    (OUT / "batches").mkdir(parents=True, exist_ok=True)
    (OUT / "items.json").write_text(json.dumps({"seed": SEED, "presence_total": presence_total, "control_pool": pool, "items": items}, ensure_ascii=False, indent=1), encoding="utf-8")
    order = list(items)
    random.Random(SEED + 1).shuffle(order)
    for bi in range(0, len(order), BATCH):
        chunk = [{"id": it["id"], "claim": it["claim"], "context": it["context"]} for it in order[bi:bi + BATCH]]
        (OUT / "batches" / f"batch_{bi // BATCH:02d}.json").write_text(json.dumps({"items": chunk}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("items", len(items), dict(Counter(i["stratum"] for i in items)), "presence_total", presence_total, "control_pool", pool, "batches", -(-len(order) // BATCH))
    print(dict(Counter((i["stratum"], i["rule"]) for i in items)))
