"""Phase 1.5 Stage 2 Step 0.5 measurement tool (measurement only; does not touch the cache implementation).

Subcommands: build | stamp <pid> start|end | check <pid> | analyze
Reuses the Step 0 A base packs byte for byte (only output_path is rewritten). Measurement-only extras
(typed facts, closed-vocabulary location) are written by the annotator to out/<pid>.m05.json and are NOT part of the pack.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import random
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import step0  # noqa: E402

WORK = HERE / "step05_work"
STEP0_WORK = HERE / "step0_work"
RESULTS = HERE / "results"
N_RUNS = 5

# Controlled predicate vocabulary for typed facts (20 + other).
PREDICATES: dict[str, str] = {
    "MOVED_TO": "人物が場所へ移動・到着する。object=場所の短い語",
    "LEFT": "人物が場所・集団から離れる。object=離れた場所/相手",
    "MET": "出会う・再会する・同席が始まる。object=相手（人物は roster id）",
    "ACQUIRED": "物・立場・能力を得る。object=得たもの",
    "LOST": "物・立場・能力を失う。object=失ったもの",
    "GAVE": "物を渡す・与える。subject=渡す側、object=渡したもの",
    "LEARNED": "知識・情報・事実を知る。object=知った内容の短い語",
    "TOLD": "情報・秘密・意図を相手に伝える/打ち明ける。object=内容の短い語",
    "SAW": "外界の出来事・物を知覚する/気付く。object=見たもの",
    "DECIDED": "決意・選択をする。object=決めた内容",
    "ASKED": "依頼・質問・要求・命令をする。object=内容の短い語",
    "PROMISED": "約束・誓いをする。object=内容の短い語",
    "CONFLICT": "口論・争い・攻撃・拒絶。object=相手（人物は roster id）",
    "EMOTION_CHANGED": "感情の変化。object=感情を表す短い語（例: 不安、安堵）",
    "RELATION_CHANGED": "人物間の関係の変化。object=相手（roster id）",
    "SITUATION_CHANGED": "周囲の状況・環境の変化。object=変化後の状況の短い語",
    "IDENTITY_CHANGED": "正体・身分・姿・呼称の変化。object=変化後の短い語",
    "BODY_CHANGED": "体調・負傷・身体状態の変化。object=状態の短い語",
    "HOOK_OPENED": "伏線・謎・未解決の問いが提示される。object=その短い語",
    "HOOK_RESOLVED": "伏線・謎・問いが解消する。object=その短い語",
    "other": "上のどれにも当てはまらない事実。object に内容の短い語",
}

# Closed location candidates per work. Built from seed locations + Step 0 observed strings (A runs) of the SAME 20 scenes,
# merging only pure surface variants (determiners, trailing particles, possessive wrappers). See the report for the rule.
LOCATIONS: dict[str, list[tuple[str, str]]] = {
    "000035_000301": [("L1", "下宿"), ("L2", "仙遊館"), ("L3", "銀座の大カフエ"), ("L4", "屋台のお鮨や"),
                      ("L5", "本所の大工さんの二階"), ("L6", "神田の屋台"), ("L7", "ビルの事務所の洋室"), ("L8", "自分の部屋")],
    "000081_000456": [("L1", "車室"), ("L2", "白鳥停車場"), ("L3", "プリオシン海岸"), ("L4", "白い岩（上・中）")],
    "001875_059897": [("L1", "安江の家")],
    "001798_056779": [("L1", "代田橋"), ("L2", "僕の部屋"), ("L3", "板の間")],
}


def _pid(tag: str) -> str:
    return "q" + hashlib.sha1(f"step05|{tag}".encode()).hexdigest()[:8]


def _manifest() -> dict[str, Any]:
    return json.loads((WORK / "manifest.json").read_text(encoding="utf-8"))


def cmd_build(a) -> int:
    from common import write_json

    for d in ("packs", "out", "m05in"):
        (WORK / d).mkdir(parents=True, exist_ok=True)
    s0 = json.loads((STEP0_WORK / "manifest.json").read_text(encoding="utf-8"))["packs"]
    base: dict[tuple[str, int], dict] = {}
    for pid, m in s0.items():
        if m["kind"] == "A" and m["run"] == 1:
            base[(m["work"], m["scene"])] = json.loads((STEP0_WORK / "packs" / f"{pid}.pack.json").read_text(encoding="utf-8"))
    items = sorted(base)
    assert len(items) == 20, len(items)
    manifest: dict[str, Any] = {"packs": {}, "batches": []}
    rng = random.Random(20261127)
    for run in range(1, N_RUNS + 1):
        order = items[:]
        rng.shuffle(order)
        for i in range(0, len(order), 5):
            pids = []
            for w, s in order[i : i + 5]:
                pid = _pid(f"{w}|{s}|{run}")
                p = copy.deepcopy(base[(w, s)])
                p["output_path"] = (WORK / "out" / f"{pid}.strong.json").as_posix()
                write_json(WORK / "packs" / f"{pid}.pack.json", p)
                write_json(WORK / "m05in" / f"{pid}.m05in.json", {
                    "work_id": w, "scene_index": s, "roster_ids": p["roster_all_ids"],
                    "predicates": PREDICATES,
                    "location_candidates": [{"id": i_, "label": l_} for i_, l_ in LOCATIONS[w]],
                    "m05_output_path": (WORK / "out" / f"{pid}.m05.json").as_posix(),
                })
                manifest["packs"][pid] = {"kind": "A", "cond": "base", "work": w, "scene": s, "run": run,
                                          "category": step0.SELECT[w][0]}
                pids.append(pid)
            manifest["batches"].append({"id": f"M{run}-{i // 5 + 1}", "pids": pids})
    write_json(WORK / "manifest.json", manifest)
    print(f"packs={len(manifest['packs'])} batches={len(manifest['batches'])}")
    for b in manifest["batches"]:
        print(b["id"], " ".join(b["pids"]))
    return 0


def validate_m05(m: Any, roster: list[str], work: str) -> list[str]:
    errs: list[str] = []
    if not isinstance(m, dict):
        return ["m05 must be an object"]
    ids = {i for i, _ in LOCATIONS[work]}
    lc = m.get("location_choice")
    if not (lc == "none" or lc in ids or (isinstance(lc, str) and lc.startswith("other:") and len(lc) > 6)):
        errs.append(f"location_choice {lc!r}: must be one of {sorted(ids)} / 'none' / 'other:<text>'")
    tf = m.get("typed_facts")
    if not isinstance(tf, list) or len(tf) > 10:
        errs.append("typed_facts must be a list of at most 10 items")
        tf = []
    for i, f in enumerate(tf):
        if not isinstance(f, dict) or set(f) != {"subject_id", "predicate", "object"}:
            errs.append(f"typed_facts[{i}]: keys must be exactly subject_id/predicate/object")
            continue
        if f["subject_id"] is not None and f["subject_id"] not in roster:
            errs.append(f"typed_facts[{i}]: subject_id {f['subject_id']!r} not in roster ids or null")
        if f["predicate"] not in PREDICATES:
            errs.append(f"typed_facts[{i}]: predicate {f['predicate']!r} not in vocabulary")
        o = f["object"]
        if not isinstance(o, str) or not o.strip() or len(o) > 20:
            errs.append(f"typed_facts[{i}]: object must be a non-empty string of at most 20 chars")
    return errs


def cmd_check(a) -> int:
    step0.WORK = WORK  # reuse the Step 0 strong-annotation validator on this work dir
    pack = json.loads((WORK / "packs" / f"{a.pid}.pack.json").read_text(encoding="utf-8"))
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = step0.cmd_check(a)
    reasons = [l[4:] for l in buf.getvalue().splitlines() if l.startswith("  - ")]
    mp = WORK / "out" / f"{a.pid}.m05.json"
    m5: list[str] = []
    try:
        m = json.loads(mp.read_text(encoding="utf-8"))
        m5 = [f"m05: {e}" for e in validate_m05(m, pack["roster_all_ids"], pack["work_id"])]
    except Exception as e:  # noqa: BLE001
        m5 = [f"m05: json_parse_error: {e}"]
    if rc == 0 and m5:  # step0.cmd_check logged this attempt as accepted; replace it with the combined verdict
        af = WORK / "attempts.jsonl"
        lines = af.read_text(encoding="utf-8").splitlines()
        last = json.loads(lines[-1])
        last.update({"accepted": False, "reasons": m5})
        af.write_text(chr(10).join(lines[:-1] + [json.dumps(last, ensure_ascii=False)]) + chr(10), encoding="utf-8")
    if rc == 0 and not m5:
        print("ACCEPT")
        return 0
    print("REJECT")
    for r in reasons + m5:
        print("  -", r)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("build"); p.set_defaults(f=cmd_build)
    p = sp.add_parser("stamp"); p.add_argument("pid"); p.add_argument("event", choices=["start", "end"])
    p.set_defaults(f=lambda a: (step0.__dict__.__setitem__("WORK", WORK), step0.cmd_stamp(a))[1])
    p = sp.add_parser("check"); p.add_argument("pid"); p.set_defaults(f=cmd_check)
    p = sp.add_parser("analyze"); p.set_defaults(f=lambda a: cmd_analyze(a))
    a = ap.parse_args()
    return a.f(a)


def cmd_analyze(a) -> int:
    import step05_analyze

    return step05_analyze.cmd_analyze(a)


if __name__ == "__main__":
    raise SystemExit(main())
