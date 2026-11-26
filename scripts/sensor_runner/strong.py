"""Strong-model annotation format (sensor.strong/1): instructions, and deterministic validation (grounding rules).

The strong model (a Claude Code session) reads each scene once and fills everything the local model + deterministic
code do not own: presence completion, time/location/pov, facts, trait variations, event/mood codes, Awase ratings.
Nothing is repaired: an invalid file is rejected whole with reasons.
"""
from __future__ import annotations

import hashlib
from typing import Any

from common import INSTRUCTION_VERSION, STRONG_SCHEMA, in_text, norm
from nagare_awase_prompt import EVENTS, MOODS

PREDICATES = ["LEARNED", "ACQUIRED", "LOST", "MOVED_TO", "STATE_CHANGED", "HOOK_OPENED", "HOOK_RESOLVED"]
EVIDENCE = ["explicit", "inferred", "ambiguous"]
RELATIVE_ORDER = ["continuous", "flashback", "parallel", "unknown"]
# STATE_CHANGED is semi-structured: target + category + free text. "unclassified" is only allowed for seed entries.
STATE_CATEGORIES = ["body_health", "emotion", "relation_status", "object_state", "identity_form", "situation", "other"]
SEED_ONLY_CATEGORY = "unclassified"

_EV = " / ".join(f"{k}={v}" for k, v in EVENTS.items())
_MO = " / ".join(f"{k}={v}" for k, v in MOODS.items())
_PR = " / ".join(PREDICATES)
_CA = " / ".join(STATE_CATEGORIES)

INSTRUCTIONS = (
    "あなたは小説の場面注釈器（強いモデル）です。1つのパックにつき1つの JSON（sensor.strong/1）を出力ファイルへ書きます。\n"
    "分担: 発話の話者・宛先・根拠はローカルモデルが、口調（politeness）は Policy X の規則が、一人称は表層規則が付けます。ここでは書きません。\n"
    "あなたが書く項目:\n"
    "1. time: relative_order（continuous/flashback/parallel/unknown。直前場面との関係）、explicit_time・elapsed_hint（本文の逐語の語句、無ければ null）、evidence（explicit/inferred/ambiguous）。\n"
    "2. location: 場面の主な場所を表す本文の逐語の語句。判断できなければ null。\n"
    "3. pov_character: 視点人物の名簿 id、無ければ null。\n"
    "4. present: その場に居合わせる人物（名簿 id）と basis（居合わせることを示す本文の逐語の語句）。ローカル側の発話者・宛先の補完を含む。話題に出るだけの人物・回想・伝聞・電話の相手方は含めない。\n"
    "5. trait_changes: 能動的な人物ごとに、台帳（baseline）の6軸値に対する変動のみを書く。changes=[{axis(0-5), delta(-2..2, 0以外), basis}]。変動が無い人物は changes=[] で記録（台帳どおりの値になる）。\n"
    "   軸: [0]生死観(-1=死・虚無／+1=生) [1]秩序・規範(-1=逸脱／+1=秩序) [2]世俗・神秘(-1=世俗／+1=神秘) [3]利己・献身(-1=利己／+1=献身) [4]理性・情念(-1=理性／+1=情念) [5]宿命・自由意志(-1=宿命／+1=自由意志)。\n"
    "   適用後の値は [-1,+1] に丸められる。台帳に無い人物は ledger_additions に baseline（6整数、各 -1..1）を添える（既存の baseline は上書き不可）。\n"
    "   pack の台帳（ledger_baseline）に無い人物は、他の場面で既に追加済みかもしれない場合でも、毎回 ledger_additions に入れること。重複は先勝ちで吸収される。\n"
    f"6. facts: 事実の更新。predicate は {_PR}。subject は名簿 id か null、evidence、basis（本文の逐語の語句、必須）。\n"
    f"   STATE_CHANGED のみ半構造化: target（対象。人物 id または物・事柄の短い語）、category（{_CA}）、text（自由記述）。他の predicate は object（短い語句）。\n"
    "7. event_top3 / mood_top3: 出来事コード・風情コードを強い順に重複なく3つ。各 basis は本文の逐語。\n"
    f"   出来事: {_EV}\n"
    f"   風情: {_MO}\n"
    "8. awase: F_valence・F_arousal・f_valence・f_arousal（各 -2..2 の整数）、F_basis・f_basis（本文の逐語）。F は出来事そのもの、f は文体・語り口の情緒。別々に評価する。\n"
    "basis の規則: 本文中に実在する連続した文字列を一字も変えずにコピーする（空白差のみ許容）。要約・言い換え・複数箇所の連結は不可。根拠が無い項目は出さない。\n"
)

OUTPUT_TEMPLATE = {
    "schema": STRONG_SCHEMA,
    "instruction_version": INSTRUCTION_VERSION,
    "scene_hash": "<pack.scene_hash をそのまま>",
    "annotator": "claude-code-session:<model id>",
    "provenance": "session",
    "time": {"relative_order": "continuous", "explicit_time": None, "elapsed_hint": None, "evidence": "inferred"},
    "location": None,
    "pov_character": None,
    "present": [{"id": "<roster id>", "basis": "<本文の逐語>"}],
    "ledger_additions": {},
    "trait_changes": [{"character_id": "<roster id>", "evidence": "inferred", "changes": [{"axis": 0, "delta": 1, "basis": "<本文の逐語>"}]}],
    "facts": [
        {"subject": "<roster id|null>", "predicate": "MOVED_TO", "object": "<短い語句>", "evidence": "explicit", "basis": "<本文の逐語>"},
        {"subject": "<roster id|null>", "predicate": "STATE_CHANGED", "target": "<対象>", "category": "emotion", "text": "<自由記述>", "evidence": "inferred", "basis": "<本文の逐語>"},
    ],
    "event_top3": [{"code": "E03", "basis": "<本文の逐語>"}, {"code": "E02", "basis": "..."}, {"code": "E10", "basis": "..."}],
    "mood_top3": [{"code": "M05", "basis": "<本文の逐語>"}, {"code": "M12", "basis": "..."}, {"code": "M10", "basis": "..."}],
    "awase": {"F_valence": 0, "F_arousal": 0, "f_valence": 0, "f_arousal": 0, "F_basis": "<本文の逐語>", "f_basis": "<本文の逐語>"},
}


def instruction_hash() -> str:
    return hashlib.sha256((INSTRUCTION_VERSION + INSTRUCTIONS).encode("utf-8")).hexdigest()


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def validate_strong(entry: Any, scene_text: str, roster_ids: list[str], ledger: dict[str, list[int]], scene_hash: str) -> list[str]:
    """Return a list of rejection reasons (empty = valid)."""
    errs: list[str] = []
    if not isinstance(entry, dict):
        return ["not_an_object"]
    seed = str(entry.get("provenance", "")).startswith("seed:")
    nt = norm(scene_text)
    ids = set(roster_ids)

    if entry.get("schema") != STRONG_SCHEMA:
        errs.append(f"schema != {STRONG_SCHEMA}")
    if entry.get("instruction_version") != INSTRUCTION_VERSION:
        errs.append(f"instruction_version != {INSTRUCTION_VERSION}")
    if entry.get("scene_hash") != scene_hash:
        errs.append("scene_hash mismatch with pack")
    for k in ("annotator", "provenance"):
        if not isinstance(entry.get(k), str) or not entry.get(k):
            errs.append(f"{k}: missing")

    def basis_ok(where: str, b: Any) -> None:
        if b is None and seed:
            return
        if not in_text(b, nt):
            errs.append(f"{where}: basis not verbatim in scene text: {b!r}")

    t = entry.get("time")
    if not isinstance(t, dict):
        errs.append("time: missing")
    else:
        if t.get("relative_order") not in RELATIVE_ORDER:
            errs.append(f"time.relative_order: {t.get('relative_order')!r}")
        if t.get("evidence") not in EVIDENCE:
            errs.append(f"time.evidence: {t.get('evidence')!r}")
        for k in ("explicit_time", "elapsed_hint"):
            v = t.get(k)
            if v is not None and not in_text(v, nt):
                errs.append(f"time.{k}: not verbatim in scene text: {v!r}")
    loc = entry.get("location")
    if loc is not None and not in_text(loc, nt):
        errs.append(f"location: not verbatim in scene text: {loc!r}")
    pov = entry.get("pov_character")
    if pov is not None and pov not in ids:
        errs.append(f"pov_character: not in roster: {pov!r}")

    pres = entry.get("present")
    if not isinstance(pres, list):
        errs.append("present: missing")
    else:
        seen: set[Any] = set()
        for i, p in enumerate(pres):
            pid = p.get("id") if isinstance(p, dict) else None
            if pid not in ids:
                errs.append(f"present[{i}]: id not in roster: {pid!r}")
            if pid in seen:
                errs.append(f"present[{i}]: duplicate id {pid!r}")
            seen.add(pid)
            basis_ok(f"present[{i}]", p.get("basis") if isinstance(p, dict) else None)

    adds = entry.get("ledger_additions") or {}
    if not isinstance(adds, dict):
        errs.append("ledger_additions: not an object")
        adds = {}
    for cid, vec in adds.items():
        if cid not in ids:
            errs.append(f"ledger_additions[{cid}]: not in roster")
        elif cid in ledger:
            errs.append(f"ledger_additions[{cid}]: baseline already exists (no overwrite)")
        if not (isinstance(vec, list) and len(vec) == 6 and all(_is_int(x) and -1 <= x <= 1 for x in vec)):
            errs.append(f"ledger_additions[{cid}]: need 6 ints in -1..1")
    known = set(ledger) | set(adds)
    tc = entry.get("trait_changes")
    if not isinstance(tc, list):
        errs.append("trait_changes: missing")
    else:
        seen = set()
        for i, c in enumerate(tc):
            cid = c.get("character_id") if isinstance(c, dict) else None
            if cid not in ids:
                errs.append(f"trait_changes[{i}]: character not in roster: {cid!r}")
            elif cid not in known:
                errs.append(f"trait_changes[{i}]: no ledger baseline for {cid!r} (add via ledger_additions)")
            if cid in seen:
                errs.append(f"trait_changes[{i}]: duplicate character {cid!r}")
            seen.add(cid)
            if not isinstance(c, dict) or c.get("evidence") not in EVIDENCE:
                errs.append(f"trait_changes[{i}]: evidence")
                continue
            axes: set[Any] = set()
            for j, ch in enumerate(c.get("changes") or []):
                ax, d = ch.get("axis"), ch.get("delta")
                if not (_is_int(ax) and 0 <= ax <= 5):
                    errs.append(f"trait_changes[{i}].changes[{j}]: axis {ax!r}")
                if not (_is_int(d) and -2 <= d <= 2 and d != 0):
                    errs.append(f"trait_changes[{i}].changes[{j}]: delta {d!r}")
                if ax in axes:
                    errs.append(f"trait_changes[{i}].changes[{j}]: duplicate axis {ax!r}")
                axes.add(ax)
                basis_ok(f"trait_changes[{i}].changes[{j}]", ch.get("basis"))

    facts = entry.get("facts")
    if not isinstance(facts, list):
        errs.append("facts: missing")
    else:
        for i, f in enumerate(facts):
            if not isinstance(f, dict):
                errs.append(f"facts[{i}]: not an object")
                continue
            if f.get("predicate") not in PREDICATES:
                errs.append(f"facts[{i}]: predicate {f.get('predicate')!r}")
            if f.get("subject") is not None and f.get("subject") not in ids:
                errs.append(f"facts[{i}]: subject not in roster: {f.get('subject')!r}")
            if f.get("evidence") not in EVIDENCE:
                errs.append(f"facts[{i}]: evidence {f.get('evidence')!r}")
            basis_ok(f"facts[{i}]", f.get("basis"))
            if f.get("predicate") == "STATE_CHANGED":
                cat = f.get("category")
                if not (cat in STATE_CATEGORIES or (seed and cat == SEED_ONLY_CATEGORY)):
                    errs.append(f"facts[{i}]: category {cat!r}")
                if not (isinstance(f.get("text"), str) and f["text"]):
                    errs.append(f"facts[{i}]: text missing")
                if not ((isinstance(f.get("target"), str) and f["target"]) or (seed and f.get("target") is None)):
                    errs.append(f"facts[{i}]: target missing")
            elif not (isinstance(f.get("object"), str) and f["object"]):
                errs.append(f"facts[{i}]: object missing")

    for key, table in (("event_top3", EVENTS), ("mood_top3", MOODS)):
        items = entry.get(key)
        if not (isinstance(items, list) and len(items) == 3):
            errs.append(f"{key}: need exactly 3 items")
            continue
        codes = [i.get("code") if isinstance(i, dict) else None for i in items]
        if len(set(codes)) != 3:
            errs.append(f"{key}: duplicate codes {codes}")
        for i, it in enumerate(items):
            if codes[i] not in table:
                errs.append(f"{key}[{i}]: bad code {codes[i]!r}")
            if not in_text(it.get("basis") if isinstance(it, dict) else None, nt):
                errs.append(f"{key}[{i}]: basis not verbatim in scene text")

    aw = entry.get("awase")
    if not isinstance(aw, dict):
        errs.append("awase: missing")
    else:
        for k in ("F_valence", "F_arousal", "f_valence", "f_arousal"):
            if not (_is_int(aw.get(k)) and -2 <= aw[k] <= 2):
                errs.append(f"awase.{k}: {aw.get(k)!r}")
        for k in ("F_basis", "f_basis"):
            if not in_text(aw.get(k), nt):
                errs.append(f"awase.{k}: not verbatim in scene text")
    return errs
