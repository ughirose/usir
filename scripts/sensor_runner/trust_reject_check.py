"""Reject-path check for `ingest`: feed deliberately invalid strong annotations (mutations of an existing valid entry) and
record, per case: accepted?, reasons, reject file written?, and whether the cache tree (strong entries, ledger, manifest) changed.

Runs in a throwaway copy of the cache and packs (nothing under scripts/sensor_runner/cache is touched).
Usage: python trust_reject_check.py WORK_ID WORKDIR OUT_JSON
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cache import Cache  # noqa: E402
from common import CACHE_ROOT, find_nishiki, read_json, write_json  # noqa: E402
from ingest import ingest_file  # noqa: E402
from prepare import prepare_work  # noqa: E402


def tree_hash(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def build_world(work_id: str, scene_hash: str, wd: Path, nishiki: str, drop_ledger_additions: list[str]) -> tuple[Cache, Path, Path, dict]:
    """Copy of the real cache with the target scene's entry removed (and its ledger additions rolled back), packs regenerated."""
    if wd.exists():
        shutil.rmtree(wd)
    shutil.copytree(CACHE_ROOT, wd / "cache")
    cache = Cache(wd / "cache")
    valid = copy.deepcopy(cache.get(scene_hash))
    cache.delete(scene_hash)
    led = cache.ledger(work_id)
    for k in drop_ledger_additions:
        led.pop(k, None)
    cache.put_ledger(work_id, led, note="trust check world")
    packs, rejects = wd / "packs", wd / "rejects"
    prepare_work(work_id, cache, nishiki, packs_dir=packs)
    return cache, packs, rejects, valid


def main() -> int:
    work_id, wdir, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    nishiki = find_nishiki()
    s3 = "6aae1e7b907f"  # scene 3: no ledger_additions; has trait_changes with changes, STATE_CHANGED, 3 events
    s4 = "79367411331b"  # scene 4: ledger_additions kikori, kohagi
    h3 = next(p.stem for p in (CACHE_ROOT / "strong").glob(f"{s3}*.json"))
    h4 = next(p.stem for p in (CACHE_ROOT / "strong").glob(f"{s4}*.json"))

    cases: list[tuple[str, str, object]] = []  # (id, description, mutator(entry)->entry|str)

    def add(cid, desc, fn):
        cases.append((cid, desc, fn))

    def m(fn):
        def run(v):
            e = copy.deepcopy(v)
            fn(e)
            return e
        return run

    state_i = lambda e: next(i for i, f in enumerate(e["facts"]) if f["predicate"] == "STATE_CHANGED")
    add("R01_basis_not_in_text", "facts[0].basis replaced by a phrase absent from the scene text", m(lambda e: e["facts"][0].__setitem__("basis", "月の裏側に広がる砂漠の民")))
    add("R02_present_id_not_in_roster", "present gets an id not in the roster", m(lambda e: e["present"].append({"id": "ghost_not_in_roster", "basis": e["present"][0]["basis"]})))
    add("R03_state_category_outside_closed_set", "STATE_CHANGED.category = 'mental_state'", m(lambda e: e["facts"][state_i(e)].__setitem__("category", "mental_state")))
    add("R04_event_code_duplicate", "event_top3 codes duplicated (E09, E09, E16)", m(lambda e: e["event_top3"][1].__setitem__("code", e["event_top3"][0]["code"])))
    add("R05_session_uses_unclassified", "STATE_CHANGED.category = 'unclassified' (seed-only) in a session entry", m(lambda e: e["facts"][state_i(e)].__setitem__("category", "unclassified")))
    add("R06_event_code_out_of_table", "event_top3[0].code = 'E99'", m(lambda e: e["event_top3"][0].__setitem__("code", "E99")))
    add("R07_trait_character_not_in_roster", "trait_changes[0].character_id not in roster", m(lambda e: e["trait_changes"][0].__setitem__("character_id", "ghost_not_in_roster")))
    add("R08_trait_delta_zero_axis_range_dup_axis", "delta=0, axis=7, duplicate axis in one character", m(lambda e: e["trait_changes"][0].__setitem__("changes", [
        {"axis": 3, "delta": 0, "basis": e["trait_changes"][0]["changes"][0]["basis"]},
        {"axis": 7, "delta": 1, "basis": e["trait_changes"][0]["changes"][0]["basis"]},
        {"axis": 3, "delta": 1, "basis": e["trait_changes"][0]["changes"][0]["basis"]}])))
    add("R09_location_not_verbatim", "location is a phrase not in the scene text", m(lambda e: e.__setitem__("location", "どこかの港町")))
    add("R10_pov_not_in_roster", "pov_character not in roster", m(lambda e: e.__setitem__("pov_character", "ghost_not_in_roster")))
    add("R11_awase_out_of_range_and_basis", "awase.F_valence=3 and f_basis not verbatim", m(lambda e: e["awase"].update({"F_valence": 3, "f_basis": "存在しない文体の根拠"})))
    add("R12_instruction_version_wrong", "instruction_version = 'strong.v0'", m(lambda e: e.__setitem__("instruction_version", "strong.v0")))
    add("R13_ledger_addition_overwrites_existing", "ledger_additions for an id that already has a baseline", m(lambda e: e.__setitem__("ledger_additions", {"zushio": [0, 0, 0, 0, 0, 0]})))
    add("R14_duplicate_present_id", "present lists the same id twice", m(lambda e: e["present"].append(copy.deepcopy(e["present"][0]))))
    add("R15_multiple_violations", "R01+R02+R03+R04 in one file (all reasons must be reported)", m(lambda e: (
        e["facts"][0].__setitem__("basis", "月の裏側に広がる砂漠の民"),
        e["present"].append({"id": "ghost_not_in_roster", "basis": e["present"][0]["basis"]}),
        e["facts"][state_i(e)].__setitem__("category", "mental_state"),
        e["event_top3"][1].__setitem__("code", e["event_top3"][0]["code"]))))
    add("R16_unknown_scene_hash", "scene_hash with no pack", m(lambda e: e.__setitem__("scene_hash", "0" * 64)))
    add("R17_not_an_object", "top-level JSON array", lambda v: [1, 2, 3])
    add("R18_json_parse_error", "truncated JSON text", lambda v: "RAW:" + json.dumps(v, ensure_ascii=False)[:200])
    add("R19_facts_missing_basis", "facts[0].basis = null in a session entry", m(lambda e: e["facts"][0].__setitem__("basis", None)))

    results = []
    cache, packs, rejects, valid = build_world(work_id, h3, wdir / "w3", nishiki, [])
    pre_manifest = cache.manifest(work_id)["cached_count"]
    for cid, desc, fn in cases:
        before = tree_hash(cache.root)
        f = wdir / "inputs" / f"{cid}.strong.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        mutated = fn(valid)
        if isinstance(mutated, str) and mutated.startswith("RAW:"):
            f.write_text(mutated[4:], encoding="utf-8")
        else:
            f.write_text(json.dumps(mutated, ensure_ascii=False), encoding="utf-8")
        res = ingest_file(f, cache, packs, rejects)
        rf = rejects / f"{f.stem}.reject.json"
        rec = {"id": cid, "desc": desc, "accepted": res["accepted"], "reasons": res["reasons"],
               "reject_file_written": rf.exists(),
               "reject_file_reasons_equal": rf.exists() and read_json(rf)["reasons"] == res["reasons"],
               "cache_tree_unchanged": tree_hash(cache.root) == before,
               "strong_entry_exists_after": cache.get(h3) is not None}
        results.append(rec)

    # ledger must not be extended by a rejected file that carries a valid ledger_additions (scene 4 world)
    cache4, packs4, rejects4, valid4 = build_world(work_id, h4, wdir / "w4", nishiki, list(valid4_adds := read_json(CACHE_ROOT / "strong" / f"{h4}.json").get("ledger_additions", {})))
    before = tree_hash(cache4.root)
    ledger_before = cache4.ledger(work_id)
    e = copy.deepcopy(valid4)
    e["facts"][0]["basis"] = "月の裏側に広がる砂漠の民"
    f = wdir / "inputs" / "R20_ledger_additions_with_bad_basis.strong.json"
    f.write_text(json.dumps(e, ensure_ascii=False), encoding="utf-8")
    res = ingest_file(f, cache4, packs4, rejects4)
    results.append({"id": "R20_ledger_additions_with_bad_basis", "desc": f"valid ledger_additions {sorted(valid4_adds)} + facts[0].basis not verbatim",
                    "accepted": res["accepted"], "reasons": res["reasons"], "reject_file_written": (rejects4 / f"{f.stem}.reject.json").exists(),
                    "reject_file_reasons_equal": read_json(rejects4 / f"{f.stem}.reject.json")["reasons"] == res["reasons"],
                    "cache_tree_unchanged": tree_hash(cache4.root) == before, "ledger_unchanged": cache4.ledger(work_id) == ledger_before,
                    "ledger_ids_added_by_file_present_after": [k for k in valid4_adds if k in cache4.ledger(work_id)],
                    "strong_entry_exists_after": cache4.get(h4) is not None})
    # control: the unmodified entry is accepted (cache gains the entry; ledger gains its additions)
    res4 = ingest_file(_write(wdir / "inputs" / "C2_valid_scene4.strong.json", valid4), cache4, packs4, rejects4)
    control4 = {"accepted": res4["accepted"], "reasons": res4["reasons"], "strong_entry_exists_after": cache4.get(h4) is not None,
                "ledger_ids_added_by_file_present_after": [k for k in valid4_adds if k in cache4.ledger(work_id)],
                "manifest_cached_count_after": cache4.manifest(work_id)["cached_count"]}

    # control (scene 3 world): valid entry accepted, then an invalid re-ingest must leave the accepted entry byte-identical
    before_valid = tree_hash(cache.root)
    ok = ingest_file(_write(wdir / "inputs" / "C1_valid_scene3.strong.json", valid), cache, packs, rejects)
    after_valid = tree_hash(cache.root)
    entry_hash = after_valid[f"strong/{h3}.json"]
    e = copy.deepcopy(valid)
    e["event_top3"][1]["code"] = e["event_top3"][0]["code"]
    bad = ingest_file(_write(wdir / "inputs" / "C3_invalid_after_accept.strong.json", e), cache, packs, rejects)
    after_bad = tree_hash(cache.root)
    control = {
        "valid_accepted": ok["accepted"], "valid_reasons": ok["reasons"],
        "files_changed_by_valid_accept": sorted(k for k in after_valid if before_valid.get(k) != after_valid[k]),
        "manifest_cached_count_before_after": [pre_manifest, cache.manifest(work_id)["cached_count"]],
        "invalid_after_accept_accepted": bad["accepted"], "invalid_after_accept_reasons": bad["reasons"],
        "cache_tree_unchanged_by_invalid_after_accept": after_bad == after_valid,
        "accepted_entry_sha256_unchanged": after_bad[f"strong/{h3}.json"] == entry_hash,
    }
    summary = {
        "work_id": work_id, "scene3_hash": h3, "scene4_hash": h4,
        "cases": len(results),
        "rejected": sum(not r["accepted"] for r in results),
        "all_rejected_with_reasons": all((not r["accepted"]) and r["reasons"] for r in results),
        "all_reject_files_written": all(r["reject_file_written"] for r in results),
        "all_reject_reasons_equal": all(r["reject_file_reasons_equal"] for r in results),
        "all_cache_unchanged": all(r["cache_tree_unchanged"] for r in results),
    }
    write_json(out, {"summary": summary, "cases": results, "control_scene3": control, "control_scene4": control4})
    print(json.dumps(summary, ensure_ascii=False))
    for r in results:
        print(("ACCEPT" if r["accepted"] else "reject"), r["id"], "cache_unchanged=", r["cache_tree_unchanged"], "| reasons:", len(r["reasons"]), r["reasons"][:1])
    print("control3", control["valid_accepted"], control["files_changed_by_valid_accept"], control["cache_tree_unchanged_by_invalid_after_accept"])
    print("control4", control4)
    return 0


def _write(p: Path, obj) -> Path:
    p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return p


if __name__ == "__main__":
    raise SystemExit(main())
