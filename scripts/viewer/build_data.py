"""Collect per-work inputs (public texts, merged annotations, characters) under data/<work>/ and run `nishiki diagnose`.

NOTE: the staging dirs inputs/ (full-length runs from branch feat/fable-v5-shootout, scripts/sensor_runner/merged + data/aozora_cache)
and inputs_seed/ (`cli.py merge --out`) were consumed on 2026-11-27 and are not committed; the resulting data/<work>/ is committed
and is the input of build_items.py / build_viewer.py. Re-running this script needs those dirs recreated.
usage: python build_data.py <nishiki.exe>
Measurement/display only: diagnose is run unmodified; its JSON is stored as-is."""
import json, shutil, subprocess, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sensor_runner"))
import common  # noqa: E402

FULL = ("000129_000689", "000148_000751")  # full-length runs (inputs/), others are the seed works (inputs_seed/)


def main(nishiki: str) -> None:
    ws = sorted(common.all_work_ids() + list(FULL))
    D = HERE / "data"
    D.mkdir(exist_ok=True)
    for w in ws:
        d = D / w
        d.mkdir(exist_ok=True)
        if w in FULL:
            src = HERE / "inputs" / w
            shutil.copy(src / "text.txt", d / "text.txt")
            shutil.copy(src / f"{w}.annotations.json", d / "annotations.json")
            shutil.copy(src / f"{w}.characters.json", d / "characters.json")
        else:
            spec = common.load_work(w)
            shutil.copy(spec["text_abs"], d / "text.txt")
            shutil.copy(HERE / "inputs_seed" / f"{w}.annotations.json", d / "annotations.json")
            shutil.copy(HERE / "inputs_seed" / f"{w}.characters.json", d / "characters.json")
        r = subprocess.run([nishiki, "diagnose", str(d / "text.txt"), "--format", "json", "--characters", str(d / "characters.json"), "--annotations", str(d / "annotations.json")], capture_output=True, check=True)
        (d / "diagnose.json").write_bytes(r.stdout)
        j = json.loads(r.stdout.decode("utf-8"))
        sp = Counter((l["id"], s["rule"]) for l in j["lenses"] for s in l["spans"])
        cc = Counter((c["lens"], c["rule"], c["reason"]) for c in j.get("span_candidates", []))
        print(w, j["schema"], "scenes", j["input"]["scenes"], "spans", dict(sp), "cands", dict(cc), "ledger_cands", len(j.get("ledger_candidates", [])))


if __name__ == "__main__":
    main(sys.argv[1])
