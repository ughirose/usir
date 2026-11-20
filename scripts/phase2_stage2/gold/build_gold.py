import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import gold_modern, gold_f9dev  # noqa
G = gold_modern.G
for k, v in G.items():
    if isinstance(v["explicit_time"], str): v["explicit_time"] = [v["explicit_time"]]
    elif v["explicit_time"] is None: v["explicit_time"] = []
out = {"note": "Stage B gold. Written by reading scene text directly, independent of sensor output. 11 modern scenes + 24 F9 Dev scenes.", "scenes": G}
Path(__file__).with_name("stage_b_gold.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(len(G), "scenes")
