"""P-2 Task A: Tier-C 64-benchmark provenance audit. Outputs numbers/hashes only (no raw text)."""
import hashlib, json, sys, importlib.util
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "python"))
spec = importlib.util.spec_from_file_location("bx", R / "scripts/build_tier_c_modern_benchmark_expanded.py")
bx = importlib.util.module_from_spec(spec); spec.loader.exec_module(bx)
from usir_py.v5.modern_text_normalizer import extract_tier_c_irreversible_profile as prof
import os; os.chdir(R)
def load(p): return {json.loads(l)["sample_id"]: json.loads(l) for l in open(R/p, encoding="utf-8")}
exp = load("data/usir_v5_eval/tier_c_modern_benchmark_expanded.jsonl")
orig = load("data/usir_v5_eval/tier_c_modern_benchmark.jsonl")
samples = bx.get_r1_fiction_texts()+bx.get_r2_essay_texts()+bx.get_r3_technical_texts()+bx.get_aozora_r4_texts()
out=[]
for s in samples:
    sid=s["sample_id"]; t=s["text"]
    p=prof(t)
    match = p==exp[sid]["profile"]
    in_orig = sid in orig
    orig_match = (orig[sid]["profile"]==exp[sid]["profile"]) if in_orig else None
    out.append(dict(sample_id=sid, register=s["register"], n=len(t), sha256_16=hashlib.sha256(t.encode()).hexdigest()[:16],
        profile_reproduces_stored=match, in_original_32=in_orig, original_equals_expanded=orig_match,
        source_type=s["source_type"]))
# R4 provenance: locate in aozora cache
spl = bx.AozoraCorpusSplitter(Path("data")); man={m["work_id"]:m for m in spl.load_manifests()}
for o in out:
    if o["register"]=="R4":
        s=[x for x in samples if x["sample_id"]==o["sample_id"]][0]
        found=[w for w,m in man.items() if s["text"] in spl.get_cached_text(m).replace("\r\n","\n")]
        o["r4_found_in_aozora_cache_works"]=found
# R1-R3: any >=20-char verbatim overlap with the local aozora corpus?
corp="".join(spl.get_cached_text(m) for m in man.values()).replace("\r\n","\n")
for o in out:
    if o["register"]!="R4":
        s=[x for x in samples if x["sample_id"]==o["sample_id"]][0]["text"]
        hits=sum(1 for i in range(0,len(s)-20,5) if s[i:i+20] in corp)
        o["verbatim20_hits_in_local_aozora"]=hits
json.dump(out, open(R/"scripts/p2/p2_taskA_audit.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)
from collections import Counter
for reg in ["R1","R2","R3","R4"]:
    rs=[o for o in out if o["register"]==reg]
    print(reg, len(rs), "reproduce", sum(o["profile_reproduces_stored"] for o in rs),
      "in_orig32", sum(o["in_original_32"] for o in rs), "orig==exp", sum(bool(o["original_equals_expanded"]) for o in rs),
      "r4_found", sum(1 for o in rs if o.get("r4_found_in_aozora_cache_works")), "ov20", sum(o.get("verbatim20_hits_in_local_aozora",0) for o in rs))
