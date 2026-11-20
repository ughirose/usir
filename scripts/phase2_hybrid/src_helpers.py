import json
E,I,A="explicit","inferred","ambiguous"
def u(s):
    """compact: spk>addr:E|I|A ; '-' = null; '~' = non-speech quote (speaker null, ambiguous)"""
    out=[]
    for t in s.split():
        if t=='~': out.append([None,None,A]); continue
        sp,rest=t.split('>'); ad,ev=rest.split(':')
        out.append([sp, None if ad=='-' else ad, {'E':E,'I':I,'A':A}[ev]])
    return out
def save(wid, scenes):
    json.dump({"scenes":scenes},open(f"src/{wid}.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
