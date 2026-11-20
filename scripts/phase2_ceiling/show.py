import json,sys
d=json.load(open('scripts/phase2_ceiling/scenes_input.json',encoding='utf-8'))
for wid in sys.argv[1:]:
    v=d[wid]
    print("=====",wid,v['group'],"narration",v['narration'],"narrator",v['narrator'],"pov",v['pov'])
    for r in v['roster']: print(" -",r['id'],r['aliases'],r['gender'])
    for s in v['scenes']:
        print("\n---",s['scene_id']); print(s['text'])
