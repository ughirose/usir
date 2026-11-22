import json,sys
raw=open('C:/dev/usir-merge/data/aozora_cache/heldout_000148_000751.txt','rb').read().decode('utf-8')
d=json.load(open('results/000148_000751.diagnose.json',encoding='utf-8'))
a=json.load(open('merged/000148_000751.annotations.json',encoding='utf-8'))
for i in map(int,sys.argv[1:]):
    s=d['scenes'][i]['span']
    print('=== s%d'%i, s)
    print(raw[s['start']:s['end']].replace('\r\n','\n'))
    m=a[i]['scene_meta']
    print('awase',m['awase'],'| event',[e['code'] for e in m['nagare']['event_top3']],'mood',[e['code'] for e in m['nagare']['mood_top3']])
