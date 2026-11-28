#!/bin/bash
# usage: next_j3.sh <run> -> PID|PACK|OUT (text = pack4 J3 round2 text)
cd /c/dev/usir-phase15/scripts/sensor_runner
export PYTHONIOENCODING=utf-8 NISHIKI_BIN=C:/dev/nishiki-mainint/target/release/nishiki.exe STEP2_WORKS_DIR=C:/dev/usir-phase15/scripts/viewer/h2/works_k2
W=000129_000689_h2b
python step2.py prep --run $1 --work $W --text C:/dev/usir-phase15/scripts/viewer/h2/text_k2.txt --local C:/dev/usir-phase15/scripts/viewer/h2/local_k2/$W.utterances.json | python -c "
import json,sys
d=json.load(sys.stdin)
for p in d['packs']: print(p['pid']+'|'+p['pack']+'|'+p['output'])
pw=d['per_work']['$W']; print('scenes',pw['scene_count'],'cached',pw['cached'],'pending',pw['pending'],file=sys.stderr)"
