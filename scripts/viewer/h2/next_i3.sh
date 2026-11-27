#!/bin/bash
# usage: next_i3.sh <run> <work_id> -> PID|PACK|OUT (text = H2 insertion text)
cd /c/dev/usir-phase15/scripts/sensor_runner
export PYTHONIOENCODING=utf-8 NISHIKI_BIN=C:/dev/nishiki-mainint/target/release/nishiki.exe STEP2_WORKS_DIR=C:/dev/usir-phase15/scripts/viewer/h2/works_i3
python step2.py prep --run $1 --work $2 --text C:/dev/usir-phase15/scripts/viewer/h2/text_h2.txt --local C:/dev/usir-phase15/scripts/viewer/h2/local/$2.utterances.json | python -c "
import json,sys
d=json.load(sys.stdin)
for p in d['packs']: print(p['pid']+'|'+p['pack']+'|'+p['output'])
pw=d['per_work']['$2']; print('scenes',pw['scene_count'],'cached',pw['cached'],'pending',pw['pending'],file=sys.stderr)"
