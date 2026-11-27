#!/bin/bash
# usage: next.sh <run> <work_id> <text> -> prints PID|PACK|OUT of next pack
cd /c/dev/usir-phase15/scripts/sensor_runner
export PYTHONIOENCODING=utf-8 NISHIKI_BIN=C:/dev/nishiki-mainint/target/release/nishiki.exe STEP2_WORKS_DIR=C:/dev/usir-phase15/scripts/viewer/h2/works
RUN=$1; W=$2; T=$3
python step2.py prep --run $RUN --work $W --text C:/dev/usir-phase15/scripts/viewer/h2/$T --local C:/dev/usir-phase15/scripts/viewer/h2/local/$W.utterances.json | python -c "
import json,sys
d=json.load(sys.stdin)
for p in d['packs']: print(p['pid']+'|'+p['pack']+'|'+p['output'])
pw=d['per_work']['$W']; print('scenes',pw['scene_count'],'cached',pw['cached'],'pending',pw['pending'],file=sys.stderr)"
