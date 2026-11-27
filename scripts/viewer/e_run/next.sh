#!/bin/bash
# prints PID|PACK|OUT of the next pack of run E1 (empty if none)
cd /c/dev/usir-phase15/scripts/sensor_runner
export PYTHONIOENCODING=utf-8 NISHIKI_BIN=C:/dev/nishiki-mainint/target/release/nishiki.exe STEP2_WORKS_DIR=C:/dev/usir-phase15/scripts/viewer/e_run/works
python step2.py prep --run E1 --work 000119_001738 --text C:/dev/usir-phase15/scripts/viewer/e_run/text_000119_001738.txt --local C:/dev/usir-phase15/scripts/viewer/e_run/local/000119_001738.utterances.json | python -c "
import json,sys
d=json.load(sys.stdin)
for p in d['packs']: print(p['pid']+'|'+p['pack']+'|'+p['output'])
pw=d['per_work']['000119_001738']; print('cached',pw['cached'],'pending',pw['pending'],'blocked',pw['l2']['blocked'],file=sys.stderr)"
