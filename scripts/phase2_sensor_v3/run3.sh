#!/bin/sh
# three runs, identical settings (seed 42, temperature 0), sequential
for i in 1 2 3; do python scripts/phase2_sensor_v3/run_v3.py scripts/phase2_sensor_v3/run$i > scripts/phase2_sensor_v3/run$i.log 2>&1; done
