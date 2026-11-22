#!/bin/bash
# usage: step.sh <hash12> <scene_index>  : ingest, log time, prepare next pack
cd /c/dev/usir-merge/python
export PYTHONIOENCODING=utf-8 NISHIKI_BIN=C:/dev/nishiki-hybrid/target/debug/nishiki.exe
R="uv run python ../scripts/sensor_runner/cli.py"
$R ingest ../scripts/sensor_runner/out/000148_000751/$1.strong.json | tee /tmp/ing_$2.txt
echo "s$2 ingested $(date +%T) $(head -1 /tmp/ing_$2.txt | cut -c1-6)" >> ../scripts/sensor_runner/results/live2/timing.log
$R prepare --work 000148_000751 --limit 1 | sed -n 1,2p\;3p
echo "s$2 next_prepared $(date +%T)" >> ../scripts/sensor_runner/results/live2/timing.log
