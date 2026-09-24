#!/bin/sh
cd /c/AI
export QBENCH_RESULTS=quality-2026-09-08-d
r=3
while [ $r -le 5 ]; do
  python qwen38-tuning/tools/quality-bench.py --rep $r --queue D:designonly
  r=$(( r + 1 ))
done
echo "DESIGN3 DONE"
