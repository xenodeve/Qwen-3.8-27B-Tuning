#!/bin/sh
cd /c/AI
export QBENCH_RESULTS=quality-2026-09-08-d
r=1
while [ $r -le 4 ]; do
  python qwen38-tuning/tools/quality-bench.py --rep $r --queue D:codegatemin@code2
  r=$(( r + 1 ))
done
echo "MIN CODE2 DONE"
