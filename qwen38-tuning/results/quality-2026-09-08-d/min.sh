#!/bin/sh
cd /c/AI
export QBENCH_RESULTS=quality-2026-09-08-d
B="qwen38-tuning/tools/quality-bench.py"
r=7
while [ $r -le 10 ]; do
  python $B --rep $r --queue D:codegatemin@code1
  r=$(( r + 1 ))
done
echo "MIN ROUNDS DONE"
