#!/bin/sh
cd /c/AI
export QBENCH_RESULTS=quality-2026-09-07-loop
B="qwen38-tuning/tools/quality-bench.py"
for r in 11 12 13 14 15 16; do
  if [ $((r % 2)) -eq 1 ]; then
    python $B --rep $r --queue A:codegate3@code2 A:codegate@code2 --deadline 15:40
  else
    python $B --rep $r --queue A:codegate@code2 A:codegate3@code2 --deadline 15:40
  fi
done
echo "V3 ROUNDS DONE"
