#!/bin/sh
# Second loop iteration: did rule 3 (light AND dark) close the dark-mode gap?
#   baseline: prefers-color-scheme present in 0 of 7 designonly pages over two days
cd /c/AI
export QBENCH_RESULTS=quality-2026-09-09-fix2
r=1
while [ $r -le 4 ]; do
  python qwen38-tuning/tools/quality-bench.py --rep $r --queue D:designonly
  r=$(( r + 1 ))
done
echo "VERIFY2 DONE"
