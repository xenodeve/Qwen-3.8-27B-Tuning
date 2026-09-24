#!/bin/sh
# Did the two rules added to using-design close the two measured gaps?
#   baseline (2026-09-08, same arm, same brief): page in 3 of 5, Thai in 0 of 3
cd /c/AI
export QBENCH_RESULTS=quality-2026-09-09-fix
r=1
while [ $r -le 4 ]; do
  python qwen38-tuning/tools/quality-bench.py --rep $r --queue D:designonly
  r=$(( r + 1 ))
done
echo "VERIFY DONE"
