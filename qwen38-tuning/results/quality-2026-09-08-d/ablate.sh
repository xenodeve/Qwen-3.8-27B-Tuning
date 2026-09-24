#!/bin/sh
# Which check produces test-before-source? no6 = full minus check 6; t6 = table + check 6.
cd /c/AI
export QBENCH_RESULTS=quality-2026-09-08-d
r=1
while [ $r -le 10 ]; do
  if [ $(( r % 2 )) -eq 1 ]; then ORDER="codegateno6 codegatet6"; else ORDER="codegatet6 codegateno6"; fi
  Q=""
  for a in $ORDER; do Q="$Q D:$a@code1"; done
  echo "== ABLATE ROUND $r order: $ORDER"
  python qwen38-tuning/tools/quality-bench.py --rep $r --queue $Q
  r=$(( r + 1 ))
done
echo "ABLATE DONE"
