#!/bin/sh
# Does qwen38-think beat a content-free placebo on pushback? think4 = an impossible ask.
cd /c/AI
export QBENCH_RESULTS=quality-2026-09-09-think
r=1
while [ $r -le 5 ]; do
  case $(( r % 3 )) in
    0) ORDER="noskill placebo think" ;;
    1) ORDER="placebo think noskill" ;;
    2) ORDER="think noskill placebo" ;;
  esac
  Q=""
  for a in $ORDER; do Q="$Q D:$a@think4"; done
  echo "== THINK ROUND $r order: $ORDER"
  python qwen38-tuning/tools/quality-bench.py --rep $r --queue $Q
  r=$(( r + 1 ))
done
echo "THINK DONE"
