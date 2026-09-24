#!/bin/sh
# Does results 13 hold on a SECOND fixture? Same four arms, task code1.
cd /c/AI
export QBENCH_RESULTS=quality-2026-09-08-d
B="qwen38-tuning/tools/quality-bench.py"
r=6
while [ $r -le 10 ]; do
  case $(( r % 4 )) in
    0) ORDER="noskill placebo codegate family" ;;
    1) ORDER="placebo codegate family noskill" ;;
    2) ORDER="codegate family noskill placebo" ;;
    3) ORDER="family noskill placebo codegate" ;;
  esac
  Q=""
  for a in $ORDER; do Q="$Q D:$a@code1"; done
  echo "== CODE1 ROUND $r order: $ORDER"
  python $B --rep $r --queue $Q
  r=$(( r + 1 ))
done
echo "CODE1 ROUNDS DONE"
