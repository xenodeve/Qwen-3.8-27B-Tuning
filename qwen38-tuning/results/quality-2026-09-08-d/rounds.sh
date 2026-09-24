#!/bin/sh
# 2026-09-08, cell D (9arm FP8 gateway, no GPU). Four arms on code2.
#
#   noskill   no skills at all -- the control
#   placebo   a skill of the same shape that says nothing about tests/order/checks
#   codegate  qwen38-code-gate
#   family    using-qwen38 + qwen38-think + qwen38-code-gate + karpathy-guidelines
#
# The arm order rotates every round so a position effect cannot favour one arm.
# D is skills_explicit: the brief asks for the Skill tool by name, because the slash
# token alone produced 0 Skill calls on this endpoint (results 12, round 4).
cd /c/AI
export QBENCH_RESULTS=quality-2026-09-08-d
B="qwen38-tuning/tools/quality-bench.py"
r=1
while [ $r -le 15 ]; do
  case $(( r % 4 )) in
    0) ORDER="noskill placebo codegate family" ;;
    1) ORDER="placebo codegate family noskill" ;;
    2) ORDER="codegate family noskill placebo" ;;
    3) ORDER="family noskill placebo codegate" ;;
  esac
  Q=""
  for a in $ORDER; do Q="$Q D:$a@code2"; done
  echo "== ROUND $r order: $ORDER"
  python $B --rep $r --queue $Q
  r=$(( r + 1 ))
done
echo "D ROUNDS DONE"
