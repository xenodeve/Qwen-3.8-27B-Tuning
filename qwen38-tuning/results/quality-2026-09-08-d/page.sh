#!/bin/sh
# Frontend track: the design arms on the gateway, now that page briefs also name the Skill tool.
cd /c/AI
export QBENCH_RESULTS=quality-2026-09-08-d
r=1
while [ $r -le 3 ]; do
  case $(( r % 3 )) in
    0) ORDER="both designonly gateonly" ;;
    1) ORDER="designonly gateonly both" ;;
    2) ORDER="gateonly both designonly" ;;
  esac
  Q=""
  for a in $ORDER; do Q="$Q D:$a"; done
  echo "== PAGE ROUND $r order: $ORDER"
  python qwen38-tuning/tools/quality-bench.py --rep $r --queue $Q
  r=$(( r + 1 ))
done
echo "PAGE DONE"
