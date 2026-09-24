#!/bin/sh
# Rounds 3-5 of the codegate vs codegate2 pairing (2026-09-07).
# Order alternates every round so a drift in the server cannot favour one arm.
cd /c/AI
export QBENCH_RESULTS=quality-2026-09-07-loop
B="qwen38-tuning/tools/quality-bench.py"
python $B --rep 3 --queue A:codegate@code2  A:codegate2@code2 --deadline 15:30
python $B --rep 4 --queue A:codegate2@code2 A:codegate@code2  --deadline 15:30
python $B --rep 5 --queue A:codegate@code2  A:codegate2@code2 --deadline 15:30
echo "ROUNDS 3-5 DONE"
