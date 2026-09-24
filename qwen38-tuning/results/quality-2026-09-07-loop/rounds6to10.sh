#!/bin/sh
cd /c/AI
export QBENCH_RESULTS=quality-2026-09-07-loop
B="qwen38-tuning/tools/quality-bench.py"
python $B --rep 6  --queue A:codegate2@code2 A:codegate@code2  --deadline 15:25
python $B --rep 7  --queue A:codegate@code2  A:codegate2@code2 --deadline 15:25
python $B --rep 8  --queue A:codegate2@code2 A:codegate@code2  --deadline 15:25
python $B --rep 9  --queue A:codegate@code2  A:codegate2@code2 --deadline 15:25
python $B --rep 10 --queue A:codegate2@code2 A:codegate@code2  --deadline 15:25
echo "ROUNDS 6-10 DONE"
