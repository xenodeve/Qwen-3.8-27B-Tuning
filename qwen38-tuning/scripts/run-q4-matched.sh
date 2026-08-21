#!/bin/bash
# Re-measure Q4 with the SAME probes and budgets the Q2 arm was measured with.
# The first Q4 pass ran before protocol_gate recorded finish_reason and before
# run_retry_bench recorded per-attempt detail, so its failures cannot be split
# into "wrong" versus "ran out of budget". Without that split the comparison
# silently penalises whichever model reasons longer.
set -e
cd /c/AI/qwen38-tuning/bench

# swap the server: Q2 -> Q4
powershell -NoProfile -Command '$c=Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue; if($c){Stop-Process -Id $c.OwningProcess -Force}'
sleep 8
nohup powershell.exe -NoProfile -ExecutionPolicy Bypass -File /c/AI/qwen38-tuning/scripts/production-q4-tuned.ps1 > /c/AI/qwen38-tuning/logs/q4-server-matched.log 2>&1 &
until curl -s -m 3 http://127.0.0.1:8080/health >/dev/null 2>&1; do sleep 10; done
# confirm the right model answered, not a stale server
curl -s -m 5 http://127.0.0.1:8080/props | grep -o 'Q4_K_XL' | head -1
sleep 3

python protocol_gate.py --label q4-mt4096 --trials 15 --temperature 0.7 --max-tokens 4096 --out protocol-budget.jsonl
python run_retry_bench.py --label q4-matched --passes 3
