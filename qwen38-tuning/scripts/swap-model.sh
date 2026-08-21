#!/bin/bash
# Swap the server on :8080 and PROVE the new model is answering before returning.
#
# /health is not sufficient. On 2026-08-19 a swap passed the health check while
# llama-server was still loading; every inference request came back HTTP 503,
# and the corpus summary reported 24.0 merged tasks/hour from 30 tasks that
# never ran -- because escalation is a constant and 30 failures still "cost"
# 90 s each. The probe must send a real request and read the model name back.
#
# usage: swap-model.sh <script.ps1> <expected-substring-in-model-path>
set -e
SCRIPT="$1"; EXPECT="$2"

# Single-owner lock on port 8080. On 2026-08-20 an armed queue woke up at
# 02:00:17, ran model_arena as its first step, and killed the server a manual
# 30-task corpus was halfway through -- 26 of 30 tasks then returned HTTP 503 in
# 0.0 s and the summary still printed "3/29 accepted, 22.0 merged tasks/hour".
# Two orchestrators cannot share this port; whoever holds the lock owns it.
LOCK=/c/AI/qwen38-tuning/.port8080.lock
if [ -f "$LOCK" ]; then
  owner=$(cat "$LOCK" 2>/dev/null)
  owner_pid=${owner%% *}
  # The lock protects against ANOTHER orchestrator, not against its own holder.
  # A sweep that swaps the model once per configuration calls this script many
  # times from the same job, and every call after the first saw a lock held by a
  # live PID -- itself -- and refused. On 2026-08-20 that killed 13 of the 14
  # arms of the sampling screen in two seconds.
  if [ "$owner_pid" = "$PPID" ]; then
    echo "re-swapping under our own lock ($owner)"
  elif kill -0 "$owner_pid" 2>/dev/null; then
    echo "port 8080 is held by: $owner -- refusing to swap under a running job"
    exit 3
  else
    echo "clearing stale lock from $owner"
  fi
fi
# The lock is held by the CALLING JOB (our parent shell), not by this script:
# a swap takes seconds while the corpus that follows it runs for an hour, and a
# lock released at swap time protects nothing. When the job's shell exits its
# PID dies and the next swap clears the lock as stale, so nothing has to
# remember to release it.
echo "$PPID $(date '+%H:%M:%S') $(basename "$SCRIPT")" > "$LOCK"
[ -n "$SCRIPT" ] && [ -n "$EXPECT" ] || { echo "usage: swap-model.sh <ps1> <expect>"; exit 2; }

# `|| true`: Get-NetTCPConnection exits 1 when nothing is listening, and
# -ErrorAction SilentlyContinue hides the message but not the exit code. Under
# `set -e` that aborted the swap in the one case needing no work at all -- the
# port already being free.
powershell -NoProfile -Command '$c=Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue; if($c){Stop-Process -Id $c.OwningProcess -Force}' || true
# wait for the port to actually be free, not just for the kill to return
for i in $(seq 1 60); do
  netstat -ano | grep -q ":8080 .*LISTENING" || break
  sleep 2
done
netstat -ano | grep -q ":8080 .*LISTENING" && { echo "port 8080 still held after 120s"; exit 1; }

nohup powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$SCRIPT" > "/c/AI/qwen38-tuning/logs/swap-$(basename $SCRIPT .ps1).log" 2>&1 &

for i in $(seq 1 180); do
  sleep 5
  P=$(curl -s -m 5 http://127.0.0.1:8080/props 2>/dev/null || true)
  echo "$P" | grep -q "$EXPECT" || continue
  # a real generation, not just metadata: 503-while-loading answers /props too
  R=$(curl -s -m 30 -X POST http://127.0.0.1:8080/completion \
       -H 'Content-Type: application/json' \
       -d '{"prompt":"def f():","n_predict":4,"temperature":0,"cache_prompt":false}' 2>/dev/null || true)
  echo "$R" | grep -q '"content"' && { echo "swap ok: $EXPECT is serving at $(date '+%H:%M:%S')"; exit 0; }
done
echo "model $EXPECT never became ready"; exit 1
