#!/bin/bash
# Executes docs/plans/01-V3-Q1-Q2-TEST-PLAN.md.
#
# Staged so each gate eliminates arms before the expensive stage. A 30-task
# corpus at max_tokens 8192 costs 45-90 minutes per arm; seven arms would be a
# night on its own.
set -u
cd /c/AI/qwen38-tuning/bench
LOGS=/c/AI/qwen38-tuning/logs
log() { echo "[$(date '+%H:%M:%S')] $*"; }
step() {
  local name="$1"; shift
  log "START $name"
  if "$@" > "$LOGS/v3-$name.log" 2>&1; then log "DONE  $name"
  else log "FAIL  $name (rc=$?) -- see v3-$name.log"; fi
}
serve() {   # serve <label> <arm-path-var>
  bash /c/AI/qwen38-tuning/scripts/swap-model.sh "$1" "$2"
}

log "waiting for the V3 downloads"
until grep -q "v3b download queue empty" "$LOGS/dl-v3-driver.log" 2>/dev/null; do sleep 60; done
sleep 60
log "link quiet; starting stage 1"

# --- Stage 1: residency and speed, paired, 3 rounds --------------------------
# The question is not which is fastest. Every V3 Q1/Q2 arm should be resident at
# 16K. It is how much VRAM each LEAVES, because depth spends exactly that.
step s1-ladder \
  python model_arena.py --rounds 3 --reps 3 \
    --arms iq2xxs-nomtp,v3-iq2xxs,v3-iq2s,v3-iq1m,v3-iq1s,v3-q2kxl \
    --out arena-v3.jsonl

# --- Stage 4 (moved early: cheap, and it uses Stage 1's headroom finding) -----
# Depth on the two smallest arms, which by construction have the most headroom.
step s4-depth-iq1s \
  python kv_sweep.py --ctx 131072 --rounds 2 --arms q4_0 --quant v3-iq1s \
    --out kv-sweep-v3.jsonl
step s4-depth-iq2xxs \
  python kv_sweep.py --ctx 131072 --rounds 2 --arms q4_0 --quant v3-iq2xxs \
    --out kv-sweep-v3.jsonl
step s4-depth-64k-iq1s \
  python kv_sweep.py --ctx 65536 --rounds 2 --arms q4_0,q8_0 --quant v3-iq1s \
    --out kv-sweep-v3.jsonl

log "afk v3 plan: stages 1 and 4 complete -- protocol and corpus follow"
