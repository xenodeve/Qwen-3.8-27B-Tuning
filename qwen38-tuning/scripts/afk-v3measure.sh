#!/bin/bash
# V3 timing measurements, gated on a quiet link.
#
# Stage 0 is already done for IQ1_S and is not repeated: SHA-256 matched the
# repo OID exactly, residency measured 1.41 % shared against 3,290 MiB free at
# 16K, and the 128K split is 65/0 with 1,436 MiB spare. Those are structural
# and were safe to take with downloads running. Decode is not.
set -u
cd /c/AI/qwen38-tuning/bench
LOGS=/c/AI/qwen38-tuning/logs
log() { echo "[$(date '+%H:%M:%S')] $*"; }
step() {
  local name="$1"; shift
  log "START $name"
  if "$@" > "$LOGS/v3m-$name.log" 2>&1; then log "DONE  $name"
  else log "FAIL  $name (rc=$?) -- see v3m-$name.log"; fi
}

log "waiting for the V3 downloads to finish"
until grep -q "v3b download queue empty" "$LOGS/dl-v3-driver.log" 2>/dev/null; do sleep 60; done
sleep 90
log "link quiet"

# S1: the whole V3 ladder against the pre-V3 artifact in production. Arms that
# have not finished downloading report FAILED TO START and the sweep continues.
step s1-v3-ladder \
  python model_arena.py --rounds 3 --reps 3 \
    --arms iq2xxs-nomtp,v3-iq1s,v3-iq1m,v3-iq2xxs,v3-iq2s,v3-q2kxl \
    --out arena-v3.jsonl

# S4 with the CORRECTED selection rule: the largest arm still fully resident
# once the 128K q4_0 cache is allocated, not the arm with the most idle VRAM.
# IQ1_S is measured because it is confirmed 65/0 at 128K with 1,436 MiB spare;
# the larger V3 rungs are measured to find where residency actually breaks.
step s4-depth-128k \
  python kv_sweep.py --ctx 131072 --rounds 2 --arms q4_0 --quant v3-iq1s \
    --out kv-sweep-v3.jsonl
step s4-depth-128k-iq2xxs \
  python kv_sweep.py --ctx 131072 --rounds 2 --arms q4_0 --quant v3-iq2xxs \
    --out kv-sweep-v3.jsonl
step s4-depth-64k \
  python kv_sweep.py --ctx 65536 --rounds 2 --arms q4_0,q8_0 --quant v3-iq1s \
    --out kv-sweep-v3.jsonl

log "afk v3 measurement: S1 and S4 complete"
