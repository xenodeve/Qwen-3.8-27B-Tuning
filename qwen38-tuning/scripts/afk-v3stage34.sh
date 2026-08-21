#!/bin/bash
# V3 stages 2-4 for the arms that survived stage 1, plus the depth runs that
# aborted when the ambiguity guard fired at import time on an unpinned pre-V3
# entry. The guard was right -- two snapshots hold Qwen3.8-27B-UD-Q2_K_XL.gguf
# and it refused to choose -- it just took three unrelated steps down with it.
#
# Stage 1 gate (full residency AND >=512 MiB free) eliminated v3-q2kxl at
# 417 MiB. Stage 2 already rejected v3-iq1s: fastest arm measured (+27.94 %
# paired) and it never leaves its own reasoning block.
set -u
cd /c/AI/qwen38-tuning/bench
LOGS=/c/AI/qwen38-tuning/logs
log() { echo "[$(date '+%H:%M:%S')] $*"; }
step() {
  local name="$1"; shift
  log "START $name"
  if "$@" > "$LOGS/v3s-$name.log" 2>&1; then log "DONE  $name"
  else log "FAIL  $name (rc=$?) -- see v3s-$name.log"; fi
}

# Stage 2 for the one arm that has not been screened.
step screen-iq2s bash -c '
  bash /c/AI/qwen38-tuning/scripts/swap-model.sh \
    /c/AI/qwen38-tuning/scripts/serve-v3-iq2s.ps1 UD-IQ2_S &&
  python /c/AI/qwen38-tuning/bench/answer_screen.py --arm v3-iq2s'

# Stage 3 on the arm that passed the screen cleanly. This is the re-run: the
# first attempt lost its server at 02:00:17 to a colliding queue and 26 of 30
# tasks returned HTTP 503. swap-model.sh now takes a lock so that cannot repeat.
step corpus-iq1m bash -c '
  bash /c/AI/qwen38-tuning/scripts/swap-model.sh \
    /c/AI/qwen38-tuning/scripts/serve-v3-iq1m.ps1 UD-IQ1_M &&
  python /c/AI/qwen38-tuning/bench/run_retry_bench.py \
    --label v3-iq1m --passes 3 --max-tokens 8192'

# Stage 3 on the mixed arm. Its screen produced two clean answers in 5 s and one
# runaway of 23,604 reasoning characters, so the corpus is where that ratio gets
# a denominator.
step corpus-iq2xxs bash -c '
  bash /c/AI/qwen38-tuning/scripts/swap-model.sh \
    /c/AI/qwen38-tuning/scripts/serve-v3-iq2xxs.ps1 UD-IQ2_XXS &&
  python /c/AI/qwen38-tuning/bench/run_retry_bench.py \
    --label v3-iq2xxs --passes 3 --max-tokens 8192'

# Stage 4, corrected selection rule: not "most idle VRAM" but the arms that
# hold residency once the 128K cache is allocated. IQ1_S is included despite
# failing stage 2 because its depth behaviour is the only thing it can still
# tell us, and it is already confirmed 65/0 at 128K with 1,436 MiB spare.
step depth-128k-iq1s \
  python kv_sweep.py --ctx 131072 --rounds 2 --arms q4_0 --quant v3-iq1s \
    --out kv-sweep-v3.jsonl
step depth-128k-iq1m \
  python kv_sweep.py --ctx 131072 --rounds 2 --arms q4_0 --quant v3-iq1m \
    --out kv-sweep-v3.jsonl
step depth-128k-iq2xxs \
  python kv_sweep.py --ctx 131072 --rounds 2 --arms q4_0 --quant v3-iq2xxs \
    --out kv-sweep-v3.jsonl

log "afk v3 stages 2-4 complete"
