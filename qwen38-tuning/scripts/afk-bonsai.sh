#!/bin/bash
# Bonsai-27B-Q1_0 is the only arm that reached 256K fully resident (report 16),
# and it is the arm with the least quality evidence in this project: none at
# all.  Its +80.12 % is a speed figure.  V3 IQ1_S was the fastest artifact ever
# measured here and emitted no usable answer in twelve of twelve attempts, so a
# residency number from an unscreened arm decides nothing.
#
# Order matters: the four-minute screen runs first because if it rejects, every
# later step on this page is wasted GPU time.
set -u
cd /c/AI/qwen38-tuning/bench
LOGS=/c/AI/qwen38-tuning/logs
SCRIPTS=/c/AI/qwen38-tuning/scripts
log() { echo "[$(date '+%H:%M:%S')] $*"; }
step() {
  local name="$1"; shift
  log "START $name"
  if "$@" > "$LOGS/bon-$name.log" 2>&1; then log "DONE  $name"
  else log "FAIL  $name (rc=$?) -- see bon-$name.log"; fi
} 2>&1

{
# 1. Does it answer at all.
step screen-bonsai-1bit bash -c "
  bash $SCRIPTS/swap-model.sh $SCRIPTS/serve-bonsai-1bit.ps1 Bonsai-27B-Q1_0 &&
  python /c/AI/qwen38-tuning/bench/answer_screen.py --arm bonsai-1bit"

# 2. 256K cleared the 512 MiB reserve by ONE megabyte on a single boot, and
#    free VRAM at boot has moved 9,326-10,732 MiB across this project's boots.
#    Two more ladders say whether 513 was the arm or the boot.
step ceil-bonsai-r2 python ctx_ceiling.py --quant bonsai-1bit --kv q4_0 \
                      --ladder 229376,262144
step ceil-bonsai-r3 python ctx_ceiling.py --quant bonsai-1bit --kv q4_0 \
                      --ladder 229376,262144

# 3. The corpus, but only if the screen said PASS.  A MIXED or REJECT verdict
#    means inspect first -- the screen has already been shown to be a floor and
#    not a gate (it passed v3-iq1m 3/3, which then scored 33.3 % on the corpus),
#    so PASS is necessary here, not sufficient.
if grep -q "PASS - proceed to the corpus" "$LOGS/bon-screen-bonsai-1bit.log"; then
  step corpus-bonsai-1bit bash -c "
    bash $SCRIPTS/swap-model.sh $SCRIPTS/serve-bonsai-1bit.ps1 Bonsai-27B-Q1_0 &&
    python /c/AI/qwen38-tuning/bench/run_retry_bench.py --label bonsai-1bit --passes 3 --max-tokens 8192"
else
  log "SKIP  corpus-bonsai-1bit -- screen did not return PASS"
fi

log "afk bonsai queue complete"
} >> "$LOGS/afk-driver.log" 2>&1
