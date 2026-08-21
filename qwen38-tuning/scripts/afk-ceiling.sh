#!/bin/bash
# How deep can this machine actually go? The developer's goal is a context
# BEYOND 128K, and every artifact measured so far collapses at 256K for the
# same reason: KV is allocated from the pool the weights live in, so context
# spends exactly the VRAM that quantization freed.
#
#     IQ2_XXS  + q4_0 @256K   43 + 22   2.23 tok/s
#     AD-IQ1_M + q4_0 @256K   46 + 19   2.29 tok/s
#
# Two properties decide the ceiling, and they are not the same property:
#
#   WEIGHT SIZE      Bonsai-27B-Q1_0 is 3.54 GiB, the smallest artifact on disk
#                    -- 2.2 GiB under V3 IQ1_S and less than half the current
#                    production model.
#   CACHE PER TOKEN  a 9B holds a much smaller cache than a 27B at the same
#                    depth: Ornith-9B measured 1,152 MiB at 128K against the
#                    27B's 2,016, 43 % smaller. Fewer layers, fewer heads.
#
# So the two best chances at 256K come from opposite directions, and the ladder
# will say which one actually gets there.
#
# Reads only the layer split from the load report -- no prefill, no generation,
# about a minute per boot instead of the ten a 256K cold prefill costs. The
# split is deterministic, so this is safe to run beside other work.
set -u
cd /c/AI/qwen38-tuning/bench
LOGS=/c/AI/qwen38-tuning/logs
log() { echo "[$(date '+%H:%M:%S')] $*"; }
step() {
  local name="$1"; shift
  log "START $name"
  if "$@" > "$LOGS/ceil-$name.log" 2>&1; then log "DONE  $name"
  else log "FAIL  $name (rc=$?) -- see ceil-$name.log"; fi
}

log "waiting for the V3 stage 2-4 queue to finish"
until grep -q "afk v3 stages 2-4 complete" "$LOGS/afk-driver.log" 2>/dev/null; do sleep 60; done
sleep 30

LADDER=131072,163840,196608,229376,262144

# Smallest weights first: if anything reaches 256K it is this.
step bonsai-1bit  python ctx_ceiling.py --quant bonsai-1bit --kv q4_0 --ladder $LADDER
# Smallest cache per token.
step ornith9b     python ctx_ceiling.py --quant ornith9b    --kv q4_0 --ladder $LADDER
# The Qwen3.8 arms, so the answer stays comparable to the rest of the project.
step v3-iq1s      python ctx_ceiling.py --quant v3-iq1s     --kv q4_0 --ladder $LADDER
step v3-iq1m      python ctx_ceiling.py --quant v3-iq1m     --kv q4_0 --ladder $LADDER
# The control, to price what the low-bit lane actually bought at depth.
step iq2xxs       python ctx_ceiling.py --quant iq2xxs      --kv q4_0 --ladder $LADDER

log "afk ceiling sweep complete"
