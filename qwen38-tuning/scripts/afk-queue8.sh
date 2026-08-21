#!/bin/bash
# Qwen3.8-27B ONLY. Supersedes queues 6 and 7.
#
# Everything else measured today -- Ornith 9B/35B, Bonsai ternary, the 35B-A3B
# MoE, gpt-oss -- stays on record as pre-V3 data and stops consuming machine
# time. The focus is the ladder Unsloth republished as Dynamic 3.0 on
# 2026-08-19T16:39:23Z, mid-session.
#
# One documented change matters before any number is read. Unsloth's own page:
#
#   "The MTP module was removed from quants Q2_K_XL and smaller to conserve
#    ~500MB disk space, available separately as Q4_0."
#
# So the V3 low-bit arms have NO built-in MTP head. `--spec-type draft-mtp`
# cannot work on them without the standalone drafter, and the pre-V3 arm this
# project has been running all day DOES have one. That is a structural
# difference between the two generations, not only a quantization difference,
# and the size drop (9.01 -> 7.27 GB) is 1.74 GB against the ~0.5 GB the MTP
# head accounts for -- so most of it is real requantization.
set -u
cd /c/AI/qwen38-tuning/bench
LOGS=/c/AI/qwen38-tuning/logs
log() { echo "[$(date '+%H:%M:%S')] $*"; }
step() {
  local name="$1"; shift
  log "START $name"
  if "$@" > "$LOGS/afk8-$name.log" 2>&1; then log "DONE  $name"
  else log "FAIL  $name (rc=$?) -- see afk8-$name.log"; fi
}

log "waiting for the Dynamic V3 downloads"
until grep -q "v3 download queue empty" "$LOGS/dl-v3-driver.log" 2>/dev/null; do sleep 60; done
sleep 60
log "downloads quiet; measuring the V3 ladder"

# 1. The whole V3 ladder against the pre-V3 artifact currently in production.
#    Same repo, same filenames, different contents -- which is why every arm is
#    pinned to an exact byte count rather than resolved by name.
step v3-ladder-16k \
  python model_arena.py --rounds 3 --reps 3 \
    --arms iq2xxs-nomtp,v3-iq2xxs,v3-iq2s,v3-iq1m,v3-iq1s \
    --out arena-v3.jsonl

# 2. V3 Q2_K_XL: Unsloth's own efficiency pick, and the largest V3 arm that
#    might still be resident. Pre-V3 Q2_K_XL was 61+4 and lost half its decode
#    to those four layers; V3 is 847 MiB smaller.
step v3-q2kxl-16k \
  python model_arena.py --rounds 3 --reps 3 \
    --arms iq2xxs-nomtp,v3-q2kxl,q2kxl-nomtp \
    --out arena-v3.jsonl

# 3. Depth on whichever V3 arm has the most headroom. IQ1_S at 5.77 GiB leaves
#    roughly 4.6 GB against the 1.18 GB production has now, and depth was
#    measured to spend exactly the VRAM that quantization frees.
step v3-iq1s-depth \
  python kv_sweep.py --ctx 131072 --rounds 2 --arms q4_0,q8_0 --quant v3-iq1s \
    --out kv-sweep-v3.jsonl

step v3-iq2xxs-depth \
  python kv_sweep.py --ctx 131072 --rounds 2 --arms q4_0 --quant v3-iq2xxs \
    --out kv-sweep-v3.jsonl

log "afk queue 8 empty"
