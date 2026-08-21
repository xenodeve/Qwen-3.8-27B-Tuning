#!/bin/bash
# Sixth queue: the control lost to the -hf stall, plus anything queue 5 could
# not reach because its artifact was still downloading.
set -u
cd /c/AI/qwen38-tuning/bench
LOGS=/c/AI/qwen38-tuning/logs
log() { echo "[$(date '+%H:%M:%S')] $*"; }
step() {
  local name="$1"; shift
  log "START $name"
  if "$@" > "$LOGS/afk6-$name.log" 2>&1; then log "DONE  $name"
  else log "FAIL  $name (rc=$?) -- see afk6-$name.log"; fi
}

# The IQ2_XXS control at 8192 was lost when its server could not boot: the
# launch script still used `-hf`, which does an online etag check per launch,
# and the link was saturated by a 10 GiB download. Every launch script now
# references a path instead. Without this row, IQ1_M's recovery from 20/30 to
# 27/30 at 8192 has no same-budget control to sit beside.
step iq2xxs-serve-8192 \
  bash /c/AI/qwen38-tuning/scripts/swap-model.sh \
    /c/AI/qwen38-tuning/scripts/production-iq2xxs.ps1 IQ2_XXS

step iq2xxs-corpus-8192 \
  python run_retry_bench.py --label iq2xxs-mt8192 --passes 3 --max-tokens 8192

# Late arrivals: whatever finished downloading after queue 5 read the cache.
step late-arrivals-arena \
  python model_arena.py --rounds 2 --reps 3 \
    --arms iq2xxs-nomtp,iq2m-nomtp,iq3xxs-nomtp,ornith9b-q8,bonsai-1bit,gptoss20b \
    --out arena-remaining.jsonl

step late-moe-arena \
  python model_arena.py --rounds 2 --reps 3 \
    --arms iq2xxs-nomtp,qwen36moe,qwen36moe-iq1,ornith35moe,ornith35-iq1s \
    --out arena-moe.jsonl

step restore-default \
  bash /c/AI/qwen38-tuning/scripts/swap-model.sh \
    /c/AI/qwen38-tuning/scripts/production-iq2xxs.ps1 IQ2_XXS

log "afk queue 6 empty"
