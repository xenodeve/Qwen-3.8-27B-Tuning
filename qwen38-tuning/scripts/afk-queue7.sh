#!/bin/bash
# Final comprehensive arena, run on a QUIET machine.
#
# Every arena before this one shared the machine with a download queue, and it
# shows: the IQ2_XXS baseline, which reproduces to +/-0.1 tok/s when nothing
# else is running, moved 42.3 -> 37.5 -> 32.4 across contaminated rounds. The
# paired design keeps both arms inside the same round so the comparison stays
# usable, but the ranges widen past the point where a 20 % effect can be called.
#
# So: wait for the last byte to land, then re-measure everything together,
# three rounds, one table. Numbers from here supersede the piecemeal ones.
set -u
cd /c/AI/qwen38-tuning/bench
LOGS=/c/AI/qwen38-tuning/logs
log() { echo "[$(date '+%H:%M:%S')] $*"; }
step() {
  local name="$1"; shift
  log "START $name"
  if "$@" > "$LOGS/afk7-$name.log" 2>&1; then log "DONE  $name"
  else log "FAIL  $name (rc=$?) -- see afk7-$name.log"; fi
}

log "waiting for the download queues to finish before measuring"
# Gate on the V3 queue, not the retry queue: that one was killed once it
# turned out to be chasing UD-IQ2_M, a file Unsloth DELETED when it
# republished the repo. Waiting on a sentinel from a dead process would
# have parked this queue forever.
until grep -q "v3 download queue empty" "$LOGS/dl-v3-driver.log" 2>/dev/null; do sleep 60; done
# and for any straggler process holding the network
sleep 90
log "downloads quiet; starting the final arena"

# Everything, one table, three rounds. ~20 boots.
step final-arena-27b \
  python model_arena.py --rounds 3 --reps 3 \
    --arms iq2xxs-nomtp,adiq2xxs,iq1m-nomtp,iq2m-nomtp,iq3xxs-nomtp,q2kxl-nomtp \
    --out arena-final.jsonl

# Dynamic V3 against the pre-V3 build of the SAME NAME. This is the comparison
# the repo turnover created and the one the handoff asks for first: V3 IQ2_XXS
# is 1.62 GiB smaller than the artifact in production, and on this card four
# CPU-resident layers were measured to cost half the decode throughput.
step final-arena-v3 \
  python model_arena.py --rounds 3 --reps 3 \
    --arms iq2xxs-nomtp,v3-iq2xxs,v3-iq2s,v3-iq1m,v3-iq1s \
    --out arena-v3.jsonl

step final-arena-alt \
  python model_arena.py --rounds 3 --reps 3 \
    --arms iq2xxs-nomtp,bonsai-g64,bonsai-1bit,ornith9b-nomtp,ornith9b-q8,gptoss20b \
    --out arena-final.jsonl

step final-arena-moe \
  python model_arena.py --rounds 3 --reps 3 \
    --arms iq2xxs-nomtp,qwen36moe,qwen36moe-iq1,ornith35moe,ornith35-iq1s \
    --out arena-final.jsonl

# Bonsai + its own DSpark drafter. The first attempt passed the drafter with
# `-hfd repo:tag`, which resolved to an empty string and killed the server
# ("loading draft model ''"). It is a path now, like everything else.
step final-bonsai-dspark   python model_arena.py --rounds 3 --reps 3     --arms bonsai-g64,bonsai-dspark --out arena-final.jsonl

step restore-default \
  bash /c/AI/qwen38-tuning/scripts/swap-model.sh \
    /c/AI/qwen38-tuning/scripts/production-iq2xxs.ps1 IQ2_XXS

log "afk queue 7 empty"
