#!/bin/bash
# Fifth queue: sweep every remaining candidate the research names.
#
# Arms resolve their artifact from the HF cache at import time, so a candidate
# still downloading reports FAILED TO START and the sweep continues. That makes
# this queue safe to run alongside the download queue: whatever has landed gets
# measured, whatever has not is simply absent from the table.
#
# Ordering follows the research's own ranking, with one change earned by
# measurement: the MoE lane moves UP. The research assumed 35B-A3B needed CPU
# expert offload at ~20 GiB, so it ranked the lane third behind two 27B
# quantizations. Unsloth ships it at 9.4-10.7 GiB, which on this card may be
# fully resident -- 35B of capacity, ~3B of work per token, no PCIe traffic.
set -u
cd /c/AI/qwen38-tuning/bench
LOGS=/c/AI/qwen38-tuning/logs
log() { echo "[$(date '+%H:%M:%S')] $*"; }
step() {
  local name="$1"; shift
  log "START $name"
  if "$@" > "$LOGS/afk5-$name.log" 2>&1; then log "DONE  $name"
  else log "FAIL  $name (rc=$?) -- see afk5-$name.log"; fi
}

# 1. Same-size quantizer battle: AtomicChat 8.36 GiB vs Unsloth 8.39 GiB. The
#    research wanted this and thought it needed a 9.9 GB file from a second
#    vendor; it needs 30 MiB of difference and the same probe.
step quantizer-battle \
  python model_arena.py --rounds 3 --reps 3 \
    --arms iq2xxs-nomtp,adiq2xxs --out arena-quantizer.jsonl

# 2. The Qwen quants never tried, filling the ladder between IQ2_XXS and Q3.
step qwen-ladder \
  python model_arena.py --rounds 2 --reps 3 \
    --arms iq2xxs-nomtp,iq2m-nomtp,iq3xxs-nomtp --out arena-ladder.jsonl

# 3. MoE at 16K. Resident or not is the whole question.
step moe-arena \
  python model_arena.py --rounds 2 --reps 3 \
    --arms iq2xxs-nomtp,qwen36moe,ornith35moe --out arena-moe.jsonl

# 4. MoE with CPU expert offload, the configuration the research actually
#    proposed. Worth measuring even if the resident form wins, because it is
#    the only one that scales past this card.
step moe-cpuoffload \
  python model_arena.py --rounds 2 --reps 3 \
    --arms qwen36moe,qwen36moe-cpu,ornith35-cpu --out arena-moe.jsonl

# 5. The remaining singles: 1-bit MoE, 1-bit ternary, Ornith Q8, gpt-oss.
step remaining-arena \
  python model_arena.py --rounds 2 --reps 3 \
    --arms iq2xxs-nomtp,qwen36moe-iq1,ornith35-iq1s,bonsai-1bit,ornith9b-q8,gptoss20b \
    --out arena-remaining.jsonl

# 6. Bonsai with its own drafter. draft-dspark is mainline in b10472 and the
#    drafter is trained against this target; Prism reports 1.34x on an H100.
#    Whether it survives a 12 GB card is a different question.
step bonsai-dspark \
  python model_arena.py --rounds 2 --reps 3 \
    --arms bonsai-g64,bonsai-dspark --out arena-bonsai-spec.jsonl

log "afk queue 5 empty"
