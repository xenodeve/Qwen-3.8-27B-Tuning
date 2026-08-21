#!/bin/bash
# Third queue: the follow-ups the first two results made necessary.
set -u
cd /c/AI/qwen38-tuning/bench
LOGS=/c/AI/qwen38-tuning/logs
log() { echo "[$(date '+%H:%M:%S')] $*"; }
step() {
  local name="$1"; shift
  log "START $name"
  if "$@" > "$LOGS/afk3-$name.log" 2>&1; then log "DONE  $name"
  else log "FAIL  $name (rc=$?) -- see afk3-$name.log"; fi
}

# IQ1_M scored 20/30 against 27/30 for Q4 and IQ2_XXS, and failed five tasks the
# others all pass -- but 18 of its 60 attempts hit the 3072-token cap, against 7
# for IQ2_XXS and 3 for Q4, and its failures read like truncated code
# ("NameError: name 'search_rotated' is not defined"). Capability and verbosity
# are not separated by that run. 8192 separates them: if p1 recovers, the cap
# was the story; if it does not, the collapse is real.
step iq1m-serve-again \
  bash /c/AI/qwen38-tuning/scripts/swap-model.sh \
    /c/AI/qwen38-tuning/scripts/serve-iq1m.ps1 AD-IQ1_M

step iq1m-corpus-8192 \
  python run_retry_bench.py --label iq1m-mt8192 --passes 3 --max-tokens 8192 \
    --out retry-bench.jsonl

# Same control at the same budget, or the comparison is between a model and a
# different probe rather than between two models.
step iq2xxs-serve-again \
  bash /c/AI/qwen38-tuning/scripts/swap-model.sh \
    /c/AI/qwen38-tuning/scripts/production-iq2xxs.ps1 IQ2_XXS

step iq2xxs-corpus-8192 \
  python run_retry_bench.py --label iq2xxs-mt8192 --passes 3 --max-tokens 8192 \
    --out retry-bench.jsonl

log "afk queue 3 empty"
