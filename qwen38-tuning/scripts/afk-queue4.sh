#!/bin/bash
# Fourth queue: extend the fair-budget re-run to the two arms that need it most.
#
# The 30-task corpus was run at max_tokens 3072 for every arm. Truncation counts
# across those 60 attempts per arm:
#
#     Q2_K_XL  2      accepted 26/30
#     Q4       3      accepted 27/30
#     IQ2_XXS  7      accepted 27/30
#     Ornith9B 10     accepted 20/30
#     IQ1_M    18     accepted 20/30
#     Bonsai   35     accepted 15/30      <- more than half of all attempts
#
# The ordering of the accepted column tracks the truncation column almost
# exactly. That is not a quality ranking, it is a budget ranking, and reporting
# it as the former would repeat today's most persistent mistake three more
# times. Queue 3 re-runs IQ1_M and IQ2_XXS at 8192; this does Ornith and Bonsai
# so all four sit on the same probe.
set -u
cd /c/AI/qwen38-tuning/bench
LOGS=/c/AI/qwen38-tuning/logs
log() { echo "[$(date '+%H:%M:%S')] $*"; }
step() {
  local name="$1"; shift
  log "START $name"
  if "$@" > "$LOGS/afk4-$name.log" 2>&1; then log "DONE  $name"
  else log "FAIL  $name (rc=$?) -- see afk4-$name.log"; fi
}

step ornith9b-serve-8192 \
  bash /c/AI/qwen38-tuning/scripts/swap-model.sh \
    /c/AI/qwen38-tuning/scripts/serve-ornith9b.ps1 ornith-1.0-9b

step ornith9b-corpus-8192 \
  python run_retry_bench.py --label ornith9b-mt8192 --passes 3 --max-tokens 8192

step bonsai-serve-8192 \
  bash /c/AI/qwen38-tuning/scripts/swap-model.sh \
    /c/AI/qwen38-tuning/scripts/serve-bonsai-g64.ps1 Ternary-Bonsai

step bonsai-corpus-8192 \
  python run_retry_bench.py --label bonsai-g64-mt8192 --passes 3 --max-tokens 8192

# Leave the machine on the profile that is currently recommended for everyday
# work, so an unattended night does not end with a random arm serving :8080.
step restore-default \
  bash /c/AI/qwen38-tuning/scripts/swap-model.sh \
    /c/AI/qwen38-tuning/scripts/production-iq2xxs.ps1 IQ2_XXS

log "afk queue 4 empty"
