#!/bin/bash
# Retry the four that failed with
#   RuntimeError: Cannot send a request, as the client has been closed
# which is an hf client-state error, not a missing file -- all four resolve
# fine against the repo listings. One process per file so a closed client
# cannot take the rest of the queue with it.
set -u
log() { echo "[$(date '+%H:%M:%S')] $*"; }
q=(
  "unsloth/Qwen3.8-27B-GGUF|Qwen3.8-27B-UD-IQ2_M.gguf"
  "unsloth/Ornith-1.0-35B-GGUF|Ornith-1.0-35B-UD-IQ1_S.gguf"
  "unsloth/Qwen3.6-35B-A3B-GGUF|Qwen3.6-35B-A3B-UD-IQ1_M.gguf"
  "unsloth/Qwen3.8-27B-GGUF|Qwen3.8-27B-UD-IQ3_XXS.gguf"
)
for item in "${q[@]}"; do
  repo="${item%%|*}"; file="${item##*|}"
  for try in 1 2 3; do
    log "START $file (attempt $try)"
    if hf download "$repo" "$file" >> /c/AI/qwen38-tuning/logs/dl-retry.log 2>&1; then
      log "DONE  $file"; break
    fi
    log "FAIL  $file attempt $try"
    sleep 20
  done
  df -h /c | tail -1 | awk '{print "        disk free: " $4}'
done
log "retry queue empty"
