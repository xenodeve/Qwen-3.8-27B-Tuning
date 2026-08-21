#!/bin/bash
# Everything the research names that is still untested, in the order the
# research itself ranks it. Serialised: two HF streams measured slower than one.
#
# The MoE lane is the biggest gap. The research calls 35B-A3B the
# "highest-ceiling architecture" -- 35B total, ~3B active per token -- and
# assumed it would need CPU expert offload at ~20 GiB. Unsloth ships IQ1_S/IQ2
# builds at 9.4-10.7 GiB, which on this card may be RESIDENT. That combination,
# 35B of capacity with 3B of per-token work and no PCIe traffic, is the one
# configuration nothing measured so far can imitate.
set -u
log() { echo "[$(date '+%H:%M:%S')] $*"; }

queue=(
  "unsloth/Qwen3.6-35B-A3B-GGUF|Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf"
  "unsloth/Ornith-1.0-35B-GGUF|Ornith-1.0-35B-UD-IQ2_XXS.gguf"
  "prism-ml/Ternary-Bonsai-27B-gguf|Ternary-Bonsai-27B-dspark-Q4_1.gguf"
  "prism-ml/Bonsai-27B-gguf|Bonsai-27B-Q1_0.gguf"
  "ornith-ai/Ornith-1.0-9B-GGUF|ornith-1.0-9b-Q8_0.gguf"
  "unsloth/Qwen3.8-27B-GGUF|Qwen3.8-27B-UD-IQ2_M.gguf"
  "unsloth/gpt-oss-20b-GGUF|gpt-oss-20b-Q4_K_M.gguf"
  "unsloth/Ornith-1.0-35B-GGUF|Ornith-1.0-35B-UD-IQ1_S.gguf"
  "unsloth/Qwen3.6-35B-A3B-GGUF|Qwen3.6-35B-A3B-UD-IQ1_M.gguf"
  "unsloth/Qwen3.8-27B-GGUF|Qwen3.8-27B-UD-IQ3_XXS.gguf"
)

for item in "${queue[@]}"; do
  repo="${item%%|*}"; file="${item##*|}"
  log "START $file"
  if hf download "$repo" "$file" >> /c/AI/qwen38-tuning/logs/dl-queue2.log 2>&1; then
    log "DONE  $file"
  else
    log "FAIL  $file (continuing)"
  fi
  df -h /c | tail -1 | awk '{print "        disk free: " $4}'
done
log "download queue 2 empty"
