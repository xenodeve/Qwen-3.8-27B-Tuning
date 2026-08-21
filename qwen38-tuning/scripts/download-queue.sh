#!/bin/bash
# Serialised download queue.
#
# Parallel is worse, measured: two Hugging Face streams delivered 0.53 MB/s
# combined against 1.13 MB/s for one, and a third (hf CLI + Xet) left the total
# unchanged at 1.30 MB/s while a concurrent Cloudflare test pulled 2.39 MB/s.
# HF throttles the aggregate, so the only thing parallelism buys is having two
# unusable half-files instead of one usable whole one.
#
# Fetched by EXACT FILENAME, never by `-hf repo:tag`: that matches by substring,
# and it silently started downloading PQ2_0.gguf when asked for Q2_0.gguf --
# two files of identical byte count in the same repo.
set -u
log() { echo "[$(date '+%H:%M:%S')] $*"; }

queue=(
  "ornith-ai/Ornith-1.0-9B-GGUF|ornith-1.0-9b-Q6_K.gguf"
  "AtomicChat/Qwen3.8-27B-GGUF|Qwen3.8-27B-AD-IQ1_M.gguf"
  "prism-ml/Ternary-Bonsai-27B-gguf|Ternary-Bonsai-27B-Q2_g64.gguf"
  "AtomicChat/Qwen3.8-27B-GGUF|Qwen3.8-27B-AD-IQ2_XXS.gguf"
)

for item in "${queue[@]}"; do
  repo="${item%%|*}"; file="${item##*|}"
  log "START $repo/$file"
  hf download "$repo" "$file" >> /c/AI/qwen38-tuning/logs/dl-queue.log 2>&1 \
    && log "DONE  $file" \
    || log "FAIL  $file (continuing to next)"
done
log "queue empty"
