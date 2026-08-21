#!/bin/bash
# Unsloth replaced the entire Qwen3.8-27B-GGUF repo with Dynamic V3 at
# 2026-08-19T16:39:23Z -- during this session. Every artifact measured today is
# the PRE-V3 build, and every V3 file has a different size and OID:
#
#     UD-IQ2_XXS   9,010,048,064  ->  7,266,070,528
#     UD-Q2_K_XL  10,676,423,744  ->  9,828,981,664
#     UD-Q4_K_XL  17,923,394,624  -> 17,559,178,144
#     UD-IQ2_M    10,319,907,904  ->  DELETED (UD-IQ2_S 8,371,970,048 replaces it)
#
# UD-IQ2_M's "download failure" earlier was not transient: the file no longer
# exists. Chasing it with retries was wrong.
#
# V3 also ships the 1-bit artifacts this project spent the day looking for, at
# sizes no other vendor offers: IQ1_S at 5.77 GiB against the 8.39 GiB currently
# in production. Given that four CPU-resident layers cost half the throughput
# here, 1.6 GiB of extra headroom is the most promising single change available.
#
# Ordered by expected value, smallest first so the most headroom lands soonest.
set -u
log() { echo "[$(date '+%H:%M:%S')] $*"; }
R=unsloth/Qwen3.8-27B-GGUF
for f in \
  Qwen3.8-27B-UD-IQ1_S.gguf \
  Qwen3.8-27B-UD-IQ1_M.gguf \
  Qwen3.8-27B-UD-IQ2_XXS.gguf \
  Qwen3.8-27B-UD-IQ2_S.gguf \
  MTP/mtp-Qwen3.8-27B-Q4_0.gguf ; do
  for try in 1 2 3; do
    log "START $f (attempt $try)"
    if hf download "$R" "$f" >> /c/AI/qwen38-tuning/logs/dl-v3.log 2>&1; then
      log "DONE  $f"; break
    fi
    log "FAIL  $f attempt $try"; sleep 20
  done
  df -h /c | tail -1 | awk '{print "        disk free: " $4}'
done
log "v3 download queue empty"
