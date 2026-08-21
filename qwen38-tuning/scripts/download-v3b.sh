#!/bin/bash
# Second half of the Dynamic V3 pull: the two larger rungs of the ladder.
# Q2_K_XL is Unsloth's own efficiency pick at 9.83 GB, and Q4_K_XL is the
# control -- both republished, both different from the copies on disk.
set -u
log() { echo "[$(date '+%H:%M:%S')] $*"; }
R=unsloth/Qwen3.8-27B-GGUF
until grep -q "v3 download queue empty" /c/AI/qwen38-tuning/logs/dl-v3-driver.log 2>/dev/null; do sleep 60; done
# V3 Q4_K_XL is deliberately skipped: 16.35 GiB and ~35 minutes of link
# time for a control this project already has, while the focus is Q1/Q2.
for f in Qwen3.8-27B-UD-Q2_K_XL.gguf ; do
  for try in 1 2 3; do
    log "START $f (attempt $try)"
    if hf download "$R" "$f" >> /c/AI/qwen38-tuning/logs/dl-v3.log 2>&1; then log "DONE  $f"; break; fi
    log "FAIL  $f attempt $try"; sleep 20
  done
  df -h /c | tail -1 | awk '{print "        disk free: " $4}'
done
log "v3b download queue empty"
