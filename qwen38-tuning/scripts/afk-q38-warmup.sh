#!/usr/bin/env bash
# P0 of plan 04: is the 160-token probe measuring speculation before it warms up?
# 2026-08-21 ~05:00.
#
# An external review of this exact model, on much larger hardware, reported:
#
#   "By the time it came to output the actual response, the MTP had gotten
#    extremely fast (91 tk/s vs 62 tk/s starting rate)"
#
# Every timed generation this project has ever run is 160 tokens. If a
# speculative decoder needs longer than that to reach its rate, then every
# decoder number we hold is understated -- and draft-mtp, draft-dflash, eagle3
# and dspark were all ELIMINATED on those numbers.
#
# This runs FIRST among the new work because it is cheap and because a broken
# ruler makes everything measured after it worthless.
#
# READING THE RESULT. Compare the SAME arm across the three lengths, not arms
# against each other:
#   * n-gram advantage flat across 160/512/1024  -> the probe is fine, decoder
#     verdicts stand, and the reviewer was seeing something else (a bigger MoE,
#     a different quant, or prompt-cache effects).
#   * advantage GROWS with length               -> the probe was too short, and
#     report 20 section 1 has to be re-run before anything stays eliminated.
#
# draft-mtp is included deliberately. It is the arm the reviewer's observation
# was actually about, and the one we eliminated most confidently: +81 % at 16K
# and -71 % at 131,072. If warm-up is real, that -71 % is the number most likely
# to be an artefact of a 160-token window.
set -u
cd /c/AI/qwen38-tuning
LOG=logs/afk-driver.log
until grep -q "q38 vram complete" "$LOG"; do sleep 60; done
step() { local n="$1"; shift
  echo "[$(date +%T)] START $n" >> "$LOG"
  if "$@" > "logs/q38-$n.log" 2>&1; then rc=0; else rc=$?; fi
  if [ "$rc" = 0 ]; then echo "[$(date +%T)] DONE  $n" >> "$LOG"
  else echo "[$(date +%T)] FAIL  $n (rc=$rc)" >> "$LOG"; fi; }

# 16,384: cheap, and the depth where every decoder was screened.
step "W1-warm-160"  python bench/kv_sweep.py --ctx 16384 --quant v3-iq2xxs \
     --rounds 2 --fixed-text --n-predict 160 \
     --arms q4_0,ngram-map-k,mtp-gpu --out kv-warmup.jsonl

step "W2-warm-512"  python bench/kv_sweep.py --ctx 16384 --quant v3-iq2xxs \
     --rounds 2 --fixed-text --n-predict 512 \
     --arms q4_0,ngram-map-k,mtp-gpu --out kv-warmup.jsonl

step "W3-warm-1024" python bench/kv_sweep.py --ctx 16384 --quant v3-iq2xxs \
     --rounds 2 --fixed-text --n-predict 1024 \
     --arms q4_0,ngram-map-k,mtp-gpu --out kv-warmup.jsonl

# One confirmation at the target depth, on the arm that would ship. If warm-up
# is real anywhere, it is worth most here: 131,072 is where a generation is long
# enough to matter and where prefill already dominates the wall clock.
step "W4-warm-128k" python bench/kv_sweep.py --ctx 131072 --quant v3-iq2xxs \
     --rounds 2 --fixed-text --n-predict 1024 \
     --arms q4_0,ngram-mod-short --out kv-warmup-128k.jsonl

echo "[$(date +%T)] q38 warmup complete" >> "$LOG"
