#!/usr/bin/env bash
# Follow-ups the first pass earned, in order of what they decide.
#
# 1. n-gram at 131,072. The 16K screen returned +45 % to +95 % from five
#    decoders that cost no VRAM and no drafter file -- the largest result of the
#    day. But report 19 says throughput at 128K is pinned at ~27 tok/s, and
#    prefill (110-127 s) is the part speculation cannot touch. Whether the win
#    survives to the target depth is the whole question.
# 2. Graded -ot on AD-IQ1_M. It is the only artifact with a good corpus (27/30)
#    and misses 65+0 at 131,072 by ONE layer. ot-ffn-tail freed 1,234 MiB and
#    ot-ssm-tail freed 168 for 19 % -- both far more than the ~125 MiB needed.
#    These find the smallest slice that clears the bar.
# 3. The four arms L-place-cpu never reached: -np 2 halved the per-slot context
#    to 8192, the 11,663-token probe returned HTTP 400, and the step died before
#    pcore-mask, prio-high, poll-0 and backend-samp ran.
set -u
cd /c/AI/qwen38-tuning
LOG=logs/afk-driver.log
until grep -q "q38 quality sweep complete" "$LOG"; do sleep 60; done
step() { local n="$1"; shift
  echo "[$(date +%T)] START $n" >> "$LOG"
  if "$@" > "logs/q38-$n.log" 2>&1; then
    echo "[$(date +%T)] DONE  $n" >> "$LOG"
  else
    rc=$?
    echo "[$(date +%T)] FAIL  $n (rc=$rc)" >> "$LOG"
  fi; }

# --fixed-text pins temperature 0 and a fixed seed for the TIMED generations.
# Without it every round writes different text, the n-gram hit rate follows the
# text, and the same arm returned +80.79 % and -30.56 % in two sweeps three
# hours apart -- BOTH passing the paired test, because the 13.6 % floor was
# built for boot-to-boot VRAM drift and cannot see variance from content.
# FOUR rounds here, not two: this is the lever that exposed the failure mode.
step "F-ngram-16k-fixed" python bench/kv_sweep.py --ctx 16384 --quant v3-iq2xxs \
     --rounds 4 --fixed-text \
     --arms q4_0,ngram-mod-short,ngram-mapk4v-wide,ngram-map-k,ngram-cache \
     --out kv-ngram-fixed.jsonl

step "F-ngram-128k-fixed" python bench/kv_sweep.py --ctx 131072 --quant v3-iq2xxs \
     --rounds 2 --fixed-text --arms q4_0,ngram-mod-short,ngram-mapk4v-wide \
     --out kv-ngram-fixed.jsonl

step "F-ot-iq1m-128k" python bench/kv_sweep.py --ctx 131072 --quant iq1m \
     --rounds 2 --arms q4_0,ot-ffn-1,ot-ssm-4,ot-ffn-2 \
     --out kv-ot-iq1m.jsonl

# np2 is deliberately absent: it is not inert, it is harmful. -np N divides the
# context by N, so at the 131,072 target each slot would get 65,536.
step "F-place-cpu-rest" python bench/kv_sweep.py --ctx 16384 --quant v3-iq2xxs \
     --rounds 2 --arms q4_0,pcore-mask,prio-high,poll-0,backend-samp \
     --out kv-layers-16k.jsonl

echo "[$(date +%T)] q38 followup complete" >> "$LOG"
