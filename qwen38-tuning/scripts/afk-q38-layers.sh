#!/usr/bin/env bash
# One arm per untouched optimization layer from report 16, on the arm that
# report 19 says to prefer: V3 UD-IQ2_XXS, the largest Qwen3.8-27B artifact
# that holds 65+0 at 131,072.
#
# Screened at 16K first because a 16K boot costs about a minute against three
# at 128K, and 19 levers x 2 rounds is 38 boots. Depth-specific levers (KV type,
# checkpoints, cache-reuse, context-shift) are NOT in this screen -- they are
# meaningless at 16K where the cache is ~288 MiB, and run separately at 131,072.
#
# Every flag was verified to parse against build 10472 before this script was
# written. Report 16 recorded a prediction for each; several are predicted inert
# and are included deliberately, because a written prediction can be refuted and
# a dropped one cannot.
set -u
cd /c/AI/qwen38-tuning
LOG=logs/afk-driver.log
until grep -q "q38 ckpt sweep complete" "$LOG"; do sleep 60; done
step() { local n="$1"; shift
  echo "[$(date +%T)] START $n" >> "$LOG"
  if "$@" > "logs/q38-$n.log" 2>&1; then
    echo "[$(date +%T)] DONE  $n" >> "$LOG"
  else
    # capture BEFORE anything else runs: $(date) resets $? and the real exit
    # code is lost. tg128k-iq1m logged "rc=0" on a genuine failure at 18:40
    # because of exactly that.
    rc=$?
    echo "[$(date +%T)] FAIL  $n (rc=$rc)" >> "$LOG"
  fi; }

# q4_0 is first in every list: kv_sweep pairs everything against the baseline
# arm, and the whole screen has to share one control to be readable.

# L8 kernels + L5 loading -- the validity questions. -fa has never been forced
# either way on any run this project has published.
step "L-kernel" python bench/kv_sweep.py --ctx 16384 --quant v3-iq2xxs --rounds 2 \
     --arms q4_0,fa-off,no-repack,no-op-offload,loadmode-none,no-host \
     --out kv-layers-16k.jsonl

# L9 decoders that need no drafter file and no download -- the cheapest
# unexplored decoders on the whole list.
step "L-ngram" python bench/kv_sweep.py --ctx 16384 --quant v3-iq2xxs --rounds 2 \
     --arms q4_0,ngram-simple,ngram-mod,ngram-map-k,ngram-map-k4v,ngram-cache \
     --out kv-layers-16k.jsonl

# L4 placement + L10 slots + L11 CPU + L12 backend sampling.
# sm-tensor and np2 are the two this project predicted INERT; pcore-mask is
# predicted near-inert at 65+0 because the CPU does almost no decode work there.
step "L-place-cpu" python bench/kv_sweep.py --ctx 16384 --quant v3-iq2xxs --rounds 2 \
     --arms q4_0,ot-ffn-tail,ot-ssm-tail,sm-tensor,np2,pcore-mask,prio-high,poll-0,backend-samp \
     --out kv-layers-16k.jsonl

echo "[$(date +%T)] q38 layer screen complete" >> "$LOG"
