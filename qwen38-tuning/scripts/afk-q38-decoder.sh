#!/usr/bin/env bash
# Layer 9 in full. This project has run four of eleven decoders and tuned only
# MTP, across twelve configurations of one drafting strategy, while ten others
# sat one line away in the same --help text.
#
# Screened at 16K: speculation either produces accepted draft tokens or it does
# not, and that shows at any depth. Whatever wins gets re-run at 131,072.
set -u
cd /c/AI/qwen38-tuning
LOG=logs/afk-driver.log
until grep -q "q38 sampling screen complete" "$LOG"; do sleep 60; done
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

# The five n-gram decoders at their defaults. None needs a drafter file.
step "dec-ngram-default" python bench/kv_sweep.py --ctx 16384 --quant v3-iq2xxs \
     --rounds 2 --arms q4_0,ngram-simple,ngram-mod,ngram-map-k,ngram-map-k4v,ngram-cache \
     --out kv-decoders.jsonl

# The tuning knobs behind them, never moved. ngram-simple scored 31 % acceptance
# at defaults with a cold table; the lookup and draft lengths decide whether a
# match is found at all.
step "dec-ngram-tuned" python bench/kv_sweep.py --ctx 16384 --quant v3-iq2xxs \
     --rounds 2 --arms q4_0,ngram-simple-wide,ngram-mod-short,ngram-mapk4v-wide \
     --out kv-decoders.jsonl

# draft-simple with a same-family 2B distill, on GPU and on CPU. If the vocab
# does not match, the server refuses at load and that IS the result.
step "dec-draft-simple" python bench/kv_sweep.py --ctx 16384 --quant v3-iq2xxs \
     --rounds 2 --arms q4_0,draft-simple,draft-simple-cpu --out kv-decoders.jsonl

# The standalone MTP head in three placements, at 16K this time so it pairs with
# everything above. V3 removed the built-in head from IQ2_XXS, so the 1.28 GiB
# drafter is the only way to drive MTP on this artifact at all.
step "dec-mtp" python bench/kv_sweep.py --ctx 16384 --quant v3-iq2xxs \
     --rounds 2 --arms q4_0,mtp-gpu,mtp-cpu,mtp-otd-cpu --out kv-decoders.jsonl

# DFlash 2. The vendor says it needs unmerged PR #27342. One boot settles
# whether the stock draft-dflash loader reads the file or rejects the
# architecture -- cheaper than assuming either way.
step "dec-dflash2" python bench/kv_sweep.py --ctx 16384 --quant v3-iq2xxs \
     --rounds 1 --arms q4_0,dflash2 --out kv-decoders.jsonl

echo "[$(date +%T)] q38 decoder sweep complete" >> "$LOG"
