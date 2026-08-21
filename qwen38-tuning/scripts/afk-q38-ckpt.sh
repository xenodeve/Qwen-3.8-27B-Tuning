#!/usr/bin/env bash
# Does --ctx-checkpoints 8 buy residency? Report 18 says the default 32 holds
# speculative VRAM an append-only agent never rewinds into. AD-IQ1_M missed
# 65+0 at 131072 by ONE layer (65+1, 338 MiB free) -- if the claim is right this
# flag alone covers it, with no change to the desktop.
# Waits for the running queue to finish so it never swaps under a live server.
cd /c/AI/qwen38-tuning
LOG=logs/afk-driver.log
until grep -q "qwen38 resident queue complete" "$LOG"; do sleep 60; done
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

# The arm that actually passes the corpus (27/31), one layer short at 131072.
step "ckpt8-iq1m"      python bench/ctx_ceiling.py --quant iq1m --kv q4_0 \
     --extra "--ctx-checkpoints 8" --tag ckpt8 \
     --ladder 131072,163840,196608 --out ctx-ceiling-q38.jsonl
# The deepest V3 arms -- does the freed VRAM buy another rung?
step "ckpt8-v3-iq1s"   python bench/ctx_ceiling.py --quant v3-iq1s --kv q4_0 \
     --extra "--ctx-checkpoints 8" --tag ckpt8 \
     --ladder 196608,229376,262144 --out ctx-ceiling-q38.jsonl
step "ckpt8-v3-iq1m"   python bench/ctx_ceiling.py --quant v3-iq1m --kv q4_0 \
     --extra "--ctx-checkpoints 8" --tag ckpt8 \
     --ladder 163840,196608,229376 --out ctx-ceiling-q38.jsonl
step "ckpt8-v3-iq2xxs" python bench/ctx_ceiling.py --quant v3-iq2xxs --kv q4_0 \
     --extra "--ctx-checkpoints 8" --tag ckpt8 \
     --ladder 131072,163840,196608 --out ctx-ceiling-q38.jsonl
# Asymmetric KV on the arm that misses: K keeps precision, V halves.
step "k8v4-iq1m"       python bench/ctx_ceiling.py --quant iq1m --kv q4_0 \
     --extra "-ctk q8_0 -ctv q4_0 --ctx-checkpoints 8" --tag k8v4ckpt8 \
     --ladder 131072,163840 --out ctx-ceiling-q38.jsonl

echo "[$(date +%T)] q38 ckpt sweep complete" >> "$LOG"
