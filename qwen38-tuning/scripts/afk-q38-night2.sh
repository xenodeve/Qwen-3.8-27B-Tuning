#!/usr/bin/env bash
# Second AFK batch, 2026-08-21 ~02:50. Runs on the harness patched tonight:
# depth-aware post() budget, kill() that waits for real VRAM release, and a
# try/finally teardown. Both steps below are also the end-to-end smoke test for
# those three changes -- `F-place-cpu-rest` died last round from exactly the
# failure they fix.
set -u
cd /c/AI/qwen38-tuning
LOG=logs/afk-driver.log
step() { local n="$1"; shift
  echo "[$(date +%T)] START $n" >> "$LOG"
  if "$@" > "logs/q38-$n.log" 2>&1; then
    rc=0
  else
    rc=$?
  fi
  if [ "$rc" = 0 ]; then echo "[$(date +%T)] DONE  $n" >> "$LOG"
  else echo "[$(date +%T)] FAIL  $n (rc=$rc)" >> "$LOG"; fi; }

# 1. The four placement arms that never produced a single row: -np 2 broke the
#    first attempt, and the second died on a port collision 30 s after the
#    previous step timed out. Cheap, at 16K, and nothing else has measured them.
step "G-place-cpu-rest" python bench/kv_sweep.py --ctx 16384 --quant v3-iq2xxs \
     --rounds 2 --fixed-text --arms q4_0,pcore-mask,prio-high,poll-0,backend-samp \
     --out kv-layers-16k.jsonl

# 2. report 22 §7 item 3. `ngram-map-k` is the strongest arm at 16K (+135.89 %)
#    and has NEVER been measured at the target depth; `ngram-mod-short` is the
#    strongest thing measured at 131,072 (+213.08 %) and was never screened at
#    16K against map-k on fixed text. Run both against one baseline so the
#    comparison is paired inside a single boot sequence.
step "G-ngram-mapk-128k" python bench/kv_sweep.py --ctx 131072 --quant v3-iq2xxs \
     --rounds 2 --fixed-text --arms q4_0,ngram-map-k,ngram-mod-short \
     --out kv-ngram-fixed.jsonl

echo "[$(date +%T)] q38 night2 complete" >> "$LOG"
