#!/usr/bin/env bash
# Tier 1 of the 16-layer programme: the levers whose value only exists AT DEPTH,
# measured at 131,072 on V3 UD-IQ2_XXS -- the largest Qwen3.8-27B artifact that
# holds 65+0 there (report 19).
#
# The 16K screen (afk-q38-layers.sh) cannot see any of these: at 16K the cache is
# ~288 MiB and every one of them is about the cache.
set -u
cd /c/AI/qwen38-tuning
LOG=logs/afk-driver.log
until grep -q "q38 layer screen complete" "$LOG"; do sleep 60; done
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

# L9 -- the comparison our own -8.8 % MTP verdict never had. Same 1.28 GiB
# drafter in three places: GPU, CPU via --spec-draft-device, CPU via -otd.
step "D-mtp-placement" python bench/kv_sweep.py --ctx 131072 --quant v3-iq2xxs \
     --rounds 2 --arms q4_0,mtp-gpu,mtp-cpu,mtp-otd-cpu --out kv-depth-levers.jsonl

# L6 -- everything that changes the cache's VRAM footprint at depth.
# k8v4 (-ctk q8_0 -ctv q4_0) is deliberately NOT in this list. The kernel
# screen only ever tested K and V at the SAME type; the mixed pair was never
# checked, and on 2026-08-20 at 19:00 it spent 45+ minutes on one 131,072
# prefill at 20 % GPU while the symmetric q4_0 arm beside it took 105 s. Same
# signature as q5_1 (144-170 tok/s prefill against 1,180): a fallback kernel,
# not a hang. It is screened at 16K below, where a slow kernel costs a minute.
step "D-kv-vram" python bench/kv_sweep.py --ctx 131072 --quant v3-iq2xxs \
     --rounds 2 --arms q4_0,q4_0-ckpt8,no-kv-unified,swa-full \
     --out kv-depth-levers.jsonl

# Mixed KV, cheaply. If prefill here sits near the symmetric q4_0 row it has a
# fast kernel and earns a depth run; if it collapses, it is settled for good.
step "D-kv-mixed-16k" python bench/kv_sweep.py --ctx 16384 --quant v3-iq2xxs \
     --rounds 2 --arms q4_0,k8v4,k8v4-ckpt8 --out kv-depth-levers.jsonl

# L6/L7 -- prefix economics and window shape. --cache-reuse attacks the largest
# single cost this project has measured (63 s at 16K, 248 s at 64K for one
# broken prefix); --context-shift changes the question from "hold the window" to
# "move it".
step "D-prefix-window" python bench/kv_sweep.py --ctx 131072 --quant v3-iq2xxs \
     --rounds 2 --arms q4_0,cache-reuse,ctx-shift --out kv-depth-levers.jsonl

# L4 -- tensor placement where it can actually pay: one rung DEEPER, where
# v3-iq2xxs is 62+3 and three layers are already on the CPU. Moving only the
# tail FFN (or only the ssm state path) should beat moving whole layers.
step "D-ot-163k" python bench/kv_sweep.py --ctx 163840 --quant v3-iq2xxs \
     --rounds 2 --arms q4_0,ot-ffn-tail,ot-ssm-tail --out kv-depth-levers.jsonl

# L11 -- the P-core mask on the one configuration where the CPU does real decode
# work. At 65+0 it cannot matter; at 62+3 it can.
step "D-pcore-163k" python bench/kv_sweep.py --ctx 163840 --quant v3-iq2xxs \
     --rounds 2 --arms q4_0,pcore-mask --out kv-depth-levers.jsonl

echo "[$(date +%T)] q38 depth levers complete" >> "$LOG"
