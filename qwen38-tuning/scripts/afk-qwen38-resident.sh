#!/usr/bin/env bash
# Qwen3.8-27B only. Goal: fully GPU-resident at >=128K, highest tok/s.
# Stage A finds the ceiling (split-only, ~1 min/boot).
# Stage B measures real decode at 128K on the arms that are resident there.
# Stage C applies the two VRAM-freeing levers from report 18 to the winner.
cd /c/AI/qwen38-tuning
LOG=logs/afk-driver.log
step() {  # step <name> <cmd...>
  local n="$1"; shift
  echo "[$(date +%T)] START $n" >> "$LOG"
  if "$@" > "logs/q38-$n.log" 2>&1; then
    echo "[$(date +%T)] DONE  $n" >> "$LOG"
  else
    echo "[$(date +%T)] FAIL  $n (rc=$?) -- see q38-$n.log" >> "$LOG"
  fi
}

# ---- Stage A: ceiling, q4_0 KV, Qwen3.8-27B arms not yet swept -------------
for q in iq1m v3-iq2xxs v3-q2kxl q2kxl; do
  step "ceil-$q" python bench/ctx_ceiling.py --quant "$q" --kv q4_0 \
       --ladder 131072,163840,196608,229376,262144 --out ctx-ceiling-q38.jsonl
done

# ---- Stage B: real decode at 128K on the deep-resident arms ----------------
for q in v3-iq1s v3-iq1m iq1m; do
  step "tg128k-$q" python bench/kv_sweep.py --ctx 131072 --quant "$q" \
       --arms q4_0 --rounds 2 --out kv-128k-q38.jsonl
done

# ---- Stage C: the two VRAM levers, paired, on the fastest resident arm -----
step "levers-128k-v3iq1s" python bench/kv_sweep.py --ctx 131072 --quant v3-iq1s \
     --arms q4_0,q4_0-ckpt8,k8v4 --rounds 2 --out kv-levers-q38.jsonl

echo "[$(date +%T)] qwen38 resident queue complete" >> "$LOG"
