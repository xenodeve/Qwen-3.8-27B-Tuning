#!/bin/bash
# Unattended experiment queue.
#
# Runs after the 128K KV sweep and works through the candidates downloaded
# today. Every step is independent: a failure logs and the queue moves on,
# because an unattended run that stops on the first surprise wastes the window
# it was given.
#
# Models are addressed by EXACT PATH, not `-hf repo:tag`. Two reasons, both
# already paid for: `-hf …:Q2_0` matched `PQ2_0.gguf` in the Bonsai repo (same
# byte count, different file, no error), and these artifacts were fetched with
# `hf download`, so a `-hf` reference risks a second full download of a file
# already on disk.
set -u
cd /c/AI/qwen38-tuning/bench
LOGS=/c/AI/qwen38-tuning/logs
log() { echo "[$(date '+%H:%M:%S')] $*"; }

step() {   # step <name> <command...>
  local name="$1"; shift
  log "START $name"
  if "$@" > "$LOGS/afk-$name.log" 2>&1; then
    log "DONE  $name"
  else
    log "FAIL  $name (rc=$?) -- see afk-$name.log"
  fi
}

# 1. Q1 at 16K: does a 1-bit build beat IQ2_XXS when BOTH are already fully
#    resident? The residency lever is spent here, so any gain is memory
#    bandwidth alone -- and any loss is quantization damage showing up.
step iq1m-arena-16k \
  python model_arena.py --rounds 3 --reps 3 \
    --arms iq2xxs-nomtp,iq1m-nomtp --out arena-iq1.jsonl

# 2. Q1 at 128K with the KV type that won today's sweep. IQ1_M is 0.48 GiB
#    smaller than IQ2_XXS, which at depth is worth a couple of layers.
step iq1m-depth-128k \
  python kv_sweep.py --ctx 131072 --rounds 2 --arms q4_0 --quant iq1m \
    --out kv-sweep-iq1m.jsonl

# 3. A different model entirely. A 9B has both smaller weights and a smaller
#    per-token cache, so 128K should be a different regime, not a better point
#    in the same one.
step ornith9b-arena-16k \
  python model_arena.py --rounds 3 --reps 3 \
    --arms iq2xxs-nomtp,ornith9b-nomtp --out arena-ornith.jsonl

step ornith9b-depth-128k \
  python kv_sweep.py --ctx 131072 --rounds 2 --arms q4_0,q8_0 --quant ornith9b \
    --out kv-sweep-ornith.jsonl

# 4. Does mainline b10472 serve Prism's group-64 ternary pack at all? Their
#    headline Q2_0 is g128 and needs their fork; Q2_g64 is described as matching
#    llama.cpp's own 64-value-group packing. Untested claim, cheap to falsify.
step bonsai-g64-arena-16k \
  python model_arena.py --rounds 2 --reps 3 \
    --arms iq2xxs-nomtp,bonsai-g64 --out arena-bonsai.jsonl

# 5. Protocol + corpus on whatever won step 1, so speed never lands in a report
#    without a quality number beside it.
# The arena leaves no server running, so boot one and PROVE it answers
# before probing: a swap that passed /health while the model was still
# loading once produced a whole corpus of HTTP 503s and a perfectly
# plausible tasks-per-hour figure out the other end.
step iq1m-serve \
  bash /c/AI/qwen38-tuning/scripts/swap-model.sh \
    /c/AI/qwen38-tuning/scripts/serve-iq1m.ps1 AD-IQ1_M

step iq1m-protocol \
  python protocol_gate.py --label iq1m --trials 15 --temperature 0.7 \
    --max-tokens 4096 --out protocol-budget.jsonl

step iq1m-corpus \
  python run_retry_bench.py --label iq1m --passes 3

log "afk queue empty"
