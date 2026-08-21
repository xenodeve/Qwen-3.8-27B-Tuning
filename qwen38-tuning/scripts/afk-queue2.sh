#!/bin/bash
# Second unattended queue, armed to start when the first one empties.
#
# Written as a separate file rather than appended to the running one on purpose:
# bash reads a script incrementally while executing it, so editing a queue
# that is mid-flight can make it jump into the middle of a line.
#
# Two things the first queue did not cover:
#   * Ornith-9B has no quality measurement at all. It won the 16K arena by
#     +44 % and has 3.9 GB of VRAM spare, which makes it the strongest 128K
#     candidate on the machine -- and a completely unmeasured one. A different
#     MODEL is a different question from a damaged quantization: the risk is a
#     capability ceiling, not selective collapse.
#   * AD-IQ1_M produced a DIFFERENT greedy hash from Q4, Q2_K_XL and IQ2_XXS on
#     a mechanical rename with one correct answer. Those three all agreed. That
#     is the first divergence this probe has ever recorded, and what the model
#     actually wrote has not been looked at.
set -u
cd /c/AI/qwen38-tuning/bench
LOGS=/c/AI/qwen38-tuning/logs
log() { echo "[$(date '+%H:%M:%S')] $*"; }

step() {
  local name="$1"; shift
  log "START $name"
  if "$@" > "$LOGS/afk2-$name.log" 2>&1; then log "DONE  $name"
  else log "FAIL  $name (rc=$?) -- see afk2-$name.log"; fi
}

ORN='C:\Users\xenod\.cache\huggingface\hub\models--ornith-ai--Ornith-1.0-9B-GGUF\snapshots\3296bc7a404871a72ac3f1903f561459c09b5c17\ornith-1.0-9b-Q6_K.gguf'

cat > /c/AI/qwen38-tuning/scripts/serve-ornith9b.ps1 <<PS1
param([int]\$Ctx = 16384, [int]\$Port = 8080)
\$ErrorActionPreference = 'Continue'
& C:\AI\llama.cpp-cuda\llama-server.exe \`
    -m "$ORN" --alias ornith9b -c \$Ctx \`
    -ngl auto --fit on --fit-target 768 -fa on -np 1 \`
    -t 18 -b 2048 -ub 256 --no-mmproj-auto \`
    --host 127.0.0.1 --port \$Port
PS1

# 1. What did IQ1_M actually write? The hash says it differs; the hash cannot
#    say whether it is wrong. Cheap, and it gates everything else about IQ1_M.
step iq1m-greedy-diff \
  python greedy_diff.py --arms iq2xxs-nomtp,iq1m-nomtp,ornith9b-nomtp \
    --out greedy-diff.jsonl

# 2. Ornith-9B at depth. A 9B has both smaller weights and a smaller per-token
#    cache, so 128K is a different regime rather than a better point in the
#    same one.
step ornith9b-depth-128k-q4 \
  python kv_sweep.py --ctx 131072 --rounds 2 --arms q4_0,q8_0 --quant ornith9b \
    --out kv-sweep-ornith.jsonl

# 3. Ornith-9B quality, same probes and budgets as every other arm.
step ornith9b-serve \
  bash /c/AI/qwen38-tuning/scripts/swap-model.sh \
    /c/AI/qwen38-tuning/scripts/serve-ornith9b.ps1 ornith-1.0-9b

step ornith9b-protocol \
  python protocol_gate.py --label ornith9b --trials 15 --temperature 0.7 \
    --max-tokens 4096 --out protocol-budget.jsonl

step ornith9b-corpus \
  python run_retry_bench.py --label ornith9b --passes 3

step ornith9b-stability \
  python stability_gate.py --label ornith9b --turns 100

# 4. 256K with q4_0 KV. At q8_0 the split was 31+34 and decode 1.71 tok/s;
#    halving the cache is aimed straight at that.
step iq2xxs-256k-q4 \
  python kv_sweep.py --ctx 262144 --rounds 1 --arms q4_0,q8_0 --quant iq2xxs \
    --out kv-sweep-256k.jsonl

step iq1m-256k-q4 \
  python kv_sweep.py --ctx 262144 --rounds 1 --arms q4_0 --quant iq1m \
    --out kv-sweep-256k.jsonl


# 5. Bonsai. Mainline b10472 DOES serve the group-64 ternary pack -- 49.86 tok/s
#    at 16K, +17.7 pct over IQ2_XXS, resolved -- which settles the open question
#    report 08 left. It is a 27B at 50 tok/s whose vendor claims 94.6 pct of
#    FP16 aggregate against 85.5 pct for a conventional IQ2_XXS. That claim is
#    exactly what this project cannot take on trust, so: same probes, same
#    budgets, same corpus as every other arm.
BON='C:\Users\xenod\.cache\huggingface\hub\models--prism-ml--Ternary-Bonsai-27B-gguf\snapshots\abbae723028d71be674e71e1a71201a6f43fab22\Ternary-Bonsai-27B-Q2_g64.gguf'
cat > /c/AI/qwen38-tuning/scripts/serve-bonsai-g64.ps1 <<PS2
param([int]\$Ctx = 16384, [int]\$Port = 8080)
\$ErrorActionPreference = 'Continue'
& C:\AI\llama.cpp-cuda\llama-server.exe \`
    -m "$BON" --alias bonsai-g64 -c \$Ctx \`
    -ngl auto --fit on --fit-target 768 -fa on -np 1 \`
    -t 18 -b 2048 -ub 256 --no-mmproj-auto \`
    --host 127.0.0.1 --port \$Port
PS2

step bonsai-serve \
  bash /c/AI/qwen38-tuning/scripts/swap-model.sh \
    /c/AI/qwen38-tuning/scripts/serve-bonsai-g64.ps1 Ternary-Bonsai

step bonsai-protocol \
  python protocol_gate.py --label bonsai-g64 --trials 15 --temperature 0.7 \
    --max-tokens 4096 --out protocol-budget.jsonl

step bonsai-corpus \
  python run_retry_bench.py --label bonsai-g64 --passes 3

step bonsai-stability \
  python stability_gate.py --label bonsai-g64 --turns 100

step bonsai-depth-128k \
  python kv_sweep.py --ctx 131072 --rounds 2 --arms q4_0,q8_0 --quant bonsai-g64 \
    --out kv-sweep-bonsai.jsonl

log "afk queue 2 empty"
