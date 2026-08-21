#!/usr/bin/env bash
# Beyond 128K, at speed. 2026-08-21 ~03:40.
#
# The goal is the deepest window this card holds AND the highest tok/s inside
# it. Report 21 answered only the first half -- it checks whether an artifact
# LOADS at a depth, never how fast it runs there. So the entire region past
# 131,072 has residency data and no throughput data at all:
#
#   v3-iq1s     65+0 to 196,608      tok/s at that depth: NEVER MEASURED
#   v3-iq1m     65+0 to 163,840      tok/s at that depth: NEVER MEASURED
#   v3-iq2xxs   65+0 to 131,072, then 62+3 at 163,840
#
# That 62+3 is the whole opportunity. Three CPU layers at depth is not a small
# tax: AD-IQ1_M at 65+1 and 131,072 decodes at 6.08 tok/s against a resident
# arm's 26.50. If `-ot` on the ssm tensors restores 65+0 -- which is exactly
# what report 20 measured it doing at 163,840 on another arm, at no throughput
# cost -- then the best-quality artifact that reaches 128K reaches 160K instead.
#
# ssm, NOT ffn. Last night's ffn slice moved 644 MiB and took prefill from 240.6
# to 8.56 tok/s. The ssm slice moves ~168 MiB. The harness now bounds this
# itself: post() is sized from the depth, so an arm that collapses is abandoned
# in ~41 min at 163,840 instead of sitting for three hours.
#
# ORDER IS BY LIKELIHOOD OF A WIN, per the goal.
#   D1  v3-iq2xxs @ 163,840 -- the artifact worth having, at a depth it nearly
#                              holds. Highest value if -ot ssm works.
#   D2  v3-iq1m   @ 163,840 -- already 65+0 there. Pure throughput question.
#   D3  v3-iq1s   @ 196,608 -- already 65+0. The deepest resident config known;
#                              its corpus is 0/12, so this measures the CEILING
#                              of what the hardware can do, not a candidate.
#   D4  v3-iq1m   @ 196,608 -- 60+5. Does -ot ssm rescue it the way D1 asks?
set -u
cd /c/AI/qwen38-tuning
LOG=logs/afk-driver.log
step() { local n="$1"; shift
  echo "[$(date +%T)] START $n" >> "$LOG"
  if "$@" > "logs/q38-$n.log" 2>&1; then rc=0; else rc=$?; fi
  if [ "$rc" = 0 ]; then echo "[$(date +%T)] DONE  $n" >> "$LOG"
  else echo "[$(date +%T)] FAIL  $n (rc=$rc)" >> "$LOG"; fi; }

# --fixed-text on every step. Without it the timed generations run at
# temperature 0.7, the n-gram hit rate follows the text rather than the
# hardware, and the same arm returned +80.79 % and -30.56 % three hours apart.
step "D1-iq2xxs-160k" python bench/kv_sweep.py --ctx 163840 --quant v3-iq2xxs \
     --rounds 2 --fixed-text \
     --arms q4_0,ot-ssm-4,ot-ssm-10,ngram-mod-short,ot-ssm-10-ngram \
     --out kv-deep-160k.jsonl

step "D2-iq1m-160k" python bench/kv_sweep.py --ctx 163840 --quant v3-iq1m \
     --rounds 2 --fixed-text --arms q4_0,ngram-mod-short \
     --out kv-deep-160k.jsonl

step "D3-iq1s-192k" python bench/kv_sweep.py --ctx 196608 --quant v3-iq1s \
     --rounds 2 --fixed-text --arms q4_0,ngram-mod-short \
     --out kv-deep-192k.jsonl

step "D4-iq1m-192k" python bench/kv_sweep.py --ctx 196608 --quant v3-iq1m \
     --rounds 2 --fixed-text --arms q4_0,ot-ssm-10,ot-ssm-10-ngram \
     --out kv-deep-192k.jsonl

echo "[$(date +%T)] q38 deep complete" >> "$LOG"
