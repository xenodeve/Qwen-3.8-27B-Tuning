#!/usr/bin/env bash
# Follow-up the D1 result earned, 2026-08-21 ~04:10.
#
# D1 at 163,840 on v3-iq2xxs produced a result that inverts the project's main
# assumption, and the follow-up is about WHY rather than what:
#
#   q4_0            62+3   19.36 / 18.83     baseline
#   ot-ssm-4        65+0   22.72 / 21.73     +16.38 %   RESIDENCY RESTORED
#   ot-ssm-10       65+0   21.40 / 20.95     +10.90 %   under the floor
#   ngram-mod-short 62+3   37.89 / 38.65    +100.48 %   <-- FASTEST, and NOT resident
#   ot-ssm-10+ngram 65+0   32.37 / 32.44     +69.74 %
#
# `-ot` on the ssm tensors does restore 65+0, exactly as report 20 promoted it
# to do. It is worth +16 % on its own. And it makes the machine SLOWER once
# speculation is on: 32.4 resident against 38.7 with three CPU layers.
#
# The mechanism that would explain it: speculative decoding verifies a batch of
# drafted tokens in one forward pass. A whole layer on the CPU processes that
# batch together, so its cost amortises across the accepted tokens. SSM state is
# recurrent and cannot batch the same way, so every drafted token pays its own
# CPU round trip -- and the more tokens speculation drafts, the worse the ssm
# slice gets. If that is right, ssm-offload and speculation are ANTAGONISTIC,
# which nothing in report 20 anticipated.
#
# Two arms decide it:
#   E1  ot-ssm-4-ngram   -- 4 blocks instead of 10. ot-ssm-4 beat ot-ssm-10
#                           without speculation; if the penalty scales with the
#                           number of offloaded ssm blocks, this lands between
#                           32.4 and 38.7 and the mechanism holds.
#   E2  acceptance       -- recorded on every row. If the ssm arms show a LOWER
#                           acceptance rate rather than the same rate at a lower
#                           speed, the explanation is different: the drafter is
#                           being starved, not taxed.
#
# E3 is the practical half, and it is the one the goal cares about: 163,840 at
# 38.65 tok/s costs less than half the speed of 131,072 at 81.46. Whether the
# window is worth the halving is a quality question, but the SHAPE of the
# trade -- where between the two depths the curve bends -- is not. 147,456 is
# the midpoint nobody has measured.
set -u
cd /c/AI/qwen38-tuning
LOG=logs/afk-driver.log
until grep -q "q38 deep complete" "$LOG"; do sleep 60; done
step() { local n="$1"; shift
  echo "[$(date +%T)] START $n" >> "$LOG"
  if "$@" > "logs/q38-$n.log" 2>&1; then rc=0; else rc=$?; fi
  if [ "$rc" = 0 ]; then echo "[$(date +%T)] DONE  $n" >> "$LOG"
  else echo "[$(date +%T)] FAIL  $n (rc=$rc)" >> "$LOG"; fi; }

step "E1-ssm-scaling-160k" python bench/kv_sweep.py --ctx 163840 --quant v3-iq2xxs \
     --rounds 2 --fixed-text --arms ngram-mod-short,ot-ssm-4-ngram,ot-ssm-10-ngram \
     --out kv-deep-160k.jsonl

step "E3-bend-147k" python bench/kv_sweep.py --ctx 147456 --quant v3-iq2xxs \
     --rounds 2 --fixed-text --arms q4_0,ngram-mod-short \
     --out kv-deep-147k.jsonl

echo "[$(date +%T)] q38 deep2 complete" >> "$LOG"
