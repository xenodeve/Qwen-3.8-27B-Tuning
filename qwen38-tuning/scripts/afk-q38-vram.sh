#!/usr/bin/env bash
# Buy the 576 MiB without moving weights. 2026-08-21 ~04:15.
#
# D1 established the shape of the problem at 163,840 on v3-iq2xxs:
#
#   * the artifact sits at 62+3 and needs roughly 576 MiB to reach 65+0
#   * `-ot` on the ssm tensors DOES free them and DOES restore 65+0
#   * and it collapses speculative acceptance from 100 % to 4 %, so the
#     resident arm ends up SLOWER than the non-resident one (32.4 vs 38.7)
#
# So residency at depth is worth having and the only route to it measured so
# far poisons the thing that makes depth usable. These arms look for MiB that
# cost neither weights nor acceptance.
#
#   --fit-target   a RESERVE the harness has passed as 768 since the first
#                  sweep, never once tested. If 192 is enough headroom, the
#                  shortfall is most of the way paid, for free.
#   -b / -ub       compute-buffer size. Smaller means less scratch VRAM and
#                  slower prefill: a real trade, and one the goal can price --
#                  unlike the -ot trade, which turned out not to be a trade.
#
# Every arm carries ngram-mod. D1 is the reason: a verdict reached with
# speculation off reversed when it was turned on, so the comparison has to be
# made in the configuration that would actually ship.
#
# WATCH TWO COLUMNS, not one. `gpu/cpu` says whether the MiB were found;
# `acceptance` says whether they cost anything. An arm that reaches 65+0 with
# acceptance near 100 % is the answer to the goal. An arm that reaches 65+0
# with acceptance in the low tens is ot-ssm again in a different costume.
set -u
cd /c/AI/qwen38-tuning
LOG=logs/afk-driver.log
until grep -q "q38 deep2 complete" "$LOG"; do sleep 60; done
step() { local n="$1"; shift
  echo "[$(date +%T)] START $n" >> "$LOG"
  if "$@" > "logs/q38-$n.log" 2>&1; then rc=0; else rc=$?; fi
  if [ "$rc" = 0 ]; then echo "[$(date +%T)] DONE  $n" >> "$LOG"
  else echo "[$(date +%T)] FAIL  $n (rc=$rc)" >> "$LOG"; fi; }

# The baseline here is ngram-mod-short at 62+3 -- the fastest thing measured at
# this depth, 37.89/38.65 -- NOT the q4_0 control. An arm has to beat what we
# would otherwise ship, not what we started from.
step "V1-vram-160k" python bench/kv_sweep.py --ctx 163840 --quant v3-iq2xxs \
     --rounds 2 --fixed-text \
     --arms ngram-mod-short,fit-384-ngram,fit-192-ngram,ub128-ngram,fit192-ub128-ngram \
     --out kv-vram-160k.jsonl

# If anything above reached 65+0 with its acceptance intact, the same lever is
# worth trying one step deeper, where v3-iq2xxs currently does not go at all.
step "V2-vram-192k" python bench/kv_sweep.py --ctx 196608 --quant v3-iq2xxs \
     --rounds 2 --fixed-text \
     --arms ngram-mod-short,fit-192-ngram,fit192-ub128-ngram,b1024ub128-ngram \
     --out kv-vram-192k.jsonl

echo "[$(date +%T)] q38 vram complete" >> "$LOG"
