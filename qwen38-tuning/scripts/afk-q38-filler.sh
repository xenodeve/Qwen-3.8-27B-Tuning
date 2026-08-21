#!/usr/bin/env bash
# Instrument fault 8: how much of the n-gram win is the benchmark's own text?
# 2026-08-21 ~05:45. Runs last, after the warm-up check.
#
# `depth_sweep.filler()` repeats one class definition with a four-digit index.
# At 147,456 that is 962 blocks, adjacent blocks 99.5 % identical, and 84.5 % of
# non-blank lines are exact duplicates of an earlier line. An n-gram decoder
# drafts from what is already in the context, so this is close to the most
# favourable text that could be constructed for it.
#
# EVERY n-gram number this project holds was measured on it:
#
#   16,384    ngram-map-k   +135.89 %   acceptance  93-100 %
#   131,072   ngram-mod     +200.22 %   acceptance      99 %
#   147,456   ngram-mod     +330.40 %   acceptance     100 %
#
# Acceptance pinned at 99-100 % across every depth is the tell. Real code is
# repetitive, but not 962 copies of one class.
#
# `--filler low` swaps in varied blocks: same structure, different identifiers,
# different types, different constants. Deterministic -- no RNG, because
# --fixed-text exists to make a sweep reproducible. It scores 73.17 % at 147,456
# against the historic 84.53 %.
#
# WHAT THE RESULT MEANS. Compare the SAME arm across the two fillers.
#   * advantage roughly unchanged -> the win is about the decoder, and the
#     published figures stand as written.
#   * advantage falls               -> the figures are an upper bound on a
#     synthetic best case, and every one of them needs a stated caveat. The SIZE
#     of the fall is the size of the correction owed.
#
# This does not test real code. It tests whether the number is SENSITIVE to how
# repetitive the prompt is. If it is, only a corpus run with n-gram enabled can
# say what production would see -- which is P5/P6 of plan 04, not this step.
set -u
cd /c/AI/qwen38-tuning
LOG=logs/afk-driver.log
until grep -q "q38 warmup complete" "$LOG"; do sleep 60; done
step() { local n="$1"; shift
  echo "[$(date +%T)] START $n" >> "$LOG"
  if "$@" > "logs/q38-$n.log" 2>&1; then rc=0; else rc=$?; fi
  if [ "$rc" = 0 ]; then echo "[$(date +%T)] DONE  $n" >> "$LOG"
  else echo "[$(date +%T)] FAIL  $n (rc=$rc)" >> "$LOG"; fi; }

# 147,456 first: it carries the largest claim (+330.40 %) and the artifact is
# fully resident there, so nothing else is moving.
step "F1-filler-low-147k" python bench/kv_sweep.py --ctx 147456 --quant v3-iq2xxs \
     --rounds 2 --fixed-text --filler low --arms q4_0,ngram-mod-short \
     --out kv-filler-147k.jsonl

step "F2-filler-low-16k" python bench/kv_sweep.py --ctx 16384 --quant v3-iq2xxs \
     --rounds 2 --fixed-text --filler low --arms q4_0,ngram-map-k,ngram-mod-short \
     --out kv-filler-16k.jsonl

echo "[$(date +%T)] q38 filler complete" >> "$LOG"
