#!/usr/bin/env bash
# Third batch, 2026-08-21 ~03:35. Quality, not speed. Speed is now settled:
# ngram-map-k at 16K (+135.89 %) and ngram-mod at 131,072 (+200.22 %), both
# byte-identical to the unaccelerated output. The blocker is format.
#
# THREE ARMS, ONE FLAG APART EACH, against a control that already exists:
#
#   control (serve-v3-iq2xxs.ps1)   19/30 accepted   58.3 % contract pass
#   -rea off alone                  15/30            58.0 %      <- measured
#   grammar + -rea off              16/27            84.3 %      <- measured
#   grammar alone                     ?                ?         <- G1, here
#   ngram alone                       ?                ?         <- G2, here
#   grammar + ngram                   ?                ?         <- G3, here
#
# G1 is the top open question in report 22 section 7: the 26-point contract jump
# has never been attributed, because the only arm that showed it changed two
# things at once. If the grammar carries it alone, the shipping config keeps
# reasoning -- which is where the accepted-task count lives.
#
# G3 runs regardless of G1's verdict, because it answers a different question:
# whether a constrained sampler and a speculative drafter compose. Watch
# `acceptance` -- 96.9-99.0 % unconstrained. If it collapses, the two flags are
# an either/or rather than a pair.
set -u
cd /c/AI/qwen38-tuning
LOG=logs/afk-driver.log
BENCH=/c/AI/qwen38-tuning/bench
SC=/c/AI/qwen38-tuning/scripts
log() { echo "[$(date +%T)] $*" >> "$LOG"; }

summary() {  # summary <label> -- print the corpus verdict line into the driver log
  python - "$1" <<'PY'
import json, sys
lab = sys.argv[1]
rows = [json.loads(l) for l in open(r'C:\AI\qwen38-tuning\results\retry-bench.jsonl',
                                    encoding='utf-8')]
s = [r for r in rows if r.get('label') == lab and r.get('kind') == 'SUMMARY']
if not s:
    print('%s NO-SUMMARY' % lab); raise SystemExit(1)
v = s[-1]
acc = v.get('accepted_of_decided', '0/0')
n, d = (int(x) for x in acc.split('/'))
pct = v.get('output_contract_pct')          # PASS rate, not violations
rate = v.get('merged_tasks_per_hour')
ok = d and n / d >= 0.80 and (pct is None or pct >= 90.0)
print('%s accepted=%s contract_pass=%s tasks_per_hour=%s -> %s'
      % (lab, acc, pct, rate, 'ENOUGH' if ok else 'NOT-ENOUGH'))
raise SystemExit(0)
PY
}

arm() {  # arm <label> <serve script>
  local lab="$1" ps1="$2"
  log "START corpus-$lab"
  if bash "$SC/swap-model.sh" "$SC/$ps1" "UD-IQ2_XXS" \
        > "logs/q38-swap-$lab.log" 2>&1 \
     && python "$BENCH/run_retry_bench.py" --label "$lab" --passes 3 \
        --max-tokens 8192 > "logs/q38-corpus-$lab.log" 2>&1; then
    log "DONE  corpus-$lab"
    summary "$lab" >> "$LOG" 2>&1
  else
    rc=$?
    log "FAIL  corpus-$lab (rc=$rc)"
  fi
}

# --max-tokens 8192, not the 3072 default. An undersized budget looks exactly
# like lost capability and did so four times in one day: 15/31 at 3072 became
# 27/31 at 8192 on the SAME artifact. Every arm here gets the budget sized for
# the most verbose one, or the comparison measures verbosity.
arm "v3-iq2xxs-gram"       serve-v3-iq2xxs-gram.ps1
arm "v3-iq2xxs-ngram"      serve-v3-iq2xxs-ngram.ps1
arm "v3-iq2xxs-gram-ngram" serve-v3-iq2xxs-gram-ngram.ps1

log "q38 night3 complete"
