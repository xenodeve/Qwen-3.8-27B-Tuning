#!/usr/bin/env bash
# Stage D. Does anything fix the format failure? Corpus, 30 tasks, the only
# instrument with a real sample size.
#
# The screen has done its job and its limit is now known: answer_screen caps at
# len(PROBES) = 3, so --trials 10 silently gives 3. It cannot separate configs
# whose effect is on a tail. It CAN reject configs that are broken, and it did:
# `--grammar-file` with `--reasoning-budget 0` returned content_chars = 0 on
# every trial (the model reasons freely, then ends its turn at the point the
# grammar starts to bind), and `--reasoning-budget 0` alone ran to 24,709
# characters despite being documented as an immediate stop.
#
# THREE ARMS, ONE ARTIFACT. V3 UD-IQ2_XXS is the largest Qwen3.8-27B artifact
# that holds 65+0 at 131,072 (report 19), and its unconstrained corpus already
# exists as the control: 19/30 accepted, 58.3 % contract violations.
#
#   1. -rea off                  -- does disabling reasoning alone fix it?
#   2. --grammar-file + -rea off -- does the grammar add anything on top?
#
# Screened at n=3 the two were indistinguishable (content 314-539 vs 314-537
# chars). Thirty tasks is the sample size that can tell them apart, and unlike
# the screen it measures whether the code RUNS, not just whether it appeared.
set -u
cd /c/AI/qwen38-tuning
LOG=logs/afk-driver.log
BENCH=/c/AI/qwen38-tuning/bench
SC=/c/AI/qwen38-tuning/scripts
until grep -q "q38 sampling second pass complete" "$LOG"; do sleep 60; done
log() { echo "[$(date +%T)] $*" >> "$LOG"; }

enough() {  # enough <label>
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
pct = v.get('output_contract_pct')
# output_contract_pct is the PASS rate, not the violation rate:
#   100 * (attempts_seen - contract_violations) / attempts_seen
# It was read as a violation rate all through 2026-08-20. The error was
# safe in one direction only -- an inverted gate passes nothing -- but it
# also meant every report of it said the opposite of the truth.
ok = d and n / d >= 0.80 and (pct is None or pct >= 90.0)
print('%s accepted_of_decided=%s contract_violation_pct=%s -> %s'
      % (lab, acc, pct, 'ENOUGH' if ok else 'NOT-ENOUGH'))
raise SystemExit(0 if ok else 1)
PY
}

run_arm() {  # run_arm <label> <serve script>
  local lab="$1" ps1="$2"
  log "START corpus-$lab"
  if bash "$SC/swap-model.sh" "$SC/$ps1" "UD-IQ2_XXS" \
        > "logs/q38-swap-$lab.log" 2>&1 \
     && python "$BENCH/run_retry_bench.py" --label "$lab" --passes 3 \
        --max-tokens 8192 > "logs/q38-corpus-$lab.log" 2>&1; then
    log "DONE  corpus-$lab"
  else
    rc=$?
    log "FAIL  corpus-$lab (rc=$rc)"; return 1
  fi
  if enough "$lab" >> "$LOG" 2>&1; then
    log "STOP  $lab is good enough"
    return 0
  fi
  log "NEXT  $lab not good enough"
  return 1
}

run_arm v3-iq2xxs-reaoff serve-v3-iq2xxs-reaoff.ps1 \
  || run_arm v3-iq2xxs-fmt serve-v3-iq2xxs-fmt.ps1

log "q38 quality sweep complete"
