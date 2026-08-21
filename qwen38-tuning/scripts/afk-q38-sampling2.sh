#!/usr/bin/env bash
# Second pass on the sampling screen, at n=10 instead of n=3.
#
# WHY: the failure being measured is the TAIL of reasoning length, not its
# centre. The control's own three trials spanned 270 to 16,277 characters -- a
# 60x range -- so three samples cannot tell whether a config trims the tail.
# The first pass is a coarse filter; this is the measurement.
#
# Selection is automatic and ranked on TWO axes, in this order:
#   1. no-fence rate  -- the thing that actually breaks the corpus (41.5-58.3 %
#      of attempts emit no fenced code block at all)
#   2. max reasoning_chars -- the tail, i.e. how badly it runs away when it does
# Length and emission are separate properties: top-n-sigma cut the MEDIAN to the
# lowest in the field (1,281) and still hit 26,076 once. Ranking on length alone
# would have promoted it.
set -u
cd /c/AI/qwen38-tuning
LOG=logs/afk-driver.log
BENCH=/c/AI/qwen38-tuning/bench
SC=/c/AI/qwen38-tuning/scripts
until grep -q "q38 sampling screen complete (rerun)" "$LOG"; do sleep 60; done
log() { echo "[$(date +%T)] $*" >> "$LOG"; }

# grammar+rea-off is added unconditionally: pass one screened grammar with
# --reasoning-budget 0 and got content_chars = 0 on all three trials, so it can
# never be selected by a rank on no-fence rate -- yet the combination it points
# to is the one the whole programme was aimed at.
gen_grammar_rea_off() {
  python - <<'PYG'
import pathlib
tpl = pathlib.Path(r'C:\AI\qwen38-tuning\scripts\serve-v3-iq2xxs-flex.ps1').read_text(encoding='utf-8')
extra = ('--grammar-file C:\AI\qwen38-tuning\grammars\python-fence.gbnf -rea off')
tpl = tpl.replace("[string]$Extra = ''", "[string]$Extra = '%s'" % extra)
pathlib.Path(r'C:\AI\qwen38-tuning\scripts\serve-tmp-grammar-reaoff.ps1').write_text(tpl, encoding='utf-8')
PYG
}
gen_grammar_rea_off

PICK=$(python - <<'PY'
import json, collections
rows = [json.loads(l) for l in
        open(r'C:\AI\qwen38-tuning\results\answer-screen-sampling.jsonl',
             encoding='utf-8-sig')]
worst = collections.defaultdict(int)
nofence = collections.Counter()
seen = collections.Counter()
for d in rows:
    a = d.get('arm')
    if not a or d.get('gate'):
        continue
    worst[a] = max(worst[a], d.get('reasoning_chars') or 0)
    if not d.get('has_fenced', d.get('contract_ok')):
        nofence[a] += 1
    seen[a] += 1
cand = [a for a in worst if seen[a] >= 3]
# no-fence RATE first, then the tail. Never the median: see the comment above.
cand.sort(key=lambda a: (nofence[a] / seen[a], worst[a]))
print(' '.join(a.replace('samp-', '') for a in cand[:5]))
PY
)
PICK="grammar-reaoff $PICK"
log "sampling2 picked: $PICK"

for tag in $PICK; do
  log "START samp2-$tag"
  if bash "$SC/swap-model.sh" "$SC/serve-tmp-$tag.ps1" "UD-IQ2_XXS" \
        > "logs/q38-swap-samp2-$tag.log" 2>&1 \
     && python "$BENCH/answer_screen.py" --arm "samp2-$tag" --trials 10 \
        --out answer-screen-sampling2.jsonl > "logs/q38-samp2-$tag.log" 2>&1; then
    log "DONE  samp2-$tag"
  else
    rc=$?
    log "FAIL  samp2-$tag (rc=$rc)"
  fi
done
log "q38 sampling second pass complete"
