#!/usr/bin/env bash
# Layer 12 (sampling) and layer 13 (chat protocol), screened with answer_screen.
#
# These levers do not change tok/s in any way worth measuring -- they change
# whether the model finishes a thought and emits an answer, which is the exact
# failure this project is stuck on (41.5-58.3 % of corpus attempts produce no
# fenced code block). answer_screen costs four minutes and measures precisely
# that, so it is the right instrument for a first pass. A corpus run costs
# 30-90 minutes and is reserved for whatever survives here.
#
# Every config is V3 UD-IQ2_XXS at 16K, one flag group changed against a control
# in the same sweep. Serve scripts are generated per config from a template so
# nothing but -Extra differs.
set -u
cd /c/AI/qwen38-tuning
LOG=logs/afk-driver.log
BENCH=/c/AI/qwen38-tuning/bench
SC=/c/AI/qwen38-tuning/scripts
until grep -q "q38 decoder sweep complete" "$LOG"; do sleep 60; done
log() { echo "[$(date +%T)] $*" >> "$LOG"; }

gen() {  # gen <tag> <extra flags>
  local tag="$1"; shift
  local extra="$*"
  python - "$tag" "$extra" <<'PY'
import sys, pathlib
tag, extra = sys.argv[1], sys.argv[2]
tpl = pathlib.Path(r'C:\AI\qwen38-tuning\scripts\serve-v3-iq2xxs-flex.ps1').read_text(encoding='utf-8')
tpl = tpl.replace("[string]$Extra = ''", "[string]$Extra = '%s'" % extra)
pathlib.Path(r'C:\AI\qwen38-tuning\scripts\serve-tmp-%s.ps1' % tag).write_text(tpl, encoding='utf-8')
PY
}

screen() {  # screen <tag> <extra flags>
  local tag="$1"; shift
  gen "$tag" "$@"
  log "START samp-$tag"
  if bash "$SC/swap-model.sh" "$SC/serve-tmp-$tag.ps1" "UD-IQ2_XXS" \
        > "logs/q38-swap-samp-$tag.log" 2>&1 \
     && python "$BENCH/answer_screen.py" --arm "samp-$tag" --out answer-screen-sampling.jsonl \
        > "logs/q38-samp-$tag.log" 2>&1; then
    log "DONE  samp-$tag"
  else
    log "FAIL  samp-$tag"
  fi
}

# control first, and again at the end -- two controls bracket the sweep so a
# drift between them is visible instead of being attributed to a flag.
screen base
# L12 anti-loop. All three are OFF by default in this build, and
# --repeat-last-n defaults to 64 tokens against loops of 4,000-8,000.
screen dry        "--dry-multiplier 0.8 --dry-base 1.75 --dry-allowed-length 2 --dry-penalty-last-n 4096"
screen rep4096    "--repeat-penalty 1.05 --repeat-last-n 4096"
screen rep-default "--repeat-penalty 1.05"
screen nsigma     "--top-n-sigma 1.0"
screen mirostat   "--mirostat 2 --mirostat-ent 5.0 --mirostat-lr 0.1"
# L13 reasoning control
screen rbudget2k  "--reasoning-budget 2048"
screen rbudget0   "--reasoning-budget 0"
screen rea-off    "-rea off"
screen prefill    "--prefill-assistant"
# the two the whole programme is aimed at, and their combination
screen grammar    "--grammar-file C:\AI\qwen38-tuning\grammars\python-fence.gbnf --reasoning-budget 0"
screen dry-rb2k   "--dry-multiplier 0.8 --dry-penalty-last-n 4096 --reasoning-budget 2048"
# L12 backend sampling -- must be checked for greedy equivalence, not just speed
screen backend    "--backend-sampling"
screen base2
log "q38 sampling screen complete (rerun)"
