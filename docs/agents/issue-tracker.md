# Issue tracker conventions

**Tracker:** GitHub — `xenodeve/Qwen-3.8-27B-Tuning`. Issues are the source of
truth; `docs/OPEN-WORK-LEDGER.md` is the discovery index over them and also
catches MD-only work that `gh issue list` cannot see.

## Bodies are bilingual

English, then a **full Thai mirror of the same depth**. A mirror is not a
summary. Identifiers — paths, flags, config keys, commit SHAs — stay English and
byte-exact inside the Thai text, because they are what the reader copies.

This applies to issue bodies, PRD bodies and PR descriptions. **It does not
apply to `docs/`**, which is English only.

## An issue for an experiment

Most work here is a measurement, so the body states:

- **the question**, phrased so it can come out either way
- **the arms** — flags, depth, artifact, rounds
- **what would falsify it** — the result that would mean "no"
- **where the row will land** — which `results/*.jsonl`

An issue that cannot name a result that would disprove it is not an experiment,
it is a plan to confirm something.

## Closing

**With evidence**: a commit SHA, a passing test, or a measured number naming the
file it came from. `state_reason` is always set. Never silently.

If the answer is "no", say so in the issue and keep it — a negative result that
is closed without its number teaches the next agent nothing and invites a re-run.
