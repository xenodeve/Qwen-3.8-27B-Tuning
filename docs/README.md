# docs — what we know, what we plan, what we were told

Six folders, and the difference between them matters. Beyond the four below,
`agents/` holds the operating standard — including
[`agents/traps.md`](agents/traps.md), item 3 of the session-start list — and
`adr/` holds architecture decisions.

| folder | what is in it | how much to trust it |
|---|---|---|
| [**`reports/`**](reports/) | **our own measurements** and what they mean | every number was measured on this machine and names the file it came from |
| [**`results/`**](results/) | **a register: has X been tried, and what happened** | one row per thing tried, each pointing at the raw file |
| [`plans/`](plans/) | what we intend to run, and briefs sent to external researchers | intent, not results |
| [`researchs/`](researchs/) | **external** material — deep-research replies, vendor docs, model cards | **unverified.** Four claims from it have already been measured wrong |

---

**Before quoting any number from `reports/`, read**
[`reports/CORRECTIONS.md`](reports/CORRECTIONS.md) — twenty-eight claims this project
published and later contradicted. `python scripts\audit-stale-claims.py` finds
every line in the tree that still matches one.

---

## → Start at## → Start at [`reports/START-HERE.md`](reports/START-HERE.md)

It covers the whole project in one document: the machine, the one mechanism that
explains most results, what was done in what order, where things stand, every
lever measured, and what is still open.

Then [`reports/README.md`](reports/README.md) is the index of all 33 numbered
reports.

---

## The rule about `researchs/`

**Nothing in that folder is evidence until it is measured here.** The record so
far, from [`reports/17`](reports/17-EXTERNAL-RESEARCH-REVIEW.md) and
[`reports/18`](reports/18-RESEARCH-ROUND2-REVIEW.md):

| external claim | measured here |
|---|---|
| MoE CPU-offload is a large win; the artifact is 20.6 GiB | the artifact is **10.02 GiB**; the config lost **46–48 %** |
| retry success `p2 ≈ 0.93` | **0.20–0.625** |
| asymmetric KV saves ~25 % VRAM | **no kernel** — prefill 29× slower, cache 44 % *larger* |
| `--ctx-checkpoints 8` frees ~900 MiB | frees **10–16 MiB** |
| drafter on CPU gives +70–85 % | **−59 %** |
| a recommended model file's exact byte count | the file **does not exist** |

The mechanisms in those replies were often sound. The numbers attached to them
were not. **Keep the mechanism, delete the number, measure it.**
