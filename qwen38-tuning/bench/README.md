# bench — the harness

**253 tests. Run them before trusting any number:**

```powershell
python -m pytest tests\ -q
```

Every primitive here raises rather than guessing. That is deliberate: this
project has published twenty-seven numbers it later had to withdraw, and each came from
code that returned something plausible instead of an error.

| file | what it measures | typical cost |
|---|---|---|
| `harness.py` | every summarising primitive — paired deltas, retry economics, layer-split parsing, output-contract checking | — |
| `model_arena.py` | paired-boot comparison between artifacts at 16K | ~1 min/boot |
| `ctx_ceiling.py` | **deepest fully-resident context.** Split only, no generation | **~1 min/boot** |
| `kv_sweep.py` | throughput at a fixed depth across arms. 48 arms defined | 1–3 min/boot; ~8 at 131,072 |
| `answer_screen.py` | 4-minute gate: does the artifact finish a thought and emit an answer | 4 min |
| `run_retry_bench.py` | **the decision metric** — 30 coding tasks, one evidence-assisted retry | 30–90 min |
| `depth_sweep.py` | context ladder with cold prefill. Owns `QUANTS`, the pinned artifact paths | expensive |
| `residency_check.py` | shared vs dedicated GPU memory during real generation | ~2 min |
| `stability_gate.py` | 100 turns with a forced prefix invalidation every tenth | ~30 min |
| `protocol_gate.py` | nested tool call and `tool_call_id` round-trip | ~10 min |
| `greedy_diff.py` | the actual greedy text, not just its hash | ~5 min |
| `kv_kernel_screen.py` | which KV types have a fast kernel | ~10 min |
| `dflash2_arena.py` | the decoder arena — every `--spec-type` and its settings, paired within a round | 1-3 min/boot |
| `real_task_bench.py` | **real GitHub issues in throwaway clones** — the project's own metric | 25-40 min/task |
| `gpu_trace.py` | VRAM, power, clocks and utilisation on an interval, attached to a run it did not launch | negligible |
| `edit_canary.py` | can the worker EDIT an existing tracked file? Found `CORRECTIONS.md` §24 | ~30 s |

---

## What gets deleted after a benchmark, and what never does

The developer's standing rule is *"ลบ code ที่พ่นตอน benchmark ให้หมดด้วย"* —
delete all the code the benchmark spat out. **It means the code a model wrote
while being measured, not the thing doing the measuring.** The line has been
misread three times, twice in the direction that would have destroyed the
evidence, so it is written down here.

| | examples | what happens to it |
|---|---|---|
| **Model output — always deleted** | worker clones under the scratch root; `bench/_work/`, `bench/_deepwork/` | deleted and **verified gone**; `bench/_*/` is gitignored |
| **The instrument — never deleted** | `dflash2_arena.py`, `kv_sweep.py`, `harness.py`, `ARM_SETS`, `tests/` | it is the apparatus. Deleting it means no future run is comparable |
| **The evidence — never deleted** | `qwen38-tuning/results/*.jsonl`, `bench/corpora/*.txt` | **85 result files, cited by 20 documents.** `CLAUDE.md`: *"A measurement names the file its number came from, or it is a hypothesis"* |

**Why the third row is not negotiable.** Deleting the JSONL rows does not tidy
the repo, it demotes **every published number in it to a guess** — including
`+34.6 %`, `+48.5 %`, and the finding that the two winners cancel. A corpus
file is evidence too: its hash is stamped into every row measured against it,
which is what makes those rows interpretable a month later.

**Enforcement.** `harness.assert_deletable` + `PROTECTED_ROOTS` bound what a
cleanup may touch; `tests/test_scratch_safety.py` pins it;
`tests/test_no_committed_worker_output.py` pins that no model output is ever
committed; `tests/test_corpus_frozen.py` pins that the corpus cannot drift.

**The reason it keeps being misread** is that both things live in this
directory and both are "benchmark stuff". The test is not where a file sits —
it is **who wrote it**. A model wrote it → it goes. A person wrote it, or a run
recorded it → it stays.

---

## Three things that will bite you

**Artifacts are pinned by byte count** in `depth_sweep.QUANTS`, and `cached()`
**raises** when a filename is ambiguous across snapshots. Unsloth replaced every
file in its repo in place on 2026-08-19 — same names, different contents. Before
that guard existed, `-hf repo:Q2_0` fetched `PQ2_0.gguf`, a different file of
*identical* byte count.

**`kv_sweep.ARMS` is where new levers go.** Each entry is a label mapped to extra
server flags, stacked on the `q4_0` control so every row in a sweep is
comparable. **Verify a flag parses before adding it:**

```sh
llama-server <your flags> -m /nonexistent.gguf --port 18080
```

An argument error appears immediately; a correct flag reaches model loading.

**`ctx_ceiling.py` kills whatever listens on port 8080** and does not take the
swap lock. Never run it beside another job.

---

## The 13.6 % floor — and the depth it was measured at

🔴 **It is a ctx 16,384 number.** At 65,536 the same arm, with per-implementation
counters byte-identical across rounds, spans up to **48.9 %** between boots — so
at depth this floor resolves pure drift as an effect. `CORRECTIONS.md` §23.

`paired_deltas()` refuses to call an effect real below **13.6 %** or with an
inconsistent sign across rounds. That number is measured: the same control
config spans 32.4–42.5 tok/s across 25 boots, because free VRAM at boot moves
9,326–10,732 MiB and `--fit` follows it.

A mean of `+9 %` with a range of `+0.2 .. +19 %` is the control drifting, not a
result.

---

## The two flat constants, removed 2026-08-21

A wait that does not know what it is waiting for is not a bound, it is a
coincidence. Both of these held for weeks only because no arm had yet been slow
enough to test them.

| was | now | why |
|---|---|---|
| `post(..., timeout=3600)` | `harness.completion_timeout_s(ctx)` -- `ctx * 0.8 / 60 tok/s + 300 s` | a flat hour spent 01:34:36 to 02:34:36, exactly, on an arm whose prefill had collapsed to 8.56 tok/s and could never finish |
| `kill()` ending `time.sleep(5)` | `depth_sweep.wait_for_vram_release()` on `harness.vram_settled(..., floor_mib=free_before + 1024)` | WDDM frees 12 GB in stages; the next server started into a full GPU, passed `/health`, and died on its first request |

Verified live at 03:15 on 2026-08-21: tearing down an 11,501 MiB server took
**9.87 s** to return 9,924 MiB. The constant it replaces was 5 s.

The **VRAM floor** is the less obvious half. "Stopped moving" cannot tell
*release finished* from *release has not begun* -- two polls taken before the
driver acts agree perfectly. So `kill()` reads free VRAM first and requires the
reading to beat it by 1,024 MiB, a tenth of the smallest artifact here (7.80
GiB). If nothing was listening on 8080 it does not wait at all.

The floor of 60 tok/s is chosen against measurements, not taste: four times
below the slowest **legitimate** prefill at 131,072 (240.6) and seven times
above the **pathological** one (8.56).

`depth_sweep.run()` now wraps the whole measurement in `try/finally`, so a raise
can never again leave a 12 GB server resident for the next queue step, and a
request that exceeds its budget is written as a row
(`note="abandoned after 2048s: ..."`) rather than raised. **An arm too slow to
finish inside its depth budget is a result.**

Full incident: `docs/reports/04-MEASUREMENT-METHODOLOGY.md` section 8.
