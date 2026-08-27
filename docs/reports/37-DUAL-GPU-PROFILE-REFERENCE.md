# 37 — The dual-GPU profile: complete reference

**Written 2026-08-27 for a reader with no prior context.** Everything needed to
understand what this configuration is, what was changed and why, how far it got,
what it is stuck on, and what should be fixed next is contained here. No other
file needs to be open.

**Status in one paragraph.** A single machine with two mismatched consumer GPUs
serves a 27B model at a 147,456-token context, fully resident, at **26 tok/s
decode and ~971 tok/s prefill**. Getting there required one flag that is worth
+59.5 %, one computed argument without which the same configuration runs at
**0.38 tok/s**, and a launch-time budget check because llama.cpp's own fitting
code is not implemented for the mode we use. The configuration works and is in
daily use. **The remaining upside is blocked on a llama.cpp limitation for
which a patch now exists but has not yet been built or tested.**

---

## 1. The hardware, and the three facts that shape everything

```
CUDA0   RTX 4070 SUPER 12 GB   sm_89    PCIe gen4 x16 under load   DRIVES THE DISPLAY
CUDA1   RTX 5060 Ti    16 GB   sm_120   PCIe gen4 x4  under load   holds 50 MiB idle
```

28 GB across two cards, **`PXB` topology, no NVLink**, 20 CPU threads. An Intel
UHD 770 iGPU exists and drives nothing.

1. **CUDA0 is the older, smaller card, it enumerates first, and `--main-gpu`
   defaults to 0.** It is also the one drawing the desktop. Every card in this
   profile is therefore addressed **by UUID, never by index** — an index is a
   position in an enumeration the driver may reorder, and after a reorder it
   keeps working while meaning a different card.
2. **The 5060 Ti's slot is x4, measured under load** across 49 samples, 34 busy.
   PCIe *generation* downtrains at idle; *width* never does. The faster, larger
   card sits behind a quarter of the other card's bandwidth.
3. **The desktop's VRAM appetite is a live variable, 1,600–2,600 MiB.** It
   decided whether a 262,144-token context loaded or ran out of memory, twice,
   hours apart, on an otherwise identical configuration.

## 2. The build

`llama.cpp` **build 10499, commit `1deefcca3`**, `CMAKE_CUDA_ARCHITECTURES=89;120`,
Ninja, Release, `BUILD_SHARED_LIBS=ON`, `GGML_NATIVE=ON`, `GGML_CUDA=ON`,
`GGML_CUDA_FA_ALL_QUANTS=OFF`, `LLAMA_CURL=OFF`.

`cuobjdump --list-elf` on `ggml-cuda.dll` gives **141 `sm_120a` + 141 `sm_89`**
cubins and **no PTX**. Both architectures are required because one of the two
cards is Ada.

**This is load-bearing and the profile enforces it.** A binary without Blackwell
SASS still runs here — through PTX JIT where PTX exists — at **2.20× the prefill
time with nothing in any log saying so**. The profile refuses to start against a
`ggml-cuda.dll` that lacks the string, matched as a **substring** on purpose:
cmake rewrites `120` to `120a` and the cubins are named `sm_120a`, so an exact
`sm_120` test would reject a correctly built binary.

**A second build directory exists and is a trap.** `C:\AI\llama.cpp-dflash2` was
built with `CMAKE_CUDA_ARCHITECTURES=89` alone — 141 `sm_89` cubins, no
`sm_120a`, no PTX. See §11.2.

## 3. The artifact

**Qwen3.8-27B, Unsloth `UD-Q4_K_XL`, 16.69 GiB on disk**, `arch = qwen35`, 65
layers, `n_ctx_train = 262,144`, `n_swa = 0`, `n_head_kv = 4`,
`n_embd_head_k = n_embd_head_v = 256`.

**It is a hybrid**: 48 layers are Gated DeltaNet with a recurrent state separate
from the attention KV cache. With `n_swa = 0` it builds `llama_memory_hybrid`.
That single fact rules out an entire class of prefill optimisation — §10.3.

**It never fit on one 16 GB card at any depth.** Across two it is `66+0` — fully
resident — at every rung to 229,376, and spills one layer at 262,144. The second
card is worth **+79.9 %** [+77.3, +82.2] to it. That is the **residency cliff**
(`55+11` becoming `66+0`), not a parallelism gain: `UD-Q2_K_XL`, already resident
on one card, gained **1.5 %** from the identical change.

## 4. How it is run

Six double-click launchers at the repository root, on two independent axes —
one card or two, loopback or exposed — plus an MTP pair:

| launcher | cards | binds | notes |
|---|---|---|---|
| `serve.bat` | one | `127.0.0.1` | `UD-Q2_K_XL` profile |
| `serve-lan.bat` | one | LAN | |
| `serve-dual.bat` | **two** | `127.0.0.1` | carries `-MaxCtx` |
| `serve-dual-lan.bat` | **two** | LAN | carries `-MaxCtx` |
| `serve-dual-mtp.bat` | **two** | `127.0.0.1` | `-MaxCtx -Mtp` |
| `serve-dual-mtp-lan.bat` | **two** | LAN | `-MaxCtx -Mtp` |

Each `.bat` is ASCII, CRLF, no BOM, anchored on `%~dp0`, runs
`powershell -ExecutionPolicy Bypass -File "%~dp0serve.ps1" ...`, and pauses on a
non-zero exit. **A launcher holds no serving flag** — the profile owns them all,
so a `.bat` cannot drift from what was benchmarked.

`serve.ps1` is the entry point and holds no serving flag either. Its switches:
`-Dual`, `-MaxCtx`, `-Mtp`, `-Lan`, `-AllowFirewall`, `-Device`, `-Port`.

`qwen38-tuning/scripts/worker-q4-dual.ps1` (703 lines, most of it the reasoning
below) is the dual profile.

**One window, one server.** A Win32 job object with `KILL_ON_JOB_CLOSE`; closing
the console kills the server, verified by doing it. Nothing stands between
llama.cpp and the console, which is why its colours survive. There is **no
detach mode** — an agent cannot start a server that outlives its own process.

## 5. The invocation, with every flag

```
llama-server.exe
  -m <UD-Q4_K_XL>
  --alias Qwen3.8-27B-Q4_K_XL
  -c <computed at launch — §7>
  -ngl auto --fit on --fit-target 768
  -fa on
  -np 1
  -sm tensor  -ts <computed at launch — §6>
  -t 18  -b 2048  -ub 1024
  --no-mmproj-auto  -lv <3 or 5>
  -ctk q4_0 -ctv q4_0
  --spec-type ngram-mod
  --spec-ngram-mod-n-match 12 --spec-ngram-mod-n-min 16 --spec-ngram-mod-n-max 32
  --chat-template-file templates/qwen38-late-system.jinja
  --reasoning-effort medium
  --sse-ping-interval 5
  --host 127.0.0.1  --port 8080
```

**Measured on this exact configuration**, ctx 147,456, real vendor code, three
paired rounds with arms rotated: **26.2 / 25.6 / 26.7 tok/s**, own spread 4.2 %.
Prefill ≈ **971 tok/s** on a 6,621-token prompt at ctx 16,384.

Why each non-obvious flag is what it is:

- **`-sm tensor`** — **+59.5 %** at 16,384 over llama.cpp's default `layer`
  split ([21.1, 21.0, 19.9] → [32.4, 33.9, 32.3]) and **+65.4 %** at 147,456.
  Same residency ceiling either way. llama.cpp's own help marks it
  *"EXPERIMENTAL"*; it ships here on a measured number with that status stated
  rather than hidden. It fails harder at the ceiling: at 262,144 `layer` spills
  one layer and `tensor` fails to load outright.
- **`-ub 1024`** — decode is flat across 128/256/512/1024 (−1.1 %, −0.6 %,
  inside the floor). **Prefill is a clean staircase**: 820 / 884 / 938 / **971**
  tok/s on an identical prompt, **+10.1 %** with non-overlapping ranges. The
  single-card profile serves 256, chosen against one card; two cards change the
  arithmetic twice, because `-sm tensor` moves activations between the cards
  *inside every layer* and the link carrying it is gen4 x4. A wider micro-batch
  amortises each transfer over more tokens — the shape of a narrow link. Costs
  ~180 MiB of compute buffer.
- **`-b 2048`** — unchanged, and deliberately not moved with `-ub`: **`-ub`
  above `-b` is silently clamped**, so raising one alone would make some arms
  identical to their neighbours with nothing in the output saying which.
- **`-ctk q4_0 -ctv q4_0`** — `q8_0` is −0.3 % at 16,384, inside any floor. Its
  "cannot load at 147,456" verdict is **confounded**, see §11.3. KV costs a
  measured **18.00 KiB per token** at these types.
- **`-fa on`** — required; `off` loses residency. The boot log confirms
  `flash_attn = enabled`. `q4_0` is one of the four types this build compiles a
  FlashAttention kernel for (`f16`, `bf16`, `q4_0`, `q8_0`), so there is no
  dequant fallback in play.
- **`--spec-type ngram-mod`** — worth **+13.3 %** over no speculation at
  147,456 ([32.4, 32.6, 33.1] against [28.1, 28.1, 28.7]). It is the only
  speculative decoder that works on this split, see §10.1.
- **`-np 1`** — one user, one slot. `-np 0` throws; slots are built once at
  startup and cannot be resized.
- **`--sse-ping-interval 5`** — llama.cpp's default is 30, which is most of a
  minute of silence on a connection that is working. This does not shorten a
  wait; it makes one visible. A client showing "waiting for response" during a
  long prefill is **prefill, not the model thinking**.
- **`--fit on --fit-target 768`** — **inert under `-sm tensor`** (§6). Kept
  because every measured row carries them, and an argv that differs from the
  benchmarked one is not the benchmarked configuration.

## 6. `-ts` is computed at launch — the 0.38 tok/s incident

**This is the single most important thing in this document.**

`serve-dual-lan.bat` shipped without `-ts` and **decoded at 0.38 tok/s** — 85×
slower than the number the profile advertised. The 5060 Ti sat at 0 %
utilisation and 45 °C while the 4070 SUPER ran at 88 %, holding 11.6 of its
12.0 GB with 0.7 GB spilled into **shared host memory**. Prefill collapsed too:
16.4 tok/s against 973.

**Root cause, from llama.cpp's source.** `-sm tensor` **splits the model EVENLY
when given no ratio** — `llama-model.cpp:707`, `ne_s * (j+1)/n_devices`. These
cards are not even, and **the smaller one draws the display**. Straight off that
boot log, the Meta buffers being per card at 8,065 model + 1,296 KV + 1,024
compute = 10,385 MiB each:

```
RTX 4070 SUPER   12,282 total - 1,579 desktop - 10,385 =   +317 MiB
RTX 5060 Ti      16,311 total -     49 desktop - 10,385 = +5,876 MiB
```

**317 MiB is not headroom.** One browser tab put it over, the driver paged to
host memory, and every token crossed PCIe.

**`--fit` does not catch it.** The boot log says
`llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort`. This
profile's header previously called that *"a hard load failure … the better
failure of the two"*. **That was wrong** — it is a **silent spill** that returns
a working server at 0.38 tok/s, which is exactly the believable-wrong-number
failure this project exists to refuse, and the profile shipped it.

**The fix is not a hardcoded ratio**, because the desktop's appetite is not
constant. At launch the profile reads free VRAM per card by UUID, subtracts a
reserve, and passes the result:

```
for each card:
    isDisplay = (used > 500 MiB)          # whichever the monitor is plugged into
    reserve   = isDisplay ? 2500 : 512    # MiB
    budget    = free - reserve
-ts = the budgets, in CUDA enumeration order
```

**2,500 MiB, not 1,600.** The reserve has to cover the desktop's *growth*, not
its resting size: the incident happened with the desktop at 1,579 MiB and
317 MiB of slack.

Proportional-to-budget is what makes it safe — each card takes a share of the
model in proportion to what it can afford, so **they run out together** instead
of one spilling while the other has 6 GB idle.

Measured after the fix, same machine, desktop running:

| `-ts` | decode | 4070 SUPER free after |
|---|---|---|
| even (the default) | **0.38 tok/s** | +317 MiB |
| `2,3` | 31–33 tok/s | 1,511 MiB |
| `1,2` | 28–30 tok/s | 2,792 MiB |

The computed ratio lands near 1:2 on this machine.

**The tensor-split RATIO is not a lever for speed.** `-ts 1,1` against the
free-VRAM default measured +1.8 %, inside the floor. It is a lever for **not
spilling**, which is a different thing and worth 85×.

## 7. `-MaxCtx` — the deepest window that fits, computed at launch

`n_ctx_train` is 262,144 and a ladder measured it fully resident. It is still
**not** a context this machine can be relied on to serve, because the answer
moves with the desktop. Measured minutes apart, same machine, same profile:

```
desktop 1,600 MiB  ->  262,144 loaded
desktop 2,575 MiB  ->  262,144 OOM, 1,696 MiB on device 1
```

So `-MaxCtx` asks for the deepest context the *current* budget supports, capped
at `n_ctx_train`, and **spends the micro-batch before the context**:

```
1.  262,144 at the requested -ub
2.  262,144 at half the -ub      # frees ~1 -ub of MiB across the pair for
                                 # about 3.5 % of prefill (971 -> 938)
3.  the deepest ctx that fits at the halved -ub

budget = sum(free per card, less reserves) - RUNTIME_RESERVE_MIB
demand = WEIGHTS(16,130 MiB) + KV(ctx x 18.00 KiB/token) + COMPUTE(2 x ubatch MiB)
```

**`RUNTIME_RESERVE_MIB = 768` is a measured line, not a model.** A successful
load is not a successful run:

```
262,144 -ub 512, 336 MiB free on card 2  ->  DIED on the first request
                                             (cuMemSetAccess ... out of memory)
262,144 -ub 512, 488 MiB free on card 2  ->  survived 135,233 tokens
```

Every depth in this project is now re-tested with a **real 135,233-token
request**, not a `/health` probe. Two launches minutes apart chose **249,856**
and **245,760** tokens — that variance is the design working.

**The budget check is approximate on purpose, and optimistic.** `-ts` governs
the *weight* slice; KV and compute do not distribute by the same ratio, so an
approved run can still finish with less headroom than the reserve implies. **The
check refuses the impossible; it does not promise comfort.** 147,456 finishes
with ~2,900 and ~2,265 MiB free; 237,568 finishes with 996 and 412, which is the
same margin that produced 0.38 tok/s.

## 8. The guards — what refuses to start, and why

Eight `FATAL` paths, each from an incident:

| refuses when | because |
|---|---|
| the server binary is missing | — |
| the model file is missing | — |
| `ggml-cuda.dll` lacks `sm_120` **or** `sm_89` SASS | PTX JIT costs 2.20× prefill and says nothing |
| a named GPU UUID is not installed | an index would silently mean another card |
| `-Device` names other than two cards | this profile is two-card by construction |
| free VRAM cannot be read for a card | refusing to guess a split |
| `-MaxCtx` finds no context that fits with the runtime reserve | — |
| the requested ctx does not fit without spilling | `--fit` is inert here and llama.cpp will not refuse |

The last one names what *would* fit, and offers the `-UBatch` trade explicitly.

## 9. Modes

| switch | effect |
|---|---|
| `-Dual` | selects this profile instead of the one-card `UD-Q2_K_XL` one |
| `-MaxCtx` | §7. The four `dual` launchers pass it |
| `-Mtp` | adds `draft-mtp` beside `ngram-mod`. **Off by default** — it loads and runs, and its rate could not be measured: every paired round was voided because the generations copy the prompt rather than answer it |
| `-Lan` | binds `0.0.0.0` instead of loopback. An act, not a default |
| `-Device` | override the two UUIDs |

## 10. What is structurally blocked — constraints, not preferences

### 10.1 No externally-loaded drafter works under `-sm tensor`

`-sm tensor` builds a virtual **`Meta` device** aggregating both cards, and it
cannot host a second model.

- `draft-dflash` (a separate 1.06 GB file via `-md`) aborts at
  **`ggml-backend-meta.cpp:543`**,
  `GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0)` — a **graph-split
  axis** assertion inside `handle_per_row`, raised at ctx 16,384 with `-ub 128`
  where memory pressure is as low as this configuration goes. Unchanged by
  `-devd CUDA1` and by `--no-spec-draft-backend-sampling`. **Structural, not
  OOM.**
- `draft-mtp` (the head baked into the model) **does load**. Its earlier failure
  at `ggml-backend-meta.cpp:1522`, `GGML_ASSERT(bufs.back() != nullptr)`, was
  the 0.38 tok/s bug wearing a different error message and went away when the
  split stopped overcommitting the display card.

**This is the largest unexploited gap in the whole configuration.** `-sm layer`
+ `draft-dflash` + `ngram-mod` at ctx 16,384 measures **42.26 / 43.65 tok/s**
against **27.66 / 26.15** for `-sm tensor` + `ngram-mod` at the same depth — the
fastest configuration measured anywhere in this work, and unavailable at the
depth we serve. **Every depth between 16,384 and 147,456 is untouched**, so
where it stops working is unknown.

### 10.2 `-sm row` cannot load, and it is not this hardware

`device CUDA0 does not support split buffers`, at model load, every attempt.

**Cause, verified in source:** `ggml/src/ggml-cuda/ggml-cuda.cu` **does not
export `ggml_backend_split_buffer_type` at all**. `src/llama-model.cpp:982-999`
looks it up through `ggml_backend_reg_get_proc_address` and throws when the
lookup returns null. The CUDA row path was removed upstream. Not `sm_89` versus
`sm_120`, not `PXB`, not x4, not the mismatched pair.

### 10.3 `--cache-reuse` is unusable on this model, and llama.cpp does not detect it

**Read from source on `1deefcca3`, not measured.** It was the largest prefill
idea available — a broken prefix costs 63 s at 16K and 248 s at 64K here.

- The server disables `--cache-reuse` when `llama_memory_can_shift()` is false
  (`server-context.cpp:1176-1185`). For a hybrid that returns
  **`mem_attn->get_can_shift()` only** (`llama-memory-hybrid.cpp:133-136`),
  whose comment reads *"Shifting is trivially supported for recurrent"* — true
  of a **position**, false of a **state**. No warning is printed.
- The reuse loop calls `seq_rm(id, head_p, head_c)` then `seq_add`
  (`server-context.cpp:3180-3181`). The hybrid tries the recurrent side first
  and will not mutate the attention cache if it fails — but for a mid-sequence
  range it **does not fail**: `llama_memory_recurrent::seq_rm`
  (`llama-memory-recurrent.cpp:150-233`) takes neither special branch and falls
  through to `return true` **having touched nothing**.
- Net: attention KV re-indexed to the new prompt, **DeltaNet state still holding
  the old prefix**, no error. A removal reaching the tail instead takes a
  bounded-rollback branch needing `rollback <= n_rs_seq`, and ours is **0**, so
  it returns false and `GGML_ABORT`s — that path crashes rather than lying.
- **`n_rs_seq` has no command-line argument.** `need_n_rs_seq()`
  (`common/common.h:386-392`) sets it only for `draft-mtp`, `draft-eagle3`,
  `draft-dflash`, `draft-dspark` — **zero for every `ngram-*` type**. `qwen35`
  *is* in `llm_arch_supports_rs_rollback`, so the capability exists and is never
  provisioned for us, and would not cover the mid-sequence path anyway.

**Do not set `--cache-reuse`.** `--slot-prompt-similarity` dies with it, since
it only decides whether a slot is reused at all.

### 10.4 Flags that are dead or inapplicable here

| flag | why not |
|---|---|
| `-mg` / `--main-gpu` | `--help` scopes it to `-sm none` or `-sm row` |
| `-dt` / `--defrag-thold` | deprecated no-op — `common/arg.cpp:2522-2531` prints a warning and does nothing |
| `--draft-max`, `--draft-min`, `--draft-n`, `--spec-ngram-size-n/-m`, `--spec-ngram-min-hits` | **removed** upstream; they call `arg_removed()` and abort startup |
| `--spec-draft-n-max` | governs a **file-loaded** drafter; none loads here |
| `--spec-draft-p-min <= 0.0625` | mathematically identical to 0.00 — `1/sum ∈ [1/16, 1]` by construction |
| `GGML_CUDA_GRAPH_OPT` | measured inert; its body contains no `cudaGraph*` call |
| `--no-repack`, `--no-op-offload`, `--load-mode`, `--no-host`, `--swa-full`, `--no-kv-unified` | all measured inert |
| thread affinity, process priority, polling, GPU-side sampling | +0.46 %, −2.02 %, +0.69 %, +2.27 % — all inert |
| `--lookup-cache-static/-dynamic` | the state behind `ngram-cache`, which is **disqualified**: its greedy hash differs from a same-depth baseline, so it changes the answer |
| CPU offload of a drafter | **−59 %** and worse-than-GPU, against an external prediction of +70–85 % |

## 11. Where it is stuck, and what is known to be wrong

### 11.1 The DFlash2 block now has a patch, unbuilt and untested

The assertion in §10.1 has a mechanism and a workaround, **verified against this
repository's own copy of llama.cpp at `1deefcca3`**:

| what | where | what it says |
|---|---|---|
| the mapping | `src/llama-model.cpp:517-524` | `output.weight` → `SPLIT_AXIS_1`; `output.bias` → `SPLIT_AXIS_0` |
| the assertion | `ggml-backend-meta.cpp:541-544` | `handle_per_row`, body is `GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0)` |
| the target axis | `ggml-backend.h:369` | `GGML_BACKEND_SPLIT_AXIS_MIRRORED = 10, // all values on all backends` |
| the arch to scope by | `llama-arch.h:154` | `LLM_ARCH_DFLASH`, name `"dflash"` |

A per-row op receives `output.bias`, which is axis 0, and aborts. **Mirroring
the output projection is the reported fix**, and a git worktree at
`C:\AI\llama.cpp-mirror` — detached at the same commit — carries it, saved as
`qwen38-tuning/patches/dflash-mirror-output-1deefcca3.patch`, 28 insertions, 0
deletions. It is **scoped to `LLM_ARCH_DFLASH` only**, narrower than the
reported workaround, so the target model's split is byte-for-byte unchanged and
every rate measured on `UD-Q4_K_XL` stays comparable against the patched binary.

**`MIRRORED` is not free** — it duplicates the tensor on both devices.

**Nothing has been built or run.** The probe order is: rebuild → verify both
architectures are in the DLL → ctx 16,384 `-ub 128` with `draft-dflash` alone →
**compare greedy output against no-speculation token for token**, because
speculative decoding is supposed to be lossless and a patch that changes the
text is a corruption, not a workaround → record per-card free VRAM → only then
ladder the depth → only then pair it with `ngram-mod`.

**Prediction, stated before the run:** a gain is likely at 16,384 and **unlikely
at 147,456**, because the same-day sweep below showed better drafting failing to
become throughput at that depth, and because DFlash2 costs 1,936 MiB resident
against ~2,210 MiB free per card there.

### 11.2 An instrument fault found and closed the same day

`dflash2_arena.py`'s default binary is `C:\AI\llama.cpp-dflash2`, which was built
`CMAKE_CUDA_ARCHITECTURES=89` and **has no Blackwell kernels and no PTX**. A
sweep launched without overriding it produced 15 rows at ctx 147,456 with `66+0`
residency, both cards holding memory, plausible rates and per-arm spreads of
1.3–3.7 % — **every log reading `CUDA : ARCHS = 890` while a compute capability
12.0 card was visible and in use**. How they ran is not established; that they
are not the served binary is, and that alone voids them.

**Blast radius, audited:** of 750 logs carrying an `ARCHS` line, 191 read
`890,1200`, 399 read the eight-architecture upstream default from the
single-card era, 160 read `890` — and of those, **exactly the 15 from that sweep
had a second CUDA device.** No historical dual-GPU row is affected. The rows are
kept with a written diagnosis beside them.

**Closed:** `harness.archs_missing_for_gpus` compares the run's own
`system_info` line against the capabilities of the cards the arm can see, and
the arena now exits on the **first** boot. Observation, not prediction — reading
cubins needs `cuobjdump` at a hardcoded path and describes what a process
*would* load; the `ARCHS` line is what it *did*. The override variable is
**`QWEN38_LLAMA_EXE`** (binary, model and effort are three separate levers on
purpose).

### 11.3 Two verdicts in the register are weaker than they read

- **`q8_0` KV "cannot load at 147,456" is CONFOUNDED, not disproved.** That run
  reports `Meta() model buffer size = 8,065.29 MiB`, and 8,065.29 × 2 =
  16,130.58 MiB is the model exactly — **it ran the even split**. The arithmetic
  says it should fit with a computed `-ts`: 5,184 MiB KV + 16,130 weights +
  ~2,048 compute + 768 reserve = 24,130 against 26,072–27,072 available. Not
  re-run.
- **The `dual-decoder` arm set carries no `-ts`**, so its 147,456 rows ran the
  even split. Report 36 §4 records those numbers and says they are *"recorded
  here only so nobody quotes them as current"*, while the register quotes them
  as the decoder verdict.

### 11.4 A comparison guard that was missing until today

The arena recorded each arm's layer split and printed it, and **`report()` never
read it** — so an arm that silently spilled to the CPU was paired against a
resident baseline and the difference attributed to whatever the arm varied.
`harness.residency_note` now refuses a delta between arms at different splits,
and names both.

## 12. What should be fixed or tried next, ranked

1. **Build and probe the DFlash mirror patch** (§11.1). The only item that could
   move the served configuration from `tensor + ngram-mod` at ~26 tok/s to
   `tensor + DFlash2 + ngram-mod`.
2. **A/B `GGML_CUDA_ALLREDUCE`.** Verified at
   `ggml/src/ggml-cuda/ggml-cuda.cu:1207-1243`: it accepts `nccl` / `internal` /
   `none`, and **the default is platform-dependent** — Linux initialises NCCL,
   everything else `internal`. **On Windows we have been running `internal` all
   along without knowing.** `internal` against `none` isolates how much of
   tensor mode's +59–65 % is the optimised collective rather than the tensor
   decomposition, and needs no hardware change. This is an **environment
   variable**, which is why a diff of all 322 CLI flags against the 20 this
   profile sets never found it.
3. **Recurrent checkpoints instead of `--cache-reuse`.** `--ctx-checkpoints`
   snapshots state during prefill; restoring the checkpoint nearest *before* an
   edit and re-prefilling only the tail fits this workload — a large immutable
   prefix followed by a small changing region. Never measured as a
   prefill-restore mechanism here; the existing row measured it for
   **residency**, which is a different question.
4. **Give `-sm layer` the same launch budget arithmetic `-sm tensor` has.** It
   survives by luck. Needed before any DFlash2 depth ladder, which runs in
   layer mode.
5. **`-b` and `-ub` together at 2048** — the only raw prefill knob whose trend
   has not been run to its end. Single-digit percent.
6. **KV type with a computed `-ts`**: `f16` / `bf16` / `q8_0` / `q4_0` at
   16,384, and `q8_0` at 147,456 to settle §11.3. `f16` was ruled "not
   measurable" under a **12 GB single-card** constraint that no longer applies —
   at 16,384 it is 1,152 MiB.
7. **Move the display to the Intel UHD 770.** Frees 1,600–2,600 MiB
   on the card that `-ts` must reserve for, and removes the live variable that
   decides whether a deep context loads. **It invalidates comparisons**, so it
   is a one-shot decision to make before or after a campaign, never during.
8. **Do NOT swap the PCIe slots expecting a bandwidth answer.** This board has
   one x16 and one x4; swapping moves which card is starved and **leaves an x4
   endpoint on the path between them**. It is not an x4-versus-x16 inter-GPU
   A/B, and a result from it must not be written up as one.

## 13. What has never been measured

- **Quality, on this project's own artifacts, at all.** Every argument for
  `UD-Q4_K_XL` over the smaller `UD-Q2_K_XL` rests on an external
  bits-per-weight ladder and an outside campaign. **Neither is our number**, and
  with decode now at parity this is no longer a trade-off to justify — it is
  simply the last unmeasured thing.
- **Any depth between 16,384 and 147,456.** A verdict at one depth does not
  transfer here: `draft-mtp` is **+81 % at 16K and −71 % at 131,072** on the
  same artifact.
- **`draft-mtp`'s rate on this configuration.** It loads; every paired round was
  voided because the generations copy the prompt.
- **One card against two for any speculative rate.** Splitting changes the
  reduction order, so the logits, so the text — and a speculative rate is partly
  a measure of how predictable the text is.

---

*Sources for every figure: `docs/results/` (the register), `docs/reports/36`
(the dual-GPU narrative), `docs/reports/CORRECTIONS.md` (33 retracted claims),
`docs/agents/traps.md` (the ways of working that failed here), and the 703-line
header of `qwen38-tuning/scripts/worker-q4-dual.ps1`, where most of the
reasoning above lives beside the code it governs.*
