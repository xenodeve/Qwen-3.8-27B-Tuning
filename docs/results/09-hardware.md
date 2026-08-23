# 09 — The machine itself, and what changed when the card did

**Every other file in this folder assumes one GPU. On 2026-08-23 that stopped
being true.** This page records which card produced which numbers, so a reader
can tell a stale figure from a current one without checking a date.

> 🔴 **Everything in files 01–08 was measured on the RTX 4070 SUPER 12 GB unless
> the row says otherwise.** `CLAUDE.md` forbids comparing raw decode across
> boots. Across hardware it is not a comparison at all — it is two different
> machines.

---

## The two cards

| | RTX 4070 SUPER | RTX 5060 Ti 16 GB |
|---|---|---|
| in service | until 2026-08-23 | from 2026-08-23 |
| VRAM, as the driver reports it | 12,281 MiB | **16,310 MiB** |
| VRAM, **as llama.cpp reports to the process** | **11,069 MiB**, in all 552 logs | **15,172 MiB** |
| compute capability | 8.9 (Ada) | **12.0 (Blackwell)** |
| memory bandwidth | ~504 GB/s | ~448 GB/s *(spec, not measured here)* |
| PCIe | gen4 x16 | gen5 x8 |

**The two VRAM rows are different measurements and only the second one matters
for `--fit`** — [`CORRECTIONS.md` §27](../reports/CORRECTIONS.md). The new card
confirmed it harder than the old one ever did: during one boot with a game
running, `nvidia-smi` reported **7,682 MiB** free while llama.cpp reported
**15,172**. A 7.5 GB gap on the same card at the same instant.

---

## What the new card allocates — measured 2026-08-23, ctx 98,304

One boot, `--spec-type ngram-mod`, corpus `real-code-deep`. **Byte-identical to
the old card on every buffer:**

| | 4070 SUPER | 5060 Ti |
|---|---:|---:|
| model, CUDA0 | 6,521.13 MiB | 6,521.13 MiB |
| model, CPU_Mapped | 397.85 MiB | 397.85 MiB |
| KV (16 attention layers, q4_0) | 1,728.00 MiB | 1,728.00 MiB |
| RS (`n_rs_seq = 0`) | 149.62 MiB | 149.62 MiB |
| compute (`-ub 256`) | 472.27 MiB | 472.27 MiB |
| split | `65+0` | `65+0` |
| `--fit` verdict | `no changes needed`, leaves **2,047** | `no changes needed`, leaves **6,150** |

**KV is 18.00 KiB per token** on this model at q4_0 with 16 attention layers.
That rate is flat and is what every projection below uses.

### What 16 GB actually buys

Projected from the measured rate, on this card:

| ctx | `ngram-mod` only | with DFlash2 (`n_rs_seq = 4`) |
|---:|---:|---:|
| 98,304 | 6,301 MiB free | 4,309 MiB free |
| 131,072 | 5,725 MiB free | — |
| 196,608 | 4,573 MiB free | 2,581 MiB free |
| **262,144** | **3,421 MiB free** | **1,429 MiB free** |

**262,144 is `n_ctx_train` for this model.** For the first time in this project
the ceiling is the model rather than the card. And the DFlash2 column matters:
on the old card the sidecar arms finished with **45–376 MiB** free and were
unreliable there ([`CORRECTIONS.md` §26](../reports/CORRECTIONS.md)); here the
same arms would have 1,429 MiB even at full native context — four to thirty
times that band.

**Whether DFlash2 now wins is unmeasured.** It lost on Ada because it competed
with the layers for a 12 GB budget. That constraint is gone; the verdict is not
transferable in either direction.

---

## ⚠️ The build targets the wrong architecture, and it is invisible

`cuobjdump --list-elf` on the shipped `ggml-cuda.dll` in **both**
`llama.cpp-cuda` and `llama.cpp-dflash2`:

```
ELF (SASS)   ggml-cuda.*.sm_89.cubin     <- Ada only
PTX          ggml-cuda.*.sm_89.ptx
```

The card is `sm_120`. The driver JIT-compiles the Ada PTX, producing kernels
tuned for neither architecture. **Measured, same corpus and flags as the old
card, at ctx 98,304:**

| | 5060 Ti (Ada PTX, JIT) | 4070 SUPER (native SASS) |
|---|---:|---:|
| prefill, 43,898 tokens | **146,155 ms** | 35,301 ms |
| decode | **22.67 tok/s** | 96.92 tok/s |

**Four times slower with three times the headroom, byte-identical allocation,
`65+0`, no OOM, and nothing in the log saying the kernels were JIT'd.** That is
the exact shape `CLAUDE.md`'s north star names: an instrument returning a
believable number instead of a failure.

**22.67 tok/s is not a property of this card and must not be quoted as one.**

**The fix is one flag.** CUDA 13.3 is installed and `nvcc --list-gpu-code` lists
`sm_120`; rebuild with `-DCMAKE_CUDA_ARCHITECTURES="89;120"`.

**The guard.** `scripts/worker-5060ti.ps1` reads the code objects out of
`ggml-cuda.dll` before launching and **refuses to start** on a binary without
`sm_120` SASS, naming what it found. Demonstrated against the real Ada build —
exit 1, not a warning. An override exists and prints
*"results are not comparable to anything"* when used.

---

## The healthy-load signature, which this project never had

`04-context-depth.md` recorded that `gpu-trace-98304.jsonl` showed **100 % GPU
utilisation at 4 % memory utilisation and 76 W** during the failing DFlash2 arms,
and noted there was **no control trace** to say whether that was abnormal.

There is one now. This card under a real 44K prefill and a 512-token decode:

```
GPU utilisation      99 %
memory utilisation   44 %
power             174.5 W   (TDP ~180 W)
```

**44 % memory utilisation is what work looks like on this model.** The old
signature — 100 % / **4 %** / 76 W — was a card spinning, not a card working,
which [`CORRECTIONS.md` §26](../reports/CORRECTIONS.md) argued from mechanism and
this now supports from a control.

---

## What transfers across the card change, and what does not

**Transfers — these are mechanisms, and a mechanism does not care which GPU runs
it:**

- `-ctk q4_0 -ctv q4_0` — no other KV type in this build has a fast kernel
- **`-cram` must never be 0** — 343× on task switching, and it caches sequence
  state in *host* RAM
- `--ctx-checkpoints` default 32 carries prefix reuse when `n_rs_seq = 0`
- an edit ahead of the suffix **zeroes** reuse rather than degrading it
- chars/token ≈ 3.4 — a property of the tokenizer and corpus
- `--fit` acts almost never, and reads a number `nvidia-smi` does not show

**Does not transfer — every rate, and every arm verdict:**

- 96.92 / 49.31 / 5.66 / 33.69 tok/s and the decoder ranking they produced
- the 45–376 MiB unreliability band
- the 13.6 % noise floor, and the 48.9 % spread at 65,536
- `-ub 64` costing 14.0 % of decode
- **`ngram-mod` as the right decoder** — it is the starting point here, not a
  verdict

*Raw: `logs/dflash2-hwbase-98304.log`, `bench/hardware_baseline.py`. Issue #40.*
