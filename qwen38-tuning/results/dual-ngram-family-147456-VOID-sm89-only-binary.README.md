# VOID — these 15 rows were taken on a binary with no Blackwell kernels

Run 2026-08-27. `dflash2_arena.py --arms dual-ngram-family --ctx 147456` was
launched with `QWEN38_TARGET` set and **`QWEN38_EXE` not set**, so it took the
module default `C:\AI\llama.cpp-dflash2\llama-server.exe`.

`build-dflash2/CMakeCache.txt` records `CMAKE_CUDA_ARCHITECTURES=89`. `cuobjdump
--list-elf` on its `ggml-cuda.dll` lists **141 `sm_89` cubins, no `sm_120a`**,
and `--list-ptx` lists **nothing** — there is no PTX to fall back on. Every one
of the 15 logs reads `CUDA : ARCHS = 890` while an RTX 5060 Ti of compute
capability **12.0** was visible and in use.

The served profile uses `C:\AI\llama.cpp-blackwell`, whose DLL carries **141
`sm_120a` beside 141 `sm_89`** and whose logs read `ARCHS = 890,1200` and
`BLACKWELL_NATIVE_FP4 = 1`.

**The rows looked right.** `66+0` residency, both cards holding memory, 19–26
tok/s at ctx 147,456, per-arm spreads of 1.3–3.7 %. **How** sm_89 cubins ran on
an sm_120 device is not established here. It does not need to be: these are not
the served binary, so they are not a measurement of the served configuration.

**Blast radius, audited the same hour.** 750 logs carry an `ARCHS` line — 191
read `890,1200`, 399 read the eight-architecture upstream default from the
single-card era, and 160 read `890`. Of those 160, **exactly these 15 had a
second CUDA device.** No historical dual-GPU row is affected.

**The hole is closed.** `dflash2_arena.start()` now compares the run's own
`system_info` line against the compute capabilities of the cards the arm can see
and exits on the **first** boot rather than collecting fifteen plausible rows —
`harness.archs_missing_for_gpus`, `gpu_device.visible_compute_caps`,
`bench/tests/test_the_binary_covers_every_visible_gpu.py`.

Kept, not deleted: a voided row is evidence about the instrument.
