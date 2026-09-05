# 10 — other engines: has an engine other than llama.cpp been run on this artifact's weights, and what happened?

> 🔴 **Reasoning effort:** not applicable on this page. Every row here is a raw
> `/completion`-style continuation of the arena corpus with no chat template,
> so there is no thinking block and no effort field — the same shape
> `bench/dflash2_arena.py` sends llama-server. **Every number is from
> 2026-09-03 and its rows are in `qwen38-tuning/results/exl3-decode.jsonl`**
> (`argv`, versions, card, `copied_frac` and `timing_source` per row).
> **Every decode figure on this page was recomputed on 2026-09-03 after an
> instrument fault** — the harness subtracted `time_prefill` from a
> `time_generate` that already excludes it, overstating every warm round by
> ~3–5 % (`CORRECTIONS.md` §47). The rows keep the old value as
> `decode_tok_s_v1_overstated`.

## ExLlama3 (EXL3), Mia-AiLab fork — issue #71

**Tried: yes, end to end, on this machine.** Weights
`Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` (15.36 GB, built-in MTP head), runtime
`MiaAI-Lab/exllamav3` @ 63b32f0 built from source against torch 2.11.0+cu130
with nvcc 13.3 and `triton-windows` 3.8.0 (how, and the four traps, in
[`researchs/exllama3-platform-2026-09-03.md`](../researchs/exllama3-platform-2026-09-03.md)).
Prompt = `bench/dflash2_arena.filler(ctx, "real-code-vendor")`, greedy
(`-temp 0 -topk 1`), `N_PREDICT` 512, three rounds, one process per arm.
Rounds with `copied_frac > 0.5` are **void** as decode measurements per
`bench/harness.generation_is_original`; they are listed and struck.

| arm | cards | ctx (tokens) | KV | MTP | decode tok/s, rounds 1–2 | prefill tok/s, cold | draft acc/rej | note |
|---|---|---:|---|---|---:|---:|---|---|
| A | 5060 Ti | 14,122 | NVFP4 | on | **50.6** (one row); ~~48.7, and the cold round~~ | 487 | 404/6 | two of three rounds `copied_frac = 1.0` — the model continued the corpus verbatim; only round 1 (`copied 0.084`) counts |
| K | 5060 Ti | 14,122 | NVFP4 | off | ~~14.6~~ | 655 | — | all three rounds `copied_frac = 1.0`; void, but it bounds the base decode of the kernel at this depth |
| I | 5060 Ti | 30,265 | NVFP4 | on | **21.1** | 549 | 328/385 | |
| J | 5060 Ti | 30,265 | **fp8** | on | **33.9–35.7** | 655 | 348/284 | **+63 % over I from the KV format alone** — NVFP4's online dequant is the largest single cost found |
| F | 5060 Ti | 61,717 | NVFP4 | on | **13.7–14.4** | 402 | 340/324 | 13.6 GB used |
| G | 5060 Ti | 61,717 | NVFP4 | off | **5.5** | 421 | — | MTP is worth 2.5x here; the base kernel decode at this depth is 5.5 tok/s |
| — | 5060 Ti | 98,304 | NVFP4 | on | OOM in prefill | — | — | model 12.4 GB + KV 2 GB + EXL3's 170 MiB fp16 weight materialisation per matrix during prefill; **single-card ceiling with NVFP4 KV lies between 65,536 and 98,304** |
| H | 5060 Ti | 61,717 | fp8 | on | "Insufficient VRAM in split" | — | — | fp8 at cache 81,920 = 2.5 GB does not fit beside the model on 15.5 GB |
| D | both, layer split (`-gs 9,15.5`) | 144,022 | NVFP4 | on | **8.3–9.5** | 191 | 367/189 | the served depth; `copied 0.0` all rounds |
| E | both, **tensor-parallel** (`-tp -tpb native`) | 144,022 | NVFP4 | on | **10.7–12.4** | 280 | 342–366/198–314 | TP beats layer split by ~30 %; no NCCL needed on Windows |
| L | both, tensor-parallel | 144,022 | **fp8** | on | **21.4–22.6**; re-run quiet display: **20.8–23.7** | 483 / 485 | 330–346/295–375 | the two winning levers together: **2x arm E**, and prefill 483 against E's 280 |

### The lever the first ladder missed: decode graph capture only runs on upstream's cache formats

`EXL3_BC_ATTN_TRACE=1` on the arms above: with NVFP4 or fp8 KV, **BC-attn (the fork's whole-step CUDA-graph decode path) DECLINES on 16 of 17 attention layers**; `build_bc_attn` accepts only `CacheLayer_fp16` and `CacheLayer_quant` (`bc_attn.py:430`), and the fork's own fp8/NVFP4 layers fall to the eager dispatch path. With fp16 KV all 17 build. That is the "100 % SM, 3–33 % memory bandwidth" signature: launch- and sync-bound, not KV-bound. Upstream's integer quant cache (`-cq k[,v]`, `cache/quant.py`) costs per token per attention layer `128·bits + 64` bytes for K and the same for V, so **`-cq 4` is 18 KiB/token — the same footprint as NVFP4 — and still graph-captured.**

| arm | cards | ctx | KV (`-cq`) | KiB/token | BC built | decode tok/s, rounds 1–2 | prefill cold | note |
|---|---|---:|---|---:|---:|---:|---:|---|
| I | one | 30,265 | nvfp4 | 18 | 1/17 | 21.1 | 549 | |
| J | one | 30,265 | fp8 | 32 | 1/17 | 33.9–35.7 | 655 | |
| cq8 | one | 30,265 | 8 | 34 | **17/17** | **41.5–42.0** | 643 | |
| cq6 | one | 30,265 | 6 | 26 | 17/17 | **44.1–44.4** | 626 | best point at 30K |
| cq4 | one | 30,265 | 4 | 18 | 17/17 | **41.0–41.6** | 634 | NVFP4's footprint, 2x its speed |
| cq8,4 | one | 30,265 | 8,4 | 26 | 17/17 | ~~56.4–63.5~~ | 640 | all rounds `copied_frac 0.992` — void; the asymmetric cache changed what the model wrote |

At depth, same discipline (BC built 17/17 on one card, 33/33 across two):

| arm | cards | ctx | KV | decode tok/s, rounds 1–2 | prefill cold | vs the NVFP4/fp8 arm at the same point |
|---|---|---:|---|---:|---:|---|
| M | one | 61,717 | `-cq 4` | **34.4–38.6** | 567 | F (NVFP4) 13.7–14.4 → **2.5x** |
| **N** | two, TP | 144,022 | `-cq 4` | **31.8–32.6** | 480 | L (fp8) 21–23 → 1.5x; E (NVFP4) 11–12 → 2.7x |
| O | two, TP | 144,022 | `-cq 6` | 30.7 | 486 | 4 bits edges 6 at depth, the reverse of 30K |

Drafter and prefill knobs on top of N (two cards, TP, `-cq 4`, 144,022), one process per arm:

| arm | change | decode tok/s, rounds 1–2 | prefill cold | draft acc/rej |
|---|---|---:|---:|---|
| N | `-ndt 4` (MTP default) | 31.8–32.6 | 480 | 332–336/345–364 |
| P | `-ndt 2` | 32.1–35.3 | 498 | 284–304/105–165 |
| **Q** | `-ndt 3` | **34.5–37.2** | 497 | 312–326/217–274 |
| R | `-dds` dynamic draft | 31.2–31.8 | 495 | 279–287/210–213 |
| S | `max_chunk_size` 4096 | 29.4–30.8 | 501 | 322–330/374–415 |

S changes only prefill chunking and should leave decode untouched; its 29.4–30.8 against N's 31.8–32.6 is the **same-arm drift across processes at this depth, ~8–10 %**. **Paired re-run, order rotated (N, Q, N, Q), rounds 1–2:**

| pair | `-ndt 4` | `-ndt 3` | delta |
|---|---:|---:|---:|
| 1 | 30.3 / 31.7 | 35.4 / 37.3 | **+17 %** |
| 2 | 29.2 / 29.0 | 32.7 / 34.6 | **+15 %** |

Same sign in both rotations and each delta above the same-arm drift *as measured at that hour*; per-arm spread inside a process 3–6 %. **Weakened later the same day:** four recipe controls run over two hours spread 30.2–35.9 (19 %, passes 3–5 below), which is wider than either pair's delta, so `-ndt 3` rests on the two rotated pairs agreeing in sign plus llama.cpp's identical verdict, not on clearing the spread. **`-ndt 3` is adopted for the depth recipe** — the same verdict llama.cpp gave `--spec-draft-n-max 3` at this depth (results 02). Acceptance explains it: at 144K the fourth draft token is rejected more often than it is accepted (acc/rej ~330/370 at `-ndt 4` against ~320/230 at 3). Prefill did not move with chunk size (501 against 480, inside that drift).

**Recipe after two optimisation passes:** `-cq 4 -mtp -ndt 3 -tp -tpb native -gs 9,15.5 -cs 163840`, with `triton-windows` and the DSA guard: **33–37 tok/s at 144,022 tokens on two cards, 2.5 GB of KV** (against llama.cpp's 39.4–42.6 on different boots — a ~15 % gap, at the edge of the same-arm drift measured below; prefill ~495 against ~830 is not).

### Passes 3–5 (2026-09-03, 13:15–15:24): every remaining lever on the recipe, paired and rotated

Recipe = `-cq 4 -mtp -ndt 3 -tp -tpb native -gs 9,15.5 -cs 163840`, 144,022
tokens, one process per arm, rounds 1–2, **decode already corrected per
§47**. The recipe itself was re-run four times through the afternoon as the
control (`EXL3_ARM=base1..4` in the rows):

| control | decode r1 / r2 | prefill | draft acc/rej |
|---|---:|---:|---|
| base1 13:20 | 31.6 / 31.5 | 497 | 310/282, 311/277 |
| base2 13:39 | 30.2 / 35.6 | 493 | 302/316, 332/195 |
| base3 14:32 | 31.4 / 31.7 | 492 | 308/290, 310/282 |
| base4 15:11 | 35.2 / 35.9 | 495 | 321/240, 325/224 |

**Same-arm spread with byte-identical argv: 30.2–35.9, i.e. 19 %.** It is not
only timing noise: the prompt is the same bytes and the sampler is greedy, yet
the accepted/rejected draft counts differ per process (rej 195–316), so the
**draft/verify sequence is not deterministic across processes** — whether the
final 512 tokens differ was not recorded (the rows keep `copied_frac`, not the
text); the native TP backend's reduction order is the likely cause,
unverified — and a round that happens to reject fewer drafts decodes faster. Every arm below is judged
against that 30–36 band, not against one control.

| lever | arm(s) | decode r1 / r2 | prefill | verdict |
|---|---|---:|---:|---|
| `-tp_linear_attn 1` | la1_a, la1_b | 33.0 / 33.7 · 34.5 / 34.6 | 515 · 516 | inside the band; prefill +4 % both times, too small to adopt |
| `EXL3_TP_SPIN_RECV=5` | spin5_a, spin5_b | 32.2 / 34.3 · 32.2 / 34.8 | 493 · 489 | inside the band |
| `-gs 12,15.5` (more on the 4070) | gs12_a, gs12_b | 28.0 / 26.1 · 35.2 / 31.0 | **320** · 545 | rejected: the two runs disagree by ~23 % and one prefill is −35 % |
| `-gs 7,15.5` (less on the 4070) | gs7_a, gs7_b | 31.5 / 32.1 · 33.2 / 32.6 | **423** · **459** | rejected: decode inside the band, prefill −7 to −14 % both times |
| `EXL3_QC_PREFILL_NS=1` / `=2` | qcns1_a, qcns2_a | 34.8 / 33.7 · 33.4 / 37.4 | 486 · 496 | no prefill effect — the fork already auto-picks the stage count per kernel family |
| `EXL3_QC_STAGING=0` | stag0_a, stag0_b | 37.7 / 35.0 · 37.2 / 34.2 | **432** · **444** | rejected: prefill −10 to −12 % both times (the source comment says so); decode at the band's top edge but the knob does not touch decode |
| `EXL3_INT8_GEMV=0` | gemv0_a | 36.8 / 35.1 | 496 | inside the band, one run |
| `-cq 5` | cq5_a, cq5_b | 33.5 / 31.9 · 34.3 / 34.7 | 479 · 495 | inside the band; 22 KiB/token for nothing |
| `-cq 3` | cq3_a | **38.3 / 36.9** | 473 | above the band in one run — confirmed or refuted in pass 6 below |

### Pass 6 (15:24–15:48): `-cq 3` against `-cq 4`, three rotated pairs

| pair | `-cq 4` r1 / r2 | `-cq 3` r1 / r2 | delta (mean) | `-cq 3` draft acc/rej |
|---|---:|---:|---:|---|
| cq3_a vs base4 | 35.2 / 35.9 | 38.3 / 36.9 | +6 % | 330/201, 322/233 |
| cq3_b vs base5 | 33.5 / 36.2 | 34.1 / 33.6 | −3 % | 318/250, 315/264 |
| cq3_c vs base6 | 33.3 / 35.0 | 41.4 / 36.0 | +13 % | **341/158**, 316/260 |

**Not adopted.** The sign flips across pairs, the mean delta (+5 %) is inside
the 19 % same-arm spread, and the one round that reached 41 tok/s is the one
whose drafter was rejected 158 times instead of ~260 — the acceptance-rate
variance again, not the cache. `-cq 3` also halves the KV precision of a
model whose quality has not been measured on any EXL3 arm, so even a real
+5 % would be a trade needing a quality number first. **Prefill unchanged**
(488–492 against 492–497).

**Where the recipe stands after six passes:** `-cq 4 -mtp -ndt 3 -tp -tpb
native -gs 9,15.5 -cs 163840` — six controls over 2.5 h, **30.2–36.2 tok/s at
144,022, prefill 489–497**. Ten further levers were tried today (`-tp_linear_attn`,
`EXL3_TP_SPIN_RECV`, two `-gs` ratios, `EXL3_QC_PREFILL_NS` 1/2, `EXL3_QC_STAGING=0`,
`EXL3_INT8_GEMV=0`, `-cq 5`, `-cq 3`) and none moved decode outside the
control band or prefill upward. The levers that are left are not flags:
DFlash2 as drafter (needs the Mia 5.0bpw download), the served sampler and
thinking, quality, and **the llama.cpp arm in the same boot**.

### The same-boot pairing, first attempt (18:54–19:32): three of four legs VOID — a game was running

Rows: `results/exl3-vs-llamacpp-147456.jsonl` (llama.cpp, arena arm set
`nvfp4-served`, byte-identical to the served config, pinned by
`tests/test_served_arm_set_is_the_served_argv.py`) and the `pair_e1` /
`pair_e2` rows of `exl3-decode.jsonl`. Order L1 → E1 → L2 → E2, 147,456 /
144,022 tokens, same corpus, greedy.

| leg | engine | decode, three rounds | prefill | free VRAM before |
|---|---|---:|---:|---:|
| L1 18:54 | llama.cpp | **38.8 / 41.9 / 45.2** | — | 25,196 MiB |
| E1 19:02 | EXL3 recipe | ~~13.5 / 15.0 / 15.0~~ | 392 | |
| L2 19:11 | llama.cpp | ~~29.3 / 24.8 / 25.2~~ | — | 24,319 MiB |
| E2 19:24 | EXL3 recipe | ~~24.6 / 24.0 / 24.2~~ | 453 | |

**Why the last three are struck.** `Riot Client` started at 18:56:29 (process
start time, read at 20:19) and the developer was playing through the window;
both engines fell together — EXL3 to half its afternoon band (30–36), llama.cpp
to 60 % of its own first leg — free VRAM before L2 was 877 MiB lower than
before L1, and EXL3's prefill dropped from ~495 to 392/453. A contention
signature on both engines at once, not a property of either. L1 alone matches
the recorded 39.4–42.6 and is kept as a llama.cpp point in this boot; **it is
not a pair**. Re-run below with the machine idle.

### The same-boot pairing, second attempt (20:20–20:48, machine idle): **EXL3 decodes at ~81 % of llama.cpp**

Rows: `results/exl3-vs-llamacpp-147456-run2.jsonl` (llama.cpp, arm
`nvfp4-served`, `tg_med` = median of three generations per round, cold one
included) and the `pair2_e1` / `pair2_e2` rows of `exl3-decode.jsonl`
(rounds 1–2). Order L1 → E1 → L2 → E2; llama.cpp at ctx 147,456 with 144,022
prompt tokens of `real-code-vendor`, EXL3 on the same tokens; both greedy,
`N_PREDICT` 512; `copied_frac` ≤ 0.03 everywhere; free VRAM before every
llama.cpp boot 25.6–26.0 GB.

| leg | engine | decode tok/s | notes |
|---|---|---:|---|
| L1 20:20 | llama.cpp `NVFP4-MTP-VERY-LOW` + `draft-mtp,ngram-mod` nm24, `-sm tensor` | 40.1 / 42.5 / 43.6 | acceptance 58.8 %, split 66+0 |
| E1 20:28 | EXL3 recipe `-cq 4 -mtp -ndt 3 -tp -tpb native -gs 9,15.5` | 36.9 / 31.9 | prefill 501; acc/rej 343/149, 309/286 |
| L2 20:34 | llama.cpp, same | 42.7 / 43.3 / 44.3 | |
| E2 20:42 | EXL3, same | 35.8 / 33.9 | prefill 495; acc/rej 323/229, 313/270 |

| | llama.cpp | EXL3 | EXL3 / llama.cpp |
|---|---:|---:|---:|
| range | 40.1–44.3 | 31.9–36.9 | |
| median of legs | 43.0 | 34.9 | **0.81** |
| best round | 44.3 | 36.9 | 0.83 |
| worst round | 40.1 | 31.9 | 0.80 |

**Verdict, same boot, rotated, both engines inside their own recorded bands
(llama.cpp 39.4–42.6 in results 02; EXL3 30.2–36.2 across six controls):
at the served depth EXL3 decodes ~19 % slower than llama.cpp, and its
prefill (~500 tok/s) is ~60 % of llama.cpp's (~830, `results/02-decoders.md`,
not re-measured in this pairing — the arena does not record it).** The
distributions do not overlap: EXL3's best round (36.9) is below llama.cpp's
worst (40.1). This closes the question the spike asked. What EXL3 keeps:
7.7 s load, 18 KiB/token KV (2.5 GB at 144K against llama.cpp's ~2.4 GB
free after its own q4_0 cache), a cross-job prompt cache. What is still not
measured on EXL3: quality, thinking, the served sampler, DFlash2 as drafter.

### Pass 7 (2026-09-04, 06:09–07:56): the DFlash2 drafter (`-dm`) against the MTP head — no gain, and it cannot run on the recipe at all

`Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw` (1.4 GB, block 8), rows `pair*`
→ `ls_*`, `ls30_*`, `sc_*` in `exl3-decode.jsonl`; host RAM and GPU
utilisation sampled every 10 s from 07:52 (`ramwatch.log`, min free 18.9 GB).

- **On the recipe (`-tp`) it does not load:** `DFlash2Model does not yet
  support tensor-parallel targets` (`architecture/dflash2.py:162`). Why, and
  the 30–40-line fix the fork already ships for DeepSeek-V4, in
  [`researchs/exl3-drafters-under-tensor-parallel-2026-09-04.md`](../researchs/exl3-drafters-under-tensor-parallel-2026-09-04.md).
- **One card at 30K:** `Insufficient VRAM in split for model and cache` — model
  12.4 GB + drafter 1.4 GB + its fp16 draft KV do not fit beside a 40,960 cache
  on 15.5 GB.
- **Layer split (`-gs 9,15.5`, no `-tp`), the only shape it runs in:**

| ctx | pair | MTP `-ndt 3` r1 / r2 | DFlash2 r1 / r2 | delta | DFlash2 draft acc/rej |
|---:|---|---:|---:|---:|---|
| 32,768 | a | 45.5 / 48.3 | 40.4 / 44.9 | **−9 %** | 329/889, 345/761 |
| 32,768 | b | 46.1 / 48.1 | 43.7 / 45.5 | **−5 %** | 348/740, 354/695 |
| 147,456 | a | 29.1 / 29.4 (prefill 460) | ~~2.3 / 2.4~~ (prefill 45, cold round 54 min) | VOID | 325/924, 328/899 |

**Verdict: not adopted, and the TP port is not worth writing.** Same sign in
both rotated pairs at 30K; the drafter proposes 7 tokens a step and has 700–900
rejected against the MTP head's 190–260, so its extra verify width is paid for
and not recovered — the same shape as llama.cpp's `draft-dflash` at this depth
(+0.2 %, `results/nvfp4-dflash-147456.jsonl`). The VENDOR +33–43 % over MTP
was measured on a GB10 at 273 GB/s, where a 1.4 GB drafter's weight reads are
cheap relative to the target's; on two PCIe cards they are not.

**The 144K row is VOID for the reason the developer spotted, not for what it
says about the drafter:** during that arm two Python processes committed over
20 GB each with the page file active and both GPUs sat under 30 % with
periodic 0 % (developer's Task Manager, 07:5x); the sampler started afterwards
shows a *single* MTP harness at 30K committing **22.2 GB** (working set 4.8
GB), so two of them plus the drafter at 144K exceed the 47.7 GB of host RAM.
That is an instrument finding — the harness's host footprint is a variable no
row before today recorded — and the row is struck rather than read.

**Nothing in passes 3–5 moves prefill up; three levers move it down.** Prefill
at ~490 against llama.cpp's ~830 remains the gap no flag on this side closes.

**With `-cq 4` the served-depth arm reaches ~81 % of llama.cpp's decode in the same boot** (34.9 against 43.0, medians of the rotated pairing above; the earlier cross-boot figure was ~75–80 %) with a KV footprint llama.cpp cannot match. **Prefill is unchanged by the cache type (~480 against ~830) and is now the larger gap.**


**Against the served llama.cpp profile** (`NVFP4-MTP-VERY-LOW` + `draft-mtp,ngram-mod`, two cards `-sm tensor`, ctx 147,456: **39.4 / 42.6 / 42.6 tok/s** paired, prefill ~830–855 tok/s on the same cards, `results/02-decoders.md` and `logs/serve-20260902-160749.log`) — **different boots, so no paired delta**, but the best fp8 arm at depth (L, 22–23 tok/s) sits at **~55 % of llama.cpp's decode and ~57 % of its prefill**, a gap far outside the 13.6 % floor, and EXL3 at 14K on one card (51.5) is in the same band as llama.cpp at 16K on one card.

**What the ladder says.** Decode falls 51 → 21 → 14 → 11–13 from 14K to 144K on the same kernels, and removing MTP at 62K drops it to 5.5; changing the KV format from NVFP4 to fp8 at 30K adds 63 %. So the cost is the attention over a long paged cache, and inside that the NVFP4 online-dequant path: fp8 KV doubles the two-card depth arm (E 11–13 → L 22–23), the drafter hides part of the rest, and the tensor split recovers ~30 % over layer split. The remaining ~1.8x against llama.cpp at 144K has no further flag on this side that was tried. Nothing here is a flag sweep on llama.cpp's side, and nothing was paired within a boot against llama.cpp — a same-day llama.cpp run on the same cards is the step that would turn the ~1.8x into a verdict.

**Instrument caveat on every two-card arm (D, E, L), found after they ran:** while the developer watched a video with RTX Video Super Resolution, `nvidia-smi dmon` on an otherwise idle machine showed the **4070 SUPER at 40–41 % SM, 28 % memory bandwidth, 4 % decoder, 137 W** for five consecutive seconds; the 5060 Ti sat at 0 %. Arms D, E and L split the model across both cards and therefore competed with that load; the single-card arms (A, I, J, F, G) ran on the 5060 Ti alone and did not. **Re-run of L with RTX VSR off and the 4070 at 1–6 % SM (`exl3-decode.jsonl`, the later three L rows): decode 20.8 / 23.7 / 22.6, prefill 485 — the same as with the video playing (21.4–22.6 / 483). The contention did not move the number; the 40 % SM the display card was spending was not on EXL3's critical path.** During the run `nvidia-smi dmon` showed both cards at 100 % SM with memory-bandwidth utilisation only 3–33 %, which says the depth cost is compute or inter-card synchronisation (the native TP backend busy-waits), not KV bandwidth. The served llama.cpp dual profile is exposed to the same contention whenever a video plays; it is a variable no row in this repo has recorded before.

**What EXL3 does better, measured:** load in 7.7 s; NVFP4 KV at 18 KiB/token (2.5 GB at 144K, against fp16's 9 GB); a cross-job prompt cache that makes rounds 2–3 prefill in 0.2–2 s.

**Not done:** quality on any of this (the critical path everywhere in this repo); a chat-templated run with thinking; DFlash2 as drafter (`-dm <dir>`), which the fork also serves; a llama.cpp arm in the same boot.

### Artifact change (2026-09-04, 15:08–16:05): turboderp's SC 4.00bpw H5 replaces the 3.5bpw file, at 262,144 with split 9,15.5

**Why.** turboderp's KL sweep (VENDOR, mirrored at `Mia-AiLab/Qwen3.8-27B-EXL3`,
one self-generated trace) puts EXL3-SC 4.00 bpw H5 at **KL 0.0062 / PPL 1.3424**
at 12.2 GiB of quantised weights. The 3.5bpw file this page was measured on has
no KL figure anywhere, and neither does the served llama.cpp file (esatapedico
`VERY-LOW`, a second quantisation of Unsloth's NVFP4 with `Q3_K` head / `Q2_K`
embeddings — the chart's 0.0092 is Unsloth's source, not that file). The
developer chose the one artifact in reach with a published figure. Branch
`SC_4.00bpw_H5`, 16.70 GB on disk (+1.36 GB over 3.5bpw), same 2,383-tensor set
including the MTP head; the recipe is unchanged.

**Fits, but not at the split G shipped with.** One 197,020-token prompt built
from `docs/**/*.md` (distinct files, sized with the server's own
`count_tokens`), streamed through `/v1/messages`, `max_tokens` 64, effort low:

| boot | file | cache | `-gs` | VRAM idle | VRAM after 197K | prefill | decode at 197K | outcome |
|---|---|---:|---|---|---|---:|---:|---|
| `exl3-serve-20260904-150808` | SC 4.0 H5 | 262,144 | 10,15.5 | 10.1 / 11.2 GB | 11.9 / 12.3 GB during | 317 tok/s by the end (700 → 61 pp_3s) | — | prefill completed, then `## Synchronization timeout in kernel: pg_all_reduce_cpu_kernel` ×12 at the first decode step; the client got an `error` event after 761 s |
| `exl3-serve-20260904-152849` | 3.5bpw (control) | 262,144 | 10,15.5 | 8.9 / 10.0 GB | 11.5 / 12.1 GB | **474.5** tok/s | **33.1** tok/s (60 tokens) | passed |
| `exl3-serve-20260904-154658` | SC 4.0 H5 | 262,144 | **9,15.5** | 8.4 / 11.9 GB | 11.0 / 13.4 GB | **431.7** tok/s | **25.0** tok/s (60 tokens) | passed, no sync timeout; Claude Code round-trip after it: 745 tok/s prefill on 30.8K, 49.4 tok/s decode |

So the failure was the 4070 SUPER at 11.9 of 12.3 GB, not depth: the control
passed the same prompt at the same cache, and moving ~1.7 GB of weights to the
5060 Ti (cap 9 instead of 10) made the 4.0bpw file pass with 1.3 GB to spare.
`-gs` under native TP is a weight cap per card, not a total; the KV cache lands
by its own rule. **Both G launchers now pass `9,15.5`**, the same split as F.

**Two attempts in between were invalid and are not evidence:** a `9,15.5` boot
and a `245,760` boot both OOMed at load with `-gs 10` and host `15.5` — the
comma in `"9,15.5"` had been split by cmd because the script was called from a
Bash tool call, where the shell strips the quotes before cmd sees them (the
same defect `test_exl3_launcher_args_survive_cmd.py` guards in the launchers).
The trial wrapper now passes the caps through an environment variable.

**What the numbers are and are not.** One 60-token decode sample per boot at a
depth this page had never reached; the 3.5bpw control and the 4.0bpw run are
different boots, so the 33.1 vs 25.0 gap (−24 %) is a hypothesis of the size
of the price — 9 % more weight bytes per token plus a 4070 that now holds less
of the model — not a paired measurement. The 147K pairing on this page was
taken on the 3.5bpw file and is **not** transferred to 4.0bpw. Quality of the
new file is VENDOR only; the three-artifact gate in the ledger is still open.

**The same timeout came back at `9,15.5` (2026-09-04 ~21:00, boot
`exl3-serve-20260904-201512`).** So "the 4070 at 11.9 of 12.3 GB, not depth"
above is weakened: cap 9 moved the failure, it did not remove it. The sequence
in the log: a Claude Code session with **two agents** alternated a 130K
conversation and a 43K one on the single cache slot (task 31: 129,280 cached;
task 32: 43K; task 33: the 130K prompt again, now a full re-prefill). Task 33
printed **no progress line at all** before `## Synchronization timeout in
kernel: pg_all_reduce_cpu_kernel` ×336 and `RuntimeError: CPU reduce process
timeout` in every TP child — the first forward pass stalled at the CPU
all-reduce past the GPU-side deadline (`SYNC_TIMEOUT 2 × 45 s = 90 s`,
`exllamav3_ext/parallel/timeout.cuh`) and the CPU-side 45 s. The stream died
after 265 s; Claude Code then retried ten times at ~6 min intervals, each
queued behind the dead generation, and all ten answered 500 together at
21:57:43 (`logs/exl3-requests.jsonl`). After the children die the server keeps
answering `/health` `ok` and 500s every completion in 0.4 s — a **restart is
the only recovery**, and `/health` does not say so (open: report the TP
children in `/health`).

Observed alongside, not shown to be the cause: with the server idle the 4070
SUPER sat at **11,682 of 12,282 MiB** — the model's ~9 GB plus ~2.6 GB of
desktop processes (`nvidia-smi` lists Wallpaper Engine, Overwolf, Discord, two
browsers, the ChatGPT and Claude desktop apps on the card). Both timeouts so
far happened with that card within 0.6 GB of full; the one pass at 197K had
1.3 GB to spare. Hypothesis, untested: WDDM paging on a full 4070 stalls the
pinned-memory reduce path past 90 s. The test is cheap — close the desktop GPU
processes, repeat the two-agent 130K/43K alternation, and look for the line.
Not a VRAM figure to plan by until then.

### The server's own guards (2026-09-05 → 06): a watchdog, a loop guard, and a Han ban — issues #75, #76, #77

Three faults that only showed up once Claude Code ran against the EXL3 server
for whole days, and what the server now does about each. All three live in
`qwen38-tuning/serving/exl3/` as sibling modules of the fork file, with their
tests in `bench/tests/test_exl3_{watchdog,loop_guard,cjk_guard}.py`.

| fault | measured | what the server does now | instrument |
|---|---|---|---|
| **TP children die, `/health` stays `ok`** (#75). After `pg_all_reduce_cpu_kernel` sync timeouts every completion 500s in 0.4 s; a human noticed after an hour | 2026-09-04 21:00 and 2026-09-05 (two boots) | `watchdog.check(e)` on the fatal signatures writes `logs/exl3-restart.flag` and exits 3; `serve-exl3.cmd` relaunches after any exit that was not asked for, refuses a held :8000, gives up after three deaths within 420 s; a `/health` self-probe catches a deaf server; `stop-exl3.cmd` is the only intended stop | the flag's reason line; the relaunch log |
| **Runaway repetition under the window-sized output cap** (#76). A Thai report locked onto tone mark U+0E48 for 127,996 tokens / 46 min; a thinking phase cycled three sentences ~1,000 times to the same count | 2026-09-05 13:19 and 19:03, 4.0bpw H5 at medium | `loop_guard.feed()` on every text chunk: ≤ 2 distinct characters in the last 512 of content, or a prose unit of 64 characters ×8 in the last 4,096 of thinking → `generator.cancel`, `finish_reason length`, `timings.stop_reason = "loop"`. Replayed over the 43 bench streams: trips on the two runaways only | `/health.loops_stopped` |
| **Chinese characters mid-Thai-sentence** (#77). 14 Han characters in 3 of 43 streams (`โมเดล前沿…`, `协作`, `这套`): sampling drift where the Thai continuation is diffuse, which no prompt line reaches | 2026-09-05 bench streams | `cjk_guard`: every vocab piece with a Han character (55,328 of 248,044) at `-inf` in `ComboSampler`'s `logit_bias`, thinking included, unless the prompt carries Han or names Chinese (`จีน`/`china`/`chinese`/`mandarin`) or `EXL3_ALLOW_CJK=1`. Live 2026-09-06 06:11: five Thai briefs of the leaking shape 0 Han each at 49–61 tok/s; a `你好` prompt 219 Han; "แปลเป็นภาษาจีน" with no Han typed answered in Chinese | `timings.cjk_chars`, `/health.cjk_chars_total` |

**What is proven and what is not.** The Han ban is proven by construction plus
the live pairs above: the mask test decodes the real `tokenizer.json`
independently and checks every Han piece is in the mask and no other piece is,
so a Han character in a banned completion would be a bug in the mask, not a
model quirk. The loop guard's thinking rule was fitted on the two incidents and
replayed on 43 streams, but **has not yet tripped live**. The watchdog's relaunch
has run (the 2026-09-05 relaunch storm is what added the held-port refusal) but
the cause of the sync timeouts is still issue #63's open hypothesis A — the 4070
within 0.6 GB of full behind desktop tenants. None of the three exists for the
llama.cpp profiles: llama.cpp has no per-token ban that scales to 55K tokens,
and the loop and dead-child modes have not been seen there.
