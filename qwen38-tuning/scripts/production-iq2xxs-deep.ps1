<#
DEEP PROFILE - Qwen3.8-27B UD-IQ2_XXS with Q8_0 KV

Measured 2026-08-19 (report 02 s7), Q8_0 KV, speculation off:

    ctx     split     decode      prompt proc   cold prefill   KV      (KV type)
    64K     61 + 4    15.81 t/s   727 t/s        64.0 s        2 040 MiB  q8_0
    128K    47 + 18    5.15 t/s   474 t/s       196.2 s        3 264 MiB  q8_0
    128K    58 +  7    7.84 t/s   672 t/s       138.6 s        2 016 MiB  q4_0  <- now default
    256K    31 + 34    1.71 t/s   284 t/s       658.1 s        4 352 MiB  q8_0

64K and 256K have NOT been re-measured with q4_0; only 128K has. The 64K row is
already at 61+4, so it has little left to gain; 256K at 31+34 has the most.

KV type q4_0, REVISED 2026-08-19 from q8_0. Paired 2-round sweep at 128K:

    KV q8_0           split 47+18   tg 4.98 / 5.22   pp 490-496   prefill 188-190s
    KV q4_0           split 58+ 7   tg 7.71 / 7.84   pp 661-672   prefill 139-141s
    q8_0 --no-kv-offload  65+0      tg 3.23 / 3.59   RESOLVED LOSS -33.2%
    q4_0 --no-kv-offload  65+0      tg 5.22 / 5.29   no better than q8_0 baseline

    q4_0 vs q8_0:  +52.50%  per-round [+54.82, +50.19]  RESOLVED

Halving the cache again returns 11 more layers to the GPU. Note what
--no-kv-offload proves: it reaches 65/0, FULL weight residency, and still loses
33%. The objective is not resident weights, it is total bytes moved per token,
and at depth the cache is bigger than the layers it evicted.

Only f16, bf16, q8_0 and q4_0 have a fast kernel in build 10472 (~1180 tok/s pp
on a shallow probe). q5_1, q5_0, q4_1 and iq4_nl fall back to ~150-170 and get
worse with depth - a 128K arm on q5_1 reached 22% of its window in 15 minutes
before being abandoned. See bench/kv_kernel_screen.py.

Speculation stays off even at depth. It costs VRAM, and at depth VRAM is the
thing already in shortest supply.

256K loads and runs without host paging (RAM free 15.4 GB), unlike the Q4
attempt that had to be stopped at 0.63 GB free with a 10.11 GB pagefile. It is
still 1.71 tok/s behind an 11-minute cold prefill: a budget for one deep
question, not for an agent loop.

WARNING: deep-context RETRIEVAL QUALITY has been verified on Q4 only (report 03,
30/30 at 64K and 10/10 at a 114K prompt). These numbers are throughput and
residency. If a task depends on finding one fact inside 100K tokens, use
production-q4-deep.ps1 until this artifact has been checked the same way.

    NOTE 2026-08-19: the model is referenced by PATH, not by `-hf repo:tag`.
    `-hf` performs an ONLINE etag check on every launch even when the file is
    fully cached, so a saturated network stalls a boot that needs no network at
    all. Observed directly: with a large download running, this server logged
    `common_pull_file: download failed ... retrying after 2 seconds` in a loop
    and an unattended queue hung on it for eleven minutes.
#>
param([int]$Ctx = 65536, [int]$Port = 8080)

$ErrorActionPreference = 'Continue'

& C:\AI\llama.cpp-cuda\llama-server.exe `
    -m "C:\Users\xenod\.cache\huggingface\hub\models--unsloth--Qwen3.8-27B-GGUF\snapshots\f1bfb127c64f7072bdd2cad55f258b9c8b2910fe\Qwen3.8-27B-UD-IQ2_XXS.gguf" --alias qwen38-iq2xxs-deep -c $Ctx `
    -ngl auto --fit on --fit-target 768 -fa on -np 1 `
    -t 18 -b 2048 -ub 256 --no-mmproj-auto `
    -ctk q4_0 -ctv q4_0 `
    --host 127.0.0.1 --port $Port
