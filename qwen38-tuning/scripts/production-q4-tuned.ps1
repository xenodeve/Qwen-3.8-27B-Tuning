<#
TUNED PRODUCTION PROFILE — Qwen3.8-27B UD-Q4_K_XL

Adds three measured runtime settings on top of production-q4.ps1. Each was swept
independently, each kept the greedy output bit-identical, and each is justified
below by the number that put it there.

  --fit-target 768   default is 1024. Measured 12.39 tok/s on the code-rewrite
                     prompt vs 11.34 at the default (+9.3%). NOT monotonic in
                     layer count: 33 layers with 584 MiB headroom beat both 32
                     layers with 867 MiB and 35 layers with 345 MiB. At 256 the
                     code prompt destabilised ([6.70, 8.28, 11.57]) — intermittent
                     driver eviction, not a lower mean.

  -t 18              default is 14 (physical cores). Measured 13.58 tok/s vs 12.70
                     (+6.9%). Throughput rose monotonically 6 -> 20, contradicting
                     the usual physical-core guidance; -t 6 (P-cores only) was
                     worst at 9.38. -t 20 takes every logical thread and costs 18%
                     of prompt processing; 18 leaves two threads for the OS and
                     wins decode, spread and pp simultaneously.

  -ub 256            default is 512, with -b left at 2048. Measured 13.49 tok/s vs
                     13.00 (+3.8%) with prompt processing unchanged (164.2 vs
                     164.4). Keeping the large logical batch protects pp while the
                     smaller physical microbatch frees compute-buffer VRAM.
                     -b 512 -ub 128 decoded slightly faster but cost 33% of pp,
                     which is paid in full on every prefix-cache invalidation.

Cumulative: RETRACTED as "+19%". That figure summed three control-first sweeps,
each of which carried its own restart drift. A paired re-test of the same three
settings against a fresh control gives +6.6% mean / +9.6% pooled. The settings
are still the right ones -- each held its sign under pairing -- but the headline
number was an artefact of the design, not of the flags. See report 04 s1.
Quality verified unchanged (greedy output bit-identical per sweep).

OPERATIONAL RULE FROM THE PREFIX-CACHE TEST — this matters more than any flag:
llama-server reuses KV for append-only turns (40 tokens evaluated instead of
3900), but the cache is prefix-EXACT. Reordering tool schemas, editing one
sentence of the system prompt, or prepending a skill block each dropped cache
reuse to zero and forced a full re-prefill. Freeze everything above the append
point: stable tool order, byte-stable system prompt, skills injected once.

Sampling and reasoning_effort caveats are unchanged — see production-q4.ps1.

    NOTE 2026-08-19: the model is referenced by PATH, not by `-hf repo:tag`.
    `-hf` performs an ONLINE etag check on every launch even when the file is
    fully cached, so a saturated network stalls a boot that needs no network at
    all. Observed directly: with a large download running, this server logged
    `common_pull_file: download failed ... retrying after 2 seconds` in a loop
    and an unattended queue hung on it for eleven minutes.
#>
param(
  [int]$Ctx  = 16384,
  [int]$Port = 8080
)

$ErrorActionPreference = 'Continue'   # llama-server logs to stderr on success

& C:\AI\llama.cpp-cuda\llama-server.exe `
    -m "C:\Users\xenod\.cache\huggingface\hub\models--unsloth--Qwen3.8-27B-GGUF\snapshots\f1bfb127c64f7072bdd2cad55f258b9c8b2910fe\Qwen3.8-27B-UD-Q4_K_XL.gguf" `
    --alias qwen38-q4 `
    -c $Ctx `
    -ngl auto `
    --fit on `
    --fit-target 768 `
    -fa on `
    -np 1 `
    -t 18 `
    -b 2048 `
    -ub 256 `
    --no-mmproj-auto `
    --spec-type draft-mtp `
    --spec-draft-n-max 2 `
    --host 127.0.0.1 `
    --port $Port
