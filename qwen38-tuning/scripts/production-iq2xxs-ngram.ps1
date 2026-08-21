<#
Production profile, 16K. Pre-V3 UD-IQ2_XXS with n-gram speculative decoding.

This is production-iq2xxs.ps1 plus ONE flag. Measured 2026-08-20 on
V3 UD-IQ2_XXS, paired, two rounds, 16K:

    control  --spec-type none        41.81 tok/s
    ngram-map-k                      93.75 tok/s     +135.89 %
    four rounds, --fixed-text, per-round [+139.61, +146.47, +133.25, +124.23]

Free in every sense that matters here:
  * greedy_hash identical to the control -- byte-for-byte the same output, sooner
  * no drafter file, nothing to download
  * no VRAM cost: free VRAM and the 65+0 split were unchanged
  * prefill untouched (1,147-1,163 tok/s against the control's 1,209)

Why it works: n-gram speculation replays token sequences already in the context.
Code is the best case -- identifiers, `self.`, indentation and the block just
written all repeat. Unlike MTP it holds no weights, so on a 12 GB card it never
competes with the layers.

TWO CORRECTIONS ARE ON THE RECORD HERE, and both were instrument faults.

1. This profile first named `ngram-map-k` at +94.69 %. A re-run three hours later
   returned +69.73 % with one round at +3.68 %, and `ngram-cache` REVERSED from
   +80.79 % to -30.56 % -- both passing the paired test. Cause: the timed
   generations ran at temperature 0.7, so every round wrote different text and
   the n-gram hit rate followed the text rather than the hardware.
2. On that evidence the profile was switched to `ngram-mod` with a shortened
   lookup, which had looked stable. That reading was also wrong: shortening the
   lookup did not stabilise anything, it just happened to land in a quiet part of
   the noise.

`kv_sweep --fixed-text` pins temperature 0 and a fixed seed for the timed
generations. Under it, four rounds put every n-gram arm inside a 9-22 point band
and `ngram-map-k` -- at its DEFAULT lookup lengths -- is the fastest of them.

NOT YET VERIFIED ON THIS ARTIFACT. The measurement above is V3 UD-IQ2_XXS; this
profile serves the pre-V3 file. The mechanism is token-level and should carry,
but that is reasoning, not measurement.

NOT YET VERIFIED AT DEPTH. At 131,072 the prefill alone is ~120 s and
speculation cannot touch it. Use production-iq2xxs-deep.ps1 until that lands.
#>
param([int]$Ctx = 16384, [int]$Port = 8080)
$ErrorActionPreference = 'Continue'
& C:\AI\llama.cpp-cuda\llama-server.exe `
    -m "C:\Users\xenod\.cache\huggingface\hub\models--unsloth--Qwen3.8-27B-GGUF\snapshots\f1bfb127c64f7072bdd2cad55f258b9c8b2910fe\Qwen3.8-27B-UD-IQ2_XXS.gguf" `
    --alias qwen38-iq2xxs-ngram -c $Ctx `
    -ngl auto --fit on --fit-target 768 -fa on -np 1 `
    -t 18 -b 2048 -ub 256 --no-mmproj-auto -lv 5 `
    -ctk q4_0 -ctv q4_0 `
    --spec-type ngram-map-k `
    --host 127.0.0.1 --port $Port
