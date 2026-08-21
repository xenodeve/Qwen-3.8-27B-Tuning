<#
WORKER PROFILE A — depth. Unsloth Dynamic V3 UD-IQ2_XXS at 131,072.

Pair with worker-iq2s-quality.ps1. They serve the same port, so one at a time;
the whole point of having both is to decide between them on real work rather
than on a benchmark.

  A (this file)   131,072 ctx   ~79 % top-1 (vendor)   45 tok/s measured
  B (the other)    98,304 ctx   ~84 % top-1 (vendor)   not yet measured

WHAT IS MEASURED HERE, and what is inherited.

  ctx 131,072       65+0 resident, confirmed in many boots. The artifact's
                    actual ceiling is 147,456 -- also 65+0 -- but the margin
                    there is the same ~600 MiB and boot-to-boot VRAM drift is
                    13.6 %, so the deeper setting is one bad boot from spilling
                    a layer, and a spilled layer at this depth costs a factor of
                    four. Raise it with -Ctx if a task genuinely needs it.

  -ctk/-ctv q4_0    the settled KV type. No other type in build 10472 has a fast
                    kernel; mixed q8_0/q4_0 was 29x slower on prefill.

  --spec-type       ngram-mod, short window. At 131,072 it measured +200 % over
    ngram-mod       no speculation on the synthetic sweep and 1.8x on real code
                    -- the sweep prompt is 84.5 % duplicate lines, so treat the
                    smaller number as the real one. Output is byte-identical
                    either way: it costs no VRAM and needs no drafter file.
                    NOT ngram-map-k -- that wins at 16K and loses by 80 points
                    here.

  --fit-target 768  left at the default deliberately. Lowering it to 192 does
                    free ~576 MiB and does buy residency, and on IQ2_S at this
                    depth it took prefill from 151 s to 825 s: the headroom it
                    reserves is not slack.

NOT MEASURED: whether 131,072 is worth anything. The longest real task in our
OpenCode corpus reached 13,741 tokens of conversation -- 10.5 % of this window.
The depth may be free headroom nobody uses, which is the question profile B
exists to settle.
#>
param([int]$Ctx = 131072, [int]$Port = 8080)
$ErrorActionPreference = 'Continue'
& C:\AI\llama.cpp-cuda\llama-server.exe `
    -m "C:\Users\xenod\.cache\huggingface\hub\models--unsloth--Qwen3.8-27B-GGUF\snapshots\27af057ecb382ddfea5d12837360a8980560e3ed\Qwen3.8-27B-UD-IQ2_XXS.gguf" `
    --alias qwen38 -c $Ctx `
    -ngl auto --fit on --fit-target 768 -fa on -np 1 `
    -t 18 -b 2048 -ub 256 --no-mmproj-auto -lv 3 `
    -ctk q4_0 -ctv q4_0 `
    --spec-type ngram-mod `
    --spec-ngram-mod-n-match 12 --spec-ngram-mod-n-min 16 --spec-ngram-mod-n-max 32 `
    --host 127.0.0.1 --port $Port
