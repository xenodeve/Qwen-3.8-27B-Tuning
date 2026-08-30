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

  --fit-target 768  ALREADY LOWERED, not a default. The default is 1024 MiB
                    (common/common.h:473, fit_params_target). An earlier version
                    of this header said "left at the default deliberately" and
                    that was simply wrong -- corrected 2026-08-22 after reading
                    the source.

                    Lowering it further to 192 does free ~576 MiB and does buy
                    residency, and on IQ2_S at this depth it took prefill from
                    151 s to 825 s: the headroom it reserves is not slack.
                    Whether 768 is the right point between 1024 and 192 has
                    never been swept.

NOT MEASURED: whether 131,072 is worth anything. The longest real task in our
OpenCode corpus reached 13,741 tokens of conversation -- 10.5 % of this window.
The depth may be free headroom nobody uses, which is the question profile B
exists to settle.

WHY --chat-template-file. Qwen3.8's stock template raises
'System message must be at the beginning.' the moment a system message appears
anywhere but the front, and Claude Code appends exactly that. Without the flag
every request 500s at sampler init: 50 consecutive failures on 2026-08-21, 0
after. The one-line change, the evidence and how to regenerate it are in
templates/README.md -- kept there rather than repeated in both profiles.

DO NOT SET -cram 0. Measured 2026-08-23, results/prompt-cache-swap.jsonl.

  --cache-ram defaults to 8192 MiB and stores the WHOLE sequence state --
  attention KV and recurrent together -- for a slot that goes idle
  (server-context.cpp:261-274). It is what makes an agent switching between
  tasks cheap, and no profile here had ever named it.

  Two disjoint 44K conversations, A-B-A-B-A, one boot per arm:

      -cram 8192   returning to A costs    118.2 ms at 100 % reuse
      -cram 0      returning to A costs 40,596.0 ms at   0 % reuse

  The cold turns agree to 0.35 %, so the arms are comparable. 343x.

  It costs 898-928 MiB of HOST RAM per cached conversation, so roughly six fit
  at this depth. Restore is a move, not a copy, and load() refuses any entry
  whose common prefix is under 25 % of its length.
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
    --chat-template-file "C:\AI\qwen38-tuning\templates\qwen38-late-system.jinja" `
    --reasoning-effort medium `
    --host 127.0.0.1 --port $Port
