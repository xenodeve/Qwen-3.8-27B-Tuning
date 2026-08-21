<#
WORKER PROFILE C -- cold start. Unsloth Dynamic V3 UD-IQ2_S at 32,768.

For a harness whose prompt is small and fixed, where the wait before the first
token is the thing you feel. Qwen Code is the case this was measured for: its
first turn is 16,796 tokens of built-in system prompt and tool schemas, with no
MCP server, no extension and no QWEN.md, so that number does not grow and cannot
be trimmed from its config.

WHAT THE WINDOW COSTS, measured 2026-08-21 on the same artifact and decoder,
both fully resident at 65+0:

                        prefill tok/s      decode tok/s     KV      free MiB
  -c 131072 --fit 192      776 - 836         23.2 - 23.9   2,304    233 - 424
  -c 32768  --fit 768    1,134 - 1,168       45.3 - 50.3     576   2,028 - 2,267

Prefill is 47 % faster and decode is roughly double. On Qwen Code's 16,796-token
first turn that is about 14.7 s of cold start instead of 21.6 s, and every token
after it arrives at twice the rate.

The window is not free, and at 131,072 you are paying for it twice: a 2,304 MiB
KV cache, and a machine left with 233-424 MiB of headroom. On 2026-08-21 a sweep
at that setting had to be abandoned when the desktop grew and free VRAM reached
92 MiB -- prefill was reading 130 tok/s, a sixth of its rate. This profile leaves
2 GB and that failure mode is simply out of reach.

WHEN NOT TO USE IT. 32,768 is about twice the longest first turn measured, which
is enough for Qwen Code and not obviously enough for a harness with a bigger or
a growing prefix. OpenCode's default profile sends 99,073 tokens before the task
starts and Claude Code measured 54,685; neither fits here. Use profile A or B for
those, and read qwen38-tuning/templates/README.md first -- Claude Code needs the
chat template patch as well.

```
  .\scripts\worker-iq2s-fast.ps1        # this file, 32,768
  .\scripts\worker-iq2xxs-deep.ps1      # profile A, 131,072, faster artifact
  .\scripts\worker-iq2s-quality.ps1     # profile B, 98,304, this artifact
```

WHY --fit-target STAYS AT ITS DEFAULT. At this depth the reserve costs nothing:
the model is resident either way and 2 GB is left over. Lowering it buys layers
only when the window is deep enough to squeeze them out, which is the whole
subject of report 25 and is not this profile's problem.
#>
param([int]$Ctx = 32768, [int]$Port = 8080)
$ErrorActionPreference = 'Continue'
& C:\AI\llama.cpp-cuda\llama-server.exe `
    -m "C:\Users\xenod\.cache\huggingface\hub\models--unsloth--Qwen3.8-27B-GGUF\snapshots\27af057ecb382ddfea5d12837360a8980560e3ed\Qwen3.8-27B-UD-IQ2_S.gguf" `
    --alias qwen38 -c $Ctx `
    -ngl auto --fit on --fit-target 768 -fa on -np 1 `
    -t 18 -b 2048 -ub 256 --no-mmproj-auto -lv 3 `
    -ctk q4_0 -ctv q4_0 `
    --spec-type ngram-mod `
    --spec-ngram-mod-n-match 12 --spec-ngram-mod-n-min 16 --spec-ngram-mod-n-max 32 `
    --chat-template-file "C:\AI\qwen38-tuning\templates\qwen38-late-system.jinja" `
    --host 127.0.0.1 --port $Port
