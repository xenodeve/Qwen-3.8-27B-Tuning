<#
WORKER PROFILE B — quality. Unsloth Dynamic V3 UD-IQ2_S at 98,304.

Pair with worker-iq2xxs-deep.ps1. Same port, so one at a time.

THE TRADE THIS PROFILE MAKES, in one line: it gives up 32,768 tokens of context
for about five points of top-1 accuracy on the vendor's own curve.

  A  UD-IQ2_XXS  7.27 GB   131,072 ctx   ~79 % top-1
  B  UD-IQ2_S    8.37 GB    98,304 ctx   ~84 % top-1     <- this file

`IQ2_XXS` -> `IQ2_S` is the steepest segment of Unsloth's whole accuracy curve,
and it is where this project's own bits-per-weight ladder is steepest too. That
is the entire case for profile B; nothing about its behaviour on real work has
been measured yet.

WHY 98,304 AND NOT 131,072 -- measured, not assumed.

  At 131,072 this artifact loads at 60+5: five layers on the CPU, prefill 875 s
  against 114 s for profile A, decode 10.35 tok/s.

  `--fit-target 192` does free enough to reach 65+0 there, and whether that is
  worth having is UNRESOLVED. Two rounds of the same arm:

      round 1   65+0   prefill 825.5 s   11.64 tok/s   free 226 MiB
      round 2   65+0   prefill 110.7 s   84.20 tok/s   free 307 MiB

  Seven times on both axes, same flags, same artifact, same depth.

  The cause is not the flag. Across all 235 sweep rows ever recorded, the six
  with free VRAM under 300 MiB are the only ones with prefill under 700 tok/s,
  and the two under 250 MiB are the only ones under 120 -- the signature of
  WDDM paging compute buffers out to system RAM rather than failing. Round 1 sat
  at 226 MiB and round 2 at 307, and the 81 MiB between them is the desktop
  moving, not a setting.

  So `--fit-target 768` stays here because it is the buffer that keeps the
  machine off that cliff, not because 192 was measured to be slower. Whether
  IQ2_S can hold 131,072 usefully needs a boot with the display off this card.

  98,304 is the deepest rung where this artifact holds 65+0 with the default
  reserve. 114,688 already spills to 63+2.

IS 98,304 ENOUGH? Measured, on our own corpus through OpenCode: the longest of
ten real tasks reached **13,741 tokens** of conversation, median 6,165. The
OpenCode lean prefix is 5,377. Real single-issue PRs across five of this team's
repos touch 2-5 files at a 2-7 KB median, so the files a task must read are
roughly 3,000 tokens and their neighbours a few times that.

So a task budget lands around 30,000-40,000 tokens and this profile leaves
~93,000. **The context is not the binding constraint at either setting** -- which
is why the decision between A and B should be made on accepted tasks per hour,
not on the context number.

Everything else is identical to profile A, so the pair is a controlled
comparison: same KV type, same decoder, same reserve, same batch.

WHY --chat-template-file. Qwen3.8's stock template raises
'System message must be at the beginning.' the moment a system message appears
anywhere but the front, and Claude Code appends exactly that. Without the flag
every request 500s at sampler init: 50 consecutive failures on 2026-08-21, 0
after. The one-line change, the evidence and how to regenerate it are in
templates/README.md -- kept there rather than repeated in both profiles.

#>
param([int]$Ctx = 98304, [int]$Port = 8080)
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
