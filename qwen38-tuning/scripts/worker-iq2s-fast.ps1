<#
WORKER PROFILE C -- small window, fast turns. UD-IQ2_S at 32,768.

CORRECTED 2026-08-21, SAME DAY IT WAS WRITTEN. This file first said it was the
profile for Qwen Code, on the reading that Qwen Code's first turn was 16,796
tokens. That number was wrong: 16,796 is what the server had left to PREFILL
after cache reuse, not the size of the request. The request is 54,499 tokens, and
pointing Qwen Code at this profile does not make it slow, it makes it fail:

    API Error: 400 request (54499 tokens) exceeds the available context size
    (32768 tokens), try increasing it

WHAT ACTUALLY FITS HERE. Only the lean OpenCode profile, whose longest measured
task reached 13,741 tokens of total conversation against a 5,377-token prefix.
Everything else measured on this machine is larger than the window:

    lean OpenCode, longest of 10 real tasks     13,741   fits
    Qwen Code, one turn                         54,499   does not fit
    Claude Code with MCP loaded, one turn       54,685   does not fit
    OpenCode default profile, prefix alone      99,073   does not fit

WHAT THE SMALL WINDOW BUYS, for the one harness that fits. Measured on the same
artifact and decoder, both resident at 65+0:

                        prefill tok/s      decode tok/s     KV      free MiB
  -c 131072 --fit 192      776 - 836         23.2 - 23.9   2,304    233 - 424
  -c 32768  --fit 768    1,134 - 1,168       45.3 - 50.3     576   2,028 - 2,267

Prefill 47 % faster, decode roughly double, and 2 GB of headroom that puts the
paging collapse out of reach entirely. Report 25.

BEFORE POINTING A HARNESS HERE, MEASURE ITS REQUEST. scripts/bench-cold-start.py
reports the largest prefill the server actually saw, per run. A window chosen
from any other number is a guess, and this header is what one looks like.
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
