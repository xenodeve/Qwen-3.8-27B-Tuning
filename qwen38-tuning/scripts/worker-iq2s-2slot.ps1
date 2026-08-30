<#
WORKER PROFILE D -- two slots, for a harness that runs subagents. UD-IQ2_S at
110,592, split into two slots of 55,296.

THE PROBLEM IT SOLVES. Qwen Code sends more than one large prompt per turn: the
main agent at 207,193 characters and a managed memory extraction subagent at
195,929, with DIFFERENT system prompts. On one slot the second evicts the first,
so every invocation re-prefills about 41,000 tokens -- 41.4 s at ~1,000 tok/s.

Two slots let each keep its own cache. Measured 2026-08-21, Qwen Code with all
its memory features left ON:

                                    prefill per invocation      wall
  -c 98304  -np 1 (profile B)       ~41,300 tok, 41.4 s        58 - 71 s
  -c 110592 -np 2 -sps 0.95         0 tok, FULL CACHE HIT       6.0 - 11.3 s

WHY -sps MATTERS AND WHY -np 2 ALONE DID NOT WORK. --slot-prompt-similarity
defaults to 0.10, which is low enough that two prompts sharing a tool catalogue
look alike to it and land on the SAME slot. An earlier measurement of -np 2 at
this exact depth, with the default, changed nothing at all. Raising it to 0.95
forces them apart.

WHAT THIS REPLACES. Turning off memory.enableManagedAutoMemory in
~/.qwen/settings.json also removes the cold start, and costs the developer a
feature: Qwen Code stops updating its own memories. This profile gets the same
result server-side and keeps the feature. Report 26.

THE PRICE, AND IT IS A HARD LIMIT. Each slot gets 55,296 tokens. That was sized
against a Qwen Code request measured at 54,499 -- from a benchmark prompt, in a
directory with no project history. A real interactive session measured 71,910
and this profile rejects it outright:

    API Error: 400 request (71910 tokens) exceeds the available context size
    (55296 tokens), try increasing it

Two slots for a 71,910-token conversation would need 143,820 of context. The
deepest this card holds fully resident is 131,072, and only at --fit-target 192,
which settles at 233-424 MiB free. **So this profile is for short sessions only,
and a session grows.** CORRECTIONS 17.

IF YOUR SESSION IS LARGER, the eviction is still real and the choices are
worker-iq2s-quality.ps1 at 98,304 with one slot and the cold start, or turning
memory.enableManagedAutoMemory off in ~/.qwen/settings.json, which removes the
eviction at any prompt size and costs Qwen Code its own memory updates.

CHECK BEFORE YOU CHOOSE. scripts/bench-cold-start.py reports the largest prefill
the server actually saw. Run it from the directory you really work in, on a
session that resembles yours -- a one-line prompt in an empty directory has
under-measured this three times.

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
param([int]$Ctx = 110592, [int]$Port = 8080)
$ErrorActionPreference = 'Continue'
& C:\AI\llama.cpp-cuda\llama-server.exe `
    -m "C:\Users\xenod\.cache\huggingface\hub\models--unsloth--Qwen3.8-27B-GGUF\snapshots\27af057ecb382ddfea5d12837360a8980560e3ed\Qwen3.8-27B-UD-IQ2_S.gguf" `
    --alias qwen38 -c $Ctx -np 2 -sps 0.95 `
    -ngl auto --fit on --fit-target 768 -fa on `
    -t 18 -b 2048 -ub 256 --no-mmproj-auto -lv 3 `
    -ctk q4_0 -ctv q4_0 `
    --spec-type ngram-mod `
    --spec-ngram-mod-n-match 12 --spec-ngram-mod-n-min 16 --spec-ngram-mod-n-max 32 `
    --chat-template-file "C:\AI\qwen38-tuning\templates\qwen38-late-system.jinja" `
    --host 127.0.0.1 --port $Port
