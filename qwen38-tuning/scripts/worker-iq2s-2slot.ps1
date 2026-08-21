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

THE PRICE. 110,592 of KV against 98,304, and a window per conversation of 55,296
rather than 98,304. That is still comfortably above the 54,499-token request
measured for Qwen Code, but it is not a lot of room, and a harness with a bigger
prefix should use profile B and pay the prefill instead.

```
  .\scripts\worker-iq2s-2slot.ps1
  .\scripts\warm-cache.ps1 -Work C:\AI     # pays the one cold prefill
```
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
