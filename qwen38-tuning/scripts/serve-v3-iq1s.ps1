<#
Unsloth Dynamic 3.0, UD-IQ1_S — 5.77 GiB, verified by SHA-256 against the repo
OID 3895b6ea…, not by filename or byte count alone.

No --spec-type: V3 removed the built-in MTP head from every artifact at 2-bit
and smaller, so there is no head to drive. A standalone 1.28 GiB drafter ships
separately and is a later, separate experiment.
#>
param([int]$Ctx = 16384, [int]$Port = 8080)
$ErrorActionPreference = 'Continue'
& C:\AI\llama.cpp-cuda\llama-server.exe `
    -m "C:\Users\xenod\.cache\huggingface\hub\models--unsloth--Qwen3.8-27B-GGUF\snapshots\27af057ecb382ddfea5d12837360a8980560e3ed\Qwen3.8-27B-UD-IQ1_S.gguf" `
    --alias qwen38-v3-iq1s -c $Ctx `
    -ngl auto --fit on --fit-target 768 -fa on -np 1 `
    -t 18 -b 2048 -ub 256 --no-mmproj-auto -lv 5 `
    --host 127.0.0.1 --port $Port
