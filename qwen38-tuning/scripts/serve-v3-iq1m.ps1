<#
Unsloth Dynamic 3.0 arm v3-iq1m, resolved from the cache by EXACT byte count --
the repo was republished in place, so two snapshots hold this filename.
No --spec-type: V3 removed the built-in MTP head at 2-bit and smaller.
#>
param([int]$Ctx = 16384, [int]$Port = 8080)
$ErrorActionPreference = 'Continue'
& C:\AI\llama.cpp-cuda\llama-server.exe `
    -m "C:\Users\xenod\.cache\huggingface\hub\models--unsloth--Qwen3.8-27B-GGUF\snapshots\27af057ecb382ddfea5d12837360a8980560e3ed\Qwen3.8-27B-UD-IQ1_M.gguf" `
    --alias v3-iq1m -c $Ctx `
    -ngl auto --fit on --fit-target 768 -fa on -np 1 `
    -t 18 -b 2048 -ub 256 --no-mmproj-auto -lv 5 `
    --host 127.0.0.1 --port $Port
