<#
AtomicChat AD-IQ1_M - the only 1-bit-class Qwen3.8-27B build that exists.
7.91 GiB, against UD-IQ2_XXS at 8.39 GiB. Both are fully resident at 16K, so
any difference here is memory bandwidth and quantization damage, not residency.
Speculation off, as on every resident artifact (report 10 s1).
#>
param([int]$Ctx = 16384, [int]$Port = 8080)
$ErrorActionPreference = 'Continue'
& C:\AI\llama.cpp-cuda\llama-server.exe `
    -m "C:\Users\xenod\.cache\huggingface\hub\models--AtomicChat--Qwen3.8-27B-GGUF\snapshots\ca10ebceb1887be9d33b838770a36b39d75a8a4c\Qwen3.8-27B-AD-IQ1_M.gguf" `
    --alias qwen38-iq1m -c $Ctx `
    -ngl auto --fit on --fit-target 768 -fa on -np 1 `
    -t 18 -b 2048 -ub 256 --no-mmproj-auto `
    --host 127.0.0.1 --port $Port
