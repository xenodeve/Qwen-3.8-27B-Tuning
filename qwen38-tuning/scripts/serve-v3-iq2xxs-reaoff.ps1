<#
REASONING-OFF variant of serve-v3-iq2xxs.ps1 -- `-rea off` and nothing else.

Exists to isolate what the GRAMMAR contributes. Screened at n=3, `-rea off`
alone produced a fenced code block on every trial with content 314-539 chars in
2.4-3.7 s, and `--grammar-file` on top of it produced 314-537 chars in 2.6-3.6 s.
At that sample size the two are indistinguishable, so the corpus has to say
whether the grammar earns its place or whether disabling reasoning is the whole
effect.

Byte-identical to serve-v3-iq2xxs-fmt.ps1 except for the missing --grammar-file.
#>
param([int]$Ctx = 16384, [int]$Port = 8080)
$ErrorActionPreference = 'Continue'
& C:\AI\llama.cpp-cuda\llama-server.exe `
    -m "C:\Users\xenod\.cache\huggingface\hub\models--unsloth--Qwen3.8-27B-GGUF\snapshots\27af057ecb382ddfea5d12837360a8980560e3ed\Qwen3.8-27B-UD-IQ2_XXS.gguf" `
    --alias v3-iq2xxs -c $Ctx `
    -ngl auto --fit on --fit-target 768 -fa on -np 1 `
    -t 18 -b 2048 -ub 256 --no-mmproj-auto -lv 5 `
    -rea off `
    --host 127.0.0.1 --port $Port
