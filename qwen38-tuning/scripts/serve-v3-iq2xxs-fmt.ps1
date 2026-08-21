<#
FORMAT-CONSTRAINED variant of serve-v3-iq2xxs.ps1, added 2026-08-20.

The measured failure on every V3 arm is format, not reasoning: 12/12 attempts on
UD-IQ1_S and 41.5 %% of attempts on UD-IQ1_M emitted no fenced code block, having
looped inside the reasoning block until the token cap. Two flags attack that
from both ends and BOTH are off by default in this build:

  --grammar-file    the sampler cannot emit a token the grammar forbids, so
                    "no fenced block" stops being a possible outcome
  -rea off          disables the reasoning block entirely, so the grammar
                    applies from the FIRST token

CORRECTED 2026-08-20 23:55. This pair originally used --grammar-file with
--reasoning-budget 0. Screened at n=3, that combination returned
content_chars = 0 on all three trials: the model wrote 350-700 characters of
reasoning and then ENDED ITS TURN without emitting anything. The grammar does
not cover the reasoning block, so the model reasons freely, and at the point the
grammar starts to bind it emits end-of-turn instead of the fence. Separately,
--reasoning-budget 0 does not end the block as documented -- screened alone it
ran to 24,709 characters. -rea off measured 0 reasoning characters and 323-527
content characters on every trial, which is what the grammar needs to bind from
token one.

Everything else is byte-identical to the unconstrained script, so the pair is a
controlled comparison rather than two different configurations.
#>
param([int]$Ctx = 16384, [int]$Port = 8080)
$ErrorActionPreference = 'Continue'
& C:\AI\llama.cpp-cuda\llama-server.exe `
    -m "C:\Users\xenod\.cache\huggingface\hub\models--unsloth--Qwen3.8-27B-GGUF\snapshots\27af057ecb382ddfea5d12837360a8980560e3ed\Qwen3.8-27B-UD-IQ2_XXS.gguf" `
    --alias v3-iq2xxs -c $Ctx `
    -ngl auto --fit on --fit-target 768 -fa on -np 1 `
    -t 18 -b 2048 -ub 256 --no-mmproj-auto -lv 5 `
    --grammar-file "C:\AI\qwen38-tuning\grammars\python-fence.gbnf" `
    -rea off `
    --host 127.0.0.1 --port $Port
