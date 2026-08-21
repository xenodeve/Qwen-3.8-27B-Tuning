<#
GRAMMAR ONLY, REASONING LEFT ON. Added 2026-08-21.

This is the arm report 22 section 7 named as the top open question and that no
corpus has ever run. `serve-v3-iq2xxs-fmt.ps1` changes TWO things at once --
`--grammar-file` AND `-rea off` -- so its 84.3 % contract pass cannot say which
flag earned it. This script changes ONE:

    serve-v3-iq2xxs.ps1  +  --grammar-file          <- this file
    serve-v3-iq2xxs.ps1  +  -rea off                <- serve-v3-iq2xxs-reaoff.ps1
    serve-v3-iq2xxs.ps1  +  both                    <- serve-v3-iq2xxs-fmt.ps1

Three arms against one control, one flag apart each. That is the design the
first pass should have had.

WHAT THE OTHER TWO ARMS ALREADY MEASURED, on 30 tasks each:

    control (neither flag)   19/30 accepted   58.3 % contract pass
    -rea off alone           15/30 accepted   58.0 % contract pass
    grammar + -rea off       16/27 accepted   84.3 % contract pass

So `-rea off` moved the contract rate by 0.3 points and cost four accepted
tasks: it does not fix the format failure, it relocates the reasoning. The model
still reasons -- into prose outside the fence, and into multiple blocks -- which
is why the contract stayed at 58 % while acceptance fell.

The 26-point jump therefore belongs to the grammar, or to the pair. This arm
tells them apart, and it is the arm that matters: if the grammar carries the
gain on its own, the shipping config keeps reasoning, which is where the
accepted-task count comes from.

WHY IT MIGHT FAIL, stated in advance so the result is not read backwards. The
n=3 screen of `--grammar-file` with `--reasoning-budget 0` returned
content_chars = 0 on every trial: the model reasoned freely, then hit the point
where the grammar binds and emitted end-of-turn instead of the fence. The
grammar does not cover the reasoning block. If reasoning is ON and unbounded,
the same collision can happen at the same seam -- the difference is that here
the model has a real budget to finish its reasoning in, rather than one that was
supposed to be zero and was not honoured.

Everything else is byte-identical to serve-v3-iq2xxs.ps1.
#>
param([int]$Ctx = 16384, [int]$Port = 8080)
$ErrorActionPreference = 'Continue'
& C:\AI\llama.cpp-cuda\llama-server.exe `
    -m "C:\Users\xenod\.cache\huggingface\hub\models--unsloth--Qwen3.8-27B-GGUF\snapshots\27af057ecb382ddfea5d12837360a8980560e3ed\Qwen3.8-27B-UD-IQ2_XXS.gguf" `
    --alias v3-iq2xxs -c $Ctx `
    -ngl auto --fit on --fit-target 768 -fa on -np 1 `
    -t 18 -b 2048 -ub 256 --no-mmproj-auto -lv 5 `
    --grammar-file "C:\AI\qwen38-tuning\grammars\python-fence.gbnf" `
    --host 127.0.0.1 --port $Port
