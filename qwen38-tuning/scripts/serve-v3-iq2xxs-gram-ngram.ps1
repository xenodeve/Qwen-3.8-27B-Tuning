<#
GRAMMAR + n-gram. Added 2026-08-21. This is the candidate shipping config.

serve-v3-iq2xxs.ps1 plus the two flags that survived their own single-flag arms:

    --grammar-file        the format fix, if serve-v3-iq2xxs-gram.ps1 earns it
    --spec-type ngram-map-k   +135.89 % decode, byte-identical output

Run this arm ONLY after the grammar arm has passed on its own. Two flags at once
is what made `serve-v3-iq2xxs-fmt.ps1` unreadable -- 84.3 % contract pass and no
way to say which flag bought it -- and repeating that with a different pair
would be the same mistake with better numbers.

THE ONE INTERACTION WORTH WATCHING, and the reason this file exists rather than
the two flags being assumed to compose. A constrained sampler and a speculative
drafter both act on the token distribution. n-gram drafts from what is already
in the context; the grammar forbids tokens the drafts may contain. If the
drafter proposes tokens the grammar rejects, acceptance collapses and the
speedup goes with it -- an arm that is correct and no longer fast.

`acceptance` in the JSONL is what reports it: 96.9-99.0 % unconstrained. If this
arm holds near that, the flags compose. If it falls to the low tens, they do not,
and the choice becomes format OR speed rather than both.
#>
param([int]$Ctx = 16384, [int]$Port = 8080)
$ErrorActionPreference = 'Continue'
& C:\AI\llama.cpp-cuda\llama-server.exe `
    -m "C:\Users\xenod\.cache\huggingface\hub\models--unsloth--Qwen3.8-27B-GGUF\snapshots\27af057ecb382ddfea5d12837360a8980560e3ed\Qwen3.8-27B-UD-IQ2_XXS.gguf" `
    --alias v3-iq2xxs -c $Ctx `
    -ngl auto --fit on --fit-target 768 -fa on -np 1 `
    -t 18 -b 2048 -ub 256 --no-mmproj-auto -lv 5 `
    --grammar-file "C:\AI\qwen38-tuning\grammars\python-fence.gbnf" `
    --spec-type ngram-map-k `
    --host 127.0.0.1 --port $Port
