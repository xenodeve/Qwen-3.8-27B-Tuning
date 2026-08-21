<#
UNCONSTRAINED + n-gram. Added 2026-08-21.

serve-v3-iq2xxs.ps1 plus one flag: `--spec-type ngram-map-k`.

WHY THIS NEEDS A CORPUS AT ALL, given the output is provably identical.
Speculative decoding verifies every drafted token against the target model, so
the accepted text is what the target would have produced anyway -- and the
greedy hash confirms it: `04E5CAB1D14525C0` on the control and on this arm, in
six separate boots across two nights. Quality cannot move.

What CAN move is the number this project actually optimises: **verified accepted
coding tasks per hour.** The standing best is 60.8/hr, and decode on this arm is
+135.89 % over four fixed-text rounds at 16K (40.60/40.11/40.18/41.81 against
97.28/98.86/93.72/93.75). Tasks per hour is not decode, though -- it also pays
prefill, escalations and retry overhead, and prefill is untouched by the flag.
So the gain is real but bounded, and nobody has measured where it lands.

Two ways this could disappoint, both worth knowing:
  * a corpus task spends much of its wall clock in prefill and orchestration, in
    which case a 2.4x decode buys far less than 2.4x throughput;
  * the retry pass re-runs failed tasks, so an arm that fails the same tasks
    just reaches the failures sooner.

Both are arguments for measuring rather than assuming, which is the point.

`-ctk q4_0 -ctv q4_0` is deliberately ABSENT: the V3 serve family runs default
KV at 16K, and the control corpus (19/30, 58.3 % contract pass) was measured
that way. Adding it here would change two things at once -- the exact fault this
night's other arm was created to repair.
#>
param([int]$Ctx = 16384, [int]$Port = 8080)
$ErrorActionPreference = 'Continue'
& C:\AI\llama.cpp-cuda\llama-server.exe `
    -m "C:\Users\xenod\.cache\huggingface\hub\models--unsloth--Qwen3.8-27B-GGUF\snapshots\27af057ecb382ddfea5d12837360a8980560e3ed\Qwen3.8-27B-UD-IQ2_XXS.gguf" `
    --alias v3-iq2xxs -c $Ctx `
    -ngl auto --fit on --fit-target 768 -fa on -np 1 `
    -t 18 -b 2048 -ub 256 --no-mmproj-auto -lv 5 `
    --spec-type ngram-map-k `
    --host 127.0.0.1 --port $Port
