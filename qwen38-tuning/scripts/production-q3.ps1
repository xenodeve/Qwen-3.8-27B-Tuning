<#
Qwen3.8-27B UD-Q3_K_XL profile -- the smaller-quant lane.

Measured against the Q4 profile on this machine:

    Q4 + MTP n=2 : 10.67 tok/s synthetic / 12.10 code-rewrite
    Q3 + MTP n=2 :  8.88 tok/s synthetic / 10.30 code-rewrite

Q3 is the SLOWER of the two once speculation is enabled, which inverts the usual
expectation. Baseline Q3 (9.01/9.25) does beat baseline Q4 (8.24/8.22), but only
by ~10% for 4.17 GiB and ~3.6 points of top-1 fidelity -- and MTP then adds
+30-47% to Q4 while adding only -1 to +11% to Q3.

The mechanism: speculative decoding amortises one forward pass across several
tokens, so it pays most when a forward pass is expensive. Q4 keeps ~42% of its
weights on the CPU and Q3 only ~22%, so Q4 has far more to amortise. MTP is a
compensation for poor VRAM fit, not an independent accelerator.

Keep this profile for:
  - contexts deep enough that Q4's KV no longer fits alongside its weights
    (untested as of this writing -- the comparison above is at 16K)
  - leaving host RAM free for other work
  - a fallback if a Q4-specific regression appears

MTP n=2 is still the best Q3 configuration (n=3 measured 7.27/9.92, clearly worse).
Sampling and reasoning-effort caveats are identical to production-q4.ps1.
#>
param(
  [int]$Ctx  = 16384,
  [int]$Port = 8080
)

$ErrorActionPreference = 'Continue'

& C:\AI\llama.cpp-cuda\llama-server.exe `
    -hf unsloth/Qwen3.8-27B-GGUF:UD-Q3_K_XL `
    --alias qwen38-q3 `
    -c $Ctx `
    -ngl auto `
    --fit on `
    -fa on `
    -np 1 `
    --no-mmproj-auto `
    --spec-type draft-mtp `
    --spec-draft-n-max 2 `
    --host 127.0.0.1 `
    --port $Port
