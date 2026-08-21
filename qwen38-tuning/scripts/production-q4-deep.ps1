<#
DEEP-CONTEXT PRODUCTION PROFILE — Qwen3.8-27B UD-Q4_K_XL at 64K with Q8_0 KV.

Use this ONLY at 64K. At 16K use production-q4-tuned.ps1 instead: Q8 KV is
measured worse there (86.7% vs 90.0% pass, and slower), because at 512 MiB of KV
there is nothing for the quantization to reclaim and only its cost remains.

At 64K the trade inverts, and both halves are measured:

  speed    KV 2304 -> 1224 MiB, freeing 2 GPU layers (27 -> 29).
           Decode 4.37 -> 5.10 tok/s (+16.7%).

  quality  A deep-context corpus of 6 execution-verified tasks whose answers
           depend on constants planted at increasing depth in a ~44K-token
           repository prefix: F16 18/18, Q8_0 18/18. Identical.
           Throughput on that corpus: 51.8 -> 57.4 verified tasks/hour.

           Caveat: both arms hit 100%, so this bounds the damage rather than
           measuring a small one. It rules out the divergence seen in the raw
           greedy comparison being task-relevant; it cannot resolve a 2-3% loss.

The short-prompt greedy hash used to validate the other flags reported Q8 KV as
"identical" and was WRONG — a 4-token probe barely touches the cache Q8
quantizes. At 46.5K of context the two diverge from the first character. Any
equivalence probe must exercise the thing being changed.

Beyond 64K: 128K runs at 2.1-2.5 tok/s with an 8-minute cold prefill, and 256K
drives the host into 10 GB of pagefile. Neither is an interactive agent.
#>
param(
  [int]$Ctx  = 65536,
  [int]$Port = 8080
)

$ErrorActionPreference = 'Continue'   # llama-server logs to stderr on success

& C:\AI\llama.cpp-cuda\llama-server.exe `
    -hf unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL `
    --alias qwen38-q4 `
    -c $Ctx `
    -ngl auto `
    --fit on `
    --fit-target 768 `
    -fa on `
    -np 1 `
    -t 18 `
    -b 2048 `
    -ub 256 `
    -ctk q8_0 `
    -ctv q8_0 `
    --no-mmproj-auto `
    --spec-type draft-mtp `
    --spec-draft-n-max 2 `
    --host 127.0.0.1 `
    --port $Port
