<#
Q2 CANDIDATE PROFILE - Qwen3.8-27B UD-Q2_K_XL

Same tuned flags as production-q4-tuned.ps1 with ONE deliberate difference:
speculative decoding is OFF.

Why MTP is off here, when it is on for Q4. Measured in the paired arena
(results\arena-quant.jsonl, 3 rounds, order counterbalanced):

    q2kxl-nomtp   21.26 / 21.84 / 21.42 tok/s    layer split 61+4
    q2kxl-mtp2    19.87 / 19.92 / 19.95 tok/s    layer split 55+10

MTP costs ~7% of decode on this artifact, and the layer split shows why: the
draft head takes VRAM, which pushes six more target layers onto the CPU. On Q4
the target forward pass is expensive enough that speculation pays for itself; at
Q2 it is cheap, and the residency it costs is worth more than the tokens it
saves. That is the same mechanism report 01 recorded in reverse.

Speed vs the Q4 control, paired across three rounds: +62.1% (+60.9..+63.6%).
This profile is a CANDIDATE, not the production default: quality has not yet
been measured on the task corpus.
#>
param([int]$Ctx = 16384, [int]$Port = 8080)

$ErrorActionPreference = 'Continue'

& C:\AI\llama.cpp-cuda\llama-server.exe `
    -hf unsloth/Qwen3.8-27B-GGUF:UD-Q2_K_XL --alias qwen38-q2 -c $Ctx `
    -ngl auto --fit on --fit-target 768 -fa on -np 1 `
    -t 18 -b 2048 -ub 256 --no-mmproj-auto `
    --host 127.0.0.1 --port $Port
