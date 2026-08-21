<#
Pay the one remaining cold start on the server's behalf, before a human waits.

    .\scripts\warm-cache.ps1                 # after starting a worker
    .\scripts\warm-cache.ps1 -Harness claude-xeno

WHY THIS IS THE LAST PIECE. With Qwen Code's managed auto-memory subagents off
(see below) the harness sends ONE request per invocation, so llama-server's
prompt cache holds the whole 51,329-token prefix and every later invocation
prefills 4 tokens in 0.1 s. Measured 2026-08-21:

    run 1   51,329 tok   49.7 s prefill   wall 54.1 s   <- fresh server
    run 2        4 tok    0.1 s           wall  3.9 s
    run 3        4 tok    0.1 s           wall  3.8 s

Only run 1 is expensive, and it is expensive exactly once per server. Firing it
from here means the developer's first turn is run 2.

THE SETTING THIS DEPENDS ON, and it is not ours to keep secret:
`memory.enableManagedAutoMemory` in ~/.qwen/settings.json must be false. With it
on, Qwen Code runs a `managed memory extraction subagent` after every turn whose
system prompt is different from the main one and nearly as large -- 195,929
characters against 207,193 -- so it evicts the main prefix from the single slot
and the next invocation re-prefills ~41,000 tokens, 41.4 s, every time.

That is a real trade: with it off, Qwen Code stops updating its own memories.
`memory.enableManagedAutoDream` and `memory.enableAutoSkill` were turned off in
the same measurement and have not been isolated from each other.

WARM IN THE DIRECTORY YOU WILL WORK IN. Qwen Code's prompt embeds the working
directory, so a warm-up run somewhere else warms a different prefix and buys
nothing. Measured: warming from a background job's own directory left the first
real turn paying 49.8 s exactly as before.

WHAT WAS RULED OUT FIRST, all measured, none of them the cause: --cache-ram -1
and --cache-reuse 256 (both regressions), -np 2 at 110,592 and at 131,072 (no
change, then a VRAM collapse), a larger -ub, and the server itself -- replaying
one captured request three times gives 53.9 s then 0.4 s then 0.4 s.
#>
param(
    [ValidateSet('qwen', 'claude-xeno')][string]$Harness = 'qwen',
    [string]$Endpoint = 'http://127.0.0.1:8080',
    [int]$TimeoutSec = 900,
    [string]$Work = (Get-Location).Path
)
$ErrorActionPreference = 'Stop'

# The server is the authority on whether there is anything to warm.
try {
    $props = Invoke-RestMethod -Uri "$Endpoint/props" -TimeoutSec 10
    $nctx = [int]$props.default_generation_settings.n_ctx
} catch {
    throw "no server on $Endpoint -- start a worker profile first"
}

$exe = if ($Harness -eq 'qwen') {
    "$env:LOCALAPPDATA\qwen-code\bin\qwen.cmd"
} else {
    "$env:USERPROFILE\.claude\claude-xeno.bat"
}
if (-not (Test-Path $exe)) { throw "not found: $exe" }

Write-Host "warming $Harness against $Endpoint (n_ctx $nctx) -- this pays the" -ForegroundColor DarkGray
Write-Host "one-time prefill so the developer's first turn does not" -ForegroundColor DarkGray
$sw = [Diagnostics.Stopwatch]::StartNew()
Push-Location $Work
try {
    & $exe -p 'reply with exactly the word: ok' | Out-Null
} finally {
    Pop-Location
}
Write-Host "warm after $([int]$sw.Elapsed.TotalSeconds)s in $Work" -ForegroundColor Green
