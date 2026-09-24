<#
.SYNOPSIS
    Serve GSQ IQ3_S-MTP at the model's 262,144-token context ceiling.

.DESCRIPTION
    This profile follows worker-q4-dual.ps1's unequal-card rule. It measures each
    GPU by UUID, leaves 2,500 MiB on the display card and 512 MiB on the idle
    card, then passes those two budgets as -ts values. The values are tensor
    split weights, not a 50:50 layer count.

    The profile is intentionally foreground-only. Ctrl+C or closing the window
    stops llama-server. -WhatIf resolves the live split without launching it.
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [int]$Ctx = 262144,
    [int]$Port = 8080,
    [ValidateSet('127.0.0.1', '0.0.0.0')]
    [string]$BindAddress = '127.0.0.1',
    # 512, not GSQ 1024: at 1024 the MTP draft compute buffer does not fit at 262K
    # (tc262-iq4xs-probe.out.err); at 512 it boots 66/66, shared 774 MiB (2026-09-24).
    [int]$UBatch = 512,
    [string]$Device = 'GPU-fba37e4b-ea9e-66e9-c3fd-a16b2e833bc4,GPU-059b90e2-2b5c-00b8-f3ba-f6dea8de083e',
    [string]$Exe = 'C:\AI\llama.cpp-blackwell\llama-server.exe',
    [int]$Ti5060ReserveMiB = 512,
    [string]$Model = 'C:\AI\models\bottlecapai-ThinkingCap-Qwen3.8-27B-IQ4_XS\ThinkingCap-Qwen3.8-27B-IQ4_XS.gguf'
)

$ErrorActionPreference = 'Stop'
if ($Ctx -ne 262144) {
    throw "ThinkingCap profile is pinned to context 262144; received $Ctx."
}
if ($UBatch -lt 256) {
    throw "UBatch must be at least 256; received $UBatch."
}
if (-not (Test-Path $Exe)) {
    throw "llama-server.exe was not found at $Exe."
}
if (-not (Test-Path $Model)) {
    throw "ThinkingCap IQ4_XS model file was not found at $Model."
}
$template = Join-Path $PSScriptRoot '..\templates\qwen38-late-system.jinja'
if (-not (Test-Path $template)) {
    throw "Claude Code-compatible chat template was not found at $template."
}

. (Join-Path $PSScriptRoot 'Get-GpuVram.ps1')
$installed = @(Get-InstalledGpu)
$wanted = @($Device -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($wanted.Count -ne 2) {
    throw "ThinkingCap needs exactly two GPUs; Device names $($wanted.Count)."
}
foreach ($uuid in $wanted) {
    if (-not ($installed | Where-Object { $_.Uuid -eq $uuid })) {
        $found = ($installed | ForEach-Object { "$($_.Uuid) $($_.Name)" }) -join '; '
        throw "GPU $uuid is not installed. Found: $found"
    }
}

# Match worker-q4-dual.ps1. The display card's desktop usage is the reason an
# equal tensor split silently fails on this pair; the idle card can carry more.
$DisplayReserveMiB = 2500
# ThinkingCap IQ4_XS is 17.4 GB: at the GSQ reserve (512) the split put ~66 % on
# the 5060 Ti and the MTP draft's 262K KV then OOMed there (tc262-probe*.out.err,
# 2026-09-24). A larger 5060 Ti reserve shifts weight to GPU0; GPU0 keeps its
# 2,500 MiB display reserve.
$IdleReserveMiB = $Ti5060ReserveMiB
$budgets = @()
$report = @()
foreach ($uuid in $wanted) {
    $gpu = Get-GpuVram -Uuid $uuid
    if (-not $gpu) {
        throw "Cannot read VRAM for $uuid; refusing to guess a tensor split."
    }
    $isDisplay = $gpu.Used -gt 500
    $reserve = if ($isDisplay) { $DisplayReserveMiB } else { $IdleReserveMiB }
    $budget = $gpu.Free - $reserve
    if ($budget -le 0) {
        throw "GPU $uuid has no usable tensor budget after reserve ($budget MiB)."
    }
    $budgets += $budget
    $report += [pscustomobject]@{
        Uuid = $uuid; Used = $gpu.Used; Free = $gpu.Free
        Reserve = $reserve; Budget = $budget; Display = $isDisplay
    }
}

$ts = $budgets -join ','
$serverArgs = @(
    '-c', "$Ctx", '-ngl', 'auto', '--fit', 'off',
    '-fa', 'on', '-np', '1', '-t', '18', '-b', '2048',
    '-ctk', 'q4_0', '-ctv', 'q4_0', '--cache-ram', '24576',
    '--ctx-checkpoints', '8', '--no-mmproj-auto', '-lv', '4',
    '--log-colors', 'on', '--min-p', '0.0', '--repeat-penalty', '1.05',
    '--reasoning-effort', 'medium', '--sse-ping-interval', '5',
    '--host', $BindAddress, '--port', "$Port",
    '-m', $Model, '--chat-template-file', $template, '--alias', 'ThinkingCap-Qwen3.8-27B-IQ4_XS-262k',
    '-sm', 'tensor', '-ts', $ts, '-ub', "$UBatch",
    '--spec-type', 'draft-mtp,ngram-mod', '--spec-draft-n-max', '3',
    '--spec-ngram-mod-n-match', '24', '--spec-ngram-mod-n-min', '16',
    '--spec-ngram-mod-n-max', '64'
)

Write-Host "ThinkingCap IQ4_XS -- context 262,144 -- tensor split $ts" -ForegroundColor Cyan
foreach ($row in $report) {
    $display = if ($row.Display) { ' display' } else { '' }
    Write-Host ("  {0}: free {1} MiB - reserve {2} = budget {3} MiB{4}" -f `
        $row.Uuid.Substring(0, 12), $row.Free, $row.Reserve, $row.Budget, $display)
}
if ($BindAddress -eq '0.0.0.0') {
    Write-Host 'WARNING: GSQ is exposed on every interface with no API key.' -ForegroundColor Yellow
}

if ($WhatIfPreference) {
    Write-Host 'WhatIf: would run' -ForegroundColor Green
    Write-Host ("  {0} {1}" -f $Exe, ($serverArgs -join ' '))
    exit 0
}

$env:CUDA_VISIBLE_DEVICES = $Device
& $Exe @serverArgs
exit $LASTEXITCODE
