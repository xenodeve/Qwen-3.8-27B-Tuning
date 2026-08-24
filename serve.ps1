<#
.SYNOPSIS
    Start the best-supported Qwen3.8-27B configuration. No arguments needed.

.DESCRIPTION
    There are 58 .ps1 files under qwen38-tuning/scripts/ and nothing in the tree
    says which one is current. Several serve artifacts that stopped being the
    default and windows that stopped being the answer. This is the one to run.

    IT DELEGATES. The configuration lives in worker-q2kxl-mtp.ps1 and only
    there. This file resolves that profile and invokes it; it declares no
    serving flag of its own, because a launcher that copies them becomes a
    second source of truth and drifts the first time one changes -- silently,
    since both files still run and both still look right.
    tests/test_serve_entrypoint.py asserts the absences.

    IT VERIFIES RATHER THAN ASSUMES. Two incidents in this repository's history
    apply directly:

      * Two orchestrators cannot share port 8080. An armed queue once killed a
        running corpus and the summary still printed a plausible number. So this
        refuses to start over a port that is already answering.

      * A projection at context 163,840 was accepted before a boot showed
        64 of 66 layers resident. `--fit` SPILLS rather than refusing, and that
        reads as success in every field except the layer count. So this reads
        the split back out of the log and treats anything but full residency as
        a failure.

.PARAMETER WhatIf
    Print the resolved command line and exit without touching the GPU.

.EXAMPLE
    .\serve.ps1

.EXAMPLE
    .\serve.ps1 -WhatIf
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [int]$Port = 8080,
    [int]$BootTimeoutSeconds = 240
)

$ErrorActionPreference = 'Stop'

$profileScript = Join-Path $PSScriptRoot 'qwen38-tuning\scripts\worker-q2kxl-mtp.ps1'
$logDir        = Join-Path $PSScriptRoot 'qwen38-tuning\logs'
$base          = "http://127.0.0.1:$Port"

if (-not (Test-Path $profileScript)) {
    Write-Host "FATAL: the profile is missing: $profileScript" -ForegroundColor Red
    Write-Host "  This launcher holds no configuration of its own on purpose." -ForegroundColor Yellow
    exit 1
}

function Get-ServerProps {
    try { Invoke-RestMethod -Uri "$base/props" -TimeoutSec 4 } catch { $null }
}

# ---- what this is, and what is still open ------------------------------------
Write-Host ""
Write-Host "Qwen3.8-27B on RTX 5060 Ti 16 GB -- the configuration the evidence supports" -ForegroundColor Cyan
Write-Host "  profile   $profileScript"
Write-Host ""
Write-Host "  artifact  UD-Q2_K_XL. An external ladder puts a 10-point cliff between this"
Write-Host "            and UD-IQ2_XXS. OUR OWN quality number for it does not exist."
Write-Host "  window    147,456, boot-verified fully resident. Real use has reached 85,923."
Write-Host "  effort    medium. Chosen on the agentic axis, where xhigh costs one point and"
Write-Host "            low costs six. NEVER MEASURED on any artifact here."
Write-Host "  KV        q4_0. Not a preference -- our build compiles only f16, bf16, q4_0"
Write-Host "            and q8_0 for flash attention (issue #43)."
Write-Host "  draft     3, the default. 7 was measured at -56 % on the MTP head."
Write-Host ""
Write-Host "  OPEN: the draft-mtp half of the decoder is under question." -ForegroundColor Yellow
Write-Host "        Forced at 147,456, removing it is worth +15.6 % and 1,490 MiB." -ForegroundColor Yellow
Write-Host "        The one natural round, at 98,304, says keeping it is worth +127 %." -ForegroundColor Yellow
Write-Host "        Two variables moved between those numbers. Issues #44 and #47." -ForegroundColor Yellow
Write-Host ""

if ($WhatIfPreference) {
    Write-Host "WhatIf: would run" -ForegroundColor Green
    Write-Host "  pwsh -NoProfile -File `"$profileScript`""
    Write-Host ""
    Write-Host "The flags themselves are in that file. Read it there, not here --" -ForegroundColor DarkGray
    Write-Host "a copy in this script is how the two stop agreeing." -ForegroundColor DarkGray
    exit 0
}

# ---- refuse to start over a port that is already answering -------------------
$existing = Get-ServerProps
if ($existing) {
    if ($existing.model_alias -eq 'qwen38') {
        Write-Host "Already serving on port $Port -- alias '$($existing.model_alias)', $($existing.model_ftype), build $($existing.build_info)." -ForegroundColor Green
        Write-Host "Restarting a healthy server is not an improvement. Nothing to do." -ForegroundColor Green
        exit 0
    }
    Write-Host "FATAL: port $Port is answering and it is NOT ours." -ForegroundColor Red
    Write-Host "  alias '$($existing.model_alias)', $($existing.model_ftype)" -ForegroundColor Yellow
    Write-Host "  Refusing to start a second server. An armed queue once killed a running" -ForegroundColor Yellow
    Write-Host "  corpus this way and the summary still printed a plausible number." -ForegroundColor Yellow
    exit 1
}

# ---- launch ------------------------------------------------------------------
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
# InvariantCulture: a Thai locale renders the year as 2569 and the log names
# stop sorting next to every other dated artifact in this repository.
$stamp  = [datetime]::Now.ToString('yyyyMMdd-HHmmss', [cultureinfo]::InvariantCulture)
$log    = Join-Path $logDir "serve-$stamp.log"
$errLog = "$log.err"

# -Verbosity 4 is asked of the PROFILE, not declared here. The served default is
# 3, which omits the tensor-assignment lines, so the first version of this
# launcher reported residency UNVERIFIED on every boot -- it was looking for a
# line the profile never writes.
#
# 4 rather than 5, measured rather than assumed: one boot writes 1.7 KB at
# verbosity 3, 24.7 KB at 4, and 511.9 KB at 5, and the layer line is present
# from 4 up. Five buys nothing here and costs 20x, on a server that runs for
# hours. The DEFAULT in the profile does not move: every served row was measured
# at 3.
Write-Host "Starting. Log: $log"
$proc = Start-Process -FilePath 'pwsh' `
    -ArgumentList '-NoProfile', '-File', $profileScript, '-Verbosity', '4' `
    -RedirectStandardOutput $log -RedirectStandardError $errLog `
    -WindowStyle Hidden -PassThru

$deadline = (Get-Date).AddSeconds($BootTimeoutSeconds)
$props = $null
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 3
    if ($proc.HasExited) {
        Write-Host "FATAL: the profile exited during boot (code $($proc.ExitCode))." -ForegroundColor Red
        foreach ($f in @($errLog, $log)) {
            if ((Test-Path $f) -and (Get-Item $f).Length -gt 0) { Get-Content $f -Tail 25 }
        }
        exit 1
    }
    $props = Get-ServerProps
    if ($props) { break }
}

if (-not $props) {
    Write-Host "FATAL: no response on $base after $BootTimeoutSeconds s." -ForegroundColor Red
    foreach ($f in @($errLog, $log)) {
        if ((Test-Path $f) -and (Get-Item $f).Length -gt 0) { Get-Content $f -Tail 25 }
    }
    exit 1
}

# ---- verify residency from the log, do not assume it -------------------------
# llama.cpp logs to STDERR. The first version of this searched stdout, found an
# empty file, and reported residency unverified while the evidence sat next to it.
$streams = @($errLog, $log) | Where-Object { Test-Path $_ }
$split = Select-String -Path $streams -Pattern 'offloaded (\d+)/(\d+) layers to GPU' |
         Select-Object -Last 1
if (-not $split) {
    Write-Host "WARNING: no layer-assignment line in either stream -- residency UNVERIFIED." -ForegroundColor Yellow
    Write-Host "  The server is up and this script cannot tell you whether it spilled." -ForegroundColor Yellow
    Write-Host "  Looked in: $($streams -join ', ')" -ForegroundColor Yellow
} else {
    $onGpu = [int]$split.Matches[0].Groups[1].Value
    $total = [int]$split.Matches[0].Groups[2].Value
    if ($onGpu -eq $total) {
        Write-Host "Resident: $onGpu/$total layers on the GPU." -ForegroundColor Green
    } else {
        Write-Host "FAILED: only $onGpu of $total layers are on the GPU." -ForegroundColor Red
        Write-Host "  --fit spilled rather than refusing. Decode past this point is not the" -ForegroundColor Yellow
        Write-Host "  configuration you asked for, and the numbers it produces are not" -ForegroundColor Yellow
        Write-Host "  comparable to anything measured while fully resident." -ForegroundColor Yellow
        exit 1
    }
}

Write-Host ""
Write-Host "Ready on $base -- alias '$($props.model_alias)', $($props.model_ftype), build $($props.build_info)." -ForegroundColor Green
Write-Host "Stop it with: Get-Process llama-server | Stop-Process" -ForegroundColor DarkGray
