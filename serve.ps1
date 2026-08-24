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
    # Bind to every interface so another machine on the developer's own network
    # can reach it. OFF by default and never implied: --host is the only access
    # control this server has (no API key, CORS '*'), so this removes it rather
    # than loosening it. Issue #49.
    [switch]$Lan,
    # Add the inbound firewall rule this needs, through a UAC prompt. SEPARATE
    # from -Lan on purpose: a single switch that binds wide AND edits the
    # firewall means nobody ever chose the second thing. The rule admits two
    # named networks and no more -- Radmin VPN's 26.0.0.0/8 and the local
    # subnet -- because those are what was asked for, and a rule wider than the
    # request is a rule nobody granted.
    [switch]$AllowFirewall,
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

function Show-ServerStatus {
    <#
      One report, called from BOTH paths. The already-serving branch used to
      print two lines and exit, which is the branch taken most often: a fresh
      boot happens once, "is it up and how is it doing" is asked all day.

      Everything here is READ, not assumed. This function may be looking at a
      server it did not start, so it takes the bind from the listening socket
      rather than from whether -Lan was passed on this invocation -- a -Lan now
      does not change a server that came up on loopback an hour ago.
    #>
    param($Props, $ResidencyLog)

    $listen = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
                ForEach-Object { $_.LocalAddress } | Sort-Object -Unique)
    $wide = $listen -contains '0.0.0.0' -or $listen -contains '::'

    Write-Host ""
    Write-Host "Serving on $base" -ForegroundColor Green
    Write-Host ("  model     {0}  ({1})" -f $Props.model_alias, $Props.model_ftype)
    Write-Host ("  build     {0}" -f $Props.build_info)
    $nctx = $Props.default_generation_settings.n_ctx
    if ($nctx) { Write-Host ("  window    {0:N0} tokens" -f $nctx) }

    # Residency: from the boot log if one is at hand. Absent rather than guessed
    # -- --fit spills instead of refusing, so an assumed 66/66 is exactly the
    # field that reads as success while the card thrashes.
    $split = $null
    if ($ResidencyLog) {
        $streams = @($ResidencyLog) | Where-Object { $_ -and (Test-Path $_) }
        if ($streams) {
            $split = Select-String -Path $streams -Pattern 'offloaded (\d+)/(\d+) layers to GPU' |
                     Select-Object -Last 1
        }
    }
    if ($split) {
        $onGpu = [int]$split.Matches[0].Groups[1].Value
        $total = [int]$split.Matches[0].Groups[2].Value
        if ($onGpu -eq $total) {
            Write-Host ("  layers    {0}/{1} on the GPU" -f $onGpu, $total) -ForegroundColor Green
        } else {
            Write-Host ("  layers    {0}/{1} -- SPILLED" -f $onGpu, $total) -ForegroundColor Red
        }
    } else {
        Write-Host "  layers    not checked -- no boot log for this process" -ForegroundColor DarkGray
    }

    $vram = & nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>$null
    if ($vram) { Write-Host ("  VRAM      {0}" -f ($vram -join '')) }

    # What the conversation has actually reached, which is the number that says
    # whether the window we paid for is being used.
    try {
        $slots = Invoke-RestMethod -Uri "$base/slots" -TimeoutSec 4
        $used = @($slots | ForEach-Object { $_.n_prompt_tokens }) | Where-Object { $_ }
        if ($used) { Write-Host ("  context   {0:N0} tokens in the live slot" -f ($used | Measure-Object -Maximum).Maximum) }
    } catch { }

    Write-Host ("  bind      {0}" -f ($listen -join ', ')) -ForegroundColor $(if ($wide) { 'Yellow' } else { 'Green' })
    if ($wide) {
        Write-Host "  reachable from another machine at:" -ForegroundColor Yellow
        Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
            Where-Object { $_.IPAddress -ne '127.0.0.1' } |
            Sort-Object InterfaceAlias |
            ForEach-Object { Write-Host ("    http://{0}:{1}   ({2})" -f $_.IPAddress, $Port, $_.InterfaceAlias) }
    }

    Write-Host ""
    Write-Host "It keeps running in the background -- this terminal is free and closing it" -ForegroundColor DarkGray
    Write-Host "does not stop the server. Stop it with: Get-Process llama-server | Stop-Process" -ForegroundColor DarkGray
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
    Write-Host "  pwsh -NoProfile -File `"$profileScript`" -Verbosity 4$(if ($Lan) { ' -BindAddress 0.0.0.0' })"
    Write-Host ""
    Write-Host "The flags themselves are in that file. Read it there, not here --" -ForegroundColor DarkGray
    Write-Host "a copy in this script is how the two stop agreeing." -ForegroundColor DarkGray
    exit 0
}

# ---- refuse to start over a port that is already answering -------------------
$existing = Get-ServerProps
if ($existing) {
    if ($existing.model_alias -eq 'qwen38') {
        Write-Host "Already serving. Restarting a healthy server is not an improvement." -ForegroundColor Green
        # The newest boot log this launcher wrote, if any. It may be a server we
        # did not start, in which case residency reads "not checked" rather than
        # being invented.
        $prev = Get-ChildItem (Join-Path $logDir 'serve-*.log.err') -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime | Select-Object -Last 1
        Show-ServerStatus -Props $existing -ResidencyLog $(if ($prev) { $prev.FullName })
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
$profileArgs = @('-NoProfile', '-File', $profileScript, '-Verbosity', '4')
if ($Lan) {
    $profileArgs += @('-BindAddress', '0.0.0.0')
    Write-Host "EXPOSING on every interface. There is no API key and no origin" -ForegroundColor Yellow
    Write-Host "restriction, so anyone who can reach this port can use the GPU and" -ForegroundColor Yellow
    Write-Host "read whatever context is loaded. Developer decision, issue #49." -ForegroundColor Yellow

    # The rule cannot be added from here -- it needs elevation. Printing the
    # command is not the same as running it, and a launcher does not get to
    # edit firewall state on its own.
    # Scope, not existence. The first version skipped this whole branch when ANY
    # inbound rule was present, so a rule created when only Radmin was wanted
    # could never be widened -- and it would report "rule present" while the LAN
    # still timed out. What matters is which remote addresses it admits.
    $wanted = @('LocalSubnet', '26.0.0.0/8')
    $rule = Get-NetFirewallPortFilter -ErrorAction SilentlyContinue |
            Where-Object { $_.LocalPort -eq $Port } |
            ForEach-Object { $_ | Get-NetFirewallRule -ErrorAction SilentlyContinue } |
            Where-Object { $_.Direction -eq 'Inbound' -and $_.Enabled -eq 'True' } |
            Where-Object {
                $scope = @($_ | Get-NetFirewallAddressFilter | ForEach-Object { $_.RemoteAddress })
                # 'Any' admits everything, so it covers both. Otherwise every
                # wanted network must appear.
                ($scope -contains 'Any') -or
                (($wanted | Where-Object { $scope -notcontains $_ }).Count -eq 0)
            }
    if (-not $rule) {
        Write-Host ""
        Write-Host "No inbound rule for port $Port admitting both LocalSubnet and 26.0.0.0/8." -ForegroundColor Yellow
        Write-Host "The bind will succeed and the connection will time out, which looks like" -ForegroundColor Yellow
        Write-Host "a model problem and is not." -ForegroundColor Yellow

        if ($AllowFirewall) {
            # -Verb RunAs, not a silent edit. The agent is not an administrator
            # and must not try to become one quietly; the consent dialog is what
            # makes this authorised rather than sneaked in.
            Write-Host "LocalSubnet means whatever network this machine is on, and the Wi-Fi" -ForegroundColor Yellow
            Write-Host "adapter is classified Public -- so the rule follows the laptop onto any" -ForegroundColor Yellow
            Write-Host "network you join, not only this one." -ForegroundColor Yellow
            Write-Host "Asking Windows for permission to add it -- accept the prompt." -ForegroundColor Cyan
            # Remove first: New-NetFirewallRule with an existing DisplayName adds
            # a SECOND rule rather than replacing it, and Windows evaluates the
            # union -- so a narrower old rule would sit there looking authoritative.
            # By PREFIX, not by exact name. The first release called this rule
            # 'llama-server 8080 (Radmin)'; removing only the current name left
            # that one behind, so two rules existed and Windows evaluated their
            # union -- a stale narrower rule sitting next to the real one and
            # looking just as authoritative.
            $add = "Remove-NetFirewallRule -DisplayName 'llama-server $Port*' -ErrorAction SilentlyContinue; " +
                   "New-NetFirewallRule -DisplayName 'llama-server $Port' " +
                   "-Direction Inbound -Protocol TCP -LocalPort $Port -Action Allow " +
                   "-Profile Any -RemoteAddress LocalSubnet,26.0.0.0/8"
            try {
                Start-Process pwsh -Verb RunAs -Wait -WindowStyle Hidden `
                    -ArgumentList '-NoProfile', '-Command', $add
            } catch {
                Write-Host "Elevation was refused or failed: $_" -ForegroundColor Red
            }

            # A UAC dialog can be dismissed. Re-check rather than reporting
            # success because a command was launched.
            $rule = Get-NetFirewallRule -ErrorAction SilentlyContinue |
                    Where-Object { $_.DisplayName -eq "llama-server $Port" -and $_.Enabled -eq 'True' }
            if ($rule) {
                $scope = @($rule | Get-NetFirewallAddressFilter | ForEach-Object { $_.RemoteAddress })
                Write-Host "Rule added and enabled. Admits: $($scope -join ', ')" -ForegroundColor Green
            } else {
                Write-Host "The rule is STILL NOT THERE. Remote machines will time out." -ForegroundColor Red
                Write-Host "  Run this yourself in an elevated shell:" -ForegroundColor Yellow
                Write-Host "  $add" -ForegroundColor Cyan
            }
        } else {
            Write-Host "Re-run with -AllowFirewall to add it, or run this elevated:" -ForegroundColor Yellow
            Write-Host ""
            Write-Host "  New-NetFirewallRule -DisplayName 'llama-server $Port' -Direction Inbound ``" -ForegroundColor Cyan
            Write-Host "    -Protocol TCP -LocalPort $Port -Action Allow -Profile Any ``" -ForegroundColor Cyan
            Write-Host "    -RemoteAddress LocalSubnet,26.0.0.0/8" -ForegroundColor Cyan
            Write-Host ""
        }
    } else {
        Write-Host "Inbound rule present: $($rule.DisplayName -join ', ')" -ForegroundColor Green
    }
}

Write-Host "Starting. Log: $log"
$proc = Start-Process -FilePath 'pwsh' `
    -ArgumentList $profileArgs `
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

Show-ServerStatus -Props $props -ResidencyLog $errLog
