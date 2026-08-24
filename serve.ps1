<#
.SYNOPSIS
    Start the best-supported Qwen3.8-27B configuration. No arguments needed.

.DESCRIPTION
    There are 58 .ps1 files under qwen38-tuning/scripts/ and nothing in the tree
    says which one is current. Several serve artifacts that stopped being the
    default and windows that stopped being the answer. This is the one to run.

    ONE SCRIPT, ONE PROCESS. The profile is invoked in THIS process and its
    output is this terminal's output. Ctrl+C stops the server; closing the
    window stops the server. An earlier version launched it detached and tailed
    the log file, which made "live terminal" mean watching a file that another
    process was writing.

    IT DELEGATES. The configuration lives in worker-q2kxl-mtp.ps1 and only
    there. This file resolves that profile and invokes it; it declares no
    serving flag of its own, because a launcher that copies them becomes a
    second source of truth and drifts the first time one changes -- silently,
    since both files still run and both still look right.
    tests/test_serve_entrypoint.py asserts the absences.

    IT VERIFIES AS THE STREAM PASSES. A foreground server never returns, so
    there is no "after boot" to check anything in. Both checks happen inline:
    the layer split when llama.cpp prints it, and the status block when
    llama.cpp says it is listening. `--fit` SPILLS rather than refusing, and
    that reads as success in every field except the layer count.

    Two orchestrators cannot share port 8080 -- an armed queue once killed a
    running corpus and the summary still printed a plausible number -- so this
    refuses to start over a port that is already answering.

.PARAMETER Lan
    Bind every interface so another machine can reach it. Off by default:
    --host is the only access control this server has (no API key, CORS '*').

.PARAMETER AllowFirewall
    Add the inbound rule, through a UAC prompt. Separate from -Lan on purpose.

.PARAMETER Detach
    Run the server in the background and hand the prompt back.

.PARAMETER WhatIf
    Print the resolved command line and exit without touching the GPU.

.EXAMPLE
    .\serve.ps1

.EXAMPLE
    .\serve.ps1 -Lan -AllowFirewall
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
    # The old behaviour, for when the terminal is needed for something else.
    # Not the default: the default should do what it looks like it does.
    [switch]$Detach,
    [int]$Port = 8080
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
      One report, called from every path. Everything here is READ, not assumed:
      this may be looking at a server it did not start, so the bind comes from
      the listening socket rather than from whether -Lan was passed now.
    #>
    param($Props, [int]$OnGpu = 0, [int]$Total = 0)

    $listen = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
                ForEach-Object { $_.LocalAddress } | Sort-Object -Unique)
    $wide = $listen -contains '0.0.0.0' -or $listen -contains '::'

    Write-Host ""
    Write-Host "Serving on $base" -ForegroundColor Green
    Write-Host ("  model     {0}  ({1})" -f $Props.model_alias, $Props.model_ftype)
    Write-Host ("  build     {0}" -f $Props.build_info)
    $nctx = $Props.default_generation_settings.n_ctx
    if ($nctx) { Write-Host ("  window    {0:N0} tokens" -f $nctx) }

    if ($Total -gt 0) {
        if ($OnGpu -eq $Total) {
            Write-Host ("  layers    {0}/{1} on the GPU" -f $OnGpu, $Total) -ForegroundColor Green
        } else {
            Write-Host ("  layers    {0}/{1} -- SPILLED" -f $OnGpu, $Total) -ForegroundColor Red
        }
    } else {
        Write-Host "  layers    not seen in this stream" -ForegroundColor DarkGray
    }

    $vram = & nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>$null
    if ($vram) { Write-Host ("  VRAM      {0}" -f ($vram -join '')) }

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

# -Verbosity 4 is asked of the PROFILE, not declared here. The served default is
# 3, which omits the tensor-assignment lines. 4 rather than 5, measured: one boot
# writes 1.7 KB at 3, 24.7 KB at 4 and 511.9 KB at 5, and the layer line is
# present from 4 up. The DEFAULT in the profile does not move -- every served row
# was measured at 3.
# A HASHTABLE, not an array. `& $script @arr` on an array splats POSITIONALLY,
# so '-Verbosity' arrived as the profile's first parameter, $Ctx, and the run
# died with: Cannot convert value "-Verbosity" to type "System.Int32". Named
# splatting needs a hashtable. Pinned by tests/test_foreground_is_the_default.py.
$profileArgs = @{ Verbosity = 4 }
if ($Lan) { $profileArgs['BindAddress'] = '0.0.0.0' }

# Start-Process wants a flat string array, so derive one rather than keeping two
# hand-written copies that can disagree about what is being launched.
$profileArgv = @($profileArgs.GetEnumerator() |
                 Sort-Object Name |
                 ForEach-Object { "-$($_.Key)", "$($_.Value)" })

if ($WhatIfPreference) {
    Write-Host "WhatIf: would run" -ForegroundColor Green
    Write-Host "  $profileScript $($profileArgv -join ' ')"
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
        Show-ServerStatus -Props $existing
        Write-Host ""
        Write-Host "It was not started by this window, so Ctrl+C here will not reach it." -ForegroundColor DarkGray
        Write-Host "Stop it with: Get-Process llama-server | Stop-Process" -ForegroundColor DarkGray
        exit 0
    }
    Write-Host "FATAL: port $Port is answering and it is NOT ours." -ForegroundColor Red
    Write-Host "  alias '$($existing.model_alias)', $($existing.model_ftype)" -ForegroundColor Yellow
    Write-Host "  Refusing to start a second server. An armed queue once killed a running" -ForegroundColor Yellow
    Write-Host "  corpus this way and the summary still printed a plausible number." -ForegroundColor Yellow
    exit 1
}

# ---- exposure, and the rule that makes it reachable --------------------------
if ($Lan) {
    Write-Host "EXPOSING on every interface. There is no API key and no origin" -ForegroundColor Yellow
    Write-Host "restriction, so anyone who can reach this port can use the GPU and" -ForegroundColor Yellow
    Write-Host "read whatever context is loaded. Developer decision, issue #49." -ForegroundColor Yellow

    # Scope, not existence. An earlier version skipped this branch whenever ANY
    # inbound rule was present, so a rule created when only Radmin was wanted
    # could never be widened -- and it reported "rule present" while the LAN
    # still timed out.
    $wanted = @('LocalSubnet', '26.0.0.0/8')
    $rule = Get-NetFirewallPortFilter -ErrorAction SilentlyContinue |
            Where-Object { $_.LocalPort -eq $Port } |
            ForEach-Object { $_ | Get-NetFirewallRule -ErrorAction SilentlyContinue } |
            Where-Object { $_.Direction -eq 'Inbound' -and $_.Enabled -eq 'True' } |
            Where-Object {
                $scope = @($_ | Get-NetFirewallAddressFilter | ForEach-Object { $_.RemoteAddress })
                ($scope -contains 'Any') -or
                (($wanted | Where-Object { $scope -notcontains $_ }).Count -eq 0)
            }

    if (-not $rule) {
        Write-Host "No inbound rule for port $Port admitting both LocalSubnet and 26.0.0.0/8." -ForegroundColor Yellow
        Write-Host "The bind will succeed and the connection will time out, which looks like" -ForegroundColor Yellow
        Write-Host "a model problem and is not." -ForegroundColor Yellow

        if ($AllowFirewall) {
            Write-Host "LocalSubnet means whatever network this machine is on, and the Wi-Fi" -ForegroundColor Yellow
            Write-Host "adapter is classified Public -- so the rule follows the laptop onto any" -ForegroundColor Yellow
            Write-Host "network you join, not only this one." -ForegroundColor Yellow
            Write-Host "Asking Windows for permission to add it -- accept the prompt." -ForegroundColor Cyan
            # By PREFIX. An earlier release named this rule 'llama-server 8080
            # (Radmin)'; removing only the current name left that one behind, so
            # two rules existed and Windows evaluated their union.
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
            Write-Host "  New-NetFirewallRule -DisplayName 'llama-server $Port' -Direction Inbound ``" -ForegroundColor Cyan
            Write-Host "    -Protocol TCP -LocalPort $Port -Action Allow -Profile Any ``" -ForegroundColor Cyan
            Write-Host "    -RemoteAddress LocalSubnet,26.0.0.0/8" -ForegroundColor Cyan
        }
    } else {
        Write-Host "Inbound rule present: $($rule.DisplayName -join ', ')" -ForegroundColor Green
    }
    Write-Host ""
}

if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
# InvariantCulture: a Thai locale renders the year as 2569 and the log names stop
# sorting next to every other dated artifact in this repository.
$stamp = [datetime]::Now.ToString('yyyyMMdd-HHmmss', [cultureinfo]::InvariantCulture)
$log   = Join-Path $logDir "serve-$stamp.log"

# ---- -Detach: the server outlives this window -------------------------------
if ($Detach) {
    Write-Host "Detaching. The server will outlive this window; the log is $log" -ForegroundColor DarkGray
    $proc = Start-Process -FilePath 'pwsh' `
        -ArgumentList (@('-NoProfile', '-File', $profileScript) + $profileArgv) `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err" `
        -WindowStyle Hidden -PassThru
    $props = $null
    for ($i = 0; $i -lt 80; $i++) {
        Start-Sleep -Seconds 3
        if ($proc.HasExited) {
            Write-Host "FATAL: the profile exited during boot (code $($proc.ExitCode))." -ForegroundColor Red
            if (Test-Path "$log.err") { Get-Content "$log.err" -Tail 25 }
            exit 1
        }
        $props = Get-ServerProps
        if ($props) { break }
    }
    if (-not $props) {
        Write-Host "FATAL: no response on $base." -ForegroundColor Red
        if (Test-Path "$log.err") { Get-Content "$log.err" -Tail 25 }
        exit 1
    }
    $split = Select-String -Path "$log.err" -Pattern 'offloaded (\d+)/(\d+) layers to GPU' |
             Select-Object -Last 1
    if ($split) {
        Show-ServerStatus -Props $props -OnGpu ([int]$split.Matches[0].Groups[1].Value) `
                          -Total ([int]$split.Matches[0].Groups[2].Value)
    } else {
        Show-ServerStatus -Props $props
    }
    Write-Host ""
    Write-Host "Stop it with: Get-Process llama-server | Stop-Process" -ForegroundColor DarkGray
    exit 0
}

# ---- foreground: this process IS the server ---------------------------------
Write-Host "Starting. Ctrl+C stops the server; so does closing this window." -ForegroundColor Cyan
Write-Host "A copy of this output is kept at $log" -ForegroundColor DarkGray
Write-Host ("-" * 78) -ForegroundColor DarkGray

$script:onGpu = 0
$script:total = 0
$script:reported = $false

# The two checks happen INLINE, because a foreground server never returns and
# there is no "after boot" to do them in.
& $profileScript @profileArgs 2>&1 | ForEach-Object {
    $line = "$_"
    Add-Content -Path $log -Value $line -ErrorAction SilentlyContinue
    Write-Host $line

    if (-not $script:reported) {
        if ($line -match 'offloaded (\d+)/(\d+) layers to GPU') {
            $script:onGpu = [int]$Matches[1]
            $script:total = [int]$Matches[2]
            if ($script:onGpu -ne $script:total) {
                Write-Host ""
                Write-Host "SPILLED: only $($script:onGpu) of $($script:total) layers are on the GPU." -ForegroundColor Red
                Write-Host "  --fit spilled rather than refusing. Numbers taken past this point are" -ForegroundColor Yellow
                Write-Host "  not comparable to anything measured while fully resident." -ForegroundColor Yellow
                Write-Host ""
            }
        }
        # llama.cpp's own readiness line. Printing the status before it would
        # describe a server that is not answering yet.
        if ($line -match 'listening on http') {
            $props = Get-ServerProps
            if ($props) {
                Show-ServerStatus -Props $props -OnGpu $script:onGpu -Total $script:total
                Write-Host ""
                Write-Host ("-" * 78) -ForegroundColor DarkGray
            }
            $script:reported = $true
        }
    }
}
