<#
.SYNOPSIS
    Start the best-supported Qwen3.8-27B configuration. No arguments needed.

.DESCRIPTION
    There are 58 .ps1 files under qwen38-tuning/scripts/ and nothing in the tree
    says which one is current. Several serve artifacts that stopped being the
    default and windows that stopped being the answer. This is the one to run.

    ONE WINDOW, ONE SERVER. The profile is invoked in THIS process and its
    output is this terminal's output. Ctrl+C stops the server; closing the
    window stops the server; there is no mode in which it survives either.

    That is a simplification, not only a behaviour. Every way a server could
    outlive its terminal needed handling -- a detach mode, a branch reporting on
    a server this window did not start, a hunt through old logs for the
    residency of a process nobody watched. Removing the situation removed all
    three.

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

.PARAMETER Dual
    Serve UD-Q4_K_XL across both cards instead of UD-Q2_K_XL on one. Costs about
    a third of raw decode and ~130 W for an artifact one card cannot hold.

.PARAMETER Dflash
    With -Dual, serve draft-dflash beside ngram-mod on the patched binary.
    Window capped at 131,072; cannot be combined with -MaxCtx or -Mtp.

.PARAMETER MaxCtx
    With -Dual, serve the deepest context the current free VRAM supports, capped
    at the model's n_ctx_train of 262,144. Computed at launch, not fixed.

.PARAMETER Mtp
    With -Dual, add draft-mtp beside ngram-mod. It runs, and its rate could not
    be measured here -- every paired round was voided because the generations
    copy the prompt.

.PARAMETER Device
    Which GPU, as a UUID. Empty means "whatever the profile serves", which is
    where the default lives -- this script holds no serving configuration.
    Pass a comma-separated list to use more than one card.

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
    # Serve UD-Q4_K_XL across BOTH cards instead of UD-Q2_K_XL on one.
    # OFF by default and it is not a performance switch: it trades about a third
    # of raw decode (20.9 vs 32.0 tok/s, ctx 16,384, speculation off) for an
    # artifact 16.69 GiB in size that one 16 GB card cannot hold at any depth,
    # and it draws roughly 130 W more. Quality is the whole reason to pay that
    # and this project has never measured it on its own artifacts.
    # Which profile is the default is the developer's call, so neither is
    # implied by the other. Issue #52.
    [switch]$Dual,
    # With -Dual, serve draft-mtp beside ngram-mod. It runs on the tensor split
    # -- measured 2026-08-27, after this project wrongly recorded that it could
    # not -- and its rate could NOT be measured: every paired round was voided
    # because the generations copy the prompt. Issue #52.
    # With -Dual, serve the deepest context the current free VRAM supports,
    # capped at the model's n_ctx_train of 262,144. The window is computed at
    # launch because the budget moves with what the desktop is holding.
    # With -Dual, serve draft-dflash beside ngram-mod on the PATCHED binary.
    # +123.8 % over ngram-mod at ctx 65,536, and it costs a window capped at
    # 131,072, a binary nobody outside this project has reviewed, and almost all
    # the headroom. Its own pair of launchers; never a default.
    [switch]$Dflash,
    [switch]$Nvfp4,
    [switch]$Deep,
    [switch]$Vision,
    [switch]$Lean,
    [switch]$MaxCtx,
    [switch]$Mtp,
    # WHICH CARD, when you want one other than the served default. Deliberately
    # EMPTY here rather than carrying the UUID: this script holds no serving
    # flag, so that a measured row and a served session cannot diverge by
    # someone editing the launcher. The default lives beside every other
    # serving default in worker-q2kxl-mtp.ps1.
    # Pinned by bench/tests/test_the_launch_names_its_gpu.py (issue #50).
    [string]$Device = '',
    [int]$Port = 8080
)

$ErrorActionPreference = 'Stop'

# Pass llama.cpp's colours through verbatim. MEASURED 2026-08-25: with
# --log-colors on it emits 1,180 escape bytes in four codes -- blue timestamps,
# green INFO, magenta warnings, reset -- and they still vanished from a capture,
# because PowerShell 7 strips ANSI at render time whenever the output is not a
# console ($PSStyle.OutputRendering defaulted to PlainText). The colours were
# being removed by the thing forwarding them.
if ($PSStyle) { $PSStyle.OutputRendering = 'Ansi' }

# The launcher SELECTS a profile; it never carries one's flags. Both files hold
# their own defaults, so a measured row and a served session cannot diverge by
# someone editing this script (pinned by bench/tests/test_serve_entrypoint.py).
$profileName   = if ($Dual) { 'worker-q4-dual.ps1' } else { 'worker-q2kxl-mtp.ps1' }
$profileScript = Join-Path $PSScriptRoot "qwen38-tuning\scripts\$profileName"
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

# One copy, dot-sourced here and by the watcher. Two processes need the same
# report and the watcher cannot call a function defined in this file.
$statusScript = Join-Path $PSScriptRoot 'qwen38-tuning\scripts\Show-ServerStatus.ps1'
. $statusScript

# ---- make the server die with this terminal ---------------------------------
# MEASURED 2026-08-25: it did not. The chain is cmd.exe -> pwsh.exe ->
# llama-server.exe, and killing the top cmd left both descendants running and
# the server still answering /props. Windows does not propagate a parent's death
# down the tree. The launcher was printing "closing this window stops the
# server" over an unchecked condition.
#
# A try/finally is not enough: it runs on a clean exit and on Ctrl+C, and not at
# all when the process is killed outright, which is what closing a console can
# amount to. A job object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE (0x2000) is
# enforced by the KERNEL -- when the last handle closes, however the holder
# died, every process in the job is terminated.
Add-Type -ErrorAction SilentlyContinue -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class KillOnClose {
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode)]
    static extern IntPtr CreateJobObject(IntPtr a, string name);
    [DllImport("kernel32.dll")]
    static extern bool SetInformationJobObject(IntPtr job, int cls, IntPtr info, uint len);
    [DllImport("kernel32.dll")]
    static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll")]
    static extern IntPtr OpenProcess(uint access, bool inherit, int pid);

    static IntPtr job = IntPtr.Zero;

    // Held for the life of this process on purpose. The handle closing is the
    // event that kills the job, so it must not be released early.
    public static bool Adopt(int pid) {
        if (job == IntPtr.Zero) {
            job = CreateJobObject(IntPtr.Zero, null);
            if (job == IntPtr.Zero) return false;
            // JOBOBJECT_EXTENDED_LIMIT_INFORMATION, LimitFlags at offset 16 on
            // both 32- and 64-bit; 0x2000 = KILL_ON_JOB_CLOSE.
            int size = Marshal.SizeOf(typeof(long)) * 6 + Marshal.SizeOf(typeof(IntPtr)) * 6 + 48;
            IntPtr info = Marshal.AllocHGlobal(size);
            for (int i = 0; i < size; i++) Marshal.WriteByte(info, i, 0);
            Marshal.WriteInt32(info, 16, 0x2000);
            SetInformationJobObject(job, 9, info, (uint)size);
        }
        IntPtr h = OpenProcess(0x1F0FFF, false, pid);   // PROCESS_ALL_ACCESS
        if (h == IntPtr.Zero) return false;
        return AssignProcessToJobObject(job, h);
    }
}
"@

# ---- what this is, and what is still open ------------------------------------
Write-Host ""
if ($Dual -and $Nvfp4) {
    Write-Host "Qwen3.8-27B NVFP4 + baked-in MTP across BOTH cards -- 13.84 GiB, ceiling 200,704" -ForegroundColor Cyan
} elseif ($Dual) {
    Write-Host "Qwen3.8-27B UD-Q4_K_XL across BOTH cards -- 16.69 GiB, resident to 229,376" -ForegroundColor Cyan
} else {
    Write-Host "Qwen3.8-27B on RTX 5060 Ti 16 GB -- the configuration the evidence supports" -ForegroundColor Cyan
}
Write-Host "  profile   $profileScript"
Write-Host ""

# The description branches with the profile. It used to be one unconditional
# block, and `-Dual -WhatIf` printed "artifact UD-Q2_K_XL" underneath a line
# that had just selected worker-q4-dual.ps1 -- a launcher stating something
# false about the run it was introducing, which is the failure shipped once
# already (commit b55699c: it printed that closing the window stops the server,
# and it did not). Pinned by test_the_dual_profile_serves_both_cards.py.
if ($Dual) {
    if ($Nvfp4) {
        Write-Host "  artifact  NVFP4 VERY-LOW, 13.84 GiB, with the nextn head INSIDE the file."
        Write-Host "            448 NVFP4 tensors -- the only weight format that reaches this"
        Write-Host "            build's Blackwell path, and the 4070 runs it too."
        Write-Host "            OUR OWN quality number for it does not exist. Nor for any"
        Write-Host "            other artifact here -- but this one is a FILE change." -ForegroundColor Yellow
    } else {
        Write-Host "  artifact  UD-Q4_K_XL, 16.69 GiB. ONE 16 GB card cannot hold it at any"
        Write-Host "            depth -- it spills 11 layers and decodes 11.7 tok/s."
        Write-Host "            OUR OWN quality number for it does not exist either."
    }
    # No window line here. The PROFILE resolves it -- with -MaxCtx it is
    # computed from free VRAM at launch -- and printing a static 147,456 beside
    # the profile's own line gave two contradictory windows four rows apart.
    # Fifth instance of trap 17: the launcher describing what it does not own.
    Write-Host "  split     -sm tensor, +29.2 % over -sm layer at this depth." -ForegroundColor Green
    Write-Host "            EXPERIMENTAL in llama.cpp's own help. The ratio is computed"
    Write-Host "            at launch from free VRAM -- an even split gave 0.38 tok/s."
    if ($Dflash) {
        # A different decoder, a different binary and a different window. Saying
        # the ngram-mod story here would be the fifth launcher lie this project
        # has caught by RUNNING the launcher rather than reading it.
        Write-Host "  rate      65.1 / 64.3 / 63.8 tok/s at 65,536, spread 2.0 %," -ForegroundColor Green
        Write-Host "            +123.8 % [+121.9, +125.1] over the ngram-mod the other" -ForegroundColor Green
        Write-Host "            dual launchers serve. Three paired rounds, real vendor code."
        Write-Host "  window    131,072 -- CAPPED, and not by a budget." -ForegroundColor Yellow
        Write-Host "            147,456 LOADS, answers /health, and dies on the first real"
        Write-Host "            request. 163,840 does not load at all."
        Write-Host "  binary    llama.cpp-mirror -- a LOCAL PATCH mirroring the output" -ForegroundColor Yellow
        Write-Host "            projection, so TOP_K can read logits the split would"
        Write-Host "            otherwise scatter across both cards. Reviewed by nobody"
        Write-Host "            outside this project. It costs 1,080 MiB, measured."
        Write-Host "  headroom  about 600 MiB per card after a large request, against" -ForegroundColor Yellow
        Write-Host "            ~2,210 for the served configuration. 336 died here; 488 lived."
    } elseif ($Nvfp4) {
        Write-Host "  rate      39.4 / 42.6 / 42.6 tok/s at 147,456, spread 8.1 %," -ForegroundColor Green
        Write-Host "            +63.1 % [+58.3, +65.6] over the incumbent measured in the" -ForegroundColor Green
        Write-Host "            SAME rounds, whose own spread was 3.3 %. Three paired"
        Write-Host "            rounds rotated, real vendor code, this binary."
        Write-Host "  decoder   draft-mtp from inside the file, beside ngram-mod at" -ForegroundColor Green
        Write-Host "            n-match 24 -- NOT the 12 every other profile serves. 12"
        Write-Host "            won on the Q4 and gives away a third of the gain here."
        if ($Deep) {
            Write-Host "  window    200,704 -- the MEASURED ceiling, not a budget answer." -ForegroundColor Yellow
            Write-Host "            91,428 tokens through it, finishing 1,133 and 654 MiB free."
            Write-Host "            229,376 LOADS, answers /health and DIES on the request."
            Write-Host "  headroom  THIS IS THE COST. 654 MiB is not far above the line this" -ForegroundColor Yellow
            Write-Host "            project measured: 336 died on a first request, 488 lived."
            Write-Host "            The profile re-checks the budget at launch and refuses"
            Write-Host "            rather than spilling, so a busy desktop stops it."
        } else {
            Write-Host "  window    147,456; the ceiling is 200,704 and serve-dual-nvfp4-deep.bat"
            Write-Host "            serves it. That rung took a 91,428-token request and"
            Write-Host "            finished with 1,133 and 654 MiB free. 229,376 does not."
            Write-Host "  headroom  MORE than the incumbent: about 2,395 MiB free after a large" -ForegroundColor Green
            Write-Host "            request against about 2,010. The smaller file is real."
        }
    } else {
        Write-Host "  rate      25.5 / 25.4 / 26.4 tok/s at 147,456, spread 3.7 %,"
        Write-Host "            against 21.8 with no speculation at all."
        Write-Host "            -Mtp adds draft-mtp: it RUNS, and every paired round of it"
        Write-Host "            was voided because the generations copy the prompt."
    }
    if ($Vision) {
        Write-Host "  images    ON. The vision tower is a SECOND MODEL of 888 MiB and it" -ForegroundColor Yellow
        Write-Host "            lands on a card -- --mmproj-offload defaults to enabled."
        Write-Host "            Without it the server answers HTTP 500 to any image:"
        Write-Host "            'image input is not supported'. The model itself was"
        Write-Host "            never the limitation; its chat template handles images."
    }
    if ($Lean) {
        Write-Host "  bundle    -Lean: UNMEASURED. Six settings borrowed whole from Unsloth" -ForegroundColor Yellow
        Write-Host "            Studio, which runs this same model file on these same two"
        Write-Host "            cards -- prompt cache and context checkpoints OFF, no mmap,"
        Write-Host "            unified KV, 2 threads instead of 18, metrics on."
        Write-Host "            THE RAM IS THE POINT: a real session here held 20.4 GB"
        Write-Host "            working set and 34.4 GB private, and 32 checkpoints at"
        Write-Host "            ~350 MiB each is where it went. It is NOT free -- those"
        Write-Host "            checkpoints were being restored, so this trades host RAM"
        Write-Host "            for re-prefill, about a minute per 50,000 tokens."
        Write-Host "            Nothing here has been measured against the default yet." -ForegroundColor Yellow
    }
    Write-Host "  effort    medium. Chosen on the agentic axis, where xhigh costs one point and"
    Write-Host "            low costs six. NEVER MEASURED on any artifact here."
    Write-Host "  KV        q4_0. Not a preference -- our build compiles only f16, bf16, q4_0"
    Write-Host "            and q8_0 for flash attention (issue #43)."
    # The decoder line is the PROFILE's to print -- it is the thing that knows
    # what it was asked for. This branch printed a static "draft-mtp is NOT set
    # here" that contradicted the profile's own line four rows below it whenever
    # -Mtp was passed. A launcher describing configuration it does not own is
    # how it ends up lying; that is now three times.
    # Pinned by bench/tests/test_the_dual_profile_serves_both_cards.py.
} else {
    Write-Host "  artifact  UD-Q2_K_XL. An external ladder puts a 10-point cliff between this"
    Write-Host "            and UD-IQ2_XXS. OUR OWN quality number for it does not exist."
    Write-Host "  window    147,456, boot-verified fully resident. Real use has reached 85,923."
    Write-Host "  effort    medium. Chosen on the agentic axis, where xhigh costs one point and"
    Write-Host "            low costs six. NEVER MEASURED on any artifact here."
    Write-Host "  KV        q4_0. Not a preference -- our build compiles only f16, bf16, q4_0"
    Write-Host "            and q8_0 for flash attention (issue #43)."
    Write-Host "  draft     3, the default. 7 was measured at -56 % on the MTP head."
}

# WHICH CARD, read from the driver rather than asserted. Two GPUs have been
# installed since 2026-08-26 and the retired 4070 SUPER enumerates FIRST, so a
# banner that named the card from memory would be right about the intention and
# wrong about the machine (issue #50).
. (Join-Path $PSScriptRoot 'qwen38-tuning\scripts\Get-GpuVram.ps1')
$installed = @(Get-InstalledGpu)
if ($installed.Count -gt 1) {
    $using = if ($Dual) { "this uses both" } else { "this uses one of them" }
    Write-Host ("  gpu       {0} cards installed; {1}" -f `
                $installed.Count, $using) -ForegroundColor Cyan
    foreach ($card in $installed) {
        $inUse = $Dual -or ($card.Uuid -eq $script:ServedGpuUuid)
        $mark  = if ($inUse) { '->' } else { '  ' }
        Write-Host ("            {0} {1}" -f $mark, $card.Name) -ForegroundColor DarkGray
    }
}
Write-Host ""
if ($Dual) {
if ($Nvfp4) {
# The rate block above already carries the measurement. This section is for what
# is NOT known, and repeating the number here made the two say the same thing
# twice -- which is how a launcher's two halves start drifting apart.
Write-Host "  OPEN: QUALITY, and it is the only thing keeping this an icon." -ForegroundColor Yellow
Write-Host "        This changes the MODEL FILE, not a flag. ngram-mod acceptance" -ForegroundColor Yellow
Write-Host "        falls 55.4 -> 22.1 on this artifact, which is direct evidence" -ForegroundColor Yellow
Write-Host "        it writes DIFFERENTLY rather than merely faster. Whether" -ForegroundColor Yellow
Write-Host "        differently is worse is exactly what nobody here knows." -ForegroundColor Yellow
Write-Host "        Also open: MID-HIGH has no rate at all, and no depth above" -ForegroundColor Yellow
Write-Host "        147,456 has a PAIRED one. Issue #50." -ForegroundColor Yellow
} elseif ($Dflash) {
Write-Host "  OPEN: the DECODE RATE at 131,072 has never been measured." -ForegroundColor Yellow
Write-Host "        The +123.8 % is at 65,536. A verdict at one depth does not" -ForegroundColor Yellow
Write-Host "        transfer here -- at 147,456 a BETTER drafter measured SLOWER," -ForegroundColor Yellow
Write-Host "        because verify cost dominates there. Expect less. Issue #52." -ForegroundColor Yellow
} else {
Write-Host "  OPEN: measured at 147,456 on 2026-08-27 -- 27.6 / 27.6 / 27.6 tok/s," -ForegroundColor Yellow
Write-Host "        spread 0.1 %. ngram-mod is the ONLY decoder that produces a" -ForegroundColor Yellow
Write-Host "        number at this depth: draft-mtp copies the prompt and" -ForegroundColor Yellow
Write-Host "        draft-dflash cannot load. Issue #52." -ForegroundColor Yellow
Write-Host "        Quality has still never been measured on our own artifacts." -ForegroundColor Yellow
}
} else {
Write-Host "  OPEN: the draft-mtp half of the decoder is under question." -ForegroundColor Yellow
Write-Host "        Forced at 147,456, removing it is worth +15.6 % and 1,490 MiB." -ForegroundColor Yellow
Write-Host "        The one natural round, at 98,304, says keeping it is worth +127 %." -ForegroundColor Yellow
Write-Host "        Two variables moved between those numbers. Issues #44 and #47." -ForegroundColor Yellow
}
Write-Host ""

# Both asked of the PROFILE, not declared here. LogColors='on' because this
# script reads the output through a pipeline, and llama.cpp's default 'auto'
# means "colour when stdout is a TTY" -- a pipe is not one, so the colours were
# being turned off by the act of reading them.
#
# -Verbosity 4 is asked of the PROFILE, not declared here. The served default is
# 3, which omits the tensor-assignment lines. 4 rather than 5, measured: one boot
# writes 1.7 KB at 3, 24.7 KB at 4 and 511.9 KB at 5, and the layer line is
# present from 4 up. The DEFAULT in the profile does not move -- every served row
# was measured at 3.
# A HASHTABLE, not an array. `& $script @arr` on an array splats POSITIONALLY,
# so '-Verbosity' arrived as the profile's first parameter, $Ctx, and the run
# died with: Cannot convert value "-Verbosity" to type "System.Int32". Named
# splatting needs a hashtable. Pinned by tests/test_foreground_is_the_default.py.
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
# InvariantCulture: a Thai locale renders the year as 2569 and the log names stop
# sorting next to every other dated artifact in this repository.
#
# Assigned BEFORE $profileArgs reads it. It used to be set a hundred lines later,
# so the profile received an empty path, no log was written, and the watcher
# polled a file nothing would ever create -- then timed out in silence.
$stamp = [datetime]::Now.ToString('yyyyMMdd-HHmmss', [cultureinfo]::InvariantCulture)
$log   = Join-Path $logDir "serve-$stamp.log"

$profileArgs = @{ Verbosity = 4; LogColors = 'on'; LogFile = $log }
if ($Lan) { $profileArgs['BindAddress'] = '0.0.0.0' }
if ($Device) { $profileArgs['Device'] = $Device }
if ($Dflash) {
    if (-not $Dual) {
        Write-Host "FATAL: -Dflash is a two-card configuration; pass -Dual too." -ForegroundColor Red
        exit 1
    }
    $profileArgs['Dflash'] = $true
}
if ($Lean) {
    if (-not $Dual) {
        Write-Host "FATAL: -Lean is a two-card bundle; pass -Dual too." -ForegroundColor Red
        exit 1
    }
    $profileArgs['Lean'] = $true
}
if ($Vision) {
    if (-not $Dual) {
        Write-Host "FATAL: -Vision applies to the two-card profile; pass -Dual too." -ForegroundColor Red
        exit 1
    }
    $profileArgs['Vision'] = $true
}
if ($Deep) {
    if (-not ($Dual -and $Nvfp4)) {
        Write-Host "FATAL: -Deep is the NVFP4 ceiling; pass -Dual -Nvfp4 too." -ForegroundColor Red
        Write-Host "  On UD-Q4_K_XL the deep question is a budget one: use -MaxCtx." -ForegroundColor Yellow
        exit 1
    }
    $profileArgs['Deep'] = $true
}
if ($Nvfp4) {
    if (-not $Dual) {
        Write-Host "FATAL: -Nvfp4 is a two-card configuration; pass -Dual too." -ForegroundColor Red
        Write-Host "  14,173 MiB of weights do not fit on either card alone." -ForegroundColor Yellow
        exit 1
    }
    $profileArgs['Nvfp4'] = $true
}
if ($MaxCtx) {
    if (-not $Dual) {
        Write-Host "FATAL: -MaxCtx applies to the two-card profile; pass -Dual too." -ForegroundColor Red
        exit 1
    }
    $profileArgs['MaxCtx'] = $true
}
if ($Mtp) {
    if (-not $Dual) {
        Write-Host "FATAL: -Mtp applies to the two-card profile; pass -Dual too." -ForegroundColor Red
        Write-Host "  worker-q2kxl-mtp.ps1 already serves draft-mtp on one card." -ForegroundColor Yellow
        exit 1
    }
    $profileArgs['Mtp'] = $true
}

# Flattened for the -WhatIf preview only; nothing launches a separate process.
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
        Write-Host "Already serving -- ANOTHER WINDOW owns it." -ForegroundColor Green
        Write-Host "A server cannot outlive the window that started it, so one is open." -ForegroundColor Green
        Show-ServerStatus -Props $existing -Port $Port
        Write-Host ""
        Write-Host "Close that window to stop it. Ctrl+C here reaches nothing." -ForegroundColor DarkGray
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

# ---- foreground: llama.cpp owns this console -------------------------------
# NOTHING between llama.cpp and the terminal. Every layer that carried the output
# dropped something from it: --log-colors auto turned colour off because a pipe
# is not a terminal, and PowerShell then stripped the codes it did forward
# because its own output was not a console. Each fix was right and the next layer
# was still there. So the launcher stops carrying it.
#
# The checks still happen, from the FILE: llama.cpp writes each entry to the
# console and then again to --log-file (common/log.cpp:170-178), so a reader can
# follow the boot without standing in the way.
# Verified by hard-kill -- Stop-Process -Force on the owning pwsh, which runs no
# cleanup code at all and is strictly harder than a window close. The
# interactive close itself was NOT exercised: a headless session has no window
# handle to deliver it to. Covered by the kill-on-close job, not by an
# observation, and those are different sentences.
Write-Host "Starting. Ctrl+C stops the server; so does closing this window." -ForegroundColor Cyan
Write-Host "A copy of this output is kept at $log" -ForegroundColor DarkGray
Write-Host ("-" * 78) -ForegroundColor DarkGray

# The watcher shares this console -- hidden or redirected, its report would land
# somewhere nobody is looking. It exits on its own once it has reported, and it
# is not the server, so it does not reintroduce anything that can outlive the
# window.
$watch = @'
param($Log, $Base, $Port, $StatusScript)
. $StatusScript
$deadline = (Get-Date).AddMinutes(10)
$warnAt = (Get-Date).AddSeconds(45)
$warned = $false
$onGpu = 0; $total = 0
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 700
    if (-not (Test-Path $Log)) {
        if ((Get-Date) -gt $warnAt -and -not $warned) {
            Write-Host ""
            Write-Host "The boot log never appeared at $Log -- residency and status" -ForegroundColor Yellow
            Write-Host "will not be reported for this run. The server itself is unaffected." -ForegroundColor Yellow
            $warned = $true
        }
        continue
    }
    try {
        $fs = [IO.File]::Open($Log, 'Open', 'Read', 'ReadWrite')
        $sr = New-Object IO.StreamReader($fs); $txt = $sr.ReadToEnd()
        $sr.Close(); $fs.Close()
    } catch { continue }

    if ($total -eq 0 -and $txt -match 'offloaded (\d+)/(\d+) layers to GPU') {
        $onGpu = [int]$Matches[1]; $total = [int]$Matches[2]
        if ($onGpu -ne $total) {
            Write-Host ""
            Write-Host "SPILLED: only $onGpu of $total layers are on the GPU." -ForegroundColor Red
            Write-Host "  --fit spilled rather than refusing. Numbers taken past this point are" -ForegroundColor Yellow
            Write-Host "  not comparable to anything measured while fully resident." -ForegroundColor Yellow
        }
    }
    if ($txt -match 'listening on http') {
        try { $props = Invoke-RestMethod "$Base/props" -TimeoutSec 4 } catch { continue }
        Show-ServerStatus -Props $props -Port $Port -OnGpu $onGpu -Total $total
        Write-Host ("-" * 78) -ForegroundColor DarkGray
        return
    }
}
'@
$watchFile = Join-Path $logDir "watch-$stamp.ps1"
Set-Content -Path $watchFile -Value $watch -Encoding UTF8
$null = Start-Process pwsh -NoNewWindow -PassThru `
    -ArgumentList '-NoProfile', '-File', $watchFile, $log, $base, "$Port", $statusScript

# Put THIS process in the job. Job membership is inherited by children, so
# llama-server joins the moment the profile starts it -- no polling for a PID,
# and no window in which the server exists outside the job. When this process
# dies, however it dies, the kernel takes everything in the job with it.
if (-not [KillOnClose]::Adopt($PID)) {
    Write-Host "WARNING: could not bind this terminal's processes together." -ForegroundColor Yellow
    Write-Host "  The server may OUTLIVE this window. Stop it with: Get-Process llama-server | Stop-Process" -ForegroundColor Yellow
}

& $profileScript @profileArgs
