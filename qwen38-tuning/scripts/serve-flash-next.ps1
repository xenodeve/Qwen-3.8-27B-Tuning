<#
.SYNOPSIS
    Serve Qwen3.8-Flash-Next GSQ-RCO Q2_0 -- best measured configuration.

.DESCRIPTION
    Profile of 2026-09-23 (evidence: qwen38-tuning\results\flash-next-optimization\
    results-v2.csv, runs L-*, T-ctrl-*, V-*, K-*). Same-run pairs, ctx 16,384:

        -sm tensor recipe (before)     decode ~10.7 tok/s   prefill 16K ~24 tok/s  (8.5 min)
        base derived here (ctx 16K)    decode ~28-31 tok/s  prefill 16K ~570 tok/s (~21 s)
        shipped long-context profiles: see PROFILES below.

    WHY -sm layer, NOT tensor. Under -sm tensor the split runs through the meta
    backend, whose offload_op is NULL (ggml/src/ggml-backend-meta.cpp:192), so the
    scheduler can never move a host-resident expert matmul to a GPU: every
    prefill token's CPU experts were computed by the i5 at ~23 tok/s. Under
    -sm layer op offload works and prefill 16K went 24 -> 452-512 tok/s on the
    same binary (runs T-ctrl-ub512 vs L-ts30_18-*).

    WHY -ts 30,18. -ncmoe takes the FIRST 24 layers' experts to host, so under a
    layer split those layers are light. 8500,16000 put the full-expert layers on
    the 5060 Ti and OOMed it (18.4 GiB); 30,18 gives GPU0 24 light + 6 full layers
    and GPU1 18 full layers. It also puts the host-expert layers -- and so the
    expert cache below -- on the 4070 SUPER, the x16 card.

    WHY THIS BINARY. Upstream e6ab7c1a + x86 VNNI Q2_0 kernel (PR #26348) + GPU
    expert cache (PR #27861). VNNI: decode 12.8 -> 21.8 tok/s, A-B-B-A same run
    (V-*). Cache 64 slots: decode 22 -> 28-31 tok/s against no-cache controls at
    both ends of the run (K-*); 96 slots is +4-10 % more but leaves <0.5 GB on
    the display card. Answers at temperature 0 matched the no-cache binary on
    short prompts and stayed coherent on a 568-token answer (logs/answer-*.txt).
    Provenance: C:\AI\llama.cpp-flash-vnni-cache\PROVENANCE.txt.
    The unsloth mirror (build-mirror) is the previous binary; the 27B builds
    (llama.cpp-blackwell, llama.cpp-dflash2) do not know arch qwen4exp.

    -ub 2048: under op offload the per-ubatch cost is streaming host experts to
    the GPU; 2048 and 4096 measured the same at 16K (452 vs 474), 4096 costs
    ~0.8 GB more VRAM.

    PROFILES (-Profile, required). Long context is kept on purpose -- this model
    serves long workflows. At 262,144 the prefill compute buffer is ~ubatch x ctx
    on EACH card (7.2 GiB at ub 2048, 3.7 at 1024, 1.85 at 512), so the ubatch
    shrinks as the window grows, and -ncmoe frees only GPU0, so -ts moves with it:
        262k  ctx 262,144  -ncmoe 28  -ts 31,17  -ub 512   (run D-D)
              prefill 221-262 tok/s at 12K-240K; decode 24.7 / 17.5 / 15.3 / 9.5
              at 12K / 60K / 120K / 240K. Peak VRAM 11.35 + 14.93 GB.
        128k  ctx 131,072  -ncmoe 26  -ts 31,17  -ub 1024  (run D-C)
              prefill 320-341 tok/s at 12K-120K; decode 27.1 / 19.8 / 14.5
              at 12K / 60K / 120K. Peak VRAM 11.93 + 14.72 GB.
    Every other split / ubatch at these windows OOMed at boot (vram262k.csv).
    The 16,384 profile above (-ts 30,18 -ub 2048 -ncmoe 24) is the base these
    were derived from; it is no longer offered.

    SPECULATION (2026-09-24, pinned binary C:\AI\llama.cpp-flash-mtp, PROVENANCE.txt):
    MTP n=3 (unsloth shared-Q4_K_M head) + ngram-mod (12/16/32 here; 24/16/64 since
    q-ng64, see the argv comment), expert cache on
    verify batches via LLAMA_MOE_CACHE_MAX_TOKENS=4. 128k-shaped config, prompts
    1-3K, vs no speculation at the same -ncmoe 32: copy-heavy +40-55 % (temp 0
    up to 53 tok/s), fresh text +17-31 %. Prefill of a 1K prompt drops ~40-50 %
    (second request 167-201 vs 336 tok/s). Deep fills with MTP NOT yet run.

    NOT MEASURED: repeats (one cold fill per depth, 18-47 decode tokens at
    depth -- decode figures there are coarse); quality (planned last).

    Foreground only. Ctrl+C or closing the window stops llama-server.
    -WhatIf prints the argv without launching.
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter(Mandatory)]
    [ValidateSet('262k', '128k')]
    [string]$Profile,
    [int]$Port = 8080,
    [int]$ExpertCache = 64,
    [ValidateSet('127.0.0.1', '0.0.0.0')]
    [string]$BindAddress = '127.0.0.1',
    [string]$Device = 'GPU-fba37e4b-ea9e-66e9-c3fd-a16b2e833bc4,GPU-059b90e2-2b5c-00b8-f3ba-f6dea8de083e',
    [string]$Exe = 'C:\AI\llama.cpp-flash-mtp\bin\llama-server.exe',
    [string]$MtpHead = 'C:\AI\models\unsloth-Qwen3.8-Flash-Next-MTP\MTP\mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf',
    [string]$Model = 'C:\AI\models\ISTA-DASLab-Qwen3.8-Flash-Next-GSQ-RCO\Q2_0\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf'
)

$ErrorActionPreference = 'Stop'
# Same console contract as serve.ps1 / serve-gsq.ps1: pwsh strips ANSI when its
# output is not a console, so llama.cpp's --log-colors would vanish (serve.ps1).
if ($PSStyle) { $PSStyle.OutputRendering = 'Ansi' }
if (-not (Test-Path $Exe))   { throw "llama-server.exe was not found at $Exe." }
if (-not (Test-Path $Model)) { throw "Flash-Next shard 1 was not found at $Model." }
if (-not (Test-Path $MtpHead)) { throw "MTP head was not found at $MtpHead." }
# Claude Code appends its SessionStart hook output as a late system message; the
# stock template raises on it (HTTP 500, templates\README.md). Flash-Next's own
# template (/props, 2026-09-24) is byte-identical to qwen38-stock.jinja, so the
# same one-line patch applies unchanged.
$template = Join-Path $PSScriptRoot '..\templates\qwen38-late-system.jinja'
if (-not (Test-Path $template)) { throw "Claude Code-compatible chat template was not found at $template." }
# Each profile is one measured row -- ctx, host-expert layers, split and ubatch
# move TOGETHER, because the compute buffer is ~ubatch x ctx on EACH card and
# -ncmoe only frees GPU0 (runs M262-*, M128k-*, D-C, D-D; 2026-09-23).
$profiles = @{
    # 128k: MTP + ngram-mod. The draft and output.weight live on the 5060 Ti
    # (nothing else uses it); the 4070 SUPER drives the desktop and its free
    # VRAM moves by ~1 GB, so a draft there (-ncmoe 32) had 1,990-2,124 MiB free
    # against ~2,530 needed and spilled silently. -ncmoe 36 frees layers 32-35 on
    # CUDA1 for the draft (2,852 free, needs 2,602; -ncmoe 34 OOMed at 1,674).
    # llama then holds 7,669 MiB on CUDA0 instead of 10,609 (q-ncmoe-hr).
    # VRAM is freed by host layers, never by shrinking the expert cache.
    # 262k: ngram-mod only, no MTP (2026-09-24). With MTP it over-committed GPU0 -- 1,864
    # MiB free before the draft, 2,825 needed; WDDM spilled the rest to host
    # memory without an error and a 14.7K prefill ran at ~42 tok/s instead of
    # ~220. Back on the measured no-MTP row (D-D) until the draft moves to GPU1.
    '262k' = @{ Ctx = 262144; NcMoE = 28; Ts = '31,17'; UBatch = 512;  Mtp = $false }
    '128k' = @{ Ctx = 131072; NcMoE = 36; Ts = '31,17'; UBatch = 1024; Mtp = $true  }
}
$p = $profiles[$Profile]
$Ctx = $p.Ctx; $NcMoE = $p.NcMoE; $ts = $p.Ts; $UBatch = $p.UBatch; $Mtp = $p.Mtp
# The no-MTP row was measured on the pre-MTP binary; keep it on that binary.
if (-not $Mtp -and -not $PSBoundParameters.ContainsKey('Exe')) { $Exe = 'C:\AI\llama.cpp-flash-vnni-cache\bin\llama-server.exe' }
if (-not (Test-Path $Exe)) { throw "llama-server.exe was not found at $Exe." }

. (Join-Path $PSScriptRoot 'Get-GpuVram.ps1')
$installed = @(Get-InstalledGpu)
$wanted = @($Device -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($wanted.Count -ne 2) { throw "Flash-Next needs exactly two GPUs; Device names $($wanted.Count)." }
foreach ($uuid in $wanted) {
    if (-not ($installed | Where-Object { $_.Uuid -eq $uuid })) {
        $found = ($installed | ForEach-Object { "$($_.Uuid) $($_.Name)" }) -join '; '
        throw "GPU $uuid is not installed. Found: $found"
    }
}

foreach ($uuid in $wanted) {
    if (-not (Test-ServedGpuPresent -Uuid $uuid)) { throw "GPU $uuid missing at launch time." }
}

# A copy of the console in qwen38-tuning\logs -- the hub's window is the only
# other place these lines exist, and an agent cannot read it.
$logDir = Join-Path (Split-Path $PSScriptRoot -Parent) 'logs'
if (-not (Test-Path $logDir)) { [void](New-Item -ItemType Directory -Path $logDir -Force) }
$stamp = (Get-Date).ToString('yyyyMMdd-HHmmss', [System.Globalization.CultureInfo]::InvariantCulture)
$logFile = Join-Path $logDir "flash-next-$Profile-$stamp.log"

$serverArgs = @(
    '-c', "$Ctx", '-ngl', 'all', '-ncmoe', "$NcMoE",
    '-sm', 'layer', '-ts', $ts, '-fit', 'off',
    '-ctk', 'q4_0', '-ctv', 'q4_0', '-b', '2048', '-ub', "$UBatch",
    '-lm', 'mmap', '--lazy-mode', 'on', '-fa', 'on', '-np', '1',
    '--moe-expert-cache', "$ExpertCache",
    # Prompt-cache budget (issue #70, CORRECTIONS 46). 8 checkpoints: restores
    # never reached past the 3rd newest, and 6 covered a compact (05-runtime-flags).
    # At the default 32 one 105K entry held 5,918 MiB, so the auto-mode
    # classifier's 30K conversation evicted it and the next turn re-prefilled
    # 66,890 tokens (flash-next-128k-20260924-052307.log). The cap stays 8192,
    # not 24576 like gsq: host experts already hold ~25-28 GB of RAM here, and
    # paging would evict expert pages rather than idle cache (UNMEASURED).
    '--cache-ram', '8192', '--ctx-checkpoints', '8',
    # Claude Code needs (test_flash_next_launcher.py): a 5 s SSE keep-alive so a
    # minutes-long prefill at ~200 tok/s is not read as a dead connection, and
    # medium effort like every other served profile since report 35 -- xhigh
    # thought 9,300 tokens in one turn here. Effort on Flash-Next is UNMEASURED.
    '--sse-ping-interval', '5', '--reasoning-effort', 'medium',
    # -lv 4: `cached n_tokens` and `forcing full prompt re-processing` do not
    # print at the default 3, and they are what says whether the prompt cache
    # held (worker-q4-dual.ps1 carries the same reason).
    '-lv', '4', '--log-colors', 'on', '--log-file', $logFile,
    '-t', '14', '-tb', '20',
    '--host', $BindAddress, '--port', "$Port",
    '-m', $Model, '--chat-template-file', $template, '--alias', 'Qwen3.8-Flash-Next-Q2_0',
    # ngram-mod 24/16/64 on both profiles (q-ng64, spec-mtp.csv ng-*, 2026-09-24):
    # over 12/16/32 on the 128k duo, copy +23 % (t0) / +13 % (t1), fresh text
    # +26 % / +6 %. 12/16/64 won copy but lost fresh text -- n-match 12 lets
    # short matches trigger 64-token drafts (acceptance 0.42-0.54). ngram is
    # tried first and only drafts on long copies (n-min 16).
    '--spec-ngram-mod-n-match', '24', '--spec-ngram-mod-n-min', '16', '--spec-ngram-mod-n-max', '64'
)
if ($Mtp) {
    # MTP n=3 drafts what ngram does not. The shared head borrows output.weight,
    # which sits with the last layer on CUDA1, so the draft goes there too
    # (-devd CUDA1, no -ot): a draft cannot run an op whose weight is elsewhere.
    $serverArgs += @(
        '-md', $MtpHead, '--spec-type', 'draft-mtp,ngram-mod', '--spec-draft-n-max', '3',
        '-ngld', '99', '-devd', 'CUDA1', '-ctkd', 'q4_0', '-ctvd', 'q4_0')
} else {
    # 262k: ngram-mod alone -- no draft model, no VRAM (MTP over-committed GPU0
    # here). UNMEASURED at 262k with 24/16/64; ngram-only 12/16/32 at 128k was
    # copy ~2x, fresh text -13 % (doc 32 section 2.7).
    $serverArgs += @('--spec-type', 'ngram-mod')
}

Write-Host "Qwen3.8-Flash-Next GSQ-RCO Q2_0 -- profile $Profile -- ctx $Ctx -- -ncmoe $NcMoE -- layer split $ts -- ub $UBatch -- expert cache $ExpertCache -- $(if ($Mtp) { 'MTP n=3 + ngram-mod 24/16/64' } else { 'ngram-mod 24/16/64' })" -ForegroundColor Cyan
Write-Host "  NOTE: quality is UNMEASURED. Depth verified with one cold fill per profile (2026-09-23)." -ForegroundColor Yellow
if ($BindAddress -eq '0.0.0.0') {
    Write-Host 'WARNING: exposed on every interface with no API key.' -ForegroundColor Yellow
}
Write-Host "A copy of this output is kept at $logFile" -ForegroundColor DarkGray

if ($WhatIfPreference) {
    Write-Host 'WhatIf: would run' -ForegroundColor Green
    Write-Host ("  {0} {1}" -f $Exe, ($serverArgs -join ' '))
    exit 0
}

# ONE working-set trim once the server is healthy. -lm mmap leaves every page it
# read to upload GPU weights in this process's working set: ~19 GB of RAM that
# duplicates what is already in VRAM. EmptyWorkingSet hands those clean pages
# back with no I/O; hot CPU experts fault back in from the file cache. 128k,
# ABBA x3 (diag-mem-*.csv, 2026-09-24): RAM free after a 60K fill 14.5 GB vs
# 1.1 GB untrimmed, and untrimmed runs collapsed to 3.8-7.2 tok/s decode at 60K
# where trimmed held 13.1-13.9. Runs beside llama.cpp in this console, never
# between it and the terminal; it exits after one trim.
$trimJob = Start-ThreadJob -ArgumentList $Port, $logFile -ScriptBlock {
    param($Port, $LogFile)
    Add-Type -Namespace W -Name Psapi -MemberDefinition '[DllImport("psapi.dll")] public static extern bool EmptyWorkingSet(System.IntPtr hProcess);'
    $deadline = (Get-Date).AddMinutes(10)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 3
        try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3 } catch { continue }
        if ($h.status -ne 'ok') { continue }
        $srv = Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" |
               Where-Object { $_.CommandLine -match "--port $Port(\s|$)" } | Select-Object -First 1
        if (-not $srv) { return }
        $p = Get-Process -Id $srv.ProcessId
        $before = [int]($p.WorkingSet64 / 1MB)
        $ok = [W.Psapi]::EmptyWorkingSet($p.Handle)
        Start-Sleep -Seconds 1; $p.Refresh()
        $line = "[flash-next] working-set trim: $before -> $([int]($p.WorkingSet64 / 1MB)) MiB (ok=$ok)"
        [Console]::WriteLine($line)
        Set-Content -Path ($LogFile -replace '\.log$', '-trim.txt') -Value $line
        return
    }
}

$env:CUDA_VISIBLE_DEVICES = $Device
# Local patch in this binary: the expert cache serves verify batches up to this
# many tokens (MTP n=3 verifies 4). Default 1 = cache off during verification.
$env:LLAMA_MOE_CACHE_MAX_TOKENS = '4'
& $Exe @serverArgs
$rc = $LASTEXITCODE
Remove-Job $trimJob -Force -ErrorAction SilentlyContinue
exit $rc
