<#
WORKER PROFILE — UD-Q4_K_XL across BOTH cards, ctx 147,456.
RTX 4070 SUPER 12 GB + RTX 5060 Ti 16 GB.

Issue #52. Every number in this header was measured on this machine on
2026-08-26 with the native sm_120a+sm_89 build, not projected.

WHY THIS PROFILE EXISTS AT ALL

  UD-Q4_K_XL is 16.69 GiB. It was refused on one 16 GB card at every depth
  since that card arrived -- `docs/results/09-hardware.md` recorded it as
  "16 GB does not unlock Q4 residency either". Across two cards it is FULLY
  RESIDENT (66+0) at every rung to 229,376, including the 147,456 this file
  serves, and spills a single layer only at 262,144, which is n_ctx_train.

  The second card is worth +79.9 % [+77.3, +82.2] to this artifact, and the
  layer split says why: 55+11 becomes 66+0. That is the residency cliff, not
  the silicon -- UD-Q2_K_XL, which was already resident on one card, gained
  1.5 % from the identical change.

WHY IT IS A SEPARATE FILE FROM worker-q2kxl-mtp.ps1

  Every row in docs/results/ from 2026-08-23 onward describes the one-card
  configuration. A -Dual switch on that profile would mean its defaults no
  longer say what was measured. Both ship; which is the default is the
  developer's call with the numbers in front of them (#52).

WHAT THIS COSTS, STATED PLAINLY

  About a third of raw decode: 20.9 tok/s here against 32.0 for UD-Q2_K_XL on
  one card, speculation off, ctx 16,384. That pair is ACROSS SWEEPS and so
  across boots -- it rests on the under-0.8 % per-arm floor measured that day,
  at that depth only, and the two arms load different files so nothing else
  about them is paired. A sizing figure, not a verdict.

  And roughly 130 W more under load: both cards sat at ~50 % utilisation
  drawing 107-114 W and 133-135 W.

  QUALITY IS THE WHOLE REASON TO PAY THAT AND IT HAS NEVER BEEN MEASURED HERE
  on this project's own artifacts. The bits-per-weight ladder and an external
  12-format campaign both point the same way; neither is our number.

WHY -sm IS NOT SET

  `-sm row` CANNOT LOAD on this pair: "device CUDA0 does not support split
  buffers", at model load, in about a second, every attempt. The cards sit at
  PXB with no NVLink. `-sm tensor` and `-ts` were swept -- see
  docs/results/09-hardware.md for which won and by how much.

WHY THE CARDS ARE NAMED BY UUID

  `--main-gpu` defaults to 0, which on this machine is the RETIRED 4070 SUPER,
  and the dll carries sm_89 beside sm_120a so the wrong card is not merely
  reachable but fully supported. An index is a position in an enumeration the
  driver can reorder; after a reorder it keeps working and means a different
  card. Issue #50.

WHAT IS NOT MEASURED

  - Nothing at 147,456. Every figure above is ctx 16,384, and CORRECTIONS 23
    says the spread can be several times wider at depth.
  - This artifact with the served decoder (draft-mtp,ngram-mod). #44 already
    shows that decoder inverting sign with depth.
  - Any speculative decode rate comparing one card to two. CORRECTIONS 32:
    splitting changes the reduction order, so the logits, so the text -- and a
    speculative rate is partly a measure of how predictable the text is.
#>
param(
    # 147,456 is boot-verified 66+0 across both cards. The residency ceiling for
    # this artifact is 229,376; this matches the one-card profile so the two are
    # comparable at the depth anybody actually serves.
    [int]$Ctx  = 147456,
    [int]$Port = 8080,
    # Same meaning as in worker-q2kxl-mtp.ps1: 3 keeps the log small, the
    # tensor-assignment lines that prove residency need 5, and serve.ps1 asks
    # for what it needs rather than this default moving.
    [int]$Verbosity = 3,
    [ValidateSet('on', 'off', 'auto')]
    [string]$LogColors = 'auto',
    [string]$LogFile = '',
    # The ONLY access control this server has -- no API key, CORS '*', and
    # middleware_validate_api_key returns true immediately when no key is set.
    # Widening this does not weaken one control among several; it removes the
    # only one. Exposure is `serve.ps1 -Lan`, an act.
    # Pinned by bench/tests/test_the_dual_profile_serves_both_cards.py.
    [string]$BindAddress = '127.0.0.1',
    # BOTH cards, by UUID. Empty resolves to the pair below; pass a
    # comma-separated list to override. Order is the CUDA enumeration order and
    # is what any -ts ratio would be indexed by.
    [string]$Device = '',
    [string]$Exe = "C:\AI\llama.cpp-blackwell\llama-server.exe",
    [string]$Model = "C:\Users\xenod\.cache\huggingface\hub\models--unsloth--Qwen3.8-27B-GGUF\snapshots\f1bfb127c64f7072bdd2cad55f258b9c8b2910fe\Qwen3.8-27B-UD-Q4_K_XL.gguf",
    [switch]$IKnowTheBuildIsWrong
)
$ErrorActionPreference = 'Stop'

# ---- the build ---------------------------------------------------------------
# A binary without Blackwell SASS runs here through PTX JIT at 2.20x the prefill
# time with nothing in any log to say so. The match is a SUBSTRING one on
# purpose: cmake rewrites 120 to 120a and the cubins are named sm_120a, so an
# exact 'sm_120' test would reject a correctly built binary.
#
# This profile needs BOTH architectures, because one of its two cards is Ada.
$dll = Join-Path (Split-Path $Exe -Parent) 'ggml-cuda.dll'
$cuobjdump = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\cuobjdump.exe'

if (-not (Test-Path $Exe)) {
    Write-Host "FATAL: no server at $Exe" -ForegroundColor Red
    Write-Host "  Rebuild with -DCMAKE_CUDA_ARCHITECTURES=`"89;120`"" -ForegroundColor Yellow
    exit 1
}
if (-not (Test-Path $Model)) {
    Write-Host "FATAL: no model at $Model" -ForegroundColor Red
    exit 1
}

if ((Test-Path $dll) -and (Test-Path $cuobjdump)) {
    $elf = & $cuobjdump --list-elf $dll 2>$null
    $arches = ($elf | Select-String -Pattern 'sm_\w+' -AllMatches |
               ForEach-Object { $_.Matches.Value } | Sort-Object -Unique)
    foreach ($needed in @('sm_120', 'sm_89')) {
        if (-not ($arches -match [regex]::Escape($needed))) {
            Write-Host "FATAL: $dll has no $needed SASS (found: $($arches -join ', '))" -ForegroundColor Red
            Write-Host "  This profile drives an Ada card AND a Blackwell card." -ForegroundColor Yellow
            Write-Host "  A missing architecture is JIT-compiled from PTX with" -ForegroundColor Yellow
            Write-Host "  nothing in the log to say so." -ForegroundColor Yellow
            Write-Host "  Rebuild, or pass -IKnowTheBuildIsWrong (never for a measurement)." -ForegroundColor Yellow
            if (-not $IKnowTheBuildIsWrong) { exit 1 }
            Write-Host "  OVERRIDDEN -- results are not comparable to anything." -ForegroundColor Magenta
        }
    }
} else {
    Write-Host "WARNING: cannot verify GPU architecture (missing $dll or cuobjdump)." -ForegroundColor Yellow
}

# ---- the cards ---------------------------------------------------------------
# Checked BEFORE the model loads. An absent UUID does not make llama-server
# fail: it reports `(none)` for devices and then runs on the CPU, producing
# correct output at a rate no row explains. With two UUIDs there are two ways to
# be wrong, and a 16.69 GiB model takes long enough to load that the wrong
# answer arrives an hour later.
. (Join-Path $PSScriptRoot 'Get-GpuVram.ps1')

if (-not $Device) {
    $Device = 'GPU-fba37e4b-ea9e-66e9-c3fd-a16b2e833bc4,' + $script:ServedGpuUuid
}

$installed = @(Get-InstalledGpu)
$wanted = @($Device -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
foreach ($uuid in $wanted) {
    if (-not ($installed | Where-Object { $_.Uuid -eq $uuid })) {
        Write-Host "FATAL: GPU $uuid is not installed." -ForegroundColor Red
        Write-Host "  Installed:" -ForegroundColor Yellow
        $installed | ForEach-Object {
            Write-Host ("    {0}  {1}" -f $_.Uuid, $_.Name) -ForegroundColor Yellow
        }
        Write-Host "  llama-server would see fewer devices than this profile" -ForegroundColor Yellow
        Write-Host "  was measured on, and UD-Q4_K_XL does not fit on one card." -ForegroundColor Yellow
        exit 1
    }
}
if ($wanted.Count -lt 2) {
    Write-Host "FATAL: this profile needs two cards; -Device names $($wanted.Count)." -ForegroundColor Red
    Write-Host "  UD-Q4_K_XL is 16.69 GiB and spills 11 layers on one 16 GB card" -ForegroundColor Yellow
    Write-Host "  -- 11.7 tok/s against 20.9. Use worker-q2kxl-mtp.ps1 instead." -ForegroundColor Yellow
    exit 1
}
$env:CUDA_VISIBLE_DEVICES = $Device

# An ARRAY, empty when no log was asked for. An inline `$(if ...)` would pass an
# empty string as a real argument and llama-server would see a flag it cannot
# parse -- the kind of failure that only shows up on the default path.
$logFileArg = if ($LogFile) { @('--log-file', $LogFile) } else { @() }

# ---- serve -------------------------------------------------------------------
# -sm is NOT set: `row` cannot load on this pair, and `layer` is the default.
# -ts is NOT set: llama.cpp splits by free VRAM, measured 41:59 on these cards.
# Both were swept -- docs/results/09-hardware.md carries the arms that lost.
& $Exe -m $Model `
    --alias qwen38 -c $Ctx `
    -ngl auto --fit on --fit-target 768 -fa on -np 1 `
    -t 18 -b 2048 -ub 256 --no-mmproj-auto -lv $Verbosity `
    --log-colors $LogColors `
    @logFileArg `
    -ctk q4_0 -ctv q4_0 `
    --spec-type ngram-mod `
    --spec-ngram-mod-n-match 12 --spec-ngram-mod-n-min 16 --spec-ngram-mod-n-max 32 `
    --chat-template-file "C:\AI\qwen38-tuning\templates\qwen38-late-system.jinja" `
    --reasoning-effort medium `
    --host $BindAddress --port $Port
