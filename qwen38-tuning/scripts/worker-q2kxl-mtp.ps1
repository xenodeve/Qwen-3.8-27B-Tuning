<#
WORKER PROFILE — UD-Q2_K_XL + baked-in MTP, ctx 147,456. RTX 5060 Ti 16 GB.

Issue #41. Every number in this header was measured on this card with the native
sm_120a build on 2026-08-24, not projected.

WHY THIS ARTIFACT

  UD-Q2_K_XL is 9,373.65 MiB on disk against UD-IQ2_XXS's 6,929.46. An external
  12-format ladder on 4x RTX 3090 (720 tasks each) puts a 10-point agent-suite
  cliff between exactly these two files -- 0.76 for IQ2_XXS against 0.86 -- and
  that cliff is the ONE conclusion of that campaign which survives its author's
  own audit of it (docs/researchs/superalesha-quant-ladder).

  Our own bits-per-weight ladder points the same way: task success rises
  monotonically with bpw across five artifacts (results/01).

  UNMEASURED HERE: this artifact has no task-success number on this machine. One
  real task was run on it and did not finish; so did one on IQ2_XXS.

WHY draft-mtp AND NO -md

  UD-Q2_K_XL reports n_layer_all = 65 and offloads 66/66: blk.64.nextn.* loads
  out of the main file, and the boot log says

      creating MTP draft context against the TARGET model '...UD-Q2_K_XL.gguf'

  Every earlier draft-mtp figure in this project (+81 % @16K, -71 % @131,072) fed
  a SEPARATE 1.3 GB head via -md to an artifact that had none, costing 564 MiB.
  This configuration had never been run here.

  DO NOT ADD -md. It moves the head into a file, adds its weights to
  fit_params_target (server-context.cpp:1074, gated only on "was -md given"), and
  brings back the 1,393.90 MiB sidecar for nothing.

WHY NOT --spec-draft-n-max 7

  DFlash2's block_size is 8 so 7 is its ceiling, and on THAT drafter 7 is worth
  25 % off the wall clock. On MTP it is a regression in every bucket:

      draft-mtp+ngram n3   short-turn 54.4 tok/s   acceptance 0.48-0.61
      draft-mtp+ngram n7   short-turn 41.0         acceptance 0.38-0.44
                           and 947.2 s -> 1,481.3 s on the same task

  The metadata says why: qwen35.nextn_predict_layers = 1. The head predicts ONE
  token ahead, so asking for seven yields drafts that are mostly rejected while
  the verify cost is paid anyway. Left at the default 3.

WHY ctx 147,456 AND NOT DEEPER — measured, and a projection got this wrong

  163,840 does NOT hold. `--fit` reports "cannot meet free memory target of
  1522 MiB, need to reduce device memory by 154 MiB" and spills to 64/66. Two CPU
  layers at depth are not a small cost: AD-IQ1_M at 65+1 decodes 6.08 tok/s
  against 26.50 resident (results/04).

  A projection said 163,840 would leave 1,790 MiB. It was wrong because THREE
  buffers that look fixed scale with context:

      buffer            98,304    131,072    163,840     rate
      target compute    472.27     616.27     777.57     ~0.0047 MiB/token
      MTP KV            384.00     512.00     640.00     4.00 KiB/token exactly
      MTP compute        82.01      98.01     114.01     ~0.0005 MiB/token

  Together ~290 MiB per 32,768 tokens that the projection did not count. Only
  target KV (18.00 KiB/token) is what everyone assumes.

  Measured at 147,456, 66/66 resident:

      model 8,965.31 | KV 2,592.00 | RS 598.50 | compute 688.27
      MTP KV 576.00  | MTP compute 106.01      | CPU_Mapped 397.85
      total on CUDA0 13,526 of the 15,172 llama.cpp sees, leaving 1,646
      --fit: "will leave 1727 >= 1450 MiB, no changes needed"

  131,072 is the safer rung at 2,078 MiB free if anything is added later.

WHY --reasoning-effort medium

  Until 2026-08-24 every server this project launched ran at the template's
  xhigh with an unlimited thinking budget -- never chosen, simply never set.
  Artificial Analysis prices this model on the AGENTIC axis at xhigh 51,
  medium 50, low 44: one point down to medium, six more to low
  (docs/researchs/artificial-analysis).

  UNMEASURED HERE. No run on this artifact exists at any level but xhigh.

WHY --chat-template-file. Qwen3.8's stock template raises 'System message must be
at the beginning.' the moment a system message appears anywhere but the front,
and Claude Code appends exactly that. 50 consecutive 500s on 2026-08-21, 0 after.

WHAT IS NOT SETTLED

  - No task-success number for this artifact on this machine.
  - The decoder choice rests on one unpaired session per arm. Overall medians
    span 35.74-40.33 tok/s across five arms against a noise floor measured up to
    9.8 % at this depth -- only the short-turn gap (54.4 vs 38.8) clears it.
  - draft-mtp hit the 8,192 request cap once. If that recurs at 147,456 it will
    fill the window faster than it did at 98,304.
  - tok/s did not predict wall clock: dflash2+ngram n7 finished the same task in
    762.3 s against draft-mtp+ngram's 947.2 s while decoding slower.
#>
param(
    [int]$Ctx  = 147456,
    [int]$Port = 8080,
    # Log verbosity. 3 is the served default and keeps the log small; the
    # tensor-assignment lines that prove residency only appear at 5, so
    # `serve.ps1` asks for 5 at boot. Changing the DEFAULT would change what
    # every future served row was measured under, so it does not move.
    [int]$Verbosity = 3,
    # llama.cpp's own colours. 'auto' is exactly what it does with the flag
    # absent -- colour when stdout is a TTY -- so this default changes nothing
    # for a profile run by hand, and no measured row changes meaning. serve.ps1
    # asks for 'on' because it reads the output through a pipeline, which is not
    # a TTY, and auto would silently turn colour off there (issue #49).
    [ValidateSet('on', 'off', 'auto')]
    [string]$LogColors = 'auto',
    # llama.cpp writes an entry to the console AND then again to this file
    # (common/log.cpp:170-178), so a reader can follow the boot without standing
    # between llama.cpp and the terminal. Empty by default: a profile run by
    # hand should not start writing files nobody asked for.
    [string]$LogFile = '',
    # The ONLY access control this server has. It runs with no API key and CORS
    # '*', and middleware_validate_api_key (server-http.cpp:208) returns true
    # immediately when no key is set -- so no route is protected and widening
    # this does not weaken one control among several, it removes the only one.
    # Every measured row was taken on a server nothing off the machine could
    # reach. The default does not move; exposure is `serve.ps1 -Lan`, an act.
    # Pinned by bench/tests/test_bind_is_opt_in.py (issue #49).
    [string]$BindAddress = '127.0.0.1',
    # WHICH CARD. Empty means the served default, which is resolved below from
    # Get-GpuVram.ps1 rather than repeated here -- the same UUID written in two
    # files is two files that can disagree, and the one that gets edited is
    # never the one that gets read. Pass a comma-separated list to use more than
    # one card (issue #51 does).
    # Pinned by bench/tests/test_the_launch_names_its_gpu.py (issue #50).
    [string]$Device = '',
    [string]$Exe = "C:\AI\llama.cpp-blackwell\llama-server.exe",
    [string]$Model = "C:\Users\xenod\.cache\huggingface\hub\models--unsloth--Qwen3.8-27B-GGUF\snapshots\27af057ecb382ddfea5d12837360a8980560e3ed\Qwen3.8-27B-UD-Q2_K_XL.gguf",
    [switch]$IKnowTheBuildIsWrong
)
$ErrorActionPreference = 'Stop'

# ---- the guard ---------------------------------------------------------------
# Same as worker-5060ti.ps1, and for the same reason: a binary without Blackwell
# SASS runs here through PTX JIT at 2.20x the prefill time with nothing in any
# log to say so. The match is a SUBSTRING one on purpose -- llama.cpp's cmake
# rewrites 120 to 120a and the cubins are named sm_120a, so an exact 'sm_120'
# test would reject a correctly built binary.
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
    if (-not ($elf | Select-String -Quiet 'sm_120')) {
        # \w not \d: a Blackwell cubin is sm_120a, and \d+ would print it as
        # "sm_120" -- naming the architecture it is complaining is absent.
        $arches = ($elf | Select-String -Pattern 'sm_\w+' -AllMatches |
                   ForEach-Object { $_.Matches.Value } | Sort-Object -Unique) -join ', '
        Write-Host "FATAL: $dll has no Blackwell SASS (found: $arches)" -ForegroundColor Red
        Write-Host "  The driver would JIT the Ada PTX: 146,155 ms prefill against" -ForegroundColor Yellow
        Write-Host "  66,582 for the native build, with nothing in the log to say so." -ForegroundColor Yellow
        Write-Host "  Rebuild, or pass -IKnowTheBuildIsWrong (never for a measurement)." -ForegroundColor Yellow
        if (-not $IKnowTheBuildIsWrong) { exit 1 }
        Write-Host "  OVERRIDDEN -- results are not comparable to anything." -ForegroundColor Magenta
    }
} else {
    Write-Host "WARNING: cannot verify GPU architecture (missing $dll or cuobjdump)." -ForegroundColor Yellow
}

# ---- the card ----------------------------------------------------------------
# Checked BEFORE the model loads. An absent UUID does not make llama-server
# fail: it reports `(none)` for devices and then runs on CPU, producing correct
# output at a rate no row explains. A 40-minute plausible answer is the failure
# mode this repository exists to prevent, so this refuses first.
. (Join-Path $PSScriptRoot 'Get-GpuVram.ps1')

if (-not $Device) { $Device = $script:ServedGpuUuid }

# Read the driver ONCE. Test-ServedGpuPresent shells out per call, and asking
# nvidia-smi the same question once per requested card is work for nothing.
$installed = @(Get-InstalledGpu)
foreach ($uuid in ($Device -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })) {
    if (-not ($installed | Where-Object { $_.Uuid -eq $uuid })) {
        Write-Host "FATAL: GPU $uuid is not installed." -ForegroundColor Red
        Write-Host "  Installed:" -ForegroundColor Yellow
        $installed | ForEach-Object {
            Write-Host ("    {0}  {1}" -f $_.Uuid, $_.Name) -ForegroundColor Yellow
        }
        Write-Host "  llama-server would see no CUDA device and run on the CPU." -ForegroundColor Yellow
        exit 1
    }
}
$env:CUDA_VISIBLE_DEVICES = $Device

# An ARRAY, empty when no log was asked for. An inline `$(if ...)` would pass an
# empty string as a real argument and llama-server would see a flag it cannot
# parse -- the kind of failure that only shows up on the default path.
$logFileArg = if ($LogFile) { @('--log-file', $LogFile) } else { @() }

# ---- serve -------------------------------------------------------------------
# -cram is NOT set: its 8192 MiB default is worth 343x on task switching and
# caches into HOST RAM. Never set it to 0.
# --ctx-checkpoints is NOT set: the default 32 is what carries prefix reuse when
# n_rs_seq is 0, and it is right.
& $Exe -m $Model `
    --alias Qwen3.8-27B-Q2_K_XL -c $Ctx `
    -ngl auto --fit on --fit-target 768 -fa on -np 1 `
    -t 18 -b 2048 -ub 256 --no-mmproj-auto -lv $Verbosity `
    --log-colors $LogColors `
    @logFileArg `
    -ctk q4_0 -ctv q4_0 `
    --spec-type draft-mtp,ngram-mod `
    --spec-ngram-mod-n-match 12 --spec-ngram-mod-n-min 16 --spec-ngram-mod-n-max 32 `
    --chat-template-file "C:\AI\qwen38-tuning\templates\qwen38-late-system.jinja" `
    --reasoning-effort medium `
    --host $BindAddress --port $Port
