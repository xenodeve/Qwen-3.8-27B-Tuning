<#
WORKER PROFILE — RTX 5060 Ti 16 GB (Blackwell, sm_120). Issue #40.

Unsloth Dynamic V3 UD-IQ2_XXS at 262,144 -- the model's full n_ctx_train, which
was unreachable on the 12 GB card this project was built around.

THE GUARD BELOW IS THE POINT OF THIS FILE, not the flags.

  The binaries in llama.cpp-cuda and llama.cpp-dflash2 were built with
  CMAKE_CUDA_ARCHITECTURES=89. `cuobjdump --list-elf` on their ggml-cuda.dll
  returns only sm_89 cubins. This card is sm_120, so the driver JIT-compiles the
  Ada PTX and produces kernels tuned for neither architecture.

  Measured 2026-08-23 at ctx 98,304, same corpus and flags as the old card:

      prefill 44K    146,155 ms   against    35,301 ms on the 4070 SUPER
      decode          22.67 tok/s against     96.92 tok/s

  Allocation was byte-identical -- model 6,521.13, KV 1,728.00, RS 149.62,
  compute 472.27, 65+0, no OOM -- and --fit left 6,150 MiB free against 2,047 on
  the old card. FOUR TIMES SLOWER WITH THREE TIMES THE HEADROOM IS NOT A
  HARDWARE RESULT. Nothing in the log says the kernels were JIT'd.

  That is precisely the failure CLAUDE.md's north star names: an instrument that
  returns a believable number instead of a failure. So this profile refuses to
  launch rather than serve four times slow and silent.

  Fix: rebuild with -DCMAKE_CUDA_ARCHITECTURES="89;120". CUDA 13.3 is installed
  and nvcc --list-gpu-code lists sm_120, so nothing is missing but the flag.

WHAT TRANSFERRED FROM THE RTX 4070 SUPER -- mechanism, not measurement.

  -ctk q4_0 -ctv q4_0     no other KV type in this build has a fast kernel
  -cram NEVER 0           worth 343x on task switching; it caches sequence state
                          in HOST RAM, so it is unaffected by the card. Returning
                          to a 44K conversation cost 118.2 ms at 100 % reuse with
                          the default and 40,596 ms at 0 % with -cram 0
  --ctx-checkpoints 32     the default, and what carries prefix reuse when
                          n_rs_seq = 0. Not set here because the default is right
  an edit ahead of the    reuse does not degrade, it ZEROES -- 0.0 %, a full
  suffix                  re-prefill. Anything injected ahead of the conversation
                          must be byte-identical on every turn
  chars/token ~3.4        a property of the tokenizer and the corpus
  --fit acts almost never  confirmed again on this card: "no changes needed"
  nvidia-smi free is NOT  what --fit reads. On this card, same boot: nvidia-smi
  the number that matters said 13,637 MiB free, llama.cpp said 15,172. With a game
                          running the gap was 7.5 GB. Read the log line
  KV = 18.00 KiB/token    16 attention layers at q4_0, measured on THIS card

WHAT DID NOT TRANSFER -- every number, and every arm verdict.

  Do not carry any of these onto this card without re-measuring:

    96.92 / 49.31 / 5.66 / 33.69 tok/s and the whole decoder ranking
    the 45-376 MiB band where DFlash2 became unreliable
    the 13.6 % noise floor, and the 48.9 % spread at ctx 65,536
    "11,069 MiB free" from all 552 logs -- this card reports 15,172
    -ub 64 costing 14.0 % of decode

  --spec-type ngram-mod IS THE STARTING POINT HERE, NOT A VERDICT. It is the
  incumbent because it won on Ada, where every drafter competed with the layers
  for a 12 GB budget. That constraint is gone: at this context the sidecar would
  still leave 1,429 MiB, four to thirty times the band where it failed before.
  Whether DFlash2 now wins is UNMEASURED and is the first thing to sweep once the
  rebuild lands.

WHY ctx 262,144. Measured on this card at 98,304 and projected on its own KV
rate, which is flat per token:

    ngram-mod only          used      free
      98,304               8,871    6,301 MiB
     262,144              11,751    3,421 MiB     <- this file
    with DFlash2 loaded
     262,144              13,743    1,429 MiB

262,144 is n_ctx_train for this model -- the ceiling is now the model, not the
card, for the first time in this project.

WHY -lv 3. Speculation statistics need -lv 4 and none of the four old profiles
had them, which is issue #28. Left at 3 here because this profile is for serving;
raise it when measuring.

WHY --chat-template-file. Qwen3.8's stock template raises 'System message must be
at the beginning.' the moment a system message appears anywhere but the front,
and Claude Code appends exactly that. 50 consecutive 500s on 2026-08-21, 0 after.
See templates/README.md.
#>
param(
    [int]$Ctx  = 262144,
    [int]$Port = 8080,
    [string]$Exe = "C:\AI\llama.cpp-blackwell\llama-server.exe",
    [switch]$IKnowTheBuildIsWrong
)
$ErrorActionPreference = 'Stop'

# ---- the guard ---------------------------------------------------------------
# Reads the actual code objects out of the shipped ggml-cuda.dll. A binary
# without sm_120 SASS runs on this card through PTX JIT and is ~4x slow with no
# error anywhere, so refusing is the only honest option.
$dll = Join-Path (Split-Path $Exe -Parent) 'ggml-cuda.dll'
$cuobjdump = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\cuobjdump.exe'

if (-not (Test-Path $Exe)) {
    Write-Host "FATAL: no server at $Exe" -ForegroundColor Red
    Write-Host "  Build it:  cmake --build <dir> --target llama-server" -ForegroundColor Yellow
    Write-Host "  with       -DCMAKE_CUDA_ARCHITECTURES=`"89;120`"" -ForegroundColor Yellow
    Write-Host "  The Ada-only builds in llama.cpp-cuda and llama.cpp-dflash2 are 4x slow here." -ForegroundColor Yellow
    exit 1
}

if ((Test-Path $dll) -and (Test-Path $cuobjdump)) {
    $elf = & $cuobjdump --list-elf $dll 2>$null
    $hasBlackwell = $elf | Select-String -Quiet 'sm_120'
    if (-not $hasBlackwell) {
        $arches = ($elf | Select-String -Pattern 'sm_\d+' -AllMatches |
                   ForEach-Object { $_.Matches.Value } | Sort-Object -Unique) -join ', '
        Write-Host "FATAL: $dll has no sm_120 SASS (found: $arches)" -ForegroundColor Red
        Write-Host "  This card is sm_120. The driver would JIT the Ada PTX and run ~4x slow" -ForegroundColor Yellow
        Write-Host "  with nothing in the log to say so -- measured 22.67 tok/s against 96.92" -ForegroundColor Yellow
        Write-Host "  on the older, SMALLER card. See issue #40." -ForegroundColor Yellow
        Write-Host "  Rebuild with -DCMAKE_CUDA_ARCHITECTURES=`"89;120`"," -ForegroundColor Yellow
        Write-Host "  or pass -IKnowTheBuildIsWrong to launch anyway (never for a measurement)." -ForegroundColor Yellow
        if (-not $IKnowTheBuildIsWrong) { exit 1 }
        Write-Host "PROCEEDING ON AN ADA-ONLY BUILD -- results are not comparable to anything." -ForegroundColor Magenta
    }
} else {
    Write-Host "WARNING: cannot verify GPU architecture (missing $dll or cuobjdump)." -ForegroundColor Yellow
    Write-Host "  Verify by hand before trusting any number from this run." -ForegroundColor Yellow
}

# ---- launch ------------------------------------------------------------------
& $Exe `
    -m "C:\Users\xenod\.cache\huggingface\hub\models--unsloth--Qwen3.8-27B-GGUF\snapshots\27af057ecb382ddfea5d12837360a8980560e3ed\Qwen3.8-27B-UD-IQ2_XXS.gguf" `
    --alias qwen38 -c $Ctx `
    -ngl auto --fit on --fit-target 768 -fa on -np 1 `
    -t 18 -b 2048 -ub 256 --no-mmproj-auto -lv 3 `
    -ctk q4_0 -ctv q4_0 `
    --spec-type ngram-mod `
    --spec-ngram-mod-n-match 12 --spec-ngram-mod-n-min 16 --spec-ngram-mod-n-max 32 `
    --chat-template-file "C:\AI\qwen38-tuning\templates\qwen38-late-system.jinja" `
    --host 127.0.0.1 --port $Port
