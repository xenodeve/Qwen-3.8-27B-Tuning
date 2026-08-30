@echo off
REM ============================================================================
REM  Start the worker  --  STUDIO'S COMMAND LINE ON STUDIO'S BINARY, loopback only
REM
REM  A BASELINE. NOT A RECOMMENDATION. NOT FOR SERVING. Window 107,899.
REM
REM  THE CONFOUND THIS ICON REMOVES
REM
REM  Every comparison this project made against Unsloth Studio assumed one
REM  binary. It is two, and their build NUMBERS differ by 180 -- which is not
REM  a count of commits between them, since both sides are forks:
REM
REM      ours     version 0.1.2-dev   build 10499   commit 1deefcca3
REM      Studio   version 0.3.0-dev   build 10679   commit b84725557
REM
REM  Icon 9 is their flags on OUR build, so it cannot tell "their flags are
REM  better" from "their build is newer". This one is their flags on THEIR
REM  build. Run both.
REM
REM      icon 9   their flags, our build      -> if this matches Studio, FLAGS
REM      icon 10  their flags, their build    -> if only this matches, BUILD
REM      neither matches                      -> neither, and eight flag sweeps
REM                                              would have found nothing
REM
REM  WHAT IS AND IS NOT DIFFERENT ABOUT THE BUILDS, checked rather than assumed:
REM
REM      flags       their --help is a strict SUPERSET -- ten they have and we
REM                  lack (--kv-unified-per-slot, --spec-synth-len,
REM                  --spec-synth-rates, --tensor-read-lazy, -ncffn, -mmdev,
REM                  --rpc, three --video-*) and NONE we have that they lack.
REM                  Every flag on their command line exists in our build.
REM      defaults    identical on every flag either side sets.
REM      SASS        theirs sm_86/89/90/100/120a, ours sm_89/120a. BOTH carry
REM                  native code for BOTH cards here, so neither is JIT-ing.
REM                  Their ggml-cuda.dll is 226 MB against our 88 MB, and that
REM                  is other people's GPUs, not better kernels for ours.
REM      features    ARCHS = 860,890,900,1000,1200 ^| USE_GRAPHS = 1 ^|
REM                  BLACKWELL_NATIVE_FP4 = 1, against our 890,1200 with the
REM                  same two feature flags.
REM
REM  What is left between them is a SOURCE/BUILD-LINEAGE DELTA.
REM
REM  THE FAULT THIS ICON HAD TO BE BUILT AROUND
REM
REM  Launched with a bare PATH, their binary reports
REM
REM      device_info:
REM        - CPU     : 13th Gen Intel(R) Core(TM) i5-13500
REM
REM  and no CUDA device at all -- then serves, from the CPU, at a speed somebody
REM  would write down. Studio prepends the loader path itself, and CUDA 13 keeps
REM  cudart64_13.dll in %%CUDA_PATH%%\bin\x64 rather than \bin. The profile
REM  prepends both and REFUSES to launch if it cannot find the runtime, because
REM  a warning would be read past and the number would look fine.
REM
REM  It holds no configuration. The flags live in
REM  qwen38-tuning\scripts\worker-q4-dual.ps1 and only there.
REM
REM  This one binds 127.0.0.1 only. Nothing outside this machine reaches it.
REM ============================================================================

setlocal
cd /d "%~dp0.."

where pwsh >nul 2>nul
if errorlevel 1 (
    echo.
    echo PowerShell 7 ^(pwsh^) was not found, and this needs it.
    echo Install it with:  winget install Microsoft.PowerShell
    echo.
    pause
    exit /b 1
)

pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\serve.ps1" -Dual -Nvfp4 -Vision -Clone -TheirBuild
set RC=%ERRORLEVEL%

if not "%RC%"=="0" (
    echo.
    echo serve.ps1 exited with code %RC%.
    pause
)
