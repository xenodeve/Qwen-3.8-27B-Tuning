@echo off
REM ============================================================================
REM  Start the worker  --  ICON 7 EXACTLY, ON UNSLOTH'S BINARY, loopback only
REM
REM  200,704 context, images, the Unsloth bundle -- every value the same as
REM  icon 7. The ONLY difference is which llama-server runs it.
REM
REM  WHY THIS ONE EXISTS
REM
REM  Icon 9 is Studio's whole command line on OUR build. Icon A is Studio's
REM  command line on THEIR build. Both are baselines pinned at -c 107,899,
REM  because that is what Studio's server happened to compute at 00:11 --
REM  useful for answering a question, useless for serving, since this machine
REM  exists to serve 200,704.
REM
REM  This is the fourth cell, and the useful one:
REM
REM                     our flags          their flags
REM      our build      1 / 2 / 7 / 8      icon 9
REM      their build    THIS ONE           icon A
REM
REM  MEASURED, at matched depth, on 2026-08-30:
REM
REM      ~48,000    icon 8 (ours)                       33.05 tok/s
REM                 icon 9 (their flags, our build)     33.00
REM                 icon A (their flags, their build)   41.58
REM
REM      ~76,000    icon 8                              28.59
REM                 icon A                              35.23
REM
REM  Read the first block twice. THEIR ENTIRE FLAG SET ON OUR BUILD CHANGED
REM  NOTHING -- 33.00 against 33.05, which is not a difference. Their BUILD
REM  read +26 percent there, from ONE boot per side. It has never been
REM  paired. A run that appeared to refute it measured ONE BINARY TWICE
REM  and its null is an artefact (CORRECTIONS 40 and 41), so the status
REM  is CONTESTED and settled in neither direction.
REM
REM  That is why this icon is worth running: back to back against icon
REM  7 it is the pairing nobody has done.
REM
REM      ours     version 0.1.2-dev   build 10499   commit 1deefcca3
REM      Studio   version 0.3.0-dev   build 10679   commit b84725557
REM
REM  The build NUMBERS differ by 180. That is not a count of commits between
REM  them: our own HEAD is 1deefcca3, a local commit "Add p_min in DFlash2",
REM  and neither b84725557 nor upstream's 50f068f is in our clone. BOTH SIDES
REM  ARE FORKS. Their --help is a strict superset of ours, no default
REM  differs on any flag either side sets, and both binaries carry native SASS
REM  for both cards here -- so what separates them is a SOURCE/BUILD-LINEAGE DELTA,
REM  not a missing kernel and not a flag we cannot reach.
REM
REM  THIS IS NOT A RESULT YET. One reading per side, taken in different boots,
REM  and this project has measured the same arm drifting 48.9 percent across
REM  boots at depth (CORRECTIONS 23). Run this and icon 7 back to back, ask
REM  them the same thing, and read the SERVER LOG rather than the feel:
REM
REM      prompt eval time = ... tokens per second     <- prefill
REM      eval time        = ... tokens per second     <- decode
REM
REM  ONE THING TO WATCH. Their binary is not built to sit beside ours: with a
REM  bare PATH it finds NO CUDA device and serves from the CPU without saying
REM  so. The profile prepends the loader path and REFUSES to launch if it
REM  cannot find cudart, because a warning would be read past and the number
REM  would look ordinary.
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

pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\serve.ps1" -Dual -Nvfp4 -Deep -Vision -Beta -TheirBuild
set RC=%ERRORLEVEL%

if not "%RC%"=="0" (
    echo.
    echo serve.ps1 exited with code %RC%.
    pause
)
