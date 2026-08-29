@echo off
REM ============================================================================
REM  Start the worker  --  BOTH GPUs, NVFP4, images, THE UNSLOTH BUNDLE MINUS
REM  --kv-unified, REACHABLE FROM YOUR NETWORK
REM
REM  THIS IS ICON 7 WITH EXACTLY ONE FLAG REMOVED. Run it against icon 7 and
REM  the only difference between the two servers is `--kv-unified`.
REM
REM  WHY THAT FLAG. Measured 2026-08-29, same machine, same model file, same
REM  evening, Discord streaming through both:
REM
REM                            Unsloth Studio        our -Beta (icon 7)
REM      prefill               728 - 1,000 tok/s     319 - 633
REM      decode                34.9 - 48.0           24.1 - 29.0
REM      decode overall        39.86                 26.17
REM      at depth ~47,000      36.63                 26.93 - 29.02
REM      mean accepted length  1.81 - 2.52           2.51 - 2.81
REM
REM  Read the last row first. OUR DRAFTING IS BETTER ON EVERY REQUEST and we
REM  are still slower, so the time is going into the target model's forward
REM  pass, not into speculation. That points at the flags which lay out
REM  attention and the KV cache, and `--kv-unified` is the first of them: we
REM  set it, Studio does not (`kv_unified = 'false'` in its boot log).
REM
REM  It is also the only candidate that would explain the OTHER unexplained
REM  difference. Studio reuses a 39,616-token prefix with
REM  `--ctx-checkpoints 0`, while that same setting made every one of our
REM  requests re-read the prompt from token 0:
REM
REM      forcing full prompt re-processing due to lack of cache data
REM      (likely due to SWA or hybrid/recurrent memory)
REM
REM  A single shared KV buffer is a plausible reason a partial sequence
REM  removal cannot be done. PLAUSIBLE. NOT MEASURED. That is what this icon
REM  is for.
REM
REM  THE SECOND SUSPECT IS NOT IN THIS ICON, deliberately: our `-c 200,704`
REM  against their `-c 107,899` is nearly double the KV allocation and it
REM  squeezes the tensor split. One flag at a time -- this project has already
REM  changed two things and read the result as one.
REM
REM  HOW TO READ WHAT YOU GET. Do not compare a rate from this window against
REM  a rate written down on another day: the same arm has drifted 48.9 percent
REM  across boots at depth here. Run this and icon 7 back to back, ask them the
REM  same thing, and look at the SERVER LOG rather than the feel:
REM
REM      prompt eval time = ... tokens per second     <- prefill
REM      eval time        = ... tokens per second     <- decode
REM      forcing full prompt re-processing            <- present or absent
REM
REM  The third line is the one that matters most. If it disappears without
REM  `--kv-unified`, that is the answer to both questions at once.
REM
REM  It holds no configuration. The flags live in
REM  qwen38-tuning\scripts\worker-q4-dual.ps1 and only there.
REM
REM  %~dp0 is this file's own folder, and this file lives in launchers\,
REM  so the paths below climb one level to reach the repository root.
REM
REM  THIS BINDS 0.0.0.0. No API key, no authentication, CORS open -- anyone who
REM  can reach this machine can use it and read every prompt. LAN you control only.
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

pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\serve.ps1" -Dual -Nvfp4 -Deep -Vision -Beta -NoKvUnified -Lan
set RC=%ERRORLEVEL%

if not "%RC%"=="0" (
    echo.
    echo serve.ps1 exited with code %RC%.
    pause
)
