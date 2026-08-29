@echo off
REM ============================================================================
REM  Start the worker  --  BOTH GPUs, NVFP4, images, THE UNSLOTH BUNDLE, loopback only
REM
REM  Double-click this. The server runs IN this window and its output is this
REM  window's output. Ctrl+C stops it; so does closing this window.
REM
REM  THIS IS serve-dual-nvfp4.bat WITH SIX SETTINGS BORROWED FROM UNSLOTH
REM  STUDIO, which runs this same model file on these same two cards:
REM
REM      --cache-ram 0          prompt cache off
REM      --ctx-checkpoints 0    context checkpoints off
REM      --load-mode none       no memory-mapped read
REM      --kv-unified           one shared KV buffer
REM      --metrics              Prometheus endpoint
REM      -t 2                   two threads instead of eighteen
REM
REM  Everything else is identical to serve-dual-nvfp4.bat: same file, same
REM  window of 147,456, same speculative head, same n-gram at n-match 24, same
REM  micro-batch. That is deliberate -- the two icons are an A/B you can run.
REM
REM  THE SPEED HALF OF THIS IS UNMEASURED. Not "probably fine", not "should
REM  help" -- nobody has run a paired comparison. Two boots exist and they
REM  generated thirty tokens each, which this project's own guard rejects as a
REM  rate. Treat any speed you see here as an anecdote until it is paired.
REM
REM  THE MEMORY HALF IS MEASURED, and it is why the bundle exists:
REM
REM      serve-dual-nvfp4.bat        15.28 GB working set
REM      this one                     2.03 GB working set
REM
REM  reproduced across two boots. Private memory did not move (25.66 against
REM  25.79 GB), so what collapses is resident pages, not commitment.
REM
REM  IT IS NOT A FREE SAVING. Those context checkpoints were doing work: the
REM  log of a real eighteen-minute session shows them being RESTORED at
REM  positions 47,940 to 50,091. Turning them off trades host RAM for
REM  re-processing the prompt -- roughly a minute per 50,000 tokens. If your
REM  machine has RAM to spare and your conversations are long, the default is
REM  probably the better side of that trade. If you are short of RAM, this is.
REM
REM  It holds no configuration. The flags live in
REM  qwen38-tuning\scripts\worker-q4-dual.ps1 and only there.
REM
REM  %~dp0 is this file's own folder, and this file lives in launchers\,
REM  so the paths below climb one level to reach the repository root.
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

pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\serve.ps1" -Dual -Nvfp4 -Vision -Lean
set RC=%ERRORLEVEL%

if not "%RC%"=="0" (
    echo.
    echo serve.ps1 exited with code %RC%.
    pause
)
