@echo off
REM ============================================================================
REM  Start the worker  --  BOTH GPUs, NVFP4, at its MEASURED CEILING, loopback only
REM
REM  Double-click this. The server runs IN this window and its output is this
REM  window's output. Ctrl+C stops it; so does closing this window.
REM
REM  THIS IS serve-dual-nvfp4.bat WITH A DEEPER WINDOW AND NOTHING ELSE CHANGED.
REM  Same model file, same speculative head, same n-gram at n-match 24. Only the
REM  context moves: 200,704 instead of 147,456.
REM
REM  200,704 IS NOT "THE DEEPEST THAT FITS", AND IT IS NOT THE DEEPEST THAT
REM  LOADS EITHER. It is the deepest rung that survived a request half the size
REM  of its own window, which is what every measured row in this project uses.
REM  Measured 2026-08-29, one boot per rung:
REM
REM      229,376   loaded, answered /health, then DIED on the request
REM                (cudaMalloc failed: out of memory), having loaded with only
REM                206 MiB free on the second card
REM      200,704   took a 91,428-token request, finished 1,133 and 654 MiB free
REM      180,224   took an 83,127-token request, finished 1,379 and 1,174 free
REM
REM  229,376 was written down as the ceiling first, because it survived a
REM  65,643-token prompt -- a QUARTER of its own window. A window is not a place
REM  to put one small prompt: a session that needs this depth will fill it.
REM  That is also why this launcher does not ask the free VRAM for a number the
REM  way the deep Q4 launchers do. At this edge, asking is the wrong question,
REM  because a window can load and pass a health check and still die.
REM
REM  WHAT IT COSTS IS THE HEADROOM, AND THIS IS THE WHOLE TRADE.
REM
REM      at 147,456   about 2,395 MiB free after a large request
REM      at 200,704   1,133 and 654 MiB
REM
REM  This project has measured a run dying with 336 MiB free and a run surviving
REM  with 488. 654 is above that line but not far above it. Whether you land on
REM  the good side depends on what your desktop is holding, which is why the
REM  profile re-checks the budget every launch and REFUSES rather than spilling
REM  to host memory -- a silent spill costs about 85x and returns a server that
REM  works.
REM
REM  So: click this when the work genuinely needs the window. Click
REM  serve-dual-nvfp4.bat when it does not; it is the same speed with room to
REM  spare.
REM
REM  QUALITY IS STILL UNMEASURED, exactly as for the shallower pair. This serves
REM  a different model file from serve-dual.bat, and no artifact in this project
REM  has ever had its output quality measured.
REM
REM  It holds no configuration. The flags live in
REM  qwen38-tuning\scripts\worker-q4-dual.ps1 and only there.
REM
REM  %~dp0 is this file's own folder. %CD% is not it when the file is opened
REM  from a shortcut or from a shell that started somewhere else.
REM
REM  This one binds 127.0.0.1 only. Nothing outside this machine reaches it.
REM ============================================================================

setlocal
cd /d "%~dp0"

where pwsh >nul 2>nul
if errorlevel 1 (
    echo.
    echo PowerShell 7 ^(pwsh^) was not found, and this needs it.
    echo Install it with:  winget install Microsoft.PowerShell
    echo.
    pause
    exit /b 1
)

pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0serve.ps1" -Dual -Nvfp4 -Deep
set RC=%ERRORLEVEL%

if not "%RC%"=="0" (
    echo.
    echo serve.ps1 exited with code %RC%.
    pause
)
