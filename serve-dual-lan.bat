@echo off
REM ============================================================================
REM  Start the worker  --  BOTH GPUs, UD-Q4_K_XL, REACHABLE FROM OTHER MACHINES
REM
REM  Double-click this. The server runs IN this window and its output is this
REM  window's output -- prompt timings, cache reuse, speculation counters,
REM  warnings.
REM
REM  Ctrl+C stops the server. So does closing this window. There is one
REM  process, not a server beside a log-watcher.
REM
REM  WHAT THIS ONE IS FOR. UD-Q4_K_XL is 16.69 GiB and does not fit on one
REM  16 GB card at any depth -- it spills 11 layers and decodes 11.7 tok/s.
REM  Across both cards it is fully resident to 229,376 and runs at 32.4 / 32.6
REM  / 33.1 tok/s at the served 147,456, which is PARITY with the single-card
REM  profile's 32.1 / 32.0 / 32.0. Issue #52.
REM
REM  WHAT IT COSTS, AND IT IS NOT NOTHING.
REM    * Roughly 130 W more. Both cards work; both draw power.
REM    * It needs BOTH cards installed. With one it refuses to start rather
REM      than quietly serving something else, and the message says which UUID
REM      it could not find.
REM    * QUALITY has never been measured here on this project's own artifacts.
REM      Every reason to prefer this artifact comes from a bits-per-weight
REM      ladder and an external campaign, neither of which is our number.
REM
REM  So this is not a strictly better serve.bat. Which icon is right is a
REM  decision, which is why there are four and none implies another.
REM
REM  THIS ONE ALSO EXPOSES THE SERVER on every interface. There is no API key
REM  and no origin restriction, so anyone who can reach port 8080 can use both
REM  GPUs and read whatever context is loaded. That is why it is a separate
REM  file: clicking it is the choice, the same as typing -Lan. Issue #49.
REM
REM  It holds no configuration. The flags live in
REM  qwen38-tuning\scripts\worker-q4-dual.ps1 and only there; a copy here
REM  would be a third place to drift.
REM
REM  %~dp0 is this file's own folder. %CD% is not it when the file is opened
REM  from a shortcut or from a shell that started somewhere else.
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

pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0serve.ps1" -Dual -Lan -AllowFirewall
set RC=%ERRORLEVEL%

if not "%RC%"=="0" (
    echo.
    echo serve.ps1 exited with code %RC%.
    pause
)
