@echo off
REM ============================================================================
REM  Start the worker  --  REACHABLE FROM OTHER MACHINES
REM
REM  Double-click this. The server runs IN this window and its output is this
REM  window's output -- prompt timings, cache reuse, speculation counters,
REM  warnings.
REM
REM  Ctrl+C stops the server. So does closing this window. There is one
REM  process, not a server beside a log-watcher.
REM
REM  THIS ONE EXPOSES THE SERVER on every interface. There is no API key and
REM  no origin restriction, so anyone who can reach port 8080 can use the GPU
REM  and read whatever context is loaded. That is why it is a separate file:
REM  clicking it is the choice, the same as typing -Lan. Issue #49.
REM
REM  It holds no configuration. The flags live in
REM  qwen38-tuning\scripts\worker-q2kxl-mtp.ps1 and only there; a copy here
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

pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0serve.ps1" -Lan -AllowFirewall
set RC=%ERRORLEVEL%

if not "%RC%"=="0" (
    echo.
    echo serve.ps1 exited with code %RC%.
    pause
)
endlocal
