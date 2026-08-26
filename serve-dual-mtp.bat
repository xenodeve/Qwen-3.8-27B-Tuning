@echo off
REM ============================================================================
REM  Start the worker  --  BOTH GPUs, UD-Q4_K_XL, with draft-mtp, loopback only
REM
REM  Double-click this. The server runs IN this window and its output is this
REM  window's output. Ctrl+C stops it; so does closing this window.
REM
REM  WHAT IS DIFFERENT FROM serve-dual.bat: this one adds llama.cpp's draft-mtp
REM  decoder beside ngram-mod, using the nextn head already inside the model
REM  file. No second model is downloaded or loaded.
REM
REM  ITS SPEED IS NOT MEASURED, AND THAT IS NOT A FORMALITY.
REM  draft-mtp loads and runs here -- verified 2026-08-27, after this project
REM  had wrongly recorded that it could not run on this split at all. But every
REM  paired measurement of it was VOIDED by our own output guard, because the
REM  generations copy the prompt instead of answering it. Three unpaired manual
REM  readings looked excellent (44.5 / 54.3 / 92.7 tok/s) and those are exactly
REM  the numbers that kind of guard exists to reject: a speculative decoder
REM  gets faster the more predictable the text is, and copying is maximally
REM  predictable.
REM
REM  So click this to TRY it. serve-dual.bat is the one with a real number
REM  behind it -- 25.5 / 25.4 / 26.4 tok/s at ctx 147,456, against 21.8 with no
REM  speculation at all.
REM
REM  It also costs about 2,750 MiB more across the two cards. The profile
REM  accounts for that before starting and refuses if the budget is gone.
REM
REM  It holds no configuration. The flags live in
REM  qwen38-tuning\scripts\worker-q4-dual.ps1 and only there.
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

pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0serve.ps1" -Dual -Mtp
set RC=%ERRORLEVEL%

if not "%RC%"=="0" (
    echo.
    echo serve.ps1 exited with code %RC%.
    pause
)
