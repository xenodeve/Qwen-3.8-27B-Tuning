@echo off
REM ============================================================================
REM  Start the worker  --  BOTH GPUs, UD-Q4_K_XL, with draft-mtp, REACHABLE FROM OTHER MACHINES
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
REM  THIS ONE ALSO EXPOSES THE SERVER on every interface. There is no API key
REM  and no origin restriction, so anyone who can reach port 8080 can use both
REM  GPUs and read whatever context is loaded. Clicking it is the choice.
REM
REM  It holds no configuration. The flags live in
REM  qwen38-tuning\scripts\worker-q4-dual.ps1 and only there.
REM
REM
REM  THE WINDOW IS COMPUTED AT LAUNCH, NOT FIXED. This asks for the deepest
REM  context the free VRAM supports, capped at the model's own 262,144. It is
REM  not a constant: 262,144 loaded on this machine when the desktop held about
REM  1,600 MiB and ran out of memory at 2,575, so the number moves with what
REM  you have open. The window it settled on is printed when it starts.
REM
REM  It also spends the micro-batch before the context -- halving -ub frees
REM  about a gigabyte across the pair for roughly 3.5 percent of prefill, where
REM  the same memory bought with context costs tens of thousands of tokens.
REM
REM  AND IT LEAVES LESS ROOM. At full depth a large request finishes with a few
REM  hundred MiB spare, against about 2,000 at the 147,456 default. A run with
REM  336 MiB free died on its first request; one with 488 survived 135,233
REM  tokens. Deep is measured, not comfortable.
REM
REM  %~dp0 is this file's own folder, and this file lives in launchers\,
REM  so the paths below climb one level to reach the repository root. %CD% is not it when the file is opened
REM  from a shortcut or from a shell that started somewhere else.
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

pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\serve.ps1" -Dual -MaxCtx -Mtp -Lan -AllowFirewall
set RC=%ERRORLEVEL%

if not "%RC%"=="0" (
    echo.
    echo serve.ps1 exited with code %RC%.
    pause
)
