@echo off
REM ============================================================================
REM  Start ThinkingCap-Qwen3.8-27B IQ4_XS -- both GPUs, 262,144 context, loopback only
REM
REM  The recipe lives in qwen38-tuning\scripts\serve-thinkingcap.ps1. The cmd wrapper
REM  carries only the loopback mode and recomputes the tensor split at launch;
REM  this launcher deliberately carries no serving flags.
REM  MTP n3 plus ngram-mod 24/16/64, -ub 512 (1024 does not fit the MTP draft
REM  at 262K). Result 35: equal to GSQ on openclink #144 at 165-209K.
REM ============================================================================

setlocal
cd /d "%~dp0.."
call "%~dp0..\qwen38-tuning\scripts\serve-thinkingcap.cmd"
set RC=%ERRORLEVEL%
if not "%RC%"=="0" (
    echo.
    echo serve-thinkingcap.cmd exited with code %RC%.
    pause
)
endlocal & exit /b %RC%
