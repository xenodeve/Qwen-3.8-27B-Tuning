@echo off
REM ============================================================================
REM  Start GSQ IQ3_S-MTP -- both GPUs, 65,536 context, loopback only
REM
REM  The recipe lives in qwen38-tuning\scripts\serve-gsq.ps1. The cmd wrapper
REM  carries only the loopback mode and recomputes the tensor split at launch;
REM  this launcher deliberately carries no serving flags.
REM  The validated operating point uses MTP plus ngram-mod, tensor split
REM  8500,15468 and ngram match 24.
REM ============================================================================

setlocal
cd /d "%~dp0.."
call "%~dp0..\qwen38-tuning\scripts\serve-gsq.cmd"
set RC=%ERRORLEVEL%
if not "%RC%"=="0" (
    echo.
    echo serve-gsq.cmd exited with code %RC%.
    pause
)
endlocal & exit /b %RC%
