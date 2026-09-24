@echo off
REM ============================================================================
REM  Start Qwen3.8-Flash-Next GSQ-RCO Q2_0 -- both GPUs, 131,072 context, LAN
REM
REM  The recipe lives in qwen38-tuning\scripts\serve-flash-next.ps1 (profile
REM  128k), reached through serve-flash-next.cmd. This launcher carries no
REM  serving flags. Measured 2026-09-23; quality is unmeasured.
REM  Intentionally separate from the loopback icon. It binds 0.0.0.0 and there
REM  is no API key, so use it only on a network you control.
REM ============================================================================

setlocal
cd /d "%~dp0.."
call "%~dp0..\qwen38-tuning\scripts\serve-flash-next.cmd" 128k lan
set RC=%ERRORLEVEL%
if not "%RC%"=="0" (
    echo.
    echo serve-flash-next.cmd exited with code %RC%.
    pause
)
endlocal & exit /b %RC%
