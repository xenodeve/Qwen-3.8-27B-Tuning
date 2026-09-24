@echo off
REM ============================================================================
REM  Start ThinkingCap-Qwen3.8-27B IQ4_XS -- both GPUs, 262,144 context, explicit LAN exposure
REM
REM  This is intentionally a separate icon from the loopback launcher. It calls
REM  the same recipe with the one explicit mode that binds 0.0.0.0; there is no
REM  API key, so use it only on a network you control.
REM ============================================================================

setlocal
cd /d "%~dp0.."
call "%~dp0..\qwen38-tuning\scripts\serve-thinkingcap.cmd" lan
set RC=%ERRORLEVEL%
if not "%RC%"=="0" (
    echo.
    echo serve-thinkingcap.cmd exited with code %RC%.
    pause
)
endlocal & exit /b %RC%
