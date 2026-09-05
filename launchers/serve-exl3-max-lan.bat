@echo off
REM ============================================================================
REM  EXL3 at the native maximum window, 262,144, ON THE LAN
REM
REM  serve-exl3-max.bat bound to every interface. NO API KEY; anyone on the
REM  network can use it and read every prompt. Only on a network you control.
REM
REM  It holds no configuration. The recipe lives in
REM  qwen38-tuning\scripts\serve-exl3.cmd and only there; the only difference
REM  from serve-exl3-max.bat is the bind address.
REM ============================================================================

setlocal
cd /d "%~dp0.."

call "%~dp0..\qwen38-tuning\scripts\serve-exl3.cmd" 262144 "9,15.5" 0.0.0.0
set RC=%ERRORLEVEL%

if not "%RC%"=="0" (
    echo.
    echo serve-exl3.cmd exited with code %RC%.
    pause
)
