@echo off
REM ============================================================================
REM  Start the EXL3 server  --  BOTH GPUs, Mia-AiLab 3.5bpw, MTP head, ON THE LAN
REM
REM  The same server as serve-exl3.bat bound to every interface. THIS SERVER HAS
REM  NO API KEY: anyone who can reach this machine can use it and read every
REM  prompt. Run it only on a network you control.
REM
REM  It holds no configuration. The recipe lives in
REM  qwen38-tuning\scripts\serve-exl3.cmd and only there; the only difference
REM  from serve-exl3.bat is the bind address.
REM ============================================================================

setlocal
cd /d "%~dp0.."

call "%~dp0..\qwen38-tuning\scripts\serve-exl3.cmd" 163840 "9,15.5" 0.0.0.0
set RC=%ERRORLEVEL%

if not "%RC%"=="0" (
    echo.
    echo serve-exl3.cmd exited with code %RC%.
    pause
)
