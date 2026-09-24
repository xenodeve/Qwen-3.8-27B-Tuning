@echo off
rem ============================================================================
rem  ThinkingCap-Qwen3.8-27B IQ4_XS serving profile -- 262,144 context
rem
rem  The recipe and unequal-card tensor split live in serve-thinkingcap.ps1. This cmd
rem  wrapper keeps the loopback/LAN entry point used by the hub and passes only
rem  the explicit exposure mode. The split is recomputed from live VRAM every
rem  launch; it never assumes a 50:50 layer distribution.
rem
rem  Usage: serve-thinkingcap.cmd [lan]
rem ============================================================================

setlocal
set "HOST=127.0.0.1"
if /I "%~1"=="lan" set "HOST=0.0.0.0"
if not "%~1"=="" if /I not "%~1"=="lan" (
    echo Invalid ThinkingCap serving mode: %~1
    echo Use no argument for loopback or lan for explicit network exposure.
    exit /b 2
)

where pwsh >nul 2>nul
if errorlevel 1 (
    echo PowerShell 7 ^(pwsh^) was not found.
    exit /b 1
)

pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0serve-thinkingcap.ps1" -Ctx 262144 -BindAddress %HOST%
set RC=%ERRORLEVEL%
if not "%RC%"=="0" echo serve-thinkingcap.ps1 exited with code %RC%.
endlocal & exit /b %RC%
