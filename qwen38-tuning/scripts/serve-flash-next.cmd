@echo off
rem ============================================================================
rem  Qwen3.8-Flash-Next GSQ-RCO Q2_0 serving profiles -- 262,144 or 131,072
rem
rem  The recipe lives in serve-flash-next.ps1 (layer split 30,18, VNNI + expert
rem  cache binary, 2026-09-23). This cmd wrapper keeps the loopback/LAN entry
rem  point used by the hub and passes only the explicit exposure mode.
rem
rem  Usage: serve-flash-next.cmd <262k|128k> [lan]
rem ============================================================================

setlocal
set "PROFILE=%~1"
if /I not "%PROFILE%"=="262k" if /I not "%PROFILE%"=="128k" (
    echo Invalid Flash-Next profile: %~1
    echo Use 262k or 128k as the first argument.
    exit /b 2
)
set "HOST=127.0.0.1"
if /I "%~2"=="lan" set "HOST=0.0.0.0"
if not "%~2"=="" if /I not "%~2"=="lan" (
    echo Invalid Flash-Next serving mode: %~2
    echo Use no second argument for loopback or lan for explicit network exposure.
    exit /b 2
)

where pwsh >nul 2>nul
if errorlevel 1 (
    echo PowerShell 7 ^(pwsh^) was not found.
    exit /b 1
)

pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0serve-flash-next.ps1" -Profile %PROFILE% -BindAddress %HOST%
set RC=%ERRORLEVEL%
if not "%RC%"=="0" echo serve-flash-next.ps1 exited with code %RC%.
endlocal & exit /b %RC%
