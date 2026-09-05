@echo off
REM ============================================================================
REM  Start the EXL3 server at the model's NATIVE MAXIMUM window, 262,144
REM
REM  serve-exl3.bat with the cache raised from 163,840 to 262,144 tokens, same
REM  split 9,15.5. MEASURED 2026-09-04 on the SC 4.00bpw H5 artifact: a 197,020-
REM  token prompt prefilled at 431.7 tok/s and decoded at 25.0 tok/s (60-token
REM  sample), VRAM 11.0 / 13.4 GB, no sync timeout. At cap 10 the same prompt
REM  ended in pg_all_reduce sync timeouts (the 4070 at 11.9 / 12.3 GB). Results 10.
REM
REM  It holds no configuration. The recipe lives in
REM  qwen38-tuning\scripts\serve-exl3.cmd and only there; this passes the window,
REM  the split caps, and the bind address.
REM ============================================================================

setlocal
cd /d "%~dp0.."

call "%~dp0..\qwen38-tuning\scripts\serve-exl3.cmd" 262144 "9,15.5" 127.0.0.1
set RC=%ERRORLEVEL%

if not "%RC%"=="0" (
    echo.
    echo serve-exl3.cmd exited with code %RC%.
    pause
)
