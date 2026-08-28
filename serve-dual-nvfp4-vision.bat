@echo off
REM ============================================================================
REM  Start the worker  --  BOTH GPUs, NVFP4, WITH IMAGES, loopback only
REM
REM  Double-click this. The server runs IN this window and its output is this
REM  window's output. Ctrl+C stops it; so does closing this window.
REM
REM  THIS IS serve-dual-nvfp4.bat PLUS THE VISION TOWER. Same model file, same
REM  speculative head, same n-gram at n-match 24, same 147,456 window. The only
REM  addition is the projector, and the only thing it changes is that images
REM  work.
REM
REM  WITHOUT IT, EVERY IMAGE IS AN ERROR. The server answers HTTP 500:
REM
REM      image input is not supported - hint: if this is unexpected, you may
REM      need to provide the mmproj
REM
REM  and that is what a real Claude Code session hit five times on 2026-08-29.
REM  The model was never the limitation -- it is a native vision-language model
REM  and its own chat template handles images. The vision tower is simply a
REM  separate file, and this project had it switched off on purpose because the
REM  benchmark work is text.
REM
REM  MEASURED, NOT ASSUMED. It was expected to fail: the tower is a second model
REM  and this split has never hosted one -- the DFlash2 drafter aborts inside
REM  llama.cpp for exactly that reason. It does not fail. Loaded on the ordinary
REM  unpatched binary and answered a real 512x512 picture correctly at 65,536,
REM  147,456 and 200,704.
REM
REM  WHAT IT COSTS: 888 MiB, on a card. At this window a large request finishes
REM  with roughly 1,205 and 2,450 MiB free, against about 2,395 without it. The
REM  profile counts the tower before starting and refuses rather than spilling.
REM
REM  WHY THERE IS NO DEEP VERSION OF THIS ICON. 200,704 loaded and answered too,
REM  but that was a small picture and a short answer, and it finished with 614
REM  MiB free on one card. This project has measured a run dying with 336 MiB
REM  free and one surviving with 488. Images beside a large text prompt have not
REM  been measured at any depth, so the deep rung stays text-only until they are.
REM
REM  It holds no configuration. The flags live in
REM  qwen38-tuning\scripts\worker-q4-dual.ps1 and only there.
REM
REM  %~dp0 is this file's own folder. %CD% is not it when the file is opened
REM  from a shortcut or from a shell that started somewhere else.
REM
REM  This one binds 127.0.0.1 only. Nothing outside this machine reaches it.
REM ============================================================================

setlocal
cd /d "%~dp0"

where pwsh >nul 2>nul
if errorlevel 1 (
    echo.
    echo PowerShell 7 ^(pwsh^) was not found, and this needs it.
    echo Install it with:  winget install Microsoft.PowerShell
    echo.
    pause
    exit /b 1
)

pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0serve.ps1" -Dual -Nvfp4 -Vision
set RC=%ERRORLEVEL%

if not "%RC%"=="0" (
    echo.
    echo serve.ps1 exited with code %RC%.
    pause
)
