@echo off
REM ============================================================================
REM  Start the worker  --  BOTH GPUs, the NVFP4 artifact, reachable on your LAN
REM
REM  Double-click this. The server runs IN this window and its output is this
REM  window's output. Ctrl+C stops it; so does closing this window.
REM
REM  THIS IS THE FASTEST CONFIGURATION MEASURED HERE. At ctx 147,456, three
REM  paired rounds rotated on real vendor code:
REM
REM      this one          39.4 / 42.6 / 42.6 tok/s
REM      serve-dual.bat    24.9 / 25.7 / 25.7 tok/s
REM                        +63.1 percent [+58.3, +65.6], baseline spread 3.3 pct
REM
REM  AND IT COSTS ALMOST NOTHING TO GET THERE, which is what makes it different
REM  from serve-dual-dflash.bat. No local patch. No second model downloaded or
REM  loaded. The same llama.cpp-blackwell binary every other launcher uses --
REM  the speculative head is inside the model file. It even finishes a large
REM  request with MORE room than the default: about 2,395 MiB against 2,010.
REM
REM  WHAT IT DOES CHANGE IS THE MODEL FILE, and that is the whole reason this
REM  is a separate icon instead of the default:
REM
REM      QUALITY HAS NOT BEEN MEASURED. Not here and not anywhere in this
REM      project -- no artifact it serves has ever had its output quality
REM      measured. What IS measured is that the n-gram decoder's acceptance
REM      falls from 55.4 to 22.1 on this file, which means it writes text the
REM      predictor cannot anticipate. That is evidence it writes DIFFERENTLY.
REM      Whether differently is worse is exactly what nobody knows.
REM
REM  So: click this to find out on your own work. serve-dual.bat is the one
REM  whose output you already trust.
REM
REM  TWO NUMBERS IN HERE ARE NOT PREFERENCES.
REM
REM  The n-gram runs at n-match 24, not the 12 every other profile serves. 12
REM  won on the Q4 artifact; on this one 24 is worth about a third more, and 24
REM  is the value that LOST on the Q4 at this same depth. The tuning belongs to
REM  the file, not to the depth.
REM
REM  The window is 147,456 and the ceiling is 229,376 -- measured by pushing a
REM  65,643-token request through it, which finished with 846 and 526 MiB free.
REM  262,144 does not come up at all. This launcher therefore does NOT ask
REM  for the deepest window that fits, the way the deep launchers do: that
REM  is the wrong question at an edge where the rung above the answer can
REM  still pass a health check before dying on the first real request.
REM
REM  IMAGES WORK HERE. The vision tower is loaded, so pasting a screenshot to
REM  an agent does what you expect. Without it the server answers HTTP 500 to
REM  every image -- "image input is not supported" -- which is what a real
REM  Claude Code session hit five times on 2026-08-29. The model was never the
REM  limitation: it is a native vision-language model and its own chat template
REM  handles images. The tower is just a separate 888 MiB file.
REM
REM  It was expected to fail. The tower is a second model and this split had
REM  never hosted one -- the DFlash2 drafter aborts inside llama.cpp for exactly
REM  that reason. It does not fail: measured on this same unpatched binary, it
REM  loaded and answered a real 512x512 picture correctly.
REM
REM  IT COSTS HEADROOM, NOT WINDOW. The context is 147,456 either way. A large
REM  request finishes with about 1,205 and 2,450 MiB free, against roughly 2,395
REM  without the tower. The profile counts it before starting and refuses rather
REM  than spilling, so a busy desktop stops it instead of costing you 85x.
REM
REM  serve-dual-nvfp4-deep.bat does NOT carry it, on purpose: 200,704 answered a
REM  SMALL picture and finished with 614 MiB free, and this project has measured
REM  a run dying with 336 and surviving with 488. Images beside a large text
REM  prompt have not been measured at any depth.
REM
REM  It holds no configuration. The flags live in
REM  qwen38-tuning\scripts\worker-q4-dual.ps1 and only there.
REM
REM  %~dp0 is this file's own folder, and this file lives in launchers\,
REM  so the paths below climb one level to reach the repository root. %CD% is not it when the file is opened
REM  from a shortcut or from a shell that started somewhere else.
REM
REM  THIS ONE BINDS 0.0.0.0. The server has NO API key, CORS '*', and no auth
REM  of any kind -- anyone who can reach this machine can use it and read
REM  every prompt. Only on a network you control.
REM ============================================================================

setlocal
cd /d "%~dp0.."

where pwsh >nul 2>nul
if errorlevel 1 (
    echo.
    echo PowerShell 7 ^(pwsh^) was not found, and this needs it.
    echo Install it with:  winget install Microsoft.PowerShell
    echo.
    pause
    exit /b 1
)

pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\serve.ps1" -Dual -Nvfp4 -Vision -Lan
set RC=%ERRORLEVEL%

if not "%RC%"=="0" (
    echo.
    echo serve.ps1 exited with code %RC%.
    pause
)
