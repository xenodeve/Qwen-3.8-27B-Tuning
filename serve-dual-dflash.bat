@echo off
REM  Qwen3.8-27B UD-Q4_K_XL on BOTH cards, with the DFlash2 drafter.
REM
REM  WHAT YOU GET. draft-dflash beside ngram-mod measured +123.8 percent over
REM  the ngram-mod the other dual launchers serve -- 65.1 / 64.3 / 63.8 tok/s
REM  against 29.0 / 29.0 / 28.4, three paired rounds on real vendor code at
REM  ctx 65,536. More than double the decode.
REM
REM  WHAT IT COSTS, and all three are the reason this is its own icon.
REM
REM  1. A DIFFERENT BINARY. It runs llama.cpp-mirror, built from a local patch
REM     that mirrors the model's output projection. Without it the drafter
REM     aborts: TOP_K cannot read logits that are split across two cards. The
REM     patch has been reviewed by nobody outside this project.
REM
REM  2. A SHALLOWER WINDOW -- 131,072 tokens, against about 250,000 from
REM     serve-dual.bat. That is not a budget the launcher can stretch: 147,456
REM     LOADS, answers a health check, and then dies on the first real request.
REM
REM  3. ALMOST ALL THE HEADROOM. It finishes a large request with roughly 600
REM     MiB free per card, against about 2,210 for the served configuration.
REM     A run measured here died with 336 MiB free and survived with 488.
REM
REM  So: click this when you know the work is under 131,072 tokens and you want
REM  the speed. Click serve-dual.bat when you want the window.
REM
REM  %~dp0 is this file's own folder. Double-clicking from a shortcut does not
REM  put %CD% here, and a relative path would resolve against the wrong place.
REM
REM  This one binds 127.0.0.1 only. Nothing outside this machine reaches it.
REM
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0serve.ps1" -Dual -Dflash
if errorlevel 1 pause
